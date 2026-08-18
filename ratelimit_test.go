package lingo

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	cohereCore "github.com/cohere-ai/cohere-go/v2/core"
	"github.com/gerdou/lingo/internal/perplexity"
	"google.golang.org/genai"
)

// ============================================================================
// RATE LIMIT RETRY
// ============================================================================
//
// Two promises are pinned here. The first is that Execute always calls the
// closure it was handed: every provider assigns its response inside that
// closure and reads it back the moment Execute returns nil, so a nil error
// beside an unassigned response is not a failure, it is a nil dereference in
// the caller's own package. The second is that a status a provider had to
// classify by hand is classified where the retry can still see it -- which
// only Ollama does, being the one provider driving raw HTTP.

// fastRetries is a retry policy whose backoff is short enough to run in a test
// while still exercising the growth and the cap.
func fastRetries(maxRetries int) *RateLimitConfig {
	return &RateLimitConfig{
		MaxRetries:        maxRetries,
		InitialBackoff:    time.Millisecond,
		MaxBackoff:        5 * time.Millisecond,
		BackoffMultiplier: 2.0,
	}
}

// TestRateLimiterAlwaysCallsFn covers the retry counts a caller can express,
// including the negative one that asks for retries off. Zero is spoken for --
// newRateLimiter reads it as "unset" and applies the default of 3 -- so a
// negative is the only way left to say it, and it must mean one attempt, never
// none.
func TestRateLimiterAlwaysCallsFn(t *testing.T) {
	tests := []struct {
		name       string
		maxRetries int
		wantCalls  int
	}{
		{"retries off via -1", -1, 1},
		{"retries off via a larger negative", -7, 1},
		{"two retries", 2, 3},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := newRateLimiter(fastRetries(tt.maxRetries), &NopLogger{})

			calls := 0
			err := r.Execute(context.Background(), func() error {
				calls++
				return &HTTPStatusError{StatusCode: http.StatusTooManyRequests, Message: "rate limit"}
			})

			if calls != tt.wantCalls {
				t.Errorf("closure called %d times, want %d", calls, tt.wantCalls)
			}
			// A closure that only ever failed must not report success: that is
			// the shape callers turn into a nil dereference.
			if err == nil {
				t.Fatal("Execute returned nil after the closure only ever failed")
			}
		})
	}
}

// TestRateLimiterNegativeRetriesThroughGateway is the caller-visible half of
// the same guarantee: a negative MaxRetries reaches a provider through a plain
// exported config field, and the request must simply be made once.
func TestRateLimiterNegativeRetriesThroughGateway(t *testing.T) {
	var c capture
	srv := oaiStub(t, &c)
	defer srv.Close()

	resp := generate(t,
		&OpenAICompatibleConfig{BaseURL: srv.URL, APIKey: "k", RateLimiter: &RateLimitConfig{MaxRetries: -1}},
		NewOpenAICompatibleModel("llama-3.3-70b"))

	if resp.Text != "hi there" {
		t.Errorf("text = %q", resp.Text)
	}
}

// TestIsRateLimitErrorClassifiedByType keeps HTTPStatusError's own verdict
// ahead of the prose scan, so a status carries the decision rather than the
// wording of whatever message happens to be wrapped around it.
func TestIsRateLimitErrorClassifiedByType(t *testing.T) {
	tests := []struct {
		name string
		err  error
		want bool
	}{
		{"429 with no telling words", &HTTPStatusError{StatusCode: http.StatusTooManyRequests, Message: "slow down"}, true},
		{"400 with no telling words", &HTTPStatusError{StatusCode: http.StatusBadRequest, Message: "bad model"}, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := isRateLimitError(tt.err); got != tt.want {
				t.Errorf("isRateLimitError = %v, want %v", got, tt.want)
			}
		})
	}
}

// ollamaStatusStub answers the first failures requests with status, then
// serves the canned chat response. It counts every request it saw.
func ollamaStatusStub(t *testing.T, status, failures int, requests *int64) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.ReadAll(r.Body)
		n := atomic.AddInt64(requests, 1)

		if int(n) <= failures {
			w.WriteHeader(status)
			_, _ = io.WriteString(w, `{"error":"server busy"}`)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"model":"qwen3","created_at":"2026-08-17T00:00:00Z",
			"message":{"role":"assistant","content":"hi there"},
			"done":true,"done_reason":"stop",
			"prompt_eval_count":11,"eval_count":7}`)
	}))
}

// TestOllamaRetriesRateLimitedStatus is the regression: net/http hands back
// (resp, nil) for a 429, so a status inspected after Execute returned could
// never be retried and OllamaConfig.RateLimiter was inert.
func TestOllamaRetriesRateLimitedStatus(t *testing.T) {
	var requests int64
	srv := ollamaStatusStub(t, http.StatusTooManyRequests, 2, &requests)
	defer srv.Close()

	resp := generate(t,
		&OllamaConfig{BaseURL: srv.URL, RateLimiter: fastRetries(3)},
		NewQwen3())

	if got := atomic.LoadInt64(&requests); got != 3 {
		t.Errorf("requests = %d, want 3 (two 429s retried, third served)", got)
	}
	if resp.Text != "hi there" {
		t.Errorf("text = %q", resp.Text)
	}
}

// TestOllamaExhaustsRetriesOnStatus pins what a 429 that never clears looks
// like from the outside: the retries are spent, and the message an operator
// reads is the one this provider has always produced for a bad status.
func TestOllamaExhaustsRetriesOnStatus(t *testing.T) {
	var requests int64
	srv := ollamaStatusStub(t, http.StatusTooManyRequests, 100, &requests)
	defer srv.Close()

	g, err := New([]ProviderConfig{&OllamaConfig{BaseURL: srv.URL, RateLimiter: fastRetries(2)}})
	if err != nil {
		t.Fatalf("gateway: %v", err)
	}
	defer g.Close()

	_, err = g.Generate(context.Background(), NewQwen3(), "hello")
	if err == nil {
		t.Fatal("Generate returned no error for a 429 that never cleared")
	}
	if got := atomic.LoadInt64(&requests); got != 3 {
		t.Errorf("requests = %d, want 3 (one attempt plus two retries)", got)
	}
	if want := "ollama API error: status 429, body: "; !strings.Contains(err.Error(), want) {
		t.Errorf("err = %q, want it to contain %q", err.Error(), want)
	}
}

// TestOllamaDoesNotRetryClientError keeps the retry narrow: a 400 is the
// server saying the request is wrong, and repeating it is only slower.
func TestOllamaDoesNotRetryClientError(t *testing.T) {
	var requests int64
	srv := ollamaStatusStub(t, http.StatusBadRequest, 100, &requests)
	defer srv.Close()

	g, err := New([]ProviderConfig{&OllamaConfig{BaseURL: srv.URL, RateLimiter: fastRetries(3)}})
	if err != nil {
		t.Fatalf("gateway: %v", err)
	}
	defer g.Close()

	_, err = g.Generate(context.Background(), NewQwen3(), "hello")
	if err == nil {
		t.Fatal("Generate returned no error for a 400")
	}
	if got := atomic.LoadInt64(&requests); got != 1 {
		t.Errorf("requests = %d, want 1 (a 400 is not retried)", got)
	}
	if want := "ollama API error: status 400"; !strings.Contains(err.Error(), want) {
		t.Errorf("err = %q, want it to contain %q", err.Error(), want)
	}
}

// ============================================================================
// RATE LIMIT CLASSIFICATION
// ============================================================================
//
// Retrying is not free: a generation request is a whole prompt, billed, and
// not idempotent. So the question "is this worth sending again" is answered
// from the status the server stated wherever a status can be read, and only
// from wording where it cannot -- because wording is where a 400 saying the
// prompt "resulted in 142900 tokens" reads as a 429.

// countingStub answers the first failures requests with status and the given
// body, then serves the canned OpenAI completion. It counts every request.
func countingStub(t *testing.T, status, failures int, body string, requests *int64) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.ReadAll(r.Body)
		n := atomic.AddInt64(requests, 1)

		w.Header().Set("Content-Type", "application/json")
		if int(n) <= failures {
			w.WriteHeader(status)
			_, _ = io.WriteString(w, body)
			return
		}
		_, _ = io.WriteString(w, `{
			"id":"c1","model":"served-model","object":"chat.completion",
			"choices":[{"index":0,"finish_reason":"stop",
				"message":{"role":"assistant","content":"hi there"}}],
			"usage":{"prompt_tokens":11,"completion_tokens":7,"total_tokens":18}}`)
	}))
}

// contextLengthBody is the real shape of an over-long prompt rejection. The
// digits 429 sit inside the token count, which is the whole trouble: the SDK
// puts this body verbatim into Error().
const contextLengthBody = `{"error":{"message":"This model's maximum context length is 128000 tokens. However, your messages resulted in 142900 tokens. Please reduce the length of the messages.","type":"invalid_request_error","param":"messages","code":"context_length_exceeded"}}`

// TestNoRetryOnDeterministicClientError is the regression: a 400 can never
// become a 200 by being sent again, and sending it again costs a full prompt
// per attempt.
func TestNoRetryOnDeterministicClientError(t *testing.T) {
	var requests int64
	srv := countingStub(t, http.StatusBadRequest, 100, contextLengthBody, &requests)
	defer srv.Close()

	g, err := New([]ProviderConfig{&OpenAICompatibleConfig{
		BaseURL: srv.URL, APIKey: "k", RateLimiter: fastRetries(3),
	}})
	if err != nil {
		t.Fatalf("gateway: %v", err)
	}
	defer g.Close()

	_, err = g.Generate(context.Background(), NewOpenAICompatibleModel("llama-3.3-70b"), "hello")
	if err == nil {
		t.Fatal("Generate returned no error for a 400")
	}
	if got := atomic.LoadInt64(&requests); got != 1 {
		t.Errorf("requests = %d, want 1 (a 400 is the server's verdict, not a wait)", got)
	}
}

// anthropicError builds the SDK error type the Anthropic client returns, well
// enough formed that its Error() can be called: it dereferences Request and
// Response unconditionally.
func anthropicError(t *testing.T, status int) *anthropic.Error {
	t.Helper()
	req, err := http.NewRequest(http.MethodPost, "https://api.anthropic.com/v1/messages", nil)
	if err != nil {
		t.Fatalf("request: %v", err)
	}
	return &anthropic.Error{
		StatusCode: status,
		Request:    req,
		Response:   &http.Response{StatusCode: status},
	}
}

// TestIsRateLimitErrorReadsStatusNotProse walks the typed errors every provider
// hands back. Each provider spells the status differently and none of them
// share a method, so each one is pinned here: a 429 is retried whatever the
// message says, and a 4xx that merely contains the digits 429 is not.
func TestIsRateLimitErrorReadsStatusNotProse(t *testing.T) {
	tooLong := "maximum context length is 128000 tokens, your messages resulted in 142900 tokens"

	tests := []struct {
		name string
		err  error
		want bool
	}{
		{"cohere 429", cohereCore.NewAPIError(http.StatusTooManyRequests, nil, errors.New("slow down")), true},
		{"cohere 400 counting 142900 tokens", cohereCore.NewAPIError(http.StatusBadRequest, nil, errors.New(tooLong)), false},

		{"genai 429", genai.APIError{Code: http.StatusTooManyRequests, Message: "Resource has been exhausted"}, true},
		{"genai 400 counting 142900 tokens", genai.APIError{Code: http.StatusBadRequest, Message: tooLong}, false},

		{"perplexity 429", &perplexity.APIError{StatusCode: http.StatusTooManyRequests, Message: "slow down"}, true},
		{"perplexity 400 counting 142900 tokens", &perplexity.APIError{StatusCode: http.StatusBadRequest, Message: tooLong}, false},

		{"anthropic 529 overloaded", anthropicError(t, 529), true},
		{"anthropic 404", anthropicError(t, http.StatusNotFound), false},

		{"503 service unavailable", &HTTPStatusError{StatusCode: http.StatusServiceUnavailable, Message: "try later"}, true},
		{"404 not found", &HTTPStatusError{StatusCode: http.StatusNotFound, Message: "no such model ft:gpt-4o:acme:run-4299"}, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := isRateLimitError(tt.err); got != tt.want {
				t.Errorf("isRateLimitError = %v, want %v", got, tt.want)
			}
		})
	}
}

// TestIsRateLimitErrorProseFallback covers what is left once no status can be
// read: a transport failure or a gateway that answered in prose. The number
// still counts, but only standing on its own -- inside a longer number it is
// a token count, a limit, or part of a fine-tune id.
func TestIsRateLimitErrorProseFallback(t *testing.T) {
	tests := []struct {
		name string
		err  error
		want bool
	}{
		{"a bare 429 in a proxy message", errors.New("upstream returned status 429, body: too busy"), true},
		{"429 at the end of the message", errors.New("nginx said 429"), true},
		{"the words", errors.New("Rate limit reached for gpt-4o"), true},
		{"throttled", errors.New("ThrottlingException: rate exceeded"), true},
		{"429 inside a token count", errors.New("your messages resulted in 142900 tokens"), false},
		{"429 inside a limit value", errors.New("max_tokens must be <= 4296"), false},
		{"429 inside a model id", errors.New("model ft:gpt-4o:acme:run-4299 does not exist"), false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := isRateLimitError(tt.err); got != tt.want {
				t.Errorf("isRateLimitError = %v, want %v", got, tt.want)
			}
		})
	}
}

// ============================================================================
// NESTED RETRY
// ============================================================================
//
// Three of the vendored SDKs retry on their own: openai-go and anthropic-sdk-go
// default to three attempts, aws-sdk-go-v2 to three. Those used to multiply
// with lingo's loop rather than add to it -- three inside each of lingo's four
// is twelve upstream requests for one Generate -- and the extra delay was the
// SDK's own backoff, which RateLimitConfig cannot reach, spent inside a timeout
// every provider applies once around the whole sequence. The tests below pin
// the split: lingo retries the statuses RateLimitConfig governs, the SDK keeps
// the ones lingo never retries at all.

// rateLimitBody is the 429 an OpenAI-dialect endpoint returns.
const rateLimitBody = `{"error":{"message":"Rate limit reached for requests","type":"rate_limit_error","code":"rate_limit_exceeded"}}`

// TestOpenAICompatibleRetriesAreNotStacked is the regression: with the SDK
// retrying underneath, four lingo attempts sent twelve requests and burned the
// caller's timeout on backoff no configuration could shorten.
func TestOpenAICompatibleRetriesAreNotStacked(t *testing.T) {
	var requests int64
	srv := countingStub(t, http.StatusTooManyRequests, 100, rateLimitBody, &requests)
	defer srv.Close()

	g, err := New([]ProviderConfig{&OpenAICompatibleConfig{
		BaseURL: srv.URL, APIKey: "k", RateLimiter: fastRetries(3),
	}})
	if err != nil {
		t.Fatalf("gateway: %v", err)
	}
	defer g.Close()

	_, err = g.Generate(context.Background(), NewOpenAICompatibleModel("llama-3.3-70b"), "hello")
	if err == nil {
		t.Fatal("Generate returned no error for a 429 that never cleared")
	}
	if got := atomic.LoadInt64(&requests); got != 4 {
		t.Errorf("requests = %d, want 4 (one attempt plus the three retries asked for)", got)
	}
	// The caller still learns what happened, and learns it from the status.
	if !strings.Contains(err.Error(), "429") {
		t.Errorf("err = %q, want it to name the 429", err.Error())
	}
}

// TestSDKKeepsRetryingWhatLingoDoesNot is the other half of the split, and the
// guard against "fix" it by turning the SDK's retries off wholesale: a 500 is
// not in retryableStatuses, so lingo sends it once and the SDK's own three
// attempts are the only ones there are.
func TestSDKKeepsRetryingWhatLingoDoesNot(t *testing.T) {
	var requests int64
	srv := countingStub(t, http.StatusInternalServerError, 100,
		`{"error":{"message":"upstream exploded","type":"server_error"}}`, &requests)
	defer srv.Close()

	g, err := New([]ProviderConfig{&OpenAICompatibleConfig{
		BaseURL: srv.URL, APIKey: "k", RateLimiter: fastRetries(3),
	}})
	if err != nil {
		t.Fatalf("gateway: %v", err)
	}
	defer g.Close()

	_, err = g.Generate(context.Background(), NewOpenAICompatibleModel("llama-3.3-70b"), "hello")
	if err == nil {
		t.Fatal("Generate returned no error for a 500")
	}
	if got := atomic.LoadInt64(&requests); got != 3 {
		t.Errorf("requests = %d, want 3 (the SDK's own attempts, and no lingo retry on top)", got)
	}
}

// anthropicRateLimitStub points the pinned anthropic-sdk-go at a server that
// only ever answers 429, through the same ANTHROPIC_BASE_URL seam the caching
// tests use, and counts every request that reaches it.
func anthropicRateLimitStub(t *testing.T, requests *int64) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.ReadAll(r.Body)
		atomic.AddInt64(requests, 1)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = io.WriteString(w, `{"type":"error","error":{"type":"rate_limit_error","message":"rate limit"}}`)
	}))
	t.Setenv("ANTHROPIC_BASE_URL", srv.URL)
	t.Cleanup(srv.Close)
	return srv
}

// TestAnthropicRetriesAreNotStacked is the same regression on the other
// Stainless SDK, which shares the header convention but not the client.
func TestAnthropicRetriesAreNotStacked(t *testing.T) {
	var requests int64
	anthropicRateLimitStub(t, &requests)

	g, err := New([]ProviderConfig{&AnthropicConfig{APIKey: "k", RateLimiter: fastRetries(3)}})
	if err != nil {
		t.Fatalf("gateway: %v", err)
	}
	defer g.Close()

	_, err = g.Generate(context.Background(), NewClaudeSonnet5(), "hello")
	if err == nil {
		t.Fatal("Generate returned no error for a 429 that never cleared")
	}
	if got := atomic.LoadInt64(&requests); got != 4 {
		t.Errorf("requests = %d, want 4 (one attempt plus the three retries asked for)", got)
	}
}

// TestBedrockRetriesAreNotStacked is the AWS half, where the split is a
// replaced retryer rather than a response header. The stub answers the
// throttling shape the SDK's own retry rules key on, so a test that passes is
// a test where those rules were consulted and declined.
func TestBedrockRetriesAreNotStacked(t *testing.T) {
	var requests int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.ReadAll(r.Body)
		atomic.AddInt64(&requests, 1)

		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("X-Amzn-Errortype", "ThrottlingException")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = io.WriteString(w, `{"__type":"ThrottlingException","message":"Too many requests"}`)
	}))

	c := bedrockClientFor(t, srv)
	c.rateLimiter = newRateLimiter(fastRetries(3), &NopLogger{})

	_, err := c.Generate(context.Background(), NewBedrockClaudeSonnet5(), "hello")
	if err == nil {
		t.Fatal("Generate returned no error for a throttle that never cleared")
	}
	if got := atomic.LoadInt64(&requests); got != 4 {
		t.Errorf("requests = %d, want 4 (one attempt plus the three retries asked for)", got)
	}
}

// TestRetryDeadlineKeepsTheCause is the second half of the same finding. The
// timeout wraps the whole retry sequence, so a deadline that lands during a
// backoff used to replace the 429 with a bare "context deadline exceeded" and
// send whoever read it looking at their network.
func TestRetryDeadlineKeepsTheCause(t *testing.T) {
	var requests int64
	srv := countingStub(t, http.StatusTooManyRequests, 100, rateLimitBody, &requests)
	defer srv.Close()

	g, err := New([]ProviderConfig{&OpenAICompatibleConfig{
		BaseURL: srv.URL,
		APIKey:  "k",
		// Generous next to a local httptest round trip, so the deadline can
		// only be reached through the backoff below, never through the request.
		Timeout: 500 * time.Millisecond,
		// A backoff far longer than the timeout, so the deadline is certain to
		// land in the wait after the first 429 rather than between attempts.
		RateLimiter: &RateLimitConfig{
			MaxRetries:        3,
			InitialBackoff:    10 * time.Second,
			MaxBackoff:        10 * time.Second,
			BackoffMultiplier: 2.0,
		},
	}})
	if err != nil {
		t.Fatalf("gateway: %v", err)
	}
	defer g.Close()

	_, err = g.Generate(context.Background(), NewOpenAICompatibleModel("llama-3.3-70b"), "hello")
	if err == nil {
		t.Fatal("Generate returned no error when the deadline passed")
	}
	// The deadline is still the deadline: anything switching on it keeps working.
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("err = %q, want it to wrap context.DeadlineExceeded", err.Error())
	}
	// And the reason the deadline was reached is in the message.
	if !strings.Contains(err.Error(), "429") {
		t.Errorf("err = %q, want it to name the 429 that caused the waiting", err.Error())
	}
	if got := atomic.LoadInt64(&requests); got != 1 {
		t.Errorf("requests = %d, want 1 (the deadline landed in the first backoff)", got)
	}
}
