package lingo

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gerdou/lingo/internal/perplexity"
)

// ============================================================================
// PERPLEXITY ERROR BODIES
// ============================================================================
//
// Perplexity has no Go SDK, so lingo owns the error path end to end. The rule
// pinned here is that whatever the API said about a failure survives to the
// caller: a body only ever loses its text when it has none. json.Unmarshal
// succeeding is not evidence the shape matched -- ErrorResponse has a single
// "error" key, so every other object decodes into the zero value just as
// happily -- and the empty message that came of trusting it left an operator
// staring at a bare status code.

// perplexityErrorStub answers every request with status and the given body.
func perplexityErrorStub(t *testing.T, status int, body string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = io.WriteString(w, body)
	}))
}

// perplexityGenerateErr runs one Generate against a stub and returns the error
// it produced. PerplexityConfig has no BaseURL, so the client is built
// directly rather than through the gateway.
func perplexityGenerateErr(t *testing.T, url string) error {
	t.Helper()
	client, err := perplexity.NewClient(perplexity.ClientConfig{
		APIKey: "k", BaseURL: url, Timeout: 5 * time.Second,
	})
	if err != nil {
		t.Fatalf("perplexity client: %v", err)
	}
	logger := &NopLogger{}
	c := &perplexityClient{
		client:      client,
		timeout:     5 * time.Second,
		logger:      logger,
		rateLimiter: newRateLimiter(&RateLimitConfig{MaxRetries: -1}, logger),
	}
	_, err = c.Generate(context.Background(), NewSonar(), "hello")
	if err == nil {
		t.Fatal("Generate returned no error for a non-200 response")
	}
	return err
}

func TestPerplexityErrorBodyReachesTheCaller(t *testing.T) {
	tests := []struct {
		name string
		body string
		// want is text from the body that the caller must still be able to read.
		want string
	}{
		// The shape lingo models. Nothing changes for it.
		{
			"nested error object",
			`{"error":{"message":"Invalid model 'sonar-x'","type":"invalid_request_error","code":"400"}}`,
			"Invalid model 'sonar-x'",
		},
		// The shapes that used to unmarshal cleanly into nothing at all.
		{
			"detail key",
			`{"detail":"Invalid model 'sonar-x'. Permitted models: sonar, sonar-pro"}`,
			"Permitted models: sonar, sonar-pro",
		},
		{
			"detail list",
			`{"detail":[{"loc":["body","model"],"msg":"field required"}]}`,
			"field required",
		},
		{
			"message key",
			`{"message":"upstream connect error"}`,
			"upstream connect error",
		},
		// Not JSON at all -- a proxy or a load balancer answering instead of the
		// API. This path always worked and must keep working.
		{
			"html from a proxy",
			`<html><body><h1>502 Bad Gateway</h1></body></html>`,
			"502 Bad Gateway",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := perplexityErrorStub(t, http.StatusBadRequest, tt.body)
			defer srv.Close()

			err := perplexityGenerateErr(t, srv.URL)
			if !strings.Contains(err.Error(), tt.want) {
				t.Errorf("err = %q, want it to contain %q", err.Error(), tt.want)
			}
		})
	}
}

// TestPerplexityErrorTypeIsKept guards the other side of the same condition:
// a well-formed error object that carries only a type must be reported as
// that type rather than fall back to the raw body.
func TestPerplexityErrorTypeIsKept(t *testing.T) {
	srv := perplexityErrorStub(t, http.StatusTooManyRequests,
		`{"error":{"type":"rate_limit_error"}}`)
	defer srv.Close()

	err := perplexityGenerateErr(t, srv.URL)
	if !strings.Contains(err.Error(), "type rate_limit_error") {
		t.Errorf("err = %q, want it to name the error type", err.Error())
	}
}
