package lingo

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"regexp"
	"strconv"
	"sync"
	"testing"
	"time"
)

var bedrockTokenHeaders = map[string]string{
	"X-Amzn-Bedrock-Input-Token-Count":  "42",
	"X-Amzn-Bedrock-Output-Token-Count": "7",
}

// Mistral's InvokeModel body is documented as {"outputs":[{"text","stop_reason"}]}
// and nothing else, so the headers are the only place its counts exist.
func TestMistralReportsTokenCountsFromResponseHeaders(t *testing.T) {
	var calls bedrockCalls
	c := bedrockStub(t, &calls, bedrockTokenHeaders)

	resp, err := c.Generate(context.Background(), NewBedrockMistralLarge2407(), "hello")
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	if resp.Usage.PromptTokens != 42 || resp.Usage.CompletionTokens != 7 || resp.Usage.TotalTokens != 49 {
		t.Errorf("usage = %+v, want 42/7/49", resp.Usage)
	}
	if resp.Metadata["usage_source"] != "response_headers" {
		t.Errorf("usage_source = %q", resp.Metadata["usage_source"])
	}
	// The bytes on the wire are unchanged: reading a response header cannot
	// move a request, and this is the check that says so.
	want := `{"prompt":"\u003cs\u003e[INST] hello [/INST]","max_tokens":8192,"temperature":0.7,"top_p":0.9}`
	if calls.len() != 1 || calls.at(t, 0).body != want {
		t.Errorf("request body =\n\t%s\nwant\n\t%s", calls.at(t, 0).body, want)
	}
}

// The headers are unmodelled, so absence is a state lingo has to survive.
func TestMistralWithoutTokenHeadersKeepsReportingZero(t *testing.T) {
	var calls bedrockCalls
	c := bedrockStub(t, &calls)

	resp, err := c.Generate(context.Background(), NewBedrockMistral7B(), "hello")
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	if resp.Usage != (TokenUsage{}) {
		t.Errorf("usage = %+v, want the zero value when nothing reported", resp.Usage)
	}
	if _, ok := resp.Metadata["usage_source"]; ok {
		t.Error("usage_source names a source that did not report")
	}
}

// Blast radius: a count a model reported for itself is the count lingo reports,
// however loudly the headers disagree. The headers fill gaps and never override,
// so the only row here whose numbers may move is the one whose body left a gap.
func TestBodyCountsAreNeverOverriddenByTheHeaders(t *testing.T) {
	loud := map[string]string{
		"X-Amzn-Bedrock-Input-Token-Count":  "999",
		"X-Amzn-Bedrock-Output-Token-Count": "888",
	}
	for _, tc := range []struct {
		name             string
		model            Model
		prompt, complete int
		total            int
		source           string
	}{
		{"claude", NewBedrockClaudeSonnet5(), 10, 7, 17, ""},
		{"llama", NewBedrockLlama33Instruct70B(), 10, 7, 17, ""},
		// Titan is the one InvokeModel family whose body is incomplete: it
		// reports a completion count and, whenever inputTextTokenCount is
		// missing as it is in this stub, no prompt count at all. The header
		// fills that one gap; the body's own 7 stands against the header's 888.
		// This row used to pin the missing prompt count as an expected 0, which
		// recorded the bug as the contract.
		{"titan", NewBedrockTitanTextPremier(), 999, 7, 1006, "body+response_headers"},
		// Nova is served by Converse, not InvokeModel, but the middleware is
		// registered on the shared client and so runs there too. Its numbers
		// come from the Converse usage object and must stay that way: 1000 is
		// the stub's inputTokens 100 plus the 900 cacheReadInputTokens that
		// withCache folds into the prompt total, which is what Nova reported
		// before this change and still reports after it.
		{"nova", NewBedrockNovaPro(), 1000, 7, 1007, ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var calls bedrockCalls
			c := bedrockStub(t, &calls, loud)
			resp, err := c.Generate(context.Background(), tc.model, "hello")
			if err != nil {
				t.Fatalf("generate: %v", err)
			}
			if resp.Usage.PromptTokens != tc.prompt || resp.Usage.CompletionTokens != tc.complete ||
				resp.Usage.TotalTokens != tc.total {
				t.Errorf("usage = %+v, want %d/%d/%d", resp.Usage, tc.prompt, tc.complete, tc.total)
			}
			if got := resp.Metadata["usage_source"]; got != tc.source {
				t.Errorf("usage_source = %q, want %q", got, tc.source)
			}
		})
	}
}

// The headers are unmodelled, so what arrives in them is whatever arrives.
// Nothing a header can say is allowed to fail a generation, and a value that
// cannot be read is "nothing was reported", never a zero someone could bill on.
func TestMistralSurvivesUnusableTokenHeaders(t *testing.T) {
	for _, tc := range []struct {
		name             string
		headers          map[string]string
		prompt, complete int
		sourced          bool
	}{{
		name:    "unparseable",
		headers: map[string]string{"X-Amzn-Bedrock-Input-Token-Count": "lots", "X-Amzn-Bedrock-Output-Token-Count": ""},
	}, {
		name:    "negative",
		headers: map[string]string{"X-Amzn-Bedrock-Input-Token-Count": "-1", "X-Amzn-Bedrock-Output-Token-Count": "-7"},
	}, {
		// Half a report is still a report: the count that arrived is the count
		// that gets used, and the one that did not stays zero.
		name:    "only the input count",
		headers: map[string]string{"X-Amzn-Bedrock-Input-Token-Count": "42"},
		prompt:  42, sourced: true,
	}, {
		name:    "padded",
		headers: map[string]string{"X-Amzn-Bedrock-Input-Token-Count": " 42 ", "X-Amzn-Bedrock-Output-Token-Count": "7"},
		prompt:  42, complete: 7, sourced: true,
	}} {
		t.Run(tc.name, func(t *testing.T) {
			var calls bedrockCalls
			c := bedrockStub(t, &calls, tc.headers)

			resp, err := c.Generate(context.Background(), NewBedrockMistralLarge2407(), "hello")
			if err != nil {
				t.Fatalf("generate: %v", err)
			}
			if resp.Usage.PromptTokens != tc.prompt || resp.Usage.CompletionTokens != tc.complete ||
				resp.Usage.TotalTokens != tc.prompt+tc.complete {
				t.Errorf("usage = %+v, want %d/%d/%d", resp.Usage, tc.prompt, tc.complete, tc.prompt+tc.complete)
			}
			if _, ok := resp.Metadata["usage_source"]; ok != tc.sourced {
				t.Errorf("usage_source present = %v, want %v", ok, tc.sourced)
			}
		})
	}
}

// ============================================================================
// TITAN: A BODY THAT REPORTS HALF
// ============================================================================
//
// Titan's InvokeModel response reports the completion count inside results[]
// and the prompt count beside them as inputTextTokenCount. Lingo modelled only
// the first, so every Titan call came back with PromptTokens 0 and a TotalTokens
// that was really the completion count -- while the middleware was capturing the
// real prompt count from the response headers and throwing it away.

// bedrockCannedStub answers every request with one body and one header set.
// bedrockStub picks its body by family and cannot be told to vary it, which is
// exactly what a test about the response rather than the request needs.
func bedrockCannedStub(t *testing.T, body string, headers map[string]string) *bedrockClient {
	t.Helper()
	return bedrockClientFor(t, httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		for k, v := range headers {
			w.Header().Set(k, v)
		}
		_, _ = io.WriteString(w, body)
	})))
}

// TestTitanFillsItsMissingPromptCountFromTheHeaders is the finding: the body
// reports the completion count and no prompt count, and the header supplies the
// one that is missing without touching the one that is not.
func TestTitanFillsItsMissingPromptCountFromTheHeaders(t *testing.T) {
	var calls bedrockCalls
	c := bedrockStub(t, &calls, map[string]string{
		"X-Amzn-Bedrock-Input-Token-Count": "42",
		// Deliberately not the body's 7: the body still wins for the count it
		// does report.
		"X-Amzn-Bedrock-Output-Token-Count": "888",
	})

	resp, err := c.Generate(context.Background(), NewBedrockTitanTextPremier(), "hello")
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	want := TokenUsage{PromptTokens: 42, CompletionTokens: 7, TotalTokens: 49}
	if resp.Usage != want {
		t.Errorf("usage = %+v, want %+v", resp.Usage, want)
	}
	if got := resp.Metadata["usage_source"]; got != "body+response_headers" {
		t.Errorf("usage_source = %q, want body+response_headers", got)
	}
	// Reading a response header cannot move a request.
	want2 := `{"inputText":"hello","textGenerationConfig":{"maxTokenCount":4096,"temperature":0.7,"topP":0.9}}`
	if calls.len() != 1 || calls.at(t, 0).body != want2 {
		t.Errorf("request body =\n\t%s\nwant\n\t%s", calls.at(t, 0).body, want2)
	}
}

// TestTitanPrefersItsOwnBodyCounts is the other half of the rule. A real Titan
// response does carry inputTextTokenCount, and where the model reported for
// itself the unmodelled header is not consulted at all.
func TestTitanPrefersItsOwnBodyCounts(t *testing.T) {
	c := bedrockCannedStub(t,
		`{"inputTextTokenCount":11,"results":[{"outputText":"hi there","completionReason":"FINISH","tokenCount":7}]}`,
		map[string]string{
			"X-Amzn-Bedrock-Input-Token-Count":  "999",
			"X-Amzn-Bedrock-Output-Token-Count": "888",
		})

	resp, err := c.Generate(context.Background(), NewBedrockTitanTextPremier(), "hello")
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	want := TokenUsage{PromptTokens: 11, CompletionTokens: 7, TotalTokens: 18}
	if resp.Usage != want {
		t.Errorf("usage = %+v, want %+v from the body", resp.Usage, want)
	}
	if _, ok := resp.Metadata["usage_source"]; ok {
		t.Error("a fully body-reported usage must not be relabelled")
	}
}

// The headers are unmodelled, so Titan has to survive their absence and their
// nonsense the same way Mistral does: with the numbers its own body gave and no
// error.
func TestTitanSurvivesMissingAndUnusableHeaders(t *testing.T) {
	for _, tc := range []struct {
		name    string
		headers map[string]string
	}{
		{"absent", nil},
		{"unparseable", map[string]string{"X-Amzn-Bedrock-Input-Token-Count": "lots"}},
		{"negative", map[string]string{"X-Amzn-Bedrock-Input-Token-Count": "-1"}},
		{"empty", map[string]string{"X-Amzn-Bedrock-Input-Token-Count": ""}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var calls bedrockCalls
			c := bedrockStub(t, &calls, tc.headers)

			resp, err := c.Generate(context.Background(), NewBedrockTitanTextPremier(), "hello")
			if err != nil {
				t.Fatalf("generate: %v", err)
			}
			// The stub's body has no inputTextTokenCount, so an unreported
			// prompt count stays zero rather than becoming a number nobody
			// said.
			want := TokenUsage{CompletionTokens: 7, TotalTokens: 7}
			if resp.Usage != want {
				t.Errorf("usage = %+v, want %+v", resp.Usage, want)
			}
			if _, ok := resp.Metadata["usage_source"]; ok {
				t.Error("usage_source names a source that did not report")
			}
		})
	}
}

// ============================================================================
// CONCURRENCY
// ============================================================================

// TestBedrockTokenHeadersDoNotLeakAcrossConcurrentRequests is the guard on the
// one thing a middleware that stashes per-request state can get catastrophically
// wrong: billing one caller for another caller's tokens.
//
// The counts are derived from the prompt, so every response carries the numbers
// that belong to the request that asked for it and a leak is a mismatch rather
// than a flake. Run with -race, the same test also covers the middleware's
// writes to middleware.Metadata.
func TestBedrockTokenHeadersDoNotLeakAcrossConcurrentRequests(t *testing.T) {
	seq := regexp.MustCompile(`seq-(\d+)-end`)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		n, _ := strconv.Atoi(string(seq.FindSubmatch(raw)[1]))

		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("X-Amzn-Bedrock-Input-Token-Count", strconv.Itoa(n))
		w.Header().Set("X-Amzn-Bedrock-Output-Token-Count", strconv.Itoa(2*n))
		// A little jitter, so the responses do not come back in the order the
		// requests went out.
		time.Sleep(time.Duration(n%5) * time.Millisecond)
		_, _ = io.WriteString(w, `{"outputs":[{"text":"hi there","stop_reason":"stop"}]}`)
	}))
	c := bedrockClientFor(t, srv)

	const calls = 32
	var wg sync.WaitGroup
	problems := make(chan string, calls)
	for i := 1; i <= calls; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			resp, err := c.Generate(context.Background(), NewBedrockMistralLarge2407(),
				fmt.Sprintf("seq-%d-end", i))
			if err != nil {
				problems <- fmt.Sprintf("request %d: %v", i, err)
				return
			}
			want := TokenUsage{PromptTokens: i, CompletionTokens: 2 * i, TotalTokens: 3 * i}
			if resp.Usage != want {
				problems <- fmt.Sprintf("request %d saw usage %+v, want %+v -- another request's counts",
					i, resp.Usage, want)
			}
		}(i)
	}
	wg.Wait()
	close(problems)
	for p := range problems {
		t.Error(p)
	}
}
