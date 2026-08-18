package lingo

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/gerdou/lingo/internal/perplexity"
)

// ============================================================================
// PERPLEXITY THINKING
// ============================================================================
//
// Perplexity had no thinking surface at all, so the baseline here is simpler
// than Cohere's: every request lingo sent before this change carried no
// reasoning_effort, and every request from a model nobody touched still does
// not. What is new on the read side is not new data -- the trace was always in
// the answer, mixed into it -- but a separation lingo's own doc comments have
// warned about for as long as the reasoning models have been in the catalogue.

// perplexityThinkingStub serves a canned chat completion and records the
// request. The content carries the <think> prefix a reasoning model emits.
func perplexityThinkingStub(t *testing.T, c *capture) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		c.record(r, raw)

		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"id":"p1","model":"sonar-deep-research","object":"chat.completion",
			"choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant",
				"content":"<think>\nthe premise, then the arithmetic\n</think>\n\nhi there"}}],
			"usage":{"prompt_tokens":11,"completion_tokens":50,"total_tokens":61,
				"reasoning_tokens":42}}`)
	}))
}

// perplexityGenerate runs one Generate against a stub. PerplexityConfig has no
// BaseURL, so the client is built directly rather than through the gateway.
func perplexityGenerate(t *testing.T, url string, m Model) *GenerationResponse {
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
		rateLimiter: newRateLimiter(nil, logger),
	}
	resp, err := c.Generate(context.Background(), m, "hello")
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	return resp
}

// perplexityWire runs one Generate against the stub and returns the body.
func perplexityWire(t *testing.T, m Model) map[string]any {
	t.Helper()
	var c capture
	srv := perplexityThinkingStub(t, &c)
	defer srv.Close()
	perplexityGenerate(t, srv.URL, m)
	return c.body
}

// The request a model nobody touched produces must be exactly the request it
// produced before thinking control existed -- including after the six-arm type
// switch that built it became one accessor.
func TestPerplexityUntouchedRequestIsUnchanged(t *testing.T) {
	wantRequestBody(t, perplexityWire(t, NewSonarDeepResearch()),
		`{"max_tokens":16384,"messages":[{"content":"hello","role":"user"}],
		  "model":"sonar-deep-research","temperature":0.2}`)
	wantRequestBody(t, perplexityWire(t, NewSonar()),
		`{"max_tokens":4096,"messages":[{"content":"hello","role":"user"}],
		  "model":"sonar","temperature":0.2}`)
	wantRequestBody(t, perplexityWire(t, NewPerplexityModel("sonar-reasoning-pro")),
		`{"max_tokens":8192,"messages":[{"content":"hello","role":"user"}],
		  "model":"sonar-reasoning-pro","temperature":0.2}`)
}

func TestPerplexityThinkingGoldenRequests(t *testing.T) {
	tests := []struct {
		name  string
		model Model
		// effort is a JSON literal; "" means the key must be absent.
		effort string
	}{
		// The opt-in guarantee: nobody touched these, so nothing is sent.
		{"sonar untouched", NewSonar(), ""},
		{"sonar pro untouched", NewSonarPro(), ""},
		{"reasoning pro untouched", NewSonarReasoningPro(), ""},
		{"deep research untouched", NewSonarDeepResearch(), ""},
		{"raw id untouched", NewPerplexityModel("sonar-deep-research"), ""},

		// Deep research is the one model Perplexity documents the knob for.
		{"deep research effort", Thinking(NewSonarDeepResearch(), WithThinkingEffort(ThinkingEffortLow)),
			`"low"`},
		{"deep research by raw id", Thinking(NewPerplexityModel("sonar-deep-research"),
			WithThinkingEffort(ThinkingEffortHigh)), `"high"`},

		// The ladder has four rungs and no "none": anything above high clamps
		// down to high, and an off request finds no off switch to spell.
		{"xhigh clamps to high", Thinking(NewSonarDeepResearch(), WithThinkingEffort(ThinkingEffortXHigh)),
			`"high"`},
		{"max clamps to high", Thinking(NewSonarDeepResearch(), WithThinkingEffort(ThinkingEffortMax)),
			`"high"`},
		{"minimal is on the ladder", Thinking(NewSonarDeepResearch(), WithThinkingEffort(ThinkingEffortMinimal)),
			`"minimal"`},
		{"no thinking is a no-op", NoThinking(NewSonarDeepResearch()), ""},

		// No token budget exists, so a neutral budget is projected onto the
		// ladder rather than dropped.
		{"budget becomes an effort", Thinking(NewSonarDeepResearch(), WithThinkingBudget(30000)),
			`"high"`},
		{"dynamic thinking has nothing to ask for", Thinking(NewSonarDeepResearch(), WithDynamicThinking()),
			""},

		// Everything else takes no request-side instruction, whatever is asked.
		{"sonar sends nothing", Thinking(NewSonar(), WithThinkingEffort(ThinkingEffortHigh)), ""},
		{"reasoning pro sends nothing", Thinking(NewSonarReasoningPro(), WithThinkingEffort(ThinkingEffortHigh)), ""},
		{"unknown raw id sends nothing", Thinking(NewPerplexityModel("sonar-tomorrow"),
			WithThinkingEffort(ThinkingEffortHigh)), ""},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			wantJSON(t, perplexityWire(t, tc.model), "reasoning_effort", tc.effort)
		})
	}
}

// The options that used to be applied by a six-arm type switch of identical
// blocks are now applied through one accessor. Every one of them must still
// land on the wire, for a named model and for a raw id alike.
func TestPerplexityOptionsSurviveTheAccessorRefactor(t *testing.T) {
	for _, m := range []Model{
		NewSonarPro().
			WithMaxTokens(512).WithTemperature(0.4).WithTopP(0.8).WithTopK(3).
			WithSystemPrompt("be terse").WithSearchRecencyFilter("day").
			WithSearchDomainFilter([]string{"example.com"}).
			WithReturnImages(true).WithReturnRelatedQuestions(true).
			WithResponseFormat(map[string]any{"type": "object"}),
		NewPerplexityModel("sonar-pro").
			WithMaxTokens(512).WithTemperature(0.4).WithTopP(0.8).WithTopK(3).
			WithSystemPrompt("be terse").WithSearchRecencyFilter("day").
			WithSearchDomainFilter([]string{"example.com"}).
			WithReturnImages(true).WithReturnRelatedQuestions(true).
			WithResponseFormat(map[string]any{"type": "object"}),
	} {
		body := perplexityWire(t, m)
		for key, want := range map[string]any{
			"model":                    "sonar-pro",
			"max_tokens":               float64(512),
			"temperature":              0.4,
			"top_p":                    0.8,
			"top_k":                    float64(3),
			"search_recency_filter":    "day",
			"return_images":            true,
			"return_related_questions": true,
		} {
			if got := body[key]; got != want {
				t.Errorf("%s = %v, want %v", key, got, want)
			}
		}
		wantJSON(t, body, "search_domain_filter", `["example.com"]`)
		wantJSON(t, body, "response_format", `{"type":"json_schema","json_schema":{"schema":{"type":"object"}}}`)
		if msgs, ok := body["messages"].([]any); !ok || len(msgs) != 2 {
			t.Errorf("messages = %v, want a system and a user message", body["messages"])
		}
	}
}

func TestPerplexityThinkingCapabilities(t *testing.T) {
	report := ThinkingCanReportTokens | ThinkingCanReportTrace

	tests := []struct {
		model Model
		want  ThinkingDimension
	}{
		{NewSonarDeepResearch(), ThinkingCanSetEffort | report},
		{NewPerplexityModel("sonar-deep-research"), ThinkingCanSetEffort | report},
		{NewSonarReasoning(), report},
		{NewSonarReasoningPro(), report},
		{NewPerplexityModel("sonar-reasoning-pro"), report},

		{NewSonar(), 0},
		{NewSonarPro(), 0},
		{NewPerplexityModel("sonar"), 0},
		{NewPerplexityModel("sonar-tomorrow"), 0},
	}

	for _, tc := range tests {
		t.Run(tc.model.ModelName(), func(t *testing.T) {
			if got := ModelThinkingDimensions(tc.model); got != tc.want {
				t.Errorf("ModelThinkingDimensions = %b, want %b", got, tc.want)
			}
		})
	}

	// There is no toggle anywhere in the Perplexity dialect: a reasoning model
	// always reasons.
	for _, m := range []Model{NewSonarDeepResearch(), NewSonarReasoningPro()} {
		if ModelThinkingDimensions(m).Has(ThinkingCanToggle) {
			t.Errorf("%s must not claim a thinking toggle", m.ModelName())
		}
	}
}

var (
	_ ThinkingModel = (*Sonar)(nil)
	_ ThinkingModel = (*SonarReasoningPro)(nil)
	_ ThinkingModel = (*SonarDeepResearch)(nil)
	_ ThinkingModel = (*PerplexityModel)(nil)
)

func TestPerplexityResponseSeparatesTheTraceFromTheAnswer(t *testing.T) {
	var c capture
	srv := perplexityThinkingStub(t, &c)
	defer srv.Close()

	resp := perplexityGenerate(t, srv.URL, NewSonarDeepResearch())

	if resp.Text != "hi there" {
		t.Errorf("Text = %q, want the answer without the <think> block", resp.Text)
	}
	if want := "the premise, then the arithmetic"; resp.Thinking != want {
		t.Errorf("Thinking = %q, want %q", resp.Thinking, want)
	}
	if resp.Metadata["reasoning_content"] != resp.Thinking {
		t.Errorf("Metadata[reasoning_content] = %q", resp.Metadata["reasoning_content"])
	}
	// reasoning_tokens is read as a subset of the completion count, so the
	// answer is what is left of it.
	if resp.Usage.ThinkingTokens != 42 {
		t.Errorf("ThinkingTokens = %d, want 42", resp.Usage.ThinkingTokens)
	}
	if resp.Usage.CompletionTokens != 50 || resp.Usage.TotalTokens != 61 {
		t.Errorf("usage = %+v, want the provider's own totals untouched", resp.Usage)
	}
	if resp.Usage.AnswerTokens() != 8 {
		t.Errorf("AnswerTokens = %d, want 8", resp.Usage.AnswerTokens())
	}
}

// Only a leading, closed block is a trace. Everything else is the answer, and
// mangling it would be worse than leaving the trace where it was.
func TestSplitPerplexityThinking(t *testing.T) {
	tests := []struct {
		name        string
		in          string
		text, think string
	}{
		{"no block", "hi there", "hi there", ""},
		{"leading block", "<think>why</think>hi there", "hi there", "why"},
		{"leading block with whitespace", "\n <think>\nwhy\n</think>\n\nhi there", "hi there", "why"},
		{"unclosed block is left alone", "<think>why hi there", "<think>why hi there", ""},
		{"a mention mid-answer is not a trace",
			"hi there, the <think> tag is html", "hi there, the <think> tag is html", ""},
		{"empty block", "<think></think>hi", "hi", ""},
		{"structured output after the block",
			"<think>why</think>{\"a\":1}", `{"a":1}`, "why"},
		{"empty content", "", "", ""},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			text, think := splitPerplexityThinking(tc.in)
			if text != tc.text || think != tc.think {
				t.Errorf("splitPerplexityThinking(%q) = (%q, %q), want (%q, %q)",
					tc.in, text, think, tc.text, tc.think)
			}
		})
	}
}
