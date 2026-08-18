package lingo

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
)

// ============================================================================
// OLLAMA THINKING
// ============================================================================
//
// Ollama is the one provider where asking for thinking can fail rather than be
// ignored: its server answers `think` on a model without the thinking
// capability with HTTP 400, while `think: false` is always accepted. That
// asymmetry is why the capability table below is conservative, and why the
// golden rows care as much about what is absent as about what is sent.

// ollamaThinkingStub serves a canned chat response and records the request.
func ollamaThinkingStub(t *testing.T, c *capture) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		c.path = r.URL.Path
		c.body = map[string]any{}
		_ = json.Unmarshal(raw, &c.body)

		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"model":"qwen3","created_at":"2026-08-17T00:00:00Z",
			"message":{"role":"assistant","content":"hi there",
				"thinking":"the premise, then the arithmetic"},
			"done":true,"done_reason":"stop",
			"total_duration":42,"load_duration":7,
			"prompt_eval_count":11,"eval_count":50}`)
	}))
}

// ollamaWire runs one Generate against the stub and returns the request body.
func ollamaWire(t *testing.T, m Model) map[string]any {
	t.Helper()
	var c capture
	srv := ollamaThinkingStub(t, &c)
	defer srv.Close()
	generate(t, &OllamaConfig{BaseURL: srv.URL}, m)
	return c.body
}

func TestOllamaThinkingGoldenRequests(t *testing.T) {
	tests := []struct {
		name  string
		model Model
		// think is a JSON literal; "" means the key must be absent, which is
		// what every request lingo sent before this feature existed carried.
		think string
	}{
		// The opt-in guarantee. Note that absent is not false: a thinking-capable
		// model reasons by default when the field is missing, which is what
		// Ollama's server arranges, and lingo must not change that.
		{"qwen3 untouched", NewQwen3(), ""},
		{"deepseek-r1 untouched", NewDeepSeekR1(), ""},
		{"llama3 untouched", NewLlama3(), ""},
		{"raw tag untouched", NewOllamaModel("qwen3:8b"), ""},

		// A bare enable is the bool; a level is the string, sent instead of the
		// bool rather than beside it.
		{"enable", Thinking(NewQwen3()), `true`},
		{"disable", NoThinking(NewQwen3()), `false`},
		{"effort high", Thinking(NewDeepSeekR1(), WithThinkingEffort(ThinkingEffortHigh)), `"high"`},
		{"effort medium", Thinking(NewQwen3(), WithThinkingEffort(ThinkingEffortMedium)), `"medium"`},
		{"effort low", Thinking(NewQwen3(), WithThinkingEffort(ThinkingEffortLow)), `"low"`},

		// The ladder is exactly low, medium and high: Ollama rejects any other
		// string outright, so the rungs it does not have are clamped rather than
		// forwarded.
		{"minimal clamps up to low", Thinking(NewQwen3(), WithThinkingEffort(ThinkingEffortMinimal)), `"low"`},
		{"xhigh clamps down to high", Thinking(NewQwen3(), WithThinkingEffort(ThinkingEffortXHigh)), `"high"`},
		{"max clamps down to high", Thinking(NewQwen3(), WithThinkingEffort(ThinkingEffortMax)), `"high"`},
		{"none switches thinking off", Thinking(NewQwen3(), WithThinkingEffort(ThinkingEffortNone)), `false`},

		// There is no token budget, so a neutral budget is projected onto the
		// ladder; a dynamic budget has nothing to ask for and degrades to enable.
		{"budget becomes a level", Thinking(NewQwen3(), WithThinkingBudget(30000)), `"high"`},
		{"small budget becomes the lowest level", Thinking(NewQwen3(), WithThinkingBudget(1000)), `"low"`},
		{"dynamic budget degrades to enable", Thinking(NewQwen3(), WithDynamicThinking()), `true`},

		// A raw tag is resolved by family, so a size suffix changes nothing.
		{"raw thinking tag enables", Thinking(NewOllamaModel("deepseek-r1:32b")), `true`},
		{"raw gpt-oss takes a level",
			Thinking(NewOllamaModel("gpt-oss:20b"), WithThinkingEffort(ThinkingEffortHigh)), `"high"`},

		// Models that cannot think are left alone, whatever is asked of them.
		// This is the never-error rule doing real work: `think` here is a 400.
		{"enable on llama3 sends nothing", Thinking(NewLlama3()), ""},
		{"effort on mistral sends nothing",
			Thinking(NewMistral(), WithThinkingEffort(ThinkingEffortHigh)), ""},
		{"disable on llama3 sends nothing", NoThinking(NewLlama3()), ""},
		{"enable on an unknown tag sends nothing", Thinking(NewOllamaModel("my-private-model:latest")), ""},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			wantJSON(t, ollamaWire(t, tc.model), "think", tc.think)
		})
	}
}

// The request a model nobody touched produces must carry exactly the keys it
// carried before, and the thinking-capable models must not have acquired a
// think key by being thinking-capable.
func TestOllamaUntouchedRequestIsUnchanged(t *testing.T) {
	body := ollamaWire(t, NewQwen3().
		WithMaxTokens(512).WithTemperature(0.4).WithTopP(0.8).WithTopK(3).
		WithNumCtx(4096).WithRepeatPenalty(1.1).WithSeed(7).WithSystemPrompt("be terse"))

	wantRequestBody(t, body, `{"model":"qwen3","stream":false,
		  "messages":[{"role":"system","content":"be terse"},{"role":"user","content":"hello"}],
		  "options":{"num_predict":512,"temperature":0.4,"top_p":0.8,"top_k":3,
			"num_ctx":4096,"repeat_penalty":1.1,"seed":7}}`)

	// And with nothing set at all, down to the option block itself.
	wantRequestBody(t, ollamaWire(t, NewQwen3()), `{"model":"qwen3","stream":false,
		  "messages":[{"role":"user","content":"hello"}],
		  "options":{"num_predict":4096,"temperature":0.8}}`)
}

// An outbound message never carries a thinking field, so adding one to the
// shared message struct cannot change a request.
func TestOllamaOutboundMessagesCarryNoThinking(t *testing.T) {
	body := ollamaWire(t, Thinking(NewQwen3().WithSystemPrompt("be terse")))
	wantJSON(t, body, "messages",
		`[{"role":"system","content":"be terse"},{"role":"user","content":"hello"}]`)
}

func TestOllamaThinkingCapabilities(t *testing.T) {
	full := ThinkingCanToggle | ThinkingCanSetEffort | ThinkingCanReportTrace

	tests := []struct {
		name  string
		model Model
		want  ThinkingDimension
	}{
		{"qwen3", NewQwen3(), full},
		{"deepseek-r1", NewDeepSeekR1(), full},
		{"raw deepseek-r1", NewOllamaModel("deepseek-r1:32b"), full},
		{"raw qwq", NewOllamaModel("qwq"), full},
		{"raw gpt-oss", NewOllamaModel("gpt-oss:120b"), full},

		{"llama3", NewLlama3(), 0},
		{"llama32", NewLlama32(), 0},
		{"mistral", NewMistral(), 0},
		{"phi4", NewPhi4(), 0},
		{"gemma3", NewGemma3(), 0},
		{"qwen2", NewQwen2(), 0},
		{"deepseek-coder", NewDeepSeekCoder(), 0},
		{"raw llama3.2", NewOllamaModel("llama3.2"), 0},
		{"unknown tag", NewOllamaModel("my-private-model:latest"), 0},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := ModelThinkingDimensions(tc.model); got != tc.want {
				t.Errorf("ModelThinkingDimensions = %b, want %b", got, tc.want)
			}
		})
	}

	// Ollama folds thinking tokens into eval_count and reports no breakdown.
	if ThinkingDimensions(ProviderOllama).Has(ThinkingCanReportTokens) {
		t.Error("Ollama must not claim to report thinking tokens")
	}
}

// TestOllamaThinkingAllowlistRespectsTheTagBoundary is the never-error rule at
// its sharpest. ollamaThinkingModels lists families, and an Ollama tag is
// "<name>:<variant>", so the match has to stop at the colon: a plain prefix test
// lets the family "qwen3" capture qwen3-coder, qwen3-embedding and
// qwen3-reranker, three separate models with no thinking capability. Ollama
// answers a truthy `think` on one of those with HTTP 400 "%q does not support
// thinking" (server/routes.go, ChatHandler) rather than ignoring it, so lingo
// putting the field on the wire turns an opt-in into a failed generation.
func TestOllamaThinkingAllowlistRespectsTheTagBoundary(t *testing.T) {
	// Every tag shape a listed family legitimately takes: the bare name, and the
	// name with any variant after the colon.
	thinks := []string{
		"qwen3", "qwen3:8b", "qwen3:30b-a3b", "qwen3:0.6b-fp16",
		"deepseek-r1", "deepseek-r1:7b", "deepseek-r1:70b-llama-distill-q4_K_M",
		"deepseek-v3.1", "deepseek-v3.1:671b",
		"gpt-oss", "gpt-oss:20b", "gpt-oss:120b-cloud",
		"qwq", "qwq:32b", "magistral:24b", "phi4-reasoning", "phi4-reasoning:plus",
		"exaone-deep:2.4b", "smollm3:3b", "cogito", "cogito:8b",
	}
	// Non-thinking models whose names merely start with a listed family. These
	// are the 400s.
	cannot := []string{
		"qwen3-coder", "qwen3-coder:30b", "qwen3-coder:480b-cloud",
		"qwen3-embedding", "qwen3-embedding:8b", "qwen3-reranker", "qwen3-reranker:4b",
		"qwen3-vl:8b",
		// A family that renames rather than tags its variant is unlisted, and an
		// unlisted tag is the documented silent no-op -- the safe direction.
		"deepseek-v3.1-terminus", "phi4-reasoning-plus", "smollm3-base",
		// And the plain negatives, which never matched either way.
		"llama3.2", "mistral", "my-private-model:latest",
	}

	for _, tag := range thinks {
		t.Run("thinks/"+tag, func(t *testing.T) {
			if got := ollamaThinkingDimensions(tag); got != ollamaThinkingDims {
				t.Errorf("ollamaThinkingDimensions(%q) = %b, want %b", tag, got, ollamaThinkingDims)
			}
		})
	}
	for _, tag := range cannot {
		t.Run("cannot/"+tag, func(t *testing.T) {
			if got := ollamaThinkingDimensions(tag); got != 0 {
				t.Errorf("ollamaThinkingDimensions(%q) = %b, want 0: Ollama answers `think` on a model "+
					"without the thinking capability with a 400, not by ignoring it", tag, got)
			}
		})
	}

	// And the same contract on the wire, which is where the 400 would happen.
	t.Run("wire", func(t *testing.T) {
		wantJSON(t, ollamaWire(t, Thinking(NewOllamaModel("qwen3-coder:30b"))), "think", "")
		wantJSON(t, ollamaWire(t, Thinking(NewOllamaModel("qwen3-embedding:8b"))), "think", "")
		wantJSON(t, ollamaWire(t, Thinking(NewOllamaModel("qwen3-reranker"))), "think", "")
		wantJSON(t, ollamaWire(t,
			Thinking(NewOllamaModel("qwen3-coder:30b"), WithThinkingEffort(ThinkingEffortHigh))), "think", "")
		// The genuine thinking tags are untouched by the boundary.
		wantJSON(t, ollamaWire(t, Thinking(NewOllamaModel("qwen3:30b-a3b"))), "think", `true`)
		wantJSON(t, ollamaWire(t, Thinking(NewOllamaModel("qwen3"))), "think", `true`)
		wantJSON(t, ollamaWire(t, Thinking(NewOllamaModel("gpt-oss:20b"))), "think", `true`)
		wantJSON(t, ollamaWire(t, Thinking(NewOllamaModel("deepseek-r1:7b"))), "think", `true`)
	})
}

var (
	_ ThinkingModel = (*OllamaModel)(nil)
	_ ThinkingModel = (*Qwen3)(nil)
	_ ThinkingModel = (*DeepSeekR1)(nil)
	_ ThinkingModel = (*Llama3)(nil)
)

func TestOllamaResponseCarriesTheTrace(t *testing.T) {
	var c capture
	srv := ollamaThinkingStub(t, &c)
	defer srv.Close()

	resp := generate(t, &OllamaConfig{BaseURL: srv.URL}, Thinking(NewQwen3()))

	if resp.Text != "hi there" {
		t.Errorf("Text = %q", resp.Text)
	}
	if want := "the premise, then the arithmetic"; resp.Thinking != want {
		t.Errorf("Thinking = %q, want %q", resp.Thinking, want)
	}
	// Ollama reports no thinking token count, so the counter stays zero and the
	// whole completion counts as answer.
	if resp.Usage.ThinkingTokens != 0 {
		t.Errorf("ThinkingTokens = %d, want 0", resp.Usage.ThinkingTokens)
	}
	if resp.Usage.AnswerTokens() != 50 {
		t.Errorf("AnswerTokens = %d, want 50", resp.Usage.AnswerTokens())
	}
	if _, ok := resp.Metadata["thinking_translation"]; ok {
		t.Errorf("a plain enable needs no translation: %q", resp.Metadata["thinking_translation"])
	}
}

// The trace is reported whether or not the caller opted in: a thinking-capable
// Ollama model reasons by default, and lingo used to throw the trace away at
// unmarshal.
func TestOllamaReportsATraceNobodyAskedFor(t *testing.T) {
	var c capture
	srv := ollamaThinkingStub(t, &c)
	defer srv.Close()

	resp := generate(t, &OllamaConfig{BaseURL: srv.URL}, NewQwen3())
	if resp.Thinking == "" {
		t.Error("Thinking must be reported even when nothing was requested")
	}
	if _, ok := c.body["think"]; ok {
		t.Errorf("think = %v, want the key to be absent", c.body["think"])
	}
}

func TestOllamaReportsWhatItTranslated(t *testing.T) {
	var c capture
	srv := ollamaThinkingStub(t, &c)
	defer srv.Close()

	resp := generate(t, &OllamaConfig{BaseURL: srv.URL},
		Thinking(NewQwen3(), WithThinkingEffort(ThinkingEffortMax)))

	if got := resp.Metadata["thinking_translation"]; got != "effort max clamped to high" {
		t.Errorf("Metadata[thinking_translation] = %q", got)
	}
}
