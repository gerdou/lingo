package lingo

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
)

// ============================================================================
// COHERE THINKING
// ============================================================================
//
// Cohere already shipped a thinking surface -- WithThinkingDisabled and
// WithThinkingBudget on three model types -- so the tests below are a baseline
// rather than a description. Every golden row was recorded against the code as
// it stood before the portable surface existed and must keep marshalling byte
// for byte; a row that changes is a behaviour change for callers who wrote
// their code first, and has to be argued for rather than discovered.

// cohereThinkingStub serves a canned v2 chat response and records the request.
func cohereThinkingStub(t *testing.T, c *capture) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		c.record(r, raw)

		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"id":"c1","finish_reason":"COMPLETE",
			"message":{"role":"assistant","content":[
				{"type":"thinking","thinking":"first I check the premise"},
				{"type":"text","text":"hi there"},
				{"type":"thinking","thinking":" and then the arithmetic"},
				{"type":"text","text":", friend"}]},
			"usage":{"tokens":{"input_tokens":100,"output_tokens":7}}}`)
	}))
}

// cohereWire runs one Generate against the stub and returns the request body.
func cohereWire(t *testing.T, m Model) map[string]any {
	t.Helper()
	var c capture
	srv := cohereThinkingStub(t, &c)
	defer srv.Close()
	generate(t, &CohereConfig{APIKey: "k", BaseURL: srv.URL}, m)
	return c.body
}

// wantRequestBody compares a whole recorded request body against a JSON literal.
// It is how the opt-in guarantee is checked: not "the thinking key is absent"
// but "these are all the keys there are".
func wantRequestBody(t *testing.T, body map[string]any, want string) {
	t.Helper()
	var wantAny any
	if err := json.Unmarshal([]byte(want), &wantAny); err != nil {
		t.Fatalf("bad want literal: %v", err)
	}
	got, _ := json.Marshal(body)
	wantRaw, _ := json.Marshal(wantAny)
	if string(got) != string(wantRaw) {
		t.Errorf("request body =\n%s\nwant\n%s", got, wantRaw)
	}
}

// The request a model nobody touched produces must be exactly the request it
// produced before thinking control existed.
func TestCohereUntouchedRequestIsUnchanged(t *testing.T) {
	wantRequestBody(t, cohereWire(t, NewCommandAPlus()),
		`{"max_tokens":4096,"messages":[{"content":"hello","role":"user"}],
		  "model":"command-a-plus-05-2026","stream":false}`)
	wantRequestBody(t, cohereWire(t, NewCommandAReasoning()),
		`{"max_tokens":8192,"messages":[{"content":"hello","role":"user"}],
		  "model":"command-a-reasoning-08-2025","stream":false}`)
	wantRequestBody(t, cohereWire(t, NewCohereModel("command-a-plus-05-2026")),
		`{"messages":[{"content":"hello","role":"user"}],
		  "model":"command-a-plus-05-2026","stream":false}`)
}

func TestCohereThinkingGoldenRequests(t *testing.T) {
	tests := []struct {
		name  string
		model Model
		// thinking is a JSON literal; "" means the key must be absent.
		thinking string
	}{
		// The opt-in guarantee: nobody touched these, so nothing is sent.
		{"command a+ untouched", NewCommandAPlus(), ""},
		{"command a reasoning untouched", NewCommandAReasoning(), ""},
		{"command r+ untouched", NewCommandRPlus(), ""},
		{"raw id untouched", NewCohereModel("command-a-plus-05-2026"), ""},

		// The two pre-existing setters, exactly as they marshalled before.
		{"disabled", NewCommandAPlus().WithThinkingDisabled(),
			`{"type":"disabled"}`},
		{"budget", NewCommandAPlus().WithThinkingBudget(2048),
			`{"type":"enabled","token_budget":2048}`},
		{"reasoning model budget", NewCommandAReasoning().WithThinkingBudget(1500),
			`{"type":"enabled","token_budget":1500}`},
		{"raw id budget", NewCohereModel("command-a-plus-05-2026").WithThinkingBudget(999),
			`{"type":"enabled","token_budget":999}`},

		// A zero or negative budget has always enabled thinking with no ceiling:
		// the old wire guard was `if thinkingBudget > 0`, applied after the
		// tuple assignment had already set thinkingSet and thinkingEnabled.
		{"zero budget still enables", NewCommandAPlus().WithThinkingBudget(0),
			`{"type":"enabled"}`},
		{"negative budget still enables", NewCommandAPlus().WithThinkingBudget(-5),
			`{"type":"enabled"}`},

		// Last setter wins, in both orders, exactly as two independent bools did.
		{"budget then disabled", NewCommandAPlus().WithThinkingBudget(2048).WithThinkingDisabled(),
			`{"type":"disabled"}`},
		{"disabled then budget", NewCommandAPlus().WithThinkingDisabled().WithThinkingBudget(2048),
			`{"type":"enabled","token_budget":2048}`},

		// A pinned budget is forwarded verbatim, including one far outside the
		// window an unpinned budget would be clamped into. The caller named a
		// Cohere knob on a Cohere type, so lingo lets the API answer.
		{"pinned budget is never clamped", NewCommandAPlus().WithThinkingBudget(1 << 20),
			`{"type":"enabled","token_budget":1048576}`},

		// A raw id lingo knows cannot reason still sends what its own setter was
		// told to send: the pin outranks the id table.
		{"pinned budget on a non-thinking id", NewCohereModel("command-r-08-2024").WithThinkingBudget(64),
			`{"type":"enabled","token_budget":64}`},

		// The portable surface reaches the same wire shapes.
		{"portable enable", Thinking(NewCommandAPlus()),
			`{"type":"enabled"}`},
		{"portable disable", NoThinking(NewCommandAReasoning()),
			`{"type":"disabled"}`},
		{"portable budget", Thinking(NewCommandAPlus().WithMaxTokens(8192), WithThinkingBudget(4000)),
			`{"type":"enabled","token_budget":4000}`},

		// An unpinned budget is clamped into the model's window, whose ceiling is
		// the model's own max_tokens.
		{"portable budget clamped to max_tokens", Thinking(NewCommandAPlus(), WithThinkingBudget(999999)),
			`{"type":"enabled","token_budget":4096}`},

		// Cohere has no effort ladder, so an effort becomes a budget: 0.30 of a
		// 4096 ceiling for medium, 0.60 for high.
		{"portable effort becomes a budget", Thinking(NewCommandAPlus(), WithThinkingEffort(ThinkingEffortMedium)),
			`{"type":"enabled","token_budget":1228}`},
		{"portable high effort becomes a bigger budget", Thinking(NewCommandAReasoning(), WithThinkingEffort(ThinkingEffortHigh)),
			`{"type":"enabled","token_budget":4915}`},

		// Cohere has no dynamic setting, so a dynamic budget degrades to a plain
		// enable rather than being sent as a number.
		{"portable dynamic budget degrades to enable", Thinking(NewCommandAPlus(), WithDynamicThinking()),
			`{"type":"enabled"}`},

		// The six Command types that take no thinking object are untouched by
		// the portable surface, whatever it is asked for.
		{"portable enable on command r+ sends nothing", Thinking(NewCommandRPlus()), ""},
		{"portable disable on command a sends nothing", NoThinking(NewCommandA()), ""},
		{"portable effort on command r7b sends nothing",
			Thinking(NewCommandR7B(), WithThinkingEffort(ThinkingEffortHigh)), ""},
		{"portable enable on a non-thinking raw id sends nothing",
			Thinking(NewCohereModel("command-r-plus-08-2024")), ""},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			wantJSON(t, cohereWire(t, tc.model), "thinking", tc.thinking)
		})
	}
}

// The other request fields must be untouched by any of this: a model carrying
// thinking configuration still marshals everything it marshalled before.
func TestCohereThinkingLeavesTheRestOfTheRequestAlone(t *testing.T) {
	body := cohereWire(t, NewCommandAPlus().
		WithMaxTokens(512).WithTemperature(0.4).WithTopP(0.8).WithTopK(3).
		WithSeed(7).WithSystemPrompt("be terse").WithSafetyMode(CohereSafetyStrict).
		WithThinkingBudget(2048))

	for key, want := range map[string]any{
		"model":       "command-a-plus-05-2026",
		"max_tokens":  float64(512),
		"temperature": 0.4,
		"p":           0.8,
		"k":           float64(3),
		"seed":        float64(7),
		"safety_mode": "STRICT",
	} {
		if got := body[key]; got != want {
			t.Errorf("%s = %v, want %v", key, got, want)
		}
	}
	if msgs, ok := body["messages"].([]any); !ok || len(msgs) != 2 {
		t.Errorf("messages = %v, want a system and a user message", body["messages"])
	}
}

// oldCohereOptions is a transcription of the three fields and the wire block
// the shims replaced, copied from the code as it stood before this change:
//
//	thinkingSet, thinkingEnabled, reasoning = true, false, false   // disabled
//	thinkingSet, thinkingEnabled, reasoning = true, true, true     // budget
//	thinkingBudget = tokens
//
//	if opts.thinkingSet {
//		thinking := &cohere.Thinking{Type: cohere.ThinkingTypeDisabled}
//		if opts.thinkingEnabled {
//			thinking.Type = cohere.ThinkingTypeEnabled
//			if opts.thinkingBudget > 0 {
//				thinking.TokenBudget = &opts.thinkingBudget
//			}
//		}
//		req.Thinking = thinking
//	}
//
// It exists so the equivalence is executed rather than asserted: the same
// setter sequences are run through both, and the JSON has to match.
type oldCohereOptions struct {
	set, enabled bool
	budget       int
}

func (o *oldCohereOptions) withThinkingDisabled() { o.set, o.enabled = true, false }
func (o *oldCohereOptions) withThinkingBudget(n int) {
	o.set, o.enabled, o.budget = true, true, n
}

// wire renders what the old block would have put in the request, "" when it
// would have left the field out.
func (o *oldCohereOptions) wire() string {
	if !o.set {
		return ""
	}
	body := `{"type":"disabled"}`
	if o.enabled {
		body = `{"type":"enabled"}`
		if o.budget > 0 {
			body = `{"type":"enabled","token_budget":` + fmt.Sprintf("%d", o.budget) + `}`
		}
	}
	return body
}

func TestCohereShimsMatchTheCodeTheyReplaced(t *testing.T) {
	// Each step names a setter and its argument; -1 in a disable step is unused.
	type step struct {
		disable bool
		budget  int
	}
	sequences := [][]step{
		{{disable: true}},
		{{budget: 2048}},
		{{budget: 0}},
		{{budget: -5}},
		{{budget: 1 << 20}},
		{{budget: 2048}, {disable: true}},
		{{disable: true}, {budget: 2048}},
		{{budget: 64}, {budget: 128}},
		{{disable: true}, {disable: true}},
	}

	for _, seq := range sequences {
		var old oldCohereOptions
		m := NewCommandAPlus()
		name := ""
		for _, s := range seq {
			if s.disable {
				old.withThinkingDisabled()
				m.WithThinkingDisabled()
				name += "disable;"
				continue
			}
			old.withThinkingBudget(s.budget)
			m.WithThinkingBudget(s.budget)
			name += "budget(" + fmt.Sprintf("%d", s.budget) + ");"
		}
		t.Run(name, func(t *testing.T) {
			wantJSON(t, cohereWire(t, m), "thinking", old.wire())
		})
	}
}

func TestCohereThinkingCapabilities(t *testing.T) {
	full := ThinkingCanToggle | ThinkingCanSetBudget | ThinkingCanReportTrace

	tests := []struct {
		model Model
		want  ThinkingDimension
	}{
		{NewCommandAPlus(), full},
		{NewCommandAReasoning(), full},
		{NewCohereModel("command-a-plus-05-2026"), full},
		// A model id newer than this package is taken at its word.
		{NewCohereModel("command-b-thinking-01-2027"), full},

		{NewCommandA(), 0},
		{NewCommandAVision(), 0},
		{NewCommandATranslate(), 0},
		{NewCommandR7B(), 0},
		{NewCommandR(), 0},
		{NewCommandRPlus(), 0},
		{NewCohereModel("command-r-08-2024"), 0},
		{NewCohereModel("command-a-03-2025"), 0},
	}

	for _, tc := range tests {
		t.Run(tc.model.ModelName(), func(t *testing.T) {
			if got := ModelThinkingDimensions(tc.model); got != tc.want {
				t.Errorf("ModelThinkingDimensions = %b, want %b", got, tc.want)
			}
		})
	}

	// Cohere reports no thinking token count, on any model.
	if ThinkingDimensions(ProviderCohere).Has(ThinkingCanReportTokens) {
		t.Error("Cohere must not claim to report thinking tokens")
	}
}

// Every Cohere model can carry thinking configuration, including the ones that
// send none of it -- the wire gate is thinkingDimensions, not the accessor.
var (
	_ ThinkingModel = (*CommandAPlus)(nil)
	_ ThinkingModel = (*CommandAReasoning)(nil)
	_ ThinkingModel = (*CommandRPlus)(nil)
	_ ThinkingModel = (*CohereModel)(nil)
)

func TestCohereSettersAndPortableSurfaceShareOneStorage(t *testing.T) {
	m := NewCommandAPlus().WithThinkingBudget(2048)
	if to := m.ThinkingOptions(); !to.Enabled() || to.Budget() != 2048 {
		t.Errorf("WithThinkingBudget is not visible through ThinkingOptions: %+v", to)
	}
	if !NewCommandAPlus().WithThinkingDisabled().ThinkingOptions().Disabled() {
		t.Error("WithThinkingDisabled is not visible through ThinkingOptions")
	}
	if !NoThinking(NewCommandAPlus()).ThinkingOptions().Disabled() {
		t.Error("NoThinking must disable")
	}
	// Thinking returns the same model with its concrete type intact.
	if got := Thinking(NewCommandAReasoning(), WithThinkingBudget(64)); !got.ThinkingOptions().Enabled() {
		t.Error("Thinking must enable")
	}
}

func TestCohereResponseCarriesTheTrace(t *testing.T) {
	var c capture
	srv := cohereThinkingStub(t, &c)
	defer srv.Close()

	resp := generate(t, &CohereConfig{APIKey: "k", BaseURL: srv.URL},
		NewCommandAPlus().WithThinkingBudget(2048))

	// Both block kinds accumulate, and the trace never lands in the answer.
	if resp.Text != "hi there, friend" {
		t.Errorf("Text = %q", resp.Text)
	}
	if want := "first I check the premise and then the arithmetic"; resp.Thinking != want {
		t.Errorf("Thinking = %q, want %q", resp.Thinking, want)
	}
	// Deprecated mirror, kept for one release.
	if resp.Metadata["reasoning_content"] != resp.Thinking {
		t.Errorf("Metadata[reasoning_content] = %q", resp.Metadata["reasoning_content"])
	}
	if resp.Metadata["is_reasoning_model"] != "true" {
		t.Errorf("Metadata[is_reasoning_model] = %q", resp.Metadata["is_reasoning_model"])
	}
	// Cohere reports no thinking token count, so the counter stays zero and the
	// whole completion counts as answer.
	if resp.Usage.ThinkingTokens != 0 {
		t.Errorf("ThinkingTokens = %d, want 0", resp.Usage.ThinkingTokens)
	}
	if resp.Usage.AnswerTokens() != 7 {
		t.Errorf("AnswerTokens = %d, want 7", resp.Usage.AnswerTokens())
	}
	if _, ok := resp.Metadata["thinking_translation"]; ok {
		t.Errorf("a pinned budget needs no translation: %q", resp.Metadata["thinking_translation"])
	}
}

// A translation is recorded whenever lingo had to adapt the request, so the
// substitution is never silent.
func TestCohereReportsWhatItTranslated(t *testing.T) {
	var c capture
	srv := cohereThinkingStub(t, &c)
	defer srv.Close()

	resp := generate(t, &CohereConfig{APIKey: "k", BaseURL: srv.URL},
		Thinking(NewCommandAPlus(), WithThinkingEffort(ThinkingEffortHigh)))

	if got := resp.Metadata["thinking_translation"]; got != "effort high mapped to budget 2457 tokens" {
		t.Errorf("Metadata[thinking_translation] = %q", got)
	}
}

// Disabling thinking must not report the model as reasoning, whichever surface
// asked for it.
func TestCohereDisabledIsNotAReasoningRequest(t *testing.T) {
	var c capture
	srv := cohereThinkingStub(t, &c)
	defer srv.Close()

	resp := generate(t, &CohereConfig{APIKey: "k", BaseURL: srv.URL},
		NewCommandAPlus().WithThinkingDisabled())
	if resp.Metadata["is_reasoning_model"] != "false" {
		t.Errorf("Metadata[is_reasoning_model] = %q, want false", resp.Metadata["is_reasoning_model"])
	}
}

// TestCohereMetadataNeverContradictsTheBody is the same claim on the model that
// exposed the hole: command-a-reasoning is constructed with the reasoning flag
// already true, so a disable that arrived through the portable surface had
// nothing to clear it. The flag then said "true" beside a body that said
// thinking={"type":"disabled"} -- and a caller reading it to decide how to price
// or route the call was reading the model's biography, not the request.
//
// The two surfaces have to agree with each other as well as with the body: the
// per-model setter and NoThinking build the same request, so they must report it
// the same way.
func TestCohereMetadataNeverContradictsTheBody(t *testing.T) {
	tests := []struct {
		name  string
		model Model
	}{
		{"portable disable on a reasoning model", NoThinking(NewCommandAReasoning())},
		{"per-model disable on a reasoning model", NewCommandAReasoning().WithThinkingDisabled()},
		{"portable disable on a general model", NoThinking(NewCommandAPlus())},
		{"portable disable after a pinned budget",
			NoThinking(NewCommandAReasoning().WithThinkingBudget(2048))},
		{"portable disable on a raw id", NoThinking(NewCohereModel("command-a-reasoning-08-2025"))},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var c capture
			srv := cohereThinkingStub(t, &c)
			defer srv.Close()

			resp := generate(t, &CohereConfig{APIKey: "k", BaseURL: srv.URL}, tc.model)

			// The body really did switch thinking off...
			wantJSON(t, c.body, "thinking", `{"type":"disabled"}`)
			// ...so the metadata beside it cannot claim otherwise.
			if got := resp.Metadata["is_reasoning_model"]; got != "false" {
				t.Errorf("Metadata[is_reasoning_model] = %q, want false: the body sent "+
					"thinking={\"type\":\"disabled\"}", got)
			}
		})
	}

	// The flag is not simply nailed to false: a request that does ask for
	// thinking still reports true.
	t.Run("an enabled request still reports true", func(t *testing.T) {
		var c capture
		srv := cohereThinkingStub(t, &c)
		defer srv.Close()

		resp := generate(t, &CohereConfig{APIKey: "k", BaseURL: srv.URL}, Thinking(NewCommandAReasoning()))
		wantJSON(t, c.body, "thinking", `{"type":"enabled"}`)
		if got := resp.Metadata["is_reasoning_model"]; got != "true" {
			t.Errorf("Metadata[is_reasoning_model] = %q, want true", got)
		}
	})
}
