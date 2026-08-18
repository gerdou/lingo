package lingo

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
)

// ============================================================================
// ANTHROPIC THINKING: GOLDEN REQUESTS
// ============================================================================
//
// The thinking half of the Anthropic request used to live in a 20-arm type
// switch, one hand-copied block per model type. It now comes from one plan
// built outside the switch, which is a much smaller surface but also the single
// easiest place in this change to alter a wire request by accident.
//
// So the table below is a baseline, not a description: every row was recorded
// against the pre-refactor code and must keep marshalling byte for byte. A row
// that changes is a behaviour change for callers who wrote their code before
// the portable thinking surface existed, and has to be argued for rather than
// discovered.

// anthropicWire runs one Generate against a stub and returns the request body.
func anthropicWire(t *testing.T, m Model) map[string]any {
	t.Helper()
	var c capture
	srv := anthropicStub(t, &c)
	defer srv.Close()
	generate(t, &AnthropicConfig{APIKey: "k"}, m)
	return c.body
}

// wantJSON compares one body field against a JSON literal, treating a "" want
// as "the key must be absent".
func wantJSON(t *testing.T, body map[string]any, key, want string) {
	t.Helper()
	got, ok := body[key]
	if want == "" {
		if ok {
			raw, _ := json.Marshal(got)
			t.Errorf("%s = %s, want the key to be absent", key, raw)
		}
		return
	}
	if !ok {
		t.Errorf("%s is absent, want %s", key, want)
		return
	}
	raw, err := json.Marshal(got)
	if err != nil {
		t.Fatalf("marshal %s: %v", key, err)
	}
	var wantAny any
	if err := json.Unmarshal([]byte(want), &wantAny); err != nil {
		t.Fatalf("bad want literal %q: %v", want, err)
	}
	wantRaw, _ := json.Marshal(wantAny)
	if string(raw) != string(wantRaw) {
		t.Errorf("%s = %s, want %s", key, raw, wantRaw)
	}
}

func TestAnthropicThinkingGoldenRequests(t *testing.T) {
	tests := []struct {
		name         string
		model        Model
		thinking     string // "" means the thinking key must be absent
		outputConfig string // "" means the output_config key must be absent
	}{
		// Claude 3.x carries no thinking configuration at all.
		{"3.5 sonnet untouched", NewClaude35Sonnet(), "", ""},

		// 3.7 through 4.5: a fixed budget is the only knob, and lingo has never
		// validated it -- a value below the API's 1024 floor is forwarded so the
		// caller sees the provider's own error.
		{"3.7 untouched", NewClaude37Sonnet(), "", ""},
		{"3.7 budget", NewClaude37Sonnet().WithThinkingBudget(2048),
			`{"type":"enabled","budget_tokens":2048}`, ""},
		{"sonnet 4 illegal budget forwarded", NewClaudeSonnet4().WithThinkingBudget(500),
			`{"type":"enabled","budget_tokens":500}`, ""},
		{"sonnet 4 zero budget sends nothing", NewClaudeSonnet4().WithThinkingBudget(0), "", ""},
		{"sonnet 4.5 budget", NewClaudeSonnet45().WithThinkingBudget(3000),
			`{"type":"enabled","budget_tokens":3000}`, ""},
		{"opus 4.1 budget", NewClaudeOpus41().WithThinkingBudget(4096),
			`{"type":"enabled","budget_tokens":4096}`, ""},
		{"haiku 4.5 budget", NewClaudeHaiku45().WithThinkingBudget(1024),
			`{"type":"enabled","budget_tokens":1024}`, ""},

		// 4.6 takes adaptive, a deprecated fixed budget, and effort. Effort on
		// its own must NOT bring a thinking key with it.
		{"opus 4.6 untouched", NewClaudeOpus46(), "", ""},
		{"opus 4.6 effort only", NewClaudeOpus46().WithEffort(EffortHigh), "", `{"effort":"high"}`},
		{"opus 4.6 adaptive", NewClaudeOpus46().WithAdaptiveThinking(), `{"type":"adaptive"}`, ""},
		{"opus 4.6 adaptive beats budget",
			NewClaudeOpus46().WithThinkingBudget(4096).WithAdaptiveThinking(),
			`{"type":"adaptive"}`, ""},
		{"opus 4.6 budget", NewClaudeOpus46().WithThinkingBudget(4096),
			`{"type":"enabled","budget_tokens":4096}`, ""},
		{"sonnet 4.6 adaptive and effort",
			NewClaudeSonnet46().WithAdaptiveThinking().WithEffort(EffortMedium),
			`{"type":"adaptive"}`, `{"effort":"medium"}`},
		// xhigh is rejected by 4.6, and lingo forwards it anyway: the caller
		// named this model's own setter, so they get this model's own error.
		{"opus 4.6 xhigh forwarded", NewClaudeOpus46().WithEffort(EffortXHigh), "", `{"effort":"xhigh"}`},

		// 4.7/4.8: adaptive only, no fixed budget setter exists.
		{"opus 4.7 untouched", NewClaudeOpus47(), "", ""},
		{"opus 4.7 adaptive and xhigh",
			NewClaudeOpus47().WithAdaptiveThinking().WithEffort(EffortXHigh),
			`{"type":"adaptive"}`, `{"effort":"xhigh"}`},
		{"opus 4.8 effort only", NewClaudeOpus48().WithEffort(EffortMax), "", `{"effort":"max"}`},

		// Fable 5 reasons server-side and must never carry a thinking config.
		{"fable 5 untouched", NewClaudeFable5(), "", ""},
		{"fable 5 effort", NewClaudeFable5().WithEffort(EffortMax), "", `{"effort":"max"}`},

		// Claude 5: thinking is on by default, so the field is sent only to
		// switch it off.
		{"opus 5 untouched", NewClaudeOpus5(), "", ""},
		{"opus 5 effort only", NewClaudeOpus5().WithEffort(EffortXHigh), "", `{"effort":"xhigh"}`},
		{"opus 5 disabled", NewClaudeOpus5().WithThinkingDisabled(), `{"type":"disabled"}`, ""},
		{"sonnet 5 disabled with effort",
			NewClaudeSonnet5().WithThinkingDisabled().WithEffort(EffortLow),
			`{"type":"disabled"}`, `{"effort":"low"}`},

		// The generic escape hatch sends whatever the caller set, whatever id it
		// was handed. Its precedence is disabled, then adaptive, then budget.
		{"generic untouched", NewAnthropicModel("claude-opus-5"), "", ""},
		{"generic budget on a 3.x id",
			NewAnthropicModel("claude-3-haiku-20240307").WithThinkingBudget(1024),
			`{"type":"enabled","budget_tokens":1024}`, ""},
		{"generic adaptive on a 4.x id",
			NewAnthropicModel("claude-sonnet-4-20250514").WithAdaptiveThinking(),
			`{"type":"adaptive"}`, ""},
		{"generic disabled beats adaptive",
			NewAnthropicModel("claude-opus-5").WithAdaptiveThinking().WithThinkingDisabled(),
			`{"type":"disabled"}`, ""},
		{"generic effort on an unknown id",
			NewAnthropicModel("claude-opus-9").WithEffort(EffortMedium), "", `{"effort":"medium"}`},

		// The three toggle setters used to own a field each and were read in a
		// fixed order, so their precedence never depended on the call order.
		// Every ordering is pinned here because one storage would otherwise
		// resolve them by whoever spoke last.
		{"4.6 budget then adaptive",
			NewClaudeOpus46().WithThinkingBudget(4096).WithAdaptiveThinking(), `{"type":"adaptive"}`, ""},
		{"4.6 adaptive then budget",
			NewClaudeOpus46().WithAdaptiveThinking().WithThinkingBudget(4096), `{"type":"adaptive"}`, ""},
		{"generic budget then disabled",
			NewAnthropicModel("claude-opus-5").WithThinkingBudget(4096).WithThinkingDisabled(),
			`{"type":"disabled"}`, ""},
		{"generic disabled then budget",
			NewAnthropicModel("claude-opus-5").WithThinkingDisabled().WithThinkingBudget(4096),
			`{"type":"disabled"}`, ""},
		{"generic adaptive then disabled",
			NewAnthropicModel("claude-opus-5").WithAdaptiveThinking().WithThinkingDisabled(),
			`{"type":"disabled"}`, ""},
		{"generic disabled then adaptive",
			NewAnthropicModel("claude-opus-5").WithThinkingDisabled().WithAdaptiveThinking(),
			`{"type":"disabled"}`, ""},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			body := anthropicWire(t, tc.model)
			wantJSON(t, body, "thinking", tc.thinking)
			wantJSON(t, body, "output_config", tc.outputConfig)
		})
	}
}

// ============================================================================
// ANTHROPIC THINKING: RESPONSE EXTRACTION
// ============================================================================

// anthropicMultiBlockStub returns a message split across several text and
// thinking blocks, which is what a real extended-thinking response looks like.
func anthropicMultiBlockStub(t *testing.T) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"msg_1","type":"message","role":"assistant",
			"model":"claude-opus-5","stop_reason":"end_turn",
			"content":[
				{"type":"thinking","thinking":"first I ","signature":"sig-a"},
				{"type":"thinking","thinking":"then I","signature":"sig-b"},
				{"type":"text","text":"hello "},
				{"type":"redacted_thinking","data":"ENCRYPTED"},
				{"type":"text","text":"world"}],
			"usage":{"input_tokens":10,"output_tokens":30,
				"output_tokens_details":{"thinking_tokens":12}}}`)
	}))
	t.Setenv("ANTHROPIC_BASE_URL", srv.URL)
	return srv
}

// TestAnthropicAccumulatesEveryContentBlock is the regression guard for a live
// bug: the extraction loop assigned rather than appended, so a response split
// across several blocks -- the normal shape once thinking is on -- returned only
// its last text block and last thinking block, silently truncating the answer.
func TestAnthropicAccumulatesEveryContentBlock(t *testing.T) {
	srv := anthropicMultiBlockStub(t)
	defer srv.Close()

	resp := generate(t, &AnthropicConfig{APIKey: "k"}, NewClaudeOpus5())

	if resp.Text != "hello world" {
		t.Errorf("Text = %q, want %q: every text block must be concatenated, not overwritten",
			resp.Text, "hello world")
	}
	if resp.Metadata["thinking"] != "first I then I" {
		t.Errorf("Metadata[thinking] = %q, want %q", resp.Metadata["thinking"], "first I then I")
	}
}

func TestAnthropicReportsTheTraceAndItsCounters(t *testing.T) {
	srv := anthropicMultiBlockStub(t)
	defer srv.Close()

	resp := generate(t, &AnthropicConfig{APIKey: "k"}, NewClaudeOpus5())

	if resp.Thinking != "first I then I" {
		t.Errorf("Thinking = %q", resp.Thinking)
	}
	// The deprecated key keeps saying exactly what the typed field says.
	if resp.Metadata["thinking"] != resp.Thinking {
		t.Errorf("Metadata[thinking] = %q, Thinking = %q: the two must not diverge",
			resp.Metadata["thinking"], resp.Thinking)
	}
	if resp.Metadata["thinking_signature"] != "sig-b" {
		t.Errorf("Metadata[thinking_signature] = %q, want the last block's signature",
			resp.Metadata["thinking_signature"])
	}
	if resp.Metadata["thinking_redacted"] != "ENCRYPTED" {
		t.Errorf("Metadata[thinking_redacted] = %q", resp.Metadata["thinking_redacted"])
	}
	// Anthropic counts thinking inside output_tokens, so the totals stay put.
	if resp.Usage.ThinkingTokens != 12 {
		t.Errorf("ThinkingTokens = %d, want 12", resp.Usage.ThinkingTokens)
	}
	if resp.Usage.CompletionTokens != 30 || resp.Usage.TotalTokens != 40 {
		t.Errorf("a subset counter must not inflate the totals: %+v", resp.Usage)
	}
	if resp.Usage.AnswerTokens() != 18 {
		t.Errorf("AnswerTokens() = %d, want 18", resp.Usage.AnswerTokens())
	}
}

// ============================================================================
// ANTHROPIC THINKING: CAPABILITIES
// ============================================================================

var (
	// Every Claude that can think carries the configuration, and only those.
	_ ThinkingModel = (*Claude37Sonnet)(nil)
	_ ThinkingModel = (*ClaudeSonnet4)(nil)
	_ ThinkingModel = (*ClaudeOpus46)(nil)
	_ ThinkingModel = (*ClaudeOpus48)(nil)
	_ ThinkingModel = (*ClaudeFable5)(nil)
	_ ThinkingModel = (*ClaudeOpus5)(nil)
	_ ThinkingModel = (*ClaudeSonnet5)(nil)
	_ ThinkingModel = (*AnthropicModel)(nil)
)

// TestClaude3xCannotCarryThinkingConfiguration is the structural guard the
// design turns on: the accessor lives on anthropicThinkingOptions, never on
// anthropicOptions, so a model whose API has no thinking field cannot be handed
// a thinking knob even by mistake.
func TestClaude3xCannotCarryThinkingConfiguration(t *testing.T) {
	for _, m := range []Model{
		NewClaude35Sonnet(), NewClaude35Haiku(),
		NewClaude3Opus(), NewClaude3Haiku(), NewClaude3Sonnet(),
	} {
		if _, ok := m.(ThinkingModel); ok {
			t.Errorf("%s implements ThinkingModel; Claude 3.x has no thinking field to configure",
				m.ModelName())
		}
		if d := ModelThinkingDimensions(m); d != 0 {
			t.Errorf("%s dimensions = %b, want 0", m.ModelName(), d)
		}
	}
}

func TestAnthropicThinkingDimensionsPerGeneration(t *testing.T) {
	const report = ThinkingCanReportTokens | ThinkingCanReportTrace

	budget := ThinkingCanToggle | ThinkingCanSetBudget | ThinkingCanHideTrace | report
	adaptiveBudget := ThinkingCanToggle | ThinkingCanSetEffort | ThinkingCanSetBudget |
		ThinkingCanHideTrace | report
	adaptive := ThinkingCanToggle | ThinkingCanSetEffort | ThinkingCanHideTrace | report
	alwaysOn := ThinkingCanSetEffort | report

	tests := []struct {
		model Model
		want  ThinkingDimension
	}{
		{NewClaude35Sonnet(), 0},
		{NewClaude3Opus(), 0},
		{NewClaude37Sonnet(), budget},
		{NewClaudeSonnet4(), budget},
		{NewClaudeOpus4(), budget},
		{NewClaudeSonnet45(), budget},
		{NewClaudeOpus45(), budget},
		{NewClaudeHaiku45(), budget},
		{NewClaudeOpus41(), budget},
		{NewClaudeOpus46(), adaptiveBudget},
		{NewClaudeSonnet46(), adaptiveBudget},
		{NewClaudeOpus47(), adaptive},
		{NewClaudeOpus48(), adaptive},
		{NewClaudeFable5(), alwaysOn},
		{NewClaudeOpus5(), adaptive},
		{NewClaudeSonnet5(), adaptive},

		// The generic type resolves from the id it was handed, which is what the
		// dead supportsThinking() it replaces got wrong: that one answered "yes"
		// for every id, Claude 3 included.
		{NewAnthropicModel("claude-3-haiku-20240307"), 0},
		{NewAnthropicModel("claude-3-7-sonnet-latest"), budget},
		{NewAnthropicModel("claude-opus-4-20250514"), budget},
		{NewAnthropicModel("claude-opus-4-5@20251101"), budget}, // Vertex id form
		{NewAnthropicModel("claude-sonnet-4-6"), adaptiveBudget},
		{NewAnthropicModel("claude-opus-4-7"), adaptive},
		{NewAnthropicModel("claude-fable-5"), alwaysOn},
		{NewAnthropicModel("claude-sonnet-5"), adaptive},
		// An id released after this build gets the current generation's dialect,
		// which is the only one no knob has ever been withdrawn from.
		{NewAnthropicModel("claude-opus-9"), adaptive},
		{NewAnthropicModel("not-a-claude"), 0},
	}

	for _, tc := range tests {
		if got := ModelThinkingDimensions(tc.model); got != tc.want {
			t.Errorf("%s dimensions = %06b, want %06b", tc.model.ModelName(), got, tc.want)
		}
	}
}

// TestZeroValueClaudeKeepsItsCapabilities guards the reason dimensions are
// resolved from ModelName() rather than stored by a constructor: a zero-value
// composite literal is constructible outside the package and must not silently
// lose its thinking support.
func TestZeroValueClaudeKeepsItsCapabilities(t *testing.T) {
	if got := ModelThinkingDimensions(&ClaudeSonnet4{}); got != ModelThinkingDimensions(NewClaudeSonnet4()) {
		t.Errorf("&ClaudeSonnet4{} dimensions = %06b", got)
	}
}

// ============================================================================
// ANTHROPIC THINKING: THE PORTABLE SURFACE
// ============================================================================

func TestAnthropicPortableThinkingRequests(t *testing.T) {
	tests := []struct {
		name         string
		model        Model
		thinking     string
		outputConfig string
	}{
		// A bare enable becomes "you decide" where the generation models it.
		{"enable on 4.6", Thinking(NewClaudeOpus46()), `{"type":"adaptive"}`, ""},
		{"enable on opus 5", Thinking(NewClaudeOpus5()), `{"type":"adaptive"}`, ""},
		// 3.7-4.5 only ever spoke in fixed budgets, so an enable has to become
		// one: 60% of the window left by max_tokens.
		{"enable on sonnet 4", Thinking(NewClaudeSonnet4()),
			`{"type":"enabled","budget_tokens":4914}`, ""},
		// Fable 5 reasons server-side; an opt-in must still send nothing.
		{"enable on fable 5", Thinking(NewClaudeFable5()), "", ""},

		// Off is honoured where there is a real switch.
		{"disable opus 5", NoThinking(NewClaudeOpus5()), `{"type":"disabled"}`, ""},
		{"disable fable 5", NoThinking(NewClaudeFable5()), "", ""},

		// Effort is clamped to the generation's ladder rather than forwarded and
		// rejected: 4.6 has no xhigh, and Anthropic has no rung below low.
		{"xhigh clamped on 4.6", Thinking(NewClaudeOpus46(), WithThinkingEffort(ThinkingEffortXHigh)),
			`{"type":"adaptive"}`, `{"effort":"high"}`},
		{"minimal clamped up", Thinking(NewClaudeFable5(), WithThinkingEffort(ThinkingEffortMinimal)),
			"", `{"effort":"low"}`},
		{"xhigh kept on 4.8", Thinking(NewClaudeOpus48(), WithThinkingEffort(ThinkingEffortXHigh)),
			`{"type":"adaptive"}`, `{"effort":"xhigh"}`},

		// An unpinned budget is clamped into the model's legal window...
		{"budget clamped up on 4.6", Thinking(NewClaudeOpus46(), WithThinkingBudget(500)),
			`{"type":"enabled","budget_tokens":1024}`, ""},
		{"budget clamped down on sonnet 4", Thinking(NewClaudeSonnet4(), WithThinkingBudget(999999)),
			`{"type":"enabled","budget_tokens":8191}`, ""},
		// ...and translated to an effort where the generation takes no budget.
		{"budget becomes effort on opus 5", Thinking(NewClaudeOpus5(), WithThinkingBudget(20000)),
			`{"type":"adaptive"}`, `{"effort":"medium"}`},

		{"dynamic on 4.7", Thinking(NewClaudeOpus47(), WithDynamicThinking()), `{"type":"adaptive"}`, ""},
		// 3.7-4.5 predate the adaptive config, so "you decide" becomes a budget.
		{"dynamic on sonnet 4", Thinking(NewClaudeSonnet4(), WithDynamicThinking()),
			`{"type":"enabled","budget_tokens":4914}`, ""},

		// Trace visibility rides on the thinking config's display field.
		{"omit the trace on opus 5",
			Thinking(NewClaudeOpus5(), WithThinkingTrace(ThinkingTraceOmit)),
			`{"type":"adaptive","display":"omitted"}`, ""},
		{"include the trace with a budget",
			Thinking(NewClaudeSonnet46(), WithThinkingBudget(4096), WithThinkingTrace(ThinkingTraceInclude)),
			`{"type":"enabled","budget_tokens":4096,"display":"summarized"}`, ""},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			body := anthropicWire(t, tc.model)
			wantJSON(t, body, "thinking", tc.thinking)
			wantJSON(t, body, "output_config", tc.outputConfig)
		})
	}
}

// TestAnthropicPinnedDimensionsBeatThePortableOnes is the backward-compatibility
// contract in one test: a value a per-model setter put there is forwarded
// verbatim, while the same value arriving through the portable surface is
// adapted to what the model accepts.
func TestAnthropicPinnedDimensionsBeatThePortableOnes(t *testing.T) {
	// Pinned: 500 is below the API's floor and goes out anyway, exactly as it
	// did before this feature existed.
	body := anthropicWire(t, NewClaudeSonnet46().WithThinkingBudget(500))
	wantJSON(t, body, "thinking", `{"type":"enabled","budget_tokens":500}`)

	// Unpinned: the same number is clamped into the legal window.
	body = anthropicWire(t, Thinking(NewClaudeSonnet46(), WithThinkingBudget(500)))
	wantJSON(t, body, "thinking", `{"type":"enabled","budget_tokens":1024}`)

	// A pinned budget survives on a generation that has no budget knob at all,
	// because the caller named the generic model's own setter.
	body = anthropicWire(t, NewAnthropicModel("claude-opus-5").WithThinkingBudget(5000))
	wantJSON(t, body, "thinking", `{"type":"enabled","budget_tokens":5000}`)

	// The same request through the portable surface is translated instead.
	body = anthropicWire(t, Thinking(NewAnthropicModel("claude-opus-5"), WithThinkingBudget(5000)))
	wantJSON(t, body, "thinking", `{"type":"adaptive"}`)
	wantJSON(t, body, "output_config", `{"effort":"low"}`)
}

// TestAnthropicPortableSettersShareStorageWithTheLegacyOnes checks the claim the
// doc comments make: there is one storage, so the two views cannot disagree.
func TestAnthropicPortableSettersShareStorageWithTheLegacyOnes(t *testing.T) {
	m := NewClaudeSonnet46().WithThinkingBudget(4096).WithEffort(EffortMax)
	to := m.ThinkingOptions()

	if to.Budget() != 4096 {
		t.Errorf("Budget() = %d, want the value WithThinkingBudget stored", to.Budget())
	}
	if to.Effort() != EffortMax {
		t.Errorf("Effort() = %q, want the value WithEffort stored", to.Effort())
	}
	if !to.Enabled() {
		t.Error("a positive thinking budget must read as enabled")
	}

	// WithAdaptiveThinking is the dynamic budget under another name.
	if !NewClaudeOpus48().WithAdaptiveThinking().ThinkingOptions().DynamicBudget() {
		t.Error("WithAdaptiveThinking must read as a dynamic budget")
	}
	// And a disable reads as a disable from either side.
	if !NewClaudeOpus5().WithThinkingDisabled().ThinkingOptions().Disabled() {
		t.Error("WithThinkingDisabled must read as disabled")
	}
	if !NoThinking(NewClaudeOpus5()).ThinkingOptions().Disabled() {
		t.Error("NoThinking must read as disabled")
	}

	// A non-positive budget clears the request rather than enabling thinking on
	// a nonsense ceiling, which is what the old `budget > 0` wire guard did.
	cleared := NewClaudeSonnet4().WithThinkingBudget(4096).WithThinkingBudget(0)
	if cleared.ThinkingOptions().Enabled() || cleared.ThinkingOptions().Budget() != 0 {
		t.Errorf("a zero budget must clear the request: %+v", cleared.ThinkingOptions())
	}
}

// TestAnthropicRecordsWhatItTranslated checks the breadcrumb, so an adaptation
// the caller did not ask for is never invisible.
func TestAnthropicRecordsWhatItTranslated(t *testing.T) {
	var c capture
	srv := anthropicStub(t, &c)
	defer srv.Close()

	resp := generate(t, &AnthropicConfig{APIKey: "k"},
		Thinking(NewClaudeOpus46(), WithThinkingEffort(ThinkingEffortXHigh)))

	if got := resp.Metadata["thinking_translation"]; got != "effort xhigh clamped to high" {
		t.Errorf("Metadata[thinking_translation] = %q", got)
	}

	// Nothing translated, nothing to report.
	resp = generate(t, &AnthropicConfig{APIKey: "k"},
		Thinking(NewClaudeOpus46(), WithThinkingEffort(ThinkingEffortHigh)))
	if got, ok := resp.Metadata["thinking_translation"]; ok {
		t.Errorf("Metadata[thinking_translation] = %q, want the key to be absent", got)
	}
}

// TestAnthropicThinkingIsNeverAnError covers the never-error posture across the
// whole catalogue: every knob, on every model type, including the ones with no
// thinking at all.
func TestAnthropicThinkingIsNeverAnError(t *testing.T) {
	var c capture
	srv := anthropicStub(t, &c)
	defer srv.Close()

	models := []Model{
		NewClaude35Sonnet(), NewClaude35Haiku(), NewClaude3Opus(), NewClaude3Haiku(),
		NewClaude3Sonnet(), NewClaude37Sonnet(), NewClaudeSonnet4(), NewClaudeOpus4(),
		NewClaudeSonnet45(), NewClaudeOpus45(), NewClaudeHaiku45(), NewClaudeOpus41(),
		NewClaudeOpus46(), NewClaudeSonnet46(), NewClaudeOpus47(), NewClaudeOpus48(),
		NewClaudeFable5(), NewClaudeOpus5(), NewClaudeSonnet5(),
		NewAnthropicModel("claude-opus-9"), NewAnthropicModel("not-a-claude"),
	}
	for _, m := range models {
		generate(t, &AnthropicConfig{APIKey: "k"},
			Thinking(m, WithThinkingEffort(ThinkingEffortMax), WithThinkingBudget(1),
				WithThinkingTrace(ThinkingTraceOmit)))
		generate(t, &AnthropicConfig{APIKey: "k"}, NoThinking(m))
	}
}
