package lingo

import (
	"encoding/json"
	"testing"
)

// ============================================================================
// COMPILE-TIME CAPABILITY ASSERTIONS
// ============================================================================
//
// The OpenAI reasoning options and the shared oaiOptions are the two places a
// ThinkingOptions accessor is promoted in this family. A type that loses one
// breaks the build here rather than silently degrading to "thinking requested,
// nothing sent" -- and the standard OpenAI models must keep NOT satisfying
// ThinkingModel, which is asserted by the runtime test below rather than here,
// because a failed interface assertion is not a compile-time error.

var (
	// First-party OpenAI: the reasoning option set only.
	_ ThinkingModel = (*O1)(nil)
	_ ThinkingModel = (*GPT5)(nil)
	_ ThinkingModel = (*GPT56Sol)(nil)
	_ ThinkingModel = (*OpenAIReasoningModel)(nil)

	// One promoted accessor on oaiOptions covers five providers.
	_ ThinkingModel = (*AzureOpenAIReasoningModel)(nil)
	_ ThinkingModel = (*AzureOpenAIModel)(nil)
	_ ThinkingModel = (*Grok43)(nil)
	_ ThinkingModel = (*Grok420NonReasoning)(nil)
	_ ThinkingModel = (*XAIModel)(nil)
	_ ThinkingModel = (*DeepSeekV4Pro)(nil)
	_ ThinkingModel = (*DeepSeekModel)(nil)
	_ ThinkingModel = (*OpenRouterModel)(nil)
	_ ThinkingModel = (*OpenAICompatibleModel)(nil)
)

// ============================================================================
// OPENAI: GOLDEN REQUESTS
// ============================================================================
//
// This provider is the one place the opt-in guarantee was already spent before
// thinking became a feature: 26 of the 27 reasoning constructors seed a
// reasoning_effort, so those models have always put the field on the wire. The
// table below is a baseline recorded against the pre-refactor code, and every
// row must keep marshalling byte for byte. A row that changes is a behaviour
// change for code written before the portable surface existed.

// openAIWire runs one Generate against a stub and returns the request body.
func openAIWire(t *testing.T, m Model) map[string]any {
	t.Helper()
	var c capture
	srv := oaiStub(t, &c)
	defer srv.Close()
	generate(t, &OpenAIConfig{APIKey: "k", BaseURL: srv.URL}, m)
	return c.body
}

// oaiCompatWire runs one Generate against a stub through any provider whose
// config carries a base URL, and returns the request body.
func oaiCompatWire(t *testing.T, cfg func(baseURL string) ProviderConfig, m Model) map[string]any {
	t.Helper()
	var c capture
	srv := oaiStub(t, &c)
	defer srv.Close()
	generate(t, cfg(srv.URL), m)
	return c.body
}

func xaiCfg(baseURL string) ProviderConfig {
	return &XAIConfig{APIKey: "k", BaseURL: baseURL}
}

func deepSeekCfg(baseURL string) ProviderConfig {
	return &DeepSeekConfig{APIKey: "k", BaseURL: baseURL}
}

func openRouterCfg(baseURL string) ProviderConfig {
	return &OpenRouterConfig{APIKey: "k", BaseURL: baseURL}
}

func oaiCompatCfg(baseURL string) ProviderConfig {
	return &OpenAICompatibleConfig{APIKey: "k", BaseURL: baseURL}
}

func TestOpenAISeededEffortsStillReachTheWire(t *testing.T) {
	// Every named reasoning constructor and the exact value it has always
	// seeded. Nothing here calls a setter: this is what an untouched model
	// sends.
	tests := []struct {
		model  Model
		effort string
	}{
		{NewO1(), "medium"},
		{NewO1Mini(), "medium"},
		{NewO1Pro(), "high"},
		{NewO1Preview(), "medium"},
		{NewO3(), "medium"},
		{NewO3Mini(), "medium"},
		{NewO3Pro(), "high"},
		{NewO4Mini(), "medium"},
		{NewGPT5(), "medium"},
		{NewGPT5Mini(), "medium"},
		{NewGPT5Nano(), "medium"},
		{NewGPT5Pro(), "high"},
		{NewGPT51(), "medium"},
		{NewGPT51Mini(), "medium"},
		{NewGPT51Nano(), "medium"},
		{NewGPT51Codex(), "medium"},
		{NewGPT51CodexMini(), "medium"},
		{NewGPT54Nano(), "medium"},
		{NewGPT54Mini(), "medium"},
		{NewGPT54(), "medium"},
		{NewGPT54Pro(), "high"},
		{NewGPT55(), "medium"},
		{NewGPT55Pro(), "high"},
		{NewGPT56Sol(), "medium"},
		{NewGPT56Terra(), "medium"},
		{NewGPT56Luna(), "low"},
	}

	for _, tc := range tests {
		t.Run(tc.model.ModelName(), func(t *testing.T) {
			if got := openAIWire(t, tc.model)["reasoning_effort"]; got != tc.effort {
				t.Errorf("reasoning_effort = %v, want %q", got, tc.effort)
			}
		})
	}
}

func TestOpenAIThinkingGoldenRequests(t *testing.T) {
	tests := []struct {
		name   string
		model  Model
		effort string // "" means the key must be absent
	}{
		// Standard models carry no thinking configuration at all: their option
		// struct is a sibling of the reasoning one, and gpt-4o rejects the field.
		{"gpt-4o untouched", NewGPT4o(), ""},
		{"gpt-4.1 untouched", NewGPT41(), ""},
		{"generic standard model", NewOpenAIModel("gpt-4o-2024-11-20"), ""},

		// The generic reasoning model is the one type that seeds nothing.
		{"generic reasoning untouched", NewOpenAIReasoningModel("gpt-5.6-sol"), ""},

		// A setter's value is forwarded exactly as given, including the rungs
		// the model does not accept and the strings lingo has never heard of:
		// the caller named an OpenAI-specific knob on an OpenAI-specific type.
		{"setter overrides the seed", NewGPT5().WithReasoningEffort("low"), "low"},
		{"xhigh forwarded to a model without it", NewGPT5().WithReasoningEffort("xhigh"), "xhigh"},
		{"minimal forwarded to 5.1", NewGPT51().WithReasoningEffort("minimal"), "minimal"},
		{"none forwarded to o1", NewO1().WithReasoningEffort("none"), "none"},
		{"max forwarded even though chat completions has no max",
			NewGPT56Sol().WithReasoningEffort("max"), "max"},
		{"an unknown word is still forwarded",
			NewOpenAIReasoningModel("gpt-9").WithReasoningEffort("obsessive"), "obsessive"},
		// Clearing the effort clears the field, as it always has.
		{"empty string clears the seed", NewGPT5().WithReasoningEffort(""), ""},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			body := openAIWire(t, tc.model)
			got, ok := body["reasoning_effort"]
			if tc.effort == "" {
				if ok {
					t.Errorf("reasoning_effort = %v, want the key to be absent", got)
				}
				return
			}
			if got != tc.effort {
				t.Errorf("reasoning_effort = %v, want %q", got, tc.effort)
			}
		})
	}
}

func TestOpenAIPortableThinkingRequests(t *testing.T) {
	tests := []struct {
		name   string
		model  Model
		effort string
	}{
		// A bare enable has nothing to send: chat completions has no toggle,
		// and the seeded effort already says how hard to think.
		{"enable on gpt-5.6", Thinking(NewGPT56Sol()), "medium"},

		// Effort is clamped to the model's own ladder rather than forwarded and
		// rejected. "max" is Responses-API only, so it can never be sent here.
		{"max clamped to xhigh on 5.6",
			Thinking(NewGPT56Sol(), WithThinkingEffort(ThinkingEffortMax)), "xhigh"},
		{"xhigh clamped to high on the original gpt-5",
			Thinking(NewGPT5(), WithThinkingEffort(ThinkingEffortXHigh)), "high"},
		{"minimal clamped up on 5.1, which withdrew it",
			Thinking(NewGPT51(), WithThinkingEffort(ThinkingEffortMinimal)), "low"},
		{"minimal kept on the original gpt-5",
			Thinking(NewGPT5(), WithThinkingEffort(ThinkingEffortMinimal)), "minimal"},
		{"gpt-5-pro takes high and nothing else",
			Thinking(NewGPT5Pro(), WithThinkingEffort(ThinkingEffortLow)), "high"},

		// Off is the rung "none" where the model has it. Where it does not, off is
		// a no-op -- never an error, and never a drop: the seeded effort these
		// constructors have always pinned stays exactly where it was, because
		// sending no reasoning_effort at all would hand the model to OpenAI's own
		// server-side default, which reasons harder than the caller asked for and
		// bills accordingly.
		{"off on 5.1 is effort none", NoThinking(NewGPT51()), "none"},
		{"off on 5.6 is effort none", NoThinking(NewGPT56Sol()), "none"},
		{"off on gpt-5 keeps the seed: the family has no none", NoThinking(NewGPT5()), "medium"},
		{"off on o3 keeps the seed", NoThinking(NewO3()), "medium"},

		// Chat completions has no thinking token budget, so a portable budget
		// degrades to the effort bucket it falls in.
		{"budget becomes an effort",
			Thinking(NewOpenAIReasoningModel("gpt-5.6-sol"), WithThinkingBudget(30000)), "high"},
		// A budget that maps to minimal clamps up to the shallowest rung 5.6
		// has, never down to "none": asking to think a little is not asking to
		// stop thinking.
		{"a small budget becomes the shallowest rung the model has",
			Thinking(NewOpenAIReasoningModel("gpt-5.6-sol"), WithThinkingBudget(1000)), "low"},
		// A seeded effort already says how hard to think, so it stands and the
		// budget is dropped with a note rather than silently overriding it.
		{"a budget does not override a seeded effort",
			Thinking(NewGPT56Sol(), WithThinkingBudget(30000)), "medium"},

		// o1-mini takes no reasoning_effort at all. A bare enable leaves the
		// pinned seed exactly where it was...
		{"o1-mini keeps its seed", Thinking(NewO1Mini()), "medium"},
		// ...while a portable depth replaces it, and is then dropped because
		// this model has nowhere to put one.
		{"a portable effort on o1-mini sends nothing",
			Thinking(NewO1Mini(), WithThinkingEffort(ThinkingEffortHigh)), ""},

		// An id this library has not seen gets the current family's ladder.
		{"unknown id gets the current ladder",
			Thinking(NewOpenAIReasoningModel("gpt-9"), WithThinkingEffort(ThinkingEffortXHigh)), "xhigh"},

		// A standard model stores nothing and sends nothing.
		{"portable thinking on gpt-4o is a no-op",
			Thinking(NewGPT4o(), WithThinkingEffort(ThinkingEffortHigh)), ""},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			body := openAIWire(t, tc.model)
			got, ok := body["reasoning_effort"]
			if tc.effort == "" {
				if ok {
					t.Errorf("reasoning_effort = %v, want the key to be absent", got)
				}
				return
			}
			if got != tc.effort {
				t.Errorf("reasoning_effort = %v, want %q", got, tc.effort)
			}
		})
	}
}

// TestOpenAIPinnedEffortBeatsThePortableOne is the backward-compatibility
// contract in one test: a value the per-model setter put there is forwarded
// verbatim, while the same value arriving through the portable surface is
// clamped to what the model accepts.
func TestOpenAIPinnedEffortBeatsThePortableOne(t *testing.T) {
	body := openAIWire(t, NewGPT56Sol().WithReasoningEffort("max"))
	if got := body["reasoning_effort"]; got != "max" {
		t.Errorf("pinned effort = %v, want it forwarded verbatim", got)
	}

	body = openAIWire(t, Thinking(NewGPT56Sol(), WithThinkingEffort(ThinkingEffortMax)))
	if got := body["reasoning_effort"]; got != "xhigh" {
		t.Errorf("portable effort = %v, want it clamped to the model's ladder", got)
	}

	// The portable surface takes the pin off the seeded default, which is the
	// only reason the row above can be clamped at all: every named reasoning
	// constructor ships a pinned effort.
	if Thinking(NewGPT56Sol(), WithThinkingEffort(ThinkingEffortMax)).
		ThinkingOptions().isPinned(ThinkingCanSetEffort) {
		t.Error("a portable WithThinkingEffort must leave the dimension unpinned")
	}
	if !NewGPT56Sol().ThinkingOptions().isPinned(ThinkingCanSetEffort) {
		t.Error("a constructor-seeded effort must be pinned")
	}
}

func TestOpenAIThinkingSharesStorageWithTheLegacySetter(t *testing.T) {
	m := NewGPT56Sol().WithReasoningEffort("high")
	if got := m.ThinkingOptions().Effort(); got != ThinkingEffortHigh {
		t.Errorf("Effort() = %q, want the value WithReasoningEffort stored", got)
	}
	// The setter records a level, not a mode: reasoning_effort is the only
	// thinking field this dialect has.
	if m.ThinkingOptions().Enabled() {
		t.Error("WithReasoningEffort must not claim thinking was toggled on")
	}
	if !NoThinking(NewGPT56Sol()).ThinkingOptions().Disabled() {
		t.Error("NoThinking must read as disabled")
	}
	// And the portable surface writes the same field the setter reads.
	if got := Thinking(NewGPT5(), WithThinkingEffort(ThinkingEffortLow)).
		ThinkingOptions().Effort(); got != ThinkingEffortLow {
		t.Errorf("Effort() = %q after a portable write", got)
	}
}

func TestOpenAIThinkingDimensionsPerModel(t *testing.T) {
	tests := []struct {
		model Model
		want  ThinkingDimension
	}{
		// Standard models: no thinking knob of any kind.
		{NewGPT4o(), 0},
		{NewGPT41Nano(), 0},
		{NewOpenAIModel("gpt-4o"), 0},

		// Reasoning models: a depth and a token count, never a budget, a toggle
		// or a trace -- chat completions returns no reasoning text at all.
		{NewGPT56Sol(), ThinkingCanSetEffort | ThinkingCanReportTokens},
		{NewO3(), ThinkingCanSetEffort | ThinkingCanReportTokens},
		{NewOpenAIReasoningModel("gpt-9"), ThinkingCanSetEffort | ThinkingCanReportTokens},

		// o1-mini is documented not to accept reasoning_effort.
		{NewO1Mini(), ThinkingCanReportTokens},
		{NewOpenAIReasoningModel("o1-mini-2024-09-12"), ThinkingCanReportTokens},
	}

	for _, tc := range tests {
		t.Run(tc.model.ModelName(), func(t *testing.T) {
			if got := ModelThinkingDimensions(tc.model); got != tc.want {
				t.Errorf("ModelThinkingDimensions = %b, want %b", got, tc.want)
			}
		})
	}
}

// TestOpenAIStandardModelsCarryNoThinkingStorage is the structural half of the
// gate: a standard model must not even be able to hold a thinking option, which
// is what stops a promoted accessor from handing gpt-4o a knob that 400s.
func TestOpenAIStandardModelsCarryNoThinkingStorage(t *testing.T) {
	for _, m := range []Model{NewGPT4o(), NewGPT4oMini(), NewGPT41(), NewGPT41Mini(),
		NewGPT41Nano(), NewGPT4(), NewGPT4Turbo(), NewGPT35Turbo(), NewOpenAIModel("gpt-4o")} {
		if _, ok := m.(ThinkingModel); ok {
			t.Errorf("%T satisfies ThinkingModel, want the standard option struct to keep it out", m)
		}
	}
}

// TestOpenAIEffortLadders pins the per-model vocabularies, which are the whole
// reason the portable surface can be clamped rather than forwarded blind.
func TestOpenAIEffortLadders(t *testing.T) {
	has := func(id string, e ThinkingEffort) bool {
		for _, a := range openAIEffortLadder(id) {
			if a == e {
				return true
			}
		}
		return false
	}
	tests := []struct {
		id   string
		e    ThinkingEffort
		want bool
	}{
		// "max" is Responses-API only and can never be sent from here.
		{"gpt-5.6-sol", ThinkingEffortMax, false},
		{"gpt-5.5", ThinkingEffortMax, false},
		// "xhigh" arrived with gpt-5.4.
		{"gpt-5.6-sol", ThinkingEffortXHigh, true},
		{"gpt-5.4", ThinkingEffortXHigh, true},
		{"gpt-5.1", ThinkingEffortXHigh, false},
		{"gpt-5.1-codex-max", ThinkingEffortXHigh, true},
		// "minimal" is the original GPT-5 family only.
		{"gpt-5", ThinkingEffortMinimal, true},
		{"gpt-5-mini", ThinkingEffortMinimal, true},
		{"gpt-5.1", ThinkingEffortMinimal, false},
		{"o3", ThinkingEffortMinimal, false},
		// "none" arrived with gpt-5.1.
		{"gpt-5.1", ThinkingEffortNone, true},
		{"gpt-5", ThinkingEffortNone, false},
		{"o1", ThinkingEffortNone, false},
		// gpt-5-pro takes high and nothing else.
		{"gpt-5-pro", ThinkingEffortHigh, true},
		{"gpt-5-pro", ThinkingEffortMedium, false},
	}
	for _, tc := range tests {
		if got := has(tc.id, tc.e); got != tc.want {
			t.Errorf("%s accepts %s = %t, want %t", tc.id, tc.e, got, tc.want)
		}
	}
	if len(openAIEffortLadder("o1-mini")) != 0 {
		t.Error("o1-mini accepts no reasoning_effort at all")
	}
}

// ============================================================================
// OPENAI: RESPONSE
// ============================================================================

func TestOpenAIReportsThinkingTokens(t *testing.T) {
	var c capture
	srv := oaiStub(t, &c)
	defer srv.Close()

	// No opt-in anywhere: reporting is unconditional.
	resp := generate(t, &OpenAIConfig{APIKey: "k", BaseURL: srv.URL}, NewGPT4o())

	if resp.Usage.ThinkingTokens != 3 {
		t.Errorf("ThinkingTokens = %d, want 3", resp.Usage.ThinkingTokens)
	}
	// Reasoning tokens are a breakdown of the completion total, so the totals
	// stay exactly as the provider reported them.
	if resp.Usage.CompletionTokens != 7 || resp.Usage.TotalTokens != 18 {
		t.Errorf("a subset counter must not inflate the totals: %+v", resp.Usage)
	}
	if resp.Usage.AnswerTokens() != 4 {
		t.Errorf("AnswerTokens() = %d, want 4", resp.Usage.AnswerTokens())
	}
	// The deprecated metadata key keeps working for one release.
	if resp.Metadata["reasoning_tokens"] != "3" {
		t.Errorf("Metadata[reasoning_tokens] = %q", resp.Metadata["reasoning_tokens"])
	}
	// Chat completions returns no trace, whatever the stub puts beside it.
	if resp.Thinking != "" {
		t.Errorf("Thinking = %q, want empty: OpenAI returns no trace on this API", resp.Thinking)
	}
}

func TestOpenAIRecordsWhatItTranslated(t *testing.T) {
	var c capture
	srv := oaiStub(t, &c)
	defer srv.Close()

	resp := generate(t, &OpenAIConfig{APIKey: "k", BaseURL: srv.URL},
		Thinking(NewGPT56Sol(), WithThinkingEffort(ThinkingEffortMax)))
	if got := resp.Metadata["thinking_translation"]; got != "effort max clamped to xhigh" {
		t.Errorf("Metadata[thinking_translation] = %q", got)
	}

	// Nothing translated, nothing to report.
	resp = generate(t, &OpenAIConfig{APIKey: "k", BaseURL: srv.URL}, NewGPT56Sol())
	if got, ok := resp.Metadata["thinking_translation"]; ok {
		t.Errorf("Metadata[thinking_translation] = %q, want the key to be absent", got)
	}
}

// ============================================================================
// THE OPENAI-COMPATIBLE FAMILY: GOLDEN REQUESTS
// ============================================================================

func TestXAIThinkingGoldenRequests(t *testing.T) {
	tests := []struct {
		name   string
		model  Model
		effort string
	}{
		// grok-4.3 has seeded reasoning_effort="low" since it shipped.
		{"grok-4.3 untouched", NewGrok43(), "low"},
		{"grok-4.3 setter", NewGrok43().WithReasoningEffort(XAIEffortHigh), "high"},
		{"grok-4.3 none", NewGrok43().WithReasoningEffort(XAIEffortNone), "none"},

		// Every other Grok sends nothing unless the raw-id type is used.
		{"grok-4.5 untouched", NewGrok45(), ""},
		{"grok 4.20 reasoning untouched", NewGrok420Reasoning(), ""},
		{"grok 4.20 non-reasoning untouched", NewGrok420NonReasoning(), ""},
		{"grok build untouched", NewGrokBuild01(), ""},
		{"raw id untouched", NewXAIModel("grok-4.6"), ""},
		{"raw id setter", NewXAIModel("grok-4.6").WithReasoningEffort("xhigh"), "xhigh"},

		// The portable surface is gated per model type, because xAI's own two
		// references disagree about which Grok takes the parameter.
		{"portable effort on 4.3", Thinking(NewGrok43(), WithThinkingEffort(ThinkingEffortMedium)), "medium"},
		{"portable effort on 4.5", Thinking(NewGrok45(), WithThinkingEffort(ThinkingEffortHigh)), "high"},
		{"xhigh clamped on 4.5", Thinking(NewGrok45(), WithThinkingEffort(ThinkingEffortXHigh)), "high"},
		{"portable effort on the multi-agent model is dropped: there the " +
			"parameter counts agents, not depth",
			Thinking(NewGrok420MultiAgent(), WithThinkingEffort(ThinkingEffortHigh)), ""},
		{"portable effort on the non-reasoning model is dropped",
			Thinking(NewGrok420NonReasoning(), WithThinkingEffort(ThinkingEffortHigh)), ""},
		{"portable effort on the reasoning 4.20, which hard-rejects unknown params",
			Thinking(NewGrok420Reasoning(), WithThinkingEffort(ThinkingEffortHigh)), ""},

		// Off is honoured only where the ladder has a rung for it.
		{"off on 4.3 is effort none", NoThinking(NewGrok43()), "none"},
		{"off on 4.5 is a silent no-op", NoThinking(NewGrok45()), ""},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			body := oaiCompatWire(t, xaiCfg, tc.model)
			got, ok := body["reasoning_effort"]
			if tc.effort == "" {
				if ok {
					t.Errorf("reasoning_effort = %v, want the key to be absent", got)
				}
				return
			}
			if got != tc.effort {
				t.Errorf("reasoning_effort = %v, want %q", got, tc.effort)
			}
		})
	}
}

func TestDeepSeekThinkingGoldenRequests(t *testing.T) {
	tests := []struct {
		name     string
		model    Model
		thinking string // "" means the key must be absent
		effort   string
	}{
		// Thinking is on by default on every V4 model, so an untouched model
		// sends neither field and the interesting request is the one that stops.
		{"untouched", NewDeepSeekV4Pro(), "", ""},
		{"disabled", NewDeepSeekV4Pro().WithThinkingDisabled(), "disabled", ""},
		{"enabled explicitly", NewDeepSeekV4Flash().WithThinkingEnabled(), "enabled", ""},
		{"effort only", NewDeepSeekV4Flash().WithReasoningEffort("max"), "", "max"},
		// The two knobs are independent fields, so both go out when both were
		// asked for, exactly as they always have.
		{"disabled with an effort",
			NewDeepSeekV4Flash().WithThinkingDisabled().WithReasoningEffort("high"), "disabled", "high"},
		{"raw id disabled", NewDeepSeekModel("deepseek-v4-pro").WithThinkingDisabled(), "disabled", ""},

		// The portable surface reaches both knobs.
		{"portable enable", Thinking(NewDeepSeekV4Pro()), "enabled", ""},
		{"portable off", NoThinking(NewDeepSeekV4Pro()), "disabled", ""},
		{"portable effort", Thinking(NewDeepSeekV4Pro(), WithThinkingEffort(ThinkingEffortMax)),
			"enabled", "max"},
		// DeepSeek's ladder is low, high, max: medium is silently folded up to
		// high by the API, so lingo clamps it down where the caller can see it.
		{"medium clamped down", Thinking(NewDeepSeekV4Pro(), WithThinkingEffort(ThinkingEffortMedium)),
			"enabled", "low"},
		// No token budget exists, so a portable budget becomes a depth.
		{"budget becomes an effort", Thinking(NewDeepSeekV4Pro(), WithThinkingBudget(70000)),
			"enabled", "high"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			body := oaiCompatWire(t, deepSeekCfg, tc.model)
			th, ok := body["thinking"]
			if tc.thinking == "" {
				if ok {
					t.Errorf("thinking = %v, want the key to be absent", th)
				}
			} else {
				obj, isObj := th.(map[string]any)
				if !isObj || obj["type"] != tc.thinking {
					t.Errorf("thinking = %v, want type %q", th, tc.thinking)
				}
			}
			got, ok := body["reasoning_effort"]
			if tc.effort == "" {
				if ok {
					t.Errorf("reasoning_effort = %v, want the key to be absent", got)
				}
				return
			}
			if got != tc.effort {
				t.Errorf("reasoning_effort = %v, want %q", got, tc.effort)
			}
		})
	}
}

// TestDeepSeekCallerExtraFieldBeatsTheDerivedOne pins the precedence rule the
// migration introduced: the setters no longer write into extraFields, so a
// caller's own escape-hatch value can no longer be clobbered by call order.
func TestDeepSeekCallerExtraFieldBeatsTheDerivedOne(t *testing.T) {
	body := oaiCompatWire(t, deepSeekCfg,
		NewDeepSeekV4Pro().
			WithThinkingDisabled().
			WithExtraField("thinking", map[string]any{"type": "enabled"}))

	th, ok := body["thinking"].(map[string]any)
	if !ok || th["type"] != "enabled" {
		t.Errorf("thinking = %v, want the caller's own value to win", body["thinking"])
	}

	// And the reverse order gives the same answer, which is the point.
	body = oaiCompatWire(t, deepSeekCfg,
		NewDeepSeekV4Pro().
			WithExtraField("thinking", map[string]any{"type": "enabled"}).
			WithThinkingDisabled())
	th, ok = body["thinking"].(map[string]any)
	if !ok || th["type"] != "enabled" {
		t.Errorf("thinking = %v, want the caller's own value to win whatever the call order", body["thinking"])
	}
}

// TestOpenRouterReasoningObject is the fidelity test for the one endpoint whose
// native request shape is already the portable one.
func TestOpenRouterReasoningObject(t *testing.T) {
	tests := []struct {
		name  string
		model Model
		want  map[string]any // nil means the reasoning key must be absent
	}{
		// The default path: an untouched model sends no reasoning object.
		{"untouched", NewOpenRouterModel("anthropic/claude-opus-5"), nil},

		// The legacy setters, each landing on its own sub-field.
		{"effort", NewOpenRouterModel("openai/gpt-5.6-sol").WithReasoningEffort("xhigh"),
			map[string]any{"effort": "xhigh"}},
		{"max tokens", NewOpenRouterModel("anthropic/claude-opus-5").WithReasoningMaxTokens(2000),
			map[string]any{"max_tokens": float64(2000)}},
		{"excluded", NewOpenRouterModel("anthropic/claude-opus-5").WithReasoningExcluded(),
			map[string]any{"exclude": true}},
		// Both depth spellings at once is the caller's choice: OpenRouter
		// normalizes whichever the upstream speaks, so lingo does not choose.
		{"effort and max tokens",
			NewOpenRouterModel("anthropic/claude-opus-5").
				WithReasoningEffort("high").WithReasoningMaxTokens(4000).WithReasoningExcluded(),
			map[string]any{"effort": "high", "max_tokens": float64(4000), "exclude": true}},
		// A pinned budget is forwarded below the 1024 floor OpenRouter
		// documents for Anthropic upstreams, so the caller sees that error.
		{"pinned budget below the floor",
			NewOpenRouterModel("anthropic/claude-opus-5").WithReasoningMaxTokens(500),
			map[string]any{"max_tokens": float64(500)}},

		// The portable surface.
		{"portable enable", Thinking(NewOpenRouterModel("anthropic/claude-opus-5")),
			map[string]any{"enabled": true}},
		{"portable off", NoThinking(NewOpenRouterModel("anthropic/claude-opus-5")),
			map[string]any{"enabled": false}},
		{"portable effort", Thinking(NewOpenRouterModel("openai/gpt-5.6-sol"),
			WithThinkingEffort(ThinkingEffortMax)), map[string]any{"effort": "max"}},
		// An unpinned budget is clamped into the documented window instead.
		{"portable budget clamped up", Thinking(NewOpenRouterModel("anthropic/claude-opus-5"),
			WithThinkingBudget(500)), map[string]any{"max_tokens": float64(1024)}},
		{"portable budget clamped down", Thinking(NewOpenRouterModel("anthropic/claude-opus-5"),
			WithThinkingBudget(999999)), map[string]any{"max_tokens": float64(128000)}},
		// Thinking() turns thinking on, so asking to hide the trace through it
		// says both things; the legacy setter above touches only the trace.
		{"portable trace omission", Thinking(NewOpenRouterModel("anthropic/claude-opus-5"),
			WithThinkingTrace(ThinkingTraceOmit)), map[string]any{"enabled": true, "exclude": true}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			body := oaiCompatWire(t, openRouterCfg, tc.model)

			// The flat OpenAI-dialect field must never appear beside the object.
			if got, ok := body["reasoning_effort"]; ok {
				t.Errorf("reasoning_effort = %v, want OpenRouter to carry the effort in its object only", got)
			}

			got, ok := body["reasoning"]
			if tc.want == nil {
				if ok {
					t.Errorf("reasoning = %v, want the key to be absent", got)
				}
				return
			}
			obj, isObj := got.(map[string]any)
			if !isObj {
				t.Fatalf("reasoning = %v, want an object", got)
			}
			if len(obj) != len(tc.want) {
				t.Errorf("reasoning = %v, want exactly %v", obj, tc.want)
			}
			for k, v := range tc.want {
				if obj[k] != v {
					t.Errorf("reasoning[%s] = %v, want %v", k, obj[k], v)
				}
			}
		})
	}
}

func TestAzureThinkingGoldenRequests(t *testing.T) {
	azure := func(baseURL string) ProviderConfig {
		return &AzureOpenAIConfig{Endpoint: baseURL, APIKey: "k", APIVersion: AzureAPIVersionV1}
	}
	tests := []struct {
		name   string
		model  Model
		effort string
	}{
		// A reasoning deployment sends nothing until asked; unlike first-party
		// OpenAI, its constructor seeds no effort.
		{"untouched", NewAzureOpenAIReasoningModel("my-gpt-5-deployment"), ""},
		{"setter", NewAzureOpenAIReasoningModel("d").WithReasoningEffort("high"), "high"},
		{"portable effort",
			Thinking(NewAzureOpenAIReasoningModel("d"), WithThinkingEffort(ThinkingEffortHigh)), "high"},
		// max is Responses-API only on Azure too.
		{"max clamped",
			Thinking(NewAzureOpenAIReasoningModel("d"), WithThinkingEffort(ThinkingEffortMax)), "xhigh"},
		// A standard deployment stores the request and sends none of it.
		{"standard deployment", Thinking(NewAzureOpenAIModel("d"),
			WithThinkingEffort(ThinkingEffortHigh)), ""},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			body := oaiCompatWire(t, azure, tc.model)
			got, ok := body["reasoning_effort"]
			if tc.effort == "" {
				if ok {
					t.Errorf("reasoning_effort = %v, want the key to be absent", got)
				}
				return
			}
			if got != tc.effort {
				t.Errorf("reasoning_effort = %v, want %q", got, tc.effort)
			}
		})
	}
}

// TestGenericEndpointSynthesizesNothing is the never-error posture where it is
// hardest: there is no catalogue behind BaseURL, so lingo forwards what the
// caller named and invents nothing.
func TestGenericEndpointSynthesizesNothing(t *testing.T) {
	tests := []struct {
		name   string
		model  Model
		effort string
	}{
		{"untouched", NewOpenAICompatibleModel("qwen3-coder"), ""},
		{"setter forwards any dialect's word",
			NewOpenAICompatibleModel("qwen3-coder").WithReasoningEffort("high"), "high"},
		{"an unknown word is forwarded too",
			NewOpenAICompatibleModel("qwen3-coder").WithReasoningEffort("deep"), "deep"},
		// The portable surface has no ladder to clamp to here, so it sends
		// nothing rather than guessing at the endpoint's vocabulary.
		{"portable effort is dropped",
			Thinking(NewOpenAICompatibleModel("qwen3-coder"), WithThinkingEffort(ThinkingEffortHigh)), ""},
		{"portable enable is dropped", Thinking(NewOpenAICompatibleModel("qwen3-coder")), ""},
		{"portable off is dropped", NoThinking(NewOpenAICompatibleModel("qwen3-coder")), ""},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			body := oaiCompatWire(t, oaiCompatCfg, tc.model)
			for _, key := range []string{"thinking", "reasoning"} {
				if got, ok := body[key]; ok {
					t.Errorf("%s = %v, want an unknown endpoint to be sent no object lingo invented", key, got)
				}
			}
			got, ok := body["reasoning_effort"]
			if tc.effort == "" {
				if ok {
					t.Errorf("reasoning_effort = %v, want the key to be absent", got)
				}
				return
			}
			if got != tc.effort {
				t.Errorf("reasoning_effort = %v, want %q", got, tc.effort)
			}
		})
	}
}

// ============================================================================
// THE OPENAI-COMPATIBLE FAMILY: RESPONSE
// ============================================================================

func TestOAICompatReportsTheTraceAndItsCounters(t *testing.T) {
	var c capture
	srv := oaiStub(t, &c)
	defer srv.Close()

	// No opt-in: both halves of the reporting are unconditional.
	resp := generate(t, &DeepSeekConfig{APIKey: "k", BaseURL: srv.URL}, NewDeepSeekV4Pro())

	if resp.Thinking != "thought" {
		t.Errorf("Thinking = %q, want the reasoning_content the endpoint returned", resp.Thinking)
	}
	if resp.Metadata["reasoning_content"] != "thought" {
		t.Errorf("Metadata[reasoning_content] = %q, want the deprecated key kept", resp.Metadata["reasoning_content"])
	}
	if resp.Usage.ThinkingTokens != 3 {
		t.Errorf("ThinkingTokens = %d, want 3", resp.Usage.ThinkingTokens)
	}
	if resp.Usage.CompletionTokens != 7 || resp.Usage.TotalTokens != 18 {
		t.Errorf("a subset counter must not inflate the totals: %+v", resp.Usage)
	}
	if resp.Metadata["reasoning_tokens"] != "3" {
		t.Errorf("Metadata[reasoning_tokens] = %q", resp.Metadata["reasoning_tokens"])
	}
}

func TestOAICompatRecordsWhatItTranslated(t *testing.T) {
	var c capture
	srv := oaiStub(t, &c)
	defer srv.Close()

	resp := generate(t, &DeepSeekConfig{APIKey: "k", BaseURL: srv.URL},
		Thinking(NewDeepSeekV4Pro(), WithThinkingEffort(ThinkingEffortMedium)))
	if got := resp.Metadata["thinking_translation"]; got != "effort medium clamped to low" {
		t.Errorf("Metadata[thinking_translation] = %q", got)
	}
}

// TestIsReasoningModelCannotContradictTheRequest checks the metadata flag is
// derived rather than merely remembered.
func TestIsReasoningModelCannotContradictTheRequest(t *testing.T) {
	var c capture
	srv := oaiStub(t, &c)
	defer srv.Close()

	resp := generate(t, &OpenAICompatibleConfig{APIKey: "k", BaseURL: srv.URL},
		Thinking(NewOpenAICompatibleModel("qwen3")))
	if resp.Metadata["is_reasoning_model"] != "true" {
		t.Errorf("is_reasoning_model = %q, want thinking asked for to count",
			resp.Metadata["is_reasoning_model"])
	}
}

// TestIsReasoningModelIsFalseWhenTheBodySwitchedThinkingOff is the other
// direction of the same claim, on the models the constructor flag hid.
//
// Every DeepSeek V4 model is built with reasoning already true, so the flag was
// only ever OR-ed upward and had nothing that could clear it. A request carrying
// thinking={"type":"disabled"} still reported is_reasoning_model=true, which is
// the model's biography rather than the request's -- and a caller reading it to
// price the call, pick a route or decide whether to expect a trace was reading
// the wrong thing. The disable lingo actually sent is the last word.
func TestIsReasoningModelIsFalseWhenTheBodySwitchedThinkingOff(t *testing.T) {
	tests := []struct {
		name  string
		cfg   func(string) ProviderConfig
		model Model
	}{
		{"deepseek portable disable", deepSeekCfg, NoThinking(NewDeepSeekV4Pro())},
		{"deepseek per-model disable", deepSeekCfg, NewDeepSeekV4Pro().WithThinkingDisabled()},
		{"deepseek flash portable disable", deepSeekCfg, NoThinking(NewDeepSeekV4Flash())},
		{"deepseek raw id portable disable", deepSeekCfg, NoThinking(NewDeepSeekModel("deepseek-v4-pro"))},
		{"openrouter portable disable", openRouterCfg, NoThinking(NewOpenRouterModel("deepseek/deepseek-v4-pro"))},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var c capture
			srv := oaiStub(t, &c)
			defer srv.Close()

			resp := generate(t, tc.cfg(srv.URL), tc.model)

			// The body really did switch thinking off, in whichever dialect.
			off := false
			if v, ok := c.body["thinking"].(map[string]any); ok && v["type"] == "disabled" {
				off = true
			}
			if v, ok := c.body["reasoning"].(map[string]any); ok && v["enabled"] == false {
				off = true
			}
			if !off {
				t.Fatalf("body did not switch thinking off: thinking=%v reasoning=%v",
					c.body["thinking"], c.body["reasoning"])
			}
			// ...so the metadata beside it cannot claim otherwise.
			if got := resp.Metadata["is_reasoning_model"]; got != "false" {
				t.Errorf("is_reasoning_model = %q, want false: the body lingo just sent "+
					"switched thinking off", got)
			}
		})
	}

	// A disable that never reached the wire leaves the flag alone: on a model
	// with no off switch NoThinking is a no-op, and a no-op cannot make a
	// reasoning model stop being one.
	t.Run("a dropped disable does not clear the flag", func(t *testing.T) {
		var c capture
		srv := oaiStub(t, &c)
		defer srv.Close()

		resp := generate(t, xaiCfg(srv.URL), NoThinking(NewXAIModel("grok-4.5").WithReasoningEffort("high")))
		if got := resp.Metadata["is_reasoning_model"]; got != "true" {
			t.Errorf("is_reasoning_model = %q, want true: nothing switched thinking off", got)
		}
	})
}

// TestIsReasoningModelIsFalseWhenOffIsSpelledAsAnEffort is the same claim for
// the other spelling of off. Not every endpoint has a toggle: xAI's grok-4.3
// switches reasoning off with reasoning_effort "none", and a plan carrying that
// rung is as much a not-reasoning request as one carrying a disabled thinking
// object.
//
// The sharp edge is that the two surfaces used to disagree about one identical
// body. NewGrok43().WithReasoningEffort(XAIEffortNone) has reported false since
// that setter shipped, because xai.go writes `m.reasoning = e != XAIEffortNone`
// beside it; the portable NoThinking(NewGrok43()) produced byte-for-byte the
// same request and reported true, because the constructor flag it inherited had
// nothing to clear it. Same bytes, two answers.
func TestIsReasoningModelIsFalseWhenOffIsSpelledAsAnEffort(t *testing.T) {
	// Every row here must put reasoning_effort "none" on the wire, so the flag
	// beside it has exactly one honest answer.
	off := []struct {
		name  string
		model Model
	}{
		{"portable disable", NoThinking(NewGrok43())},
		// The control: the per-model setter, which has always reported false.
		{"per-model setter", NewGrok43().WithReasoningEffort(XAIEffortNone)},
		{"portable effort none", Thinking(NewGrok43(), WithThinkingEffort(ThinkingEffortNone))},
		// A raw id told the literal keeps it, pinned, through a later disable.
		{"pinned none on a raw id",
			NoThinking(NewXAIModel("grok-4.3").WithReasoningEffort(XAIEffortNone))},
		{"per-model setter on a raw id", NewXAIModel("grok-4.3").WithReasoningEffort(XAIEffortNone)},
	}

	for _, tc := range off {
		t.Run(tc.name, func(t *testing.T) {
			var c capture
			srv := oaiStub(t, &c)
			defer srv.Close()

			resp := generate(t, xaiCfg(srv.URL), tc.model)

			if got := c.body["reasoning_effort"]; got != "none" {
				t.Fatalf("reasoning_effort = %v, want none: this row is meant to send off", got)
			}
			if got := resp.Metadata["is_reasoning_model"]; got != "false" {
				t.Errorf("is_reasoning_model = %q, want false: the body lingo just sent "+
					"carries reasoning_effort \"none\"", got)
			}
		})
	}

	// The two surfaces must agree on an identical body, which is the whole
	// point: same bytes, same answer.
	t.Run("both surfaces agree on identical bodies", func(t *testing.T) {
		var portableBody, legacyBody map[string]any
		var portableFlag, legacyFlag string

		var c1 capture
		srv1 := oaiStub(t, &c1)
		portableFlag = generate(t, xaiCfg(srv1.URL), NoThinking(NewGrok43())).Metadata["is_reasoning_model"]
		portableBody = c1.body
		srv1.Close()

		var c2 capture
		srv2 := oaiStub(t, &c2)
		legacyFlag = generate(t, xaiCfg(srv2.URL),
			NewGrok43().WithReasoningEffort(XAIEffortNone)).Metadata["is_reasoning_model"]
		legacyBody = c2.body
		srv2.Close()

		got, _ := json.Marshal(portableBody)
		want, _ := json.Marshal(legacyBody)
		if string(got) != string(want) {
			t.Fatalf("the two surfaces built different requests:\n portable %s\n legacy   %s", got, want)
		}
		if portableFlag != legacyFlag {
			t.Errorf("identical bodies reported differently: NoThinking said %q, "+
				"WithReasoningEffort(none) said %q", portableFlag, legacyFlag)
		}
		if portableFlag != "false" {
			t.Errorf("is_reasoning_model = %q, want false", portableFlag)
		}
	})

	// The positive controls, so the test cannot pass by nailing the flag to
	// false. A request that really does reason still reports true, whichever
	// surface asked and whether or not a depth was named.
	on := []struct {
		name       string
		model      Model
		wantEffort string
	}{
		{"untouched, seeded effort", NewGrok43(), "low"},
		{"portable enable", Thinking(NewGrok43()), "low"},
		{"per-model setter, a real rung", NewGrok43().WithReasoningEffort(XAIEffortHigh), "high"},
		{"portable effort", Thinking(NewGrok43(), WithThinkingEffort(ThinkingEffortMedium)), "medium"},
	}

	for _, tc := range on {
		t.Run("still true/"+tc.name, func(t *testing.T) {
			var c capture
			srv := oaiStub(t, &c)
			defer srv.Close()

			resp := generate(t, xaiCfg(srv.URL), tc.model)

			if got := c.body["reasoning_effort"]; got != tc.wantEffort {
				t.Fatalf("reasoning_effort = %v, want %q", got, tc.wantEffort)
			}
			if got := resp.Metadata["is_reasoning_model"]; got != "true" {
				t.Errorf("is_reasoning_model = %q, want true: the body asks for reasoning at %q",
					got, tc.wantEffort)
			}
		})
	}

	// And the case that has no off at all on the wire: grok-4.5 has no "none"
	// rung, so NoThinking sends nothing and cannot make the flag lie either way.
	t.Run("a dropped off leaves the flag alone", func(t *testing.T) {
		var c capture
		srv := oaiStub(t, &c)
		defer srv.Close()

		resp := generate(t, xaiCfg(srv.URL), NoThinking(NewXAIModel("grok-4.5").WithReasoningEffort("medium")))
		if got := c.body["reasoning_effort"]; got != "medium" {
			t.Fatalf("reasoning_effort = %v, want medium", got)
		}
		if got := resp.Metadata["is_reasoning_model"]; got != "true" {
			t.Errorf("is_reasoning_model = %q, want true: this request still reasons, at medium", got)
		}
	})
}

// TestPortableEffortNoneIsNotClampedIntoRealThinking guards the seam between the
// two fixes above. planThinking now falls through to the depth section when an
// off has nowhere to go, and an unpinned "none" sitting in that section would be
// handed to clampEffort, which raises it to the ladder's lowest real rung --
// turning "do not think" into "think a little" and putting a body on the wire
// that the is_reasoning_model flag beside it would then deny.
//
// "Think less" is not "off", which is the rule the toggle section already
// applies when it refuses to clamp into a none rung. The depth section has to
// apply it too.
func TestPortableEffortNoneIsNotClampedIntoRealThinking(t *testing.T) {
	const dropped = "thinking off dropped: model has no off switch"

	t.Run("plan", func(t *testing.T) {
		// Unpinned: the none was the off request, so it goes with it.
		var portable ThinkingOptions
		portable.WithEffort(ThinkingEffortNone)
		checkPlan(t, planThinking(&portable, dimsEffortOnly, budgetRange{}, ladderNoOff...),
			wantPlan{notes: dropped})

		// Pinned: a literal the caller named on this model's own setter, still
		// forwarded verbatim even though the ladder has no such rung.
		var pinned ThinkingOptions
		pinned.WithEffort(ThinkingEffortNone).pin(ThinkingCanSetEffort)
		pinned.Disable()
		checkPlan(t, planThinking(&pinned, dimsEffortOnly, budgetRange{}, ladderNoOff...),
			wantPlan{effort: ThinkingEffortNone, notes: dropped})

		// A real depth beside a dropped off is still kept and still clamped: this
		// is not a retreat from the no-op rule, only a refusal to invent thinking.
		var depth ThinkingOptions
		depth.WithEffort(ThinkingEffortMax).Disable()
		checkPlan(t, planThinking(&depth, dimsEffortOnly, budgetRange{}, ladderNoOff...),
			wantPlan{effort: ThinkingEffortHigh, notes: dropped + "; effort max clamped to high"})
	})

	t.Run("wire", func(t *testing.T) {
		// The raw-id escape hatch is given the ladder with no none rung, because
		// lingo cannot tell which model an id points at.
		wantJSON(t, oaiCompatWire(t, xaiCfg,
			Thinking(NewXAIModel("grok-4.3"), WithThinkingEffort(ThinkingEffortNone))),
			"reasoning_effort", "")
		// The pinned literal still reaches the wire.
		wantJSON(t, oaiCompatWire(t, xaiCfg,
			NoThinking(NewXAIModel("grok-4.3").WithReasoningEffort(XAIEffortNone))),
			"reasoning_effort", `"none"`)
	})
}

// ============================================================================
// NEVER AN ERROR
// ============================================================================

// TestOpenAIFamilyThinkingIsNeverAnError covers the never-error posture across
// every model type in this lane, including the ones with no thinking at all and
// the ones whose endpoint hard-rejects parameters it does not know.
func TestOpenAIFamilyThinkingIsNeverAnError(t *testing.T) {
	var c capture
	srv := oaiStub(t, &c)
	defer srv.Close()

	openAI := []Model{
		NewGPT4o(), NewGPT4oMini(), NewGPT4Turbo(), NewGPT4(), NewGPT41(), NewGPT41Mini(),
		NewGPT41Nano(), NewGPT35Turbo(), NewOpenAIModel("gpt-4o"),
		NewO1(), NewO1Mini(), NewO1Pro(), NewO1Preview(), NewO3(), NewO3Mini(), NewO3Pro(),
		NewO4Mini(), NewGPT5(), NewGPT5Mini(), NewGPT5Nano(), NewGPT5Pro(), NewGPT51(),
		NewGPT51Mini(), NewGPT51Nano(), NewGPT51Codex(), NewGPT51CodexMini(), NewGPT54Nano(),
		NewGPT54Mini(), NewGPT54(), NewGPT54Pro(), NewGPT55(), NewGPT55Pro(), NewGPT56Sol(),
		NewGPT56Terra(), NewGPT56Luna(), NewOpenAIReasoningModel("gpt-9"),
		NewOpenAIReasoningModel(""),
	}
	for _, m := range openAI {
		cfg := func() ProviderConfig { return &OpenAIConfig{APIKey: "k", BaseURL: srv.URL} }
		generate(t, cfg(), Thinking(m, WithThinkingEffort(ThinkingEffortMax),
			WithThinkingBudget(1), WithThinkingTrace(ThinkingTraceOmit)))
		generate(t, cfg(), NoThinking(m))
	}

	compat := []struct {
		cfg   func(string) ProviderConfig
		model Model
	}{
		{xaiCfg, NewGrok45()}, {xaiCfg, NewGrok43()}, {xaiCfg, NewGrok420Reasoning()},
		{xaiCfg, NewGrok420NonReasoning()}, {xaiCfg, NewGrok420MultiAgent()},
		{xaiCfg, NewGrokBuild01()}, {xaiCfg, NewXAIModel("grok-9")},
		{deepSeekCfg, NewDeepSeekV4Flash()}, {deepSeekCfg, NewDeepSeekV4Pro()},
		{deepSeekCfg, NewDeepSeekModel("deepseek-chat")},
		{openRouterCfg, NewOpenRouterModel("anthropic/claude-opus-5")},
		{oaiCompatCfg, NewOpenAICompatibleModel("qwen3")},
	}
	for _, tc := range compat {
		generate(t, tc.cfg(srv.URL), Thinking(tc.model, WithThinkingEffort(ThinkingEffortMax),
			WithThinkingBudget(1), WithThinkingTrace(ThinkingTraceOmit)))
		generate(t, tc.cfg(srv.URL), NoThinking(tc.model))
		generate(t, tc.cfg(srv.URL), Thinking(tc.model, WithDynamicThinking()))
	}
}
