package lingo

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// ============================================================================
// COMPILE-TIME CAPABILITY ASSERTIONS
// ============================================================================
//
// One representative model per provider that can carry thinking configuration,
// including the reasoning variants whose option structs are siblings of the
// standard ones rather than extensions and so need their own accessor. A
// provider that loses its ThinkingOptions accessor breaks the build here rather
// than silently degrading to "thinking requested, nothing sent".
//
// The reverse -- a model that must NOT satisfy ThinkingModel, because its API
// has no thinking field and storing a knob on it would be a lie -- cannot be
// asserted at compile time, since a failed interface assertion is not a build
// error. TestModelsThatCannotThinkCarryNoConfiguration does that at runtime.

var (
	// Anthropic: only the Claudes that can think embed the thinking options.
	_ ThinkingModel = (*Claude37Sonnet)(nil)
	_ ThinkingModel = (*ClaudeSonnet46)(nil)
	_ ThinkingModel = (*ClaudeOpus5)(nil)
	_ ThinkingModel = (*ClaudeFable5)(nil)
	_ ThinkingModel = (*AnthropicModel)(nil)

	// OpenAI: the standard and reasoning option sets are siblings, and only the
	// reasoning one carries thinking.
	_ ThinkingModel = (*GPT5)(nil)
	_ ThinkingModel = (*O3)(nil)
	_ ThinkingModel = (*OpenAIReasoningModel)(nil)

	// Google: one promoted accessor covers all nineteen Gemini types.
	_ ThinkingModel = (*Gemini25Flash)(nil)
	_ ThinkingModel = (*Gemini3Pro)(nil)
	_ ThinkingModel = (*GoogleModel)(nil)

	// Bedrock: the Claude and Nova families, plus the flat generic model.
	_ ThinkingModel = (*BedrockClaudeSonnet5)(nil)
	_ ThinkingModel = (*BedrockClaude37Sonnet)(nil)
	_ ThinkingModel = (*BedrockNovaPro)(nil)
	_ ThinkingModel = (*BedrockModel)(nil)

	// The oaicompat family, all sharing one accessor on oaiOptions.
	_ ThinkingModel = (*AzureOpenAIReasoningModel)(nil)
	_ ThinkingModel = (*Grok43)(nil)
	_ ThinkingModel = (*XAIModel)(nil)
	_ ThinkingModel = (*DeepSeekV4Pro)(nil)
	_ ThinkingModel = (*OpenRouterModel)(nil)
	_ ThinkingModel = (*OpenAICompatibleModel)(nil)

	// The three providers caching does not reach at all still carry thinking.
	_ ThinkingModel = (*CommandAPlus)(nil)
	_ ThinkingModel = (*CommandAReasoning)(nil)
	_ ThinkingModel = (*CohereModel)(nil)
	_ ThinkingModel = (*SonarDeepResearch)(nil)
	_ ThinkingModel = (*PerplexityModel)(nil)
	_ ThinkingModel = (*Qwen3)(nil)
	_ ThinkingModel = (*OllamaModel)(nil)
)

// ============================================================================
// SUPPORT MATRIX
// ============================================================================

func TestThinkingSupportPerProvider(t *testing.T) {
	tests := []struct {
		provider ProviderType
		want     ThinkSupport
		label    string
	}{
		{ProviderAnthropic, ThinkSupportControl, "control"},
		{ProviderOpenAI, ThinkSupportControl, "control"},
		{ProviderGoogle, ThinkSupportControl, "control"},
		{ProviderBedrock, ThinkSupportControl, "control"},
		{ProviderAzure, ThinkSupportControl, "control"},
		{ProviderXAI, ThinkSupportControl, "control"},
		{ProviderDeepSeek, ThinkSupportControl, "control"},
		{ProviderOpenRouter, ThinkSupportControl, "control"},
		{ProviderCohere, ThinkSupportControl, "control"},
		{ProviderOllama, ThinkSupportControl, "control"},
		{ProviderPerplexity, ThinkSupportControl, "control"},
		// The endpoint behind BaseURL decides, so lingo forwards only what the
		// caller named and synthesizes nothing.
		{ProviderOpenAICompatible, ThinkSupportUsageOnly, "usage-only"},
		{ProviderType("not-a-provider"), ThinkSupportNone, "none"},
	}

	for _, tc := range tests {
		if got := ThinkingSupport(tc.provider); got != tc.want {
			t.Errorf("ThinkingSupport(%q) = %v, want %v", tc.provider, got, tc.want)
		}
		if got := tc.want.String(); got != tc.label {
			t.Errorf("%v.String() = %q, want %q", tc.want, got, tc.label)
		}
	}
	if got := ThinkSupport(99).String(); got != "none" {
		t.Errorf("ThinkSupport(99).String() = %q, want the none label", got)
	}
}

func TestThinkingDimensionsPerProvider(t *testing.T) {
	const report = ThinkingCanReportTokens | ThinkingCanReportTrace

	tests := []struct {
		provider ProviderType
		want     ThinkingDimension
	}{
		{ProviderAnthropic, ThinkingCanToggle | ThinkingCanSetEffort | ThinkingCanSetBudget | ThinkingCanHideTrace | report},
		{ProviderGoogle, ThinkingCanToggle | ThinkingCanSetEffort | ThinkingCanSetBudget | ThinkingCanHideTrace | report},
		{ProviderOpenRouter, ThinkingCanToggle | ThinkingCanSetEffort | ThinkingCanSetBudget | ThinkingCanHideTrace | report},
		{ProviderBedrock, ThinkingCanToggle | ThinkingCanSetEffort | ThinkingCanSetBudget | ThinkingCanReportTrace},
		{ProviderCohere, ThinkingCanToggle | ThinkingCanSetBudget | ThinkingCanReportTrace},
		{ProviderDeepSeek, ThinkingCanToggle | ThinkingCanSetEffort | report},
		{ProviderOllama, ThinkingCanToggle | ThinkingCanSetEffort | ThinkingCanReportTrace},
		{ProviderXAI, ThinkingCanSetEffort | report},
		{ProviderOpenAI, ThinkingCanSetEffort | ThinkingCanReportTokens},
		{ProviderAzure, ThinkingCanSetEffort | ThinkingCanReportTokens},
		{ProviderPerplexity, ThinkingCanSetEffort | ThinkingCanReportTokens},
		{ProviderOpenAICompatible, report},
		{ProviderType("not-a-provider"), 0},
	}

	for _, tc := range tests {
		if got := ThinkingDimensions(tc.provider); got != tc.want {
			t.Errorf("ThinkingDimensions(%q) = %06b, want %06b", tc.provider, got, tc.want)
		}
	}

	// Has is "every bit in want", not "any".
	d := ThinkingCanToggle | ThinkingCanSetBudget
	if !d.Has(ThinkingCanToggle) || !d.Has(ThinkingCanToggle|ThinkingCanSetBudget) {
		t.Error("Has must report the bits that are present")
	}
	if d.Has(ThinkingCanSetEffort) || d.Has(ThinkingCanToggle|ThinkingCanSetEffort) {
		t.Error("Has must not report a bit that is absent, even beside one that is present")
	}
	if !d.Has(0) {
		t.Error("Has(0) must be vacuously true")
	}
}

// thinkingRequestDimensions are the four knobs that change a request. The other
// two only describe what comes back, so a model can have them and still be
// unable to take any instruction.
const thinkingRequestDimensions = ThinkingCanToggle | ThinkingCanSetEffort |
	ThinkingCanSetBudget | ThinkingCanHideTrace

// TestModelsThatCannotThinkCarryNoConfiguration is the honesty half of the
// capability surface. A model whose API has no thinking field must advertise
// nothing, whether it declines to carry configuration at all (GPT-4o, Claude
// 3.x, the non-Claude Bedrock families) or carries it because it shares an
// option struct with a sibling that can think (Command R, Gemini 1.5, Ollama's
// non-thinking tags). The second group is the dangerous one: the storage is
// there, so only thinkingDimensions stops it reaching the wire.
func TestModelsThatCannotThinkCarryNoConfiguration(t *testing.T) {
	for _, m := range []Model{
		NewGPT4o(),
		NewClaude3Opus(),
		NewClaude35Sonnet(),
		NewGemini15Pro(),
		NewGemini20Flash(),
		NewGoogleModel("gemini-9-not-a-real-model"),
		NewBedrockClaude35Sonnet(),
		NewBedrockNovaPro(),
		NewBedrockTitanTextPremier(),
		NewBedrockLlama33Instruct70B(),
		NewBedrockMistralLarge(),
		NewCommandR(),
		NewCohereModel("command-r-08-2024"),
		NewSonar(),
		NewSonarPro(),
		NewPerplexityModel("sonar"),
		NewLlama33(),
		NewOllamaModel("mistral"),
		NewAzureOpenAIModel("my-deployment"),
	} {
		if got := ModelThinkingDimensions(m); got != 0 {
			t.Errorf("%s/%s: ModelThinkingDimensions = %06b, want 0",
				m.Provider(), m.ModelName(), got)
		}
	}

	// The report-only models are a weaker claim: they reason on their own terms
	// and lingo can only relay what came back, so they must advertise reporting
	// and nothing on the request side.
	for _, m := range []Model{
		NewO1Mini(),
		NewGrok420NonReasoning(),
		NewGrok420MultiAgent(),
		NewSonarReasoning(),
		NewSonarReasoningPro(),
	} {
		d := ModelThinkingDimensions(m)
		if d&thinkingRequestDimensions != 0 {
			t.Errorf("%s/%s: dimensions %06b claim a request-side knob this model has none of",
				m.Provider(), m.ModelName(), d)
		}
		if d == 0 {
			t.Errorf("%s/%s: dimensions = 0, want the reporting bits", m.Provider(), m.ModelName())
		}
	}

	// A nil model is the coarse answer's floor, not a panic.
	if got := ModelThinkingDimensions(nil); got != 0 {
		t.Errorf("ModelThinkingDimensions(nil) = %06b, want 0", got)
	}
	// A model that declares nothing falls back to its provider's answer, which
	// is what makes the escape hatches useful before their tables are updated.
	if got := ModelThinkingDimensions(&unknownProviderModel{}); got != 0 {
		t.Errorf("ModelThinkingDimensions of an unknown provider = %06b, want 0", got)
	}
}

// unknownProviderModel implements nothing but Model, which is what an external
// implementation of the interface looks like.
type unknownProviderModel struct{}

func (unknownProviderModel) ModelName() string      { return "someone-elses-model" }
func (unknownProviderModel) Provider() ProviderType { return ProviderType("elsewhere") }
func (unknownProviderModel) SystemPrompt() string   { return "" }

// ============================================================================
// THE NEUTRAL LADDER
// ============================================================================

func TestThinkingEffortRank(t *testing.T) {
	ladder := []ThinkingEffort{
		ThinkingEffortNone, ThinkingEffortMinimal, ThinkingEffortLow,
		ThinkingEffortMedium, ThinkingEffortHigh, ThinkingEffortXHigh, ThinkingEffortMax,
	}
	for i, e := range ladder {
		if got := e.rank(); got != i {
			t.Errorf("%q.rank() = %d, want %d", e, got, i)
		}
	}
	// The type is open on purpose: anything a per-model setter forwards ranks
	// off-ladder rather than being coerced onto it.
	for _, e := range []ThinkingEffort{"", "obsessive", "MEDIUM", "xxhigh"} {
		if got := e.rank(); got != -1 {
			t.Errorf("%q.rank() = %d, want -1 for an off-ladder value", e, got)
		}
	}
}

func TestClampEffort(t *testing.T) {
	var (
		full     = []ThinkingEffort{ThinkingEffortNone, ThinkingEffortMinimal, ThinkingEffortLow, ThinkingEffortMedium, ThinkingEffortHigh, ThinkingEffortXHigh, ThinkingEffortMax}
		noNone   = []ThinkingEffort{ThinkingEffortLow, ThinkingEffortMedium, ThinkingEffortHigh}
		withNone = []ThinkingEffort{ThinkingEffortNone, ThinkingEffortLow, ThinkingEffortMedium, ThinkingEffortHigh}
		deep     = []ThinkingEffort{ThinkingEffortMedium, ThinkingEffortHigh, ThinkingEffortMax}
	)

	tests := []struct {
		name    string
		in      ThinkingEffort
		allowed []ThinkingEffort
		want    ThinkingEffort
		wantOK  bool
	}{
		{"exact rung is kept", ThinkingEffortMedium, full, ThinkingEffortMedium, true},
		{"above the top clamps down", ThinkingEffortMax, noNone, ThinkingEffortHigh, true},
		{"between rungs clamps down", ThinkingEffortXHigh, noNone, ThinkingEffortHigh, true},
		{"below the bottom clamps up", ThinkingEffortMinimal, deep, ThinkingEffortMedium, true},
		// The floor rule: a request to think a little must never become a
		// request not to think. Without it, minimal on a ladder that has none
		// but no minimal would silently switch reasoning off.
		{"minimal never clamps down to none", ThinkingEffortMinimal, withNone, ThinkingEffortLow, true},
		{"low never clamps down to none", ThinkingEffortLow, []ThinkingEffort{ThinkingEffortNone, ThinkingEffortHigh}, ThinkingEffortHigh, true},
		// none is a candidate only when it is what was asked for.
		{"none is honoured when the ladder has it", ThinkingEffortNone, withNone, ThinkingEffortNone, true},
		// On a ladder without an off switch, asking for none clamps up like any
		// other rung; planThinking is what refuses to send the result.
		{"none on a ladder without it clamps up", ThinkingEffortNone, noNone, ThinkingEffortLow, true},
		{"off-ladder values are dropped", "obsessive", full, "", false},
		{"an empty ladder drops everything", ThinkingEffortHigh, nil, "", false},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := clampEffort(tc.in, tc.allowed...)
			if got != tc.want || ok != tc.wantOK {
				t.Errorf("clampEffort(%q, %v) = %q, %t, want %q, %t",
					tc.in, tc.allowed, got, ok, tc.want, tc.wantOK)
			}
		})
	}
}

func TestThinkingBudgetForEffort(t *testing.T) {
	// The published shares, applied to a 32768-token window.
	tests := []struct {
		effort   ThinkingEffort
		min, max int
		want     int
	}{
		{ThinkingEffortMinimal, 1024, 32768, 1638},
		{ThinkingEffortLow, 1024, 32768, 3932},
		{ThinkingEffortMedium, 1024, 32768, 9830},
		{ThinkingEffortHigh, 1024, 32768, 19660},
		{ThinkingEffortXHigh, 1024, 32768, 27852},
		{ThinkingEffortMax, 1024, 32768, 32768},
		// The window is a clamp at both ends.
		{ThinkingEffortMinimal, 4096, 32768, 4096},
		{ThinkingEffortMax, 1, 2048, 2048},
		// No budget equivalent: the field is dropped rather than guessed.
		{ThinkingEffortNone, 1024, 32768, 0},
		{"obsessive", 1024, 32768, 0},
		{"", 1024, 32768, 0},
		// A model that takes no budget, or a nonsense window.
		{ThinkingEffortHigh, 1024, 0, 0},
		{ThinkingEffortHigh, 4096, 1024, 0},
	}

	for _, tc := range tests {
		if got := ThinkingBudgetForEffort(tc.effort, tc.min, tc.max); got != tc.want {
			t.Errorf("ThinkingBudgetForEffort(%q, %d, %d) = %d, want %d",
				tc.effort, tc.min, tc.max, got, tc.want)
		}
	}
}

func TestThinkingEffortForBudget(t *testing.T) {
	tests := []struct {
		tokens int
		want   ThinkingEffort
	}{
		{0, ""},
		{-7, ""},
		{ThinkingBudgetDynamic, ""},
		{1, ThinkingEffortMinimal},
		{2048, ThinkingEffortMinimal},
		{2049, ThinkingEffortLow},
		{8192, ThinkingEffortLow},
		{8193, ThinkingEffortMedium},
		{24576, ThinkingEffortMedium},
		{24577, ThinkingEffortHigh},
		{65536, ThinkingEffortHigh},
		{65537, ThinkingEffortXHigh},
		{1 << 30, ThinkingEffortXHigh},
	}

	for _, tc := range tests {
		if got := ThinkingEffortForBudget(tc.tokens); got != tc.want {
			t.Errorf("ThinkingEffortForBudget(%d) = %q, want %q", tc.tokens, got, tc.want)
		}
	}
}

// ============================================================================
// THE TRANSLATION MATRIX
// ============================================================================
//
// planThinking is where the three vocabularies meet, and every provider goes
// through it, so this table is the closest thing the feature has to a spec.
// Each row is one set of neutral options projected onto one set of model
// capabilities, and asserts both what goes on the wire and the breadcrumb that
// explains it -- an adaptation nobody can see is the failure mode the notes
// exist to prevent.

// Capability fixtures. Real providers are tested against their own tables in
// the per-provider files; these are the shapes, not any one model.
const (
	dimsNothing      ThinkingDimension = 0
	dimsEffortOnly                     = ThinkingCanSetEffort
	dimsBudgetOnly                     = ThinkingCanSetBudget
	dimsToggleOnly                     = ThinkingCanToggle
	dimsToggleEffort                   = ThinkingCanToggle | ThinkingCanSetEffort
	dimsToggleBudget                   = ThinkingCanToggle | ThinkingCanSetBudget
	dimsEverything                     = ThinkingCanToggle | ThinkingCanSetEffort |
		ThinkingCanSetBudget | ThinkingCanHideTrace
)

var (
	ladderFull = []ThinkingEffort{
		ThinkingEffortNone, ThinkingEffortMinimal, ThinkingEffortLow, ThinkingEffortMedium,
		ThinkingEffortHigh, ThinkingEffortXHigh, ThinkingEffortMax,
	}
	// ladderNoOff is the common shape: three depths and no way to say "off".
	ladderNoOff = []ThinkingEffort{ThinkingEffortLow, ThinkingEffortMedium, ThinkingEffortHigh}
	// ladderWithOff is the shape that spells off as a depth.
	ladderWithOff = []ThinkingEffort{
		ThinkingEffortNone, ThinkingEffortLow, ThinkingEffortMedium, ThinkingEffortHigh,
	}
)

// window32k is a typical budget window: a floor the API rejects below and a
// ceiling the reply length imposes.
var window32k = budgetRange{min: 1024, max: 32768}

// wantPlan is the expected wire outcome of one projection, notes included.
type wantPlan struct {
	enable, disable, dynamic bool
	effort                   ThinkingEffort
	budget                   int
	hideTrace, showTrace     bool
	notes                    string
}

func checkPlan(t *testing.T, got thinkingPlan, want wantPlan) {
	t.Helper()
	if got.enable != want.enable || got.disable != want.disable || got.dynamic != want.dynamic {
		t.Errorf("toggle = enable:%t disable:%t dynamic:%t, want enable:%t disable:%t dynamic:%t",
			got.enable, got.disable, got.dynamic, want.enable, want.disable, want.dynamic)
	}
	if got.effort != want.effort {
		t.Errorf("effort = %q, want %q", got.effort, want.effort)
	}
	if got.budget != want.budget {
		t.Errorf("budget = %d, want %d", got.budget, want.budget)
	}
	if got.hideTrace != want.hideTrace || got.showTrace != want.showTrace {
		t.Errorf("trace = hide:%t show:%t, want hide:%t show:%t",
			got.hideTrace, got.showTrace, want.hideTrace, want.showTrace)
	}
	if got.translation() != want.notes {
		t.Errorf("translation = %q, want %q", got.translation(), want.notes)
	}
}

func TestPlanThinkingTranslationMatrix(t *testing.T) {
	tests := []struct {
		name    string
		opts    func(*ThinkingOptions)
		dims    ThinkingDimension
		br      budgetRange
		efforts []ThinkingEffort
		want    wantPlan
	}{
		// --- the untouched case, which is the whole opt-in guarantee ---------
		{
			name: "options nobody touched send nothing",
			opts: func(o *ThinkingOptions) {},
			dims: dimsEverything, br: window32k, efforts: ladderFull,
			want: wantPlan{},
		},

		// --- toggle ----------------------------------------------------------
		{
			name: "on reaches a model with a toggle",
			opts: func(o *ThinkingOptions) { o.Enable() },
			dims: dimsToggleEffort, efforts: ladderNoOff,
			want: wantPlan{enable: true},
		}, {
			// Nothing to say and nothing lost: a model that always reasons is
			// already doing what was asked.
			name: "on is silent on a model with no toggle",
			opts: func(o *ThinkingOptions) { o.Enable() },
			dims: dimsEffortOnly, efforts: ladderNoOff,
			want: wantPlan{},
		}, {
			name: "off reaches a model with a toggle",
			opts: func(o *ThinkingOptions) { o.Disable() },
			dims: dimsToggleBudget, br: window32k,
			want: wantPlan{disable: true},
		}, {
			// The one lossy direction that is not lossy at all: an effort ladder
			// with a none rung is a real off switch.
			name: "off becomes effort=none where the ladder has one",
			opts: func(o *ThinkingOptions) { o.Disable() },
			dims: dimsEffortOnly, efforts: ladderWithOff,
			want: wantPlan{effort: ThinkingEffortNone, notes: "thinking off sent as effort=none"},
		}, {
			// clampEffort would happily return low here; planThinking refuses
			// anything but an exact none, because "think less" is not "off".
			name: "off is dropped where the ladder has no none rung",
			opts: func(o *ThinkingOptions) { o.Disable() },
			dims: dimsEffortOnly, efforts: ladderNoOff,
			want: wantPlan{notes: "thinking off dropped: model has no off switch"},
		}, {
			name: "off is dropped on a model with no knobs at all",
			opts: func(o *ThinkingOptions) { o.Disable() },
			dims: dimsNothing,
			want: wantPlan{notes: "thinking off dropped: model has no off switch"},
		}, {
			// Depth is meaningless once thinking is off, so it is not sent even
			// where the model would accept it.
			name: "off suppresses a depth that was set earlier",
			opts: func(o *ThinkingOptions) { o.WithEffort(ThinkingEffortHigh).WithBudget(4096).Disable() },
			dims: dimsEverything, br: window32k, efforts: ladderFull,
			want: wantPlan{disable: true},
		},

		// --- effort onto a model that takes an effort -------------------------
		{
			name: "an exact rung is forwarded",
			opts: func(o *ThinkingOptions) { o.WithEffort(ThinkingEffortMedium) },
			dims: dimsToggleEffort, efforts: ladderNoOff,
			want: wantPlan{enable: true, effort: ThinkingEffortMedium},
		}, {
			name: "a rung above the ladder clamps down",
			opts: func(o *ThinkingOptions) { o.WithEffort(ThinkingEffortMax) },
			dims: dimsEffortOnly, efforts: ladderNoOff,
			want: wantPlan{effort: ThinkingEffortHigh, notes: "effort max clamped to high"},
		}, {
			name: "a rung below the ladder clamps up",
			opts: func(o *ThinkingOptions) { o.WithEffort(ThinkingEffortMinimal) },
			dims: dimsEffortOnly, efforts: []ThinkingEffort{ThinkingEffortMedium, ThinkingEffortMax},
			want: wantPlan{effort: ThinkingEffortMedium, notes: "effort minimal clamped to medium"},
		}, {
			name: "a shallow rung never clamps into the off rung",
			opts: func(o *ThinkingOptions) { o.WithEffort(ThinkingEffortMinimal) },
			dims: dimsEffortOnly, efforts: ladderWithOff,
			want: wantPlan{effort: ThinkingEffortLow, notes: "effort minimal clamped to low"},
		}, {
			// An off-ladder value came from somewhere lingo cannot reason about,
			// so the portable surface drops it rather than forwarding a guess.
			name: "an off-ladder effort is dropped when it came from the portable surface",
			opts: func(o *ThinkingOptions) { o.WithEffort("obsessive") },
			dims: dimsEffortOnly, efforts: ladderNoOff,
			want: wantPlan{notes: "effort obsessive dropped: not on this model's ladder"},
		}, {
			// The effort setter spells off as a level, and that reaches the mode.
			name: "effort none is an off switch",
			opts: func(o *ThinkingOptions) { o.WithEffort(ThinkingEffortNone) },
			dims: dimsToggleEffort, efforts: ladderWithOff,
			want: wantPlan{disable: true},
		},

		// --- effort onto a model that takes only a budget --------------------
		{
			name: "effort becomes a share of the budget window",
			opts: func(o *ThinkingOptions) { o.WithEffort(ThinkingEffortHigh) },
			dims: dimsBudgetOnly, br: window32k,
			want: wantPlan{budget: 19660, notes: "effort high mapped to budget 19660 tokens"},
		}, {
			name: "a shallow effort is floored at the window's minimum",
			opts: func(o *ThinkingOptions) { o.WithEffort(ThinkingEffortMinimal) },
			dims: dimsToggleBudget, br: budgetRange{min: 4096, max: 32768},
			want: wantPlan{enable: true, budget: 4096, notes: "effort minimal mapped to budget 4096 tokens"},
		}, {
			// none has no budget equivalent; it is an off switch, and this model
			// has one, so the toggle carries it and the depth is not translated.
			name: "effort none on a budget model is the toggle, not a budget",
			opts: func(o *ThinkingOptions) { o.WithEffort(ThinkingEffortNone) },
			dims: dimsToggleBudget, br: window32k,
			want: wantPlan{disable: true},
		}, {
			name: "an off-ladder effort has no budget equivalent",
			opts: func(o *ThinkingOptions) { o.WithEffort("obsessive") },
			dims: dimsBudgetOnly, br: window32k,
			want: wantPlan{notes: "effort obsessive dropped: no budget equivalent"},
		},

		// --- budget onto a model that takes only an effort -------------------
		{
			name: "budget becomes a depth by absolute thresholds",
			opts: func(o *ThinkingOptions) { o.WithBudget(10000) },
			dims: dimsEffortOnly, efforts: ladderNoOff,
			want: wantPlan{effort: ThinkingEffortMedium, notes: "budget 10000 mapped to effort medium"},
		}, {
			name: "a derived depth is clamped to the model's ladder",
			opts: func(o *ThinkingOptions) { o.WithBudget(1_000_000) },
			dims: dimsToggleEffort, efforts: ladderNoOff,
			want: wantPlan{enable: true, effort: ThinkingEffortHigh, notes: "budget 1000000 mapped to effort high"},
		}, {
			name: "a derived depth is dropped when the ladder is empty",
			opts: func(o *ThinkingOptions) { o.WithBudget(10000) },
			dims: dimsEffortOnly,
			want: wantPlan{notes: "budget 10000 dropped: no effort equivalent"},
		},

		// --- budget onto a model that takes a budget --------------------------
		{
			name: "a budget inside the window is forwarded",
			opts: func(o *ThinkingOptions) { o.WithBudget(4096) },
			dims: dimsBudgetOnly, br: window32k,
			want: wantPlan{budget: 4096},
		}, {
			name: "a budget below the floor is clamped up",
			opts: func(o *ThinkingOptions) { o.WithBudget(500) },
			dims: dimsBudgetOnly, br: window32k,
			want: wantPlan{budget: 1024, notes: "budget 500 clamped to 1024"},
		}, {
			name: "a budget above the ceiling is clamped down",
			opts: func(o *ThinkingOptions) { o.WithBudget(100000) },
			dims: dimsBudgetOnly, br: window32k,
			want: wantPlan{budget: 32768, notes: "budget 100000 clamped to 32768"},
		},

		// --- dynamic ---------------------------------------------------------
		{
			name: "dynamic reaches a model that can decide for itself",
			opts: func(o *ThinkingOptions) { o.WithDynamicBudget() },
			dims: dimsToggleBudget, br: window32k,
			want: wantPlan{enable: true, dynamic: true},
		}, {
			name: "dynamic degrades to a plain enable",
			opts: func(o *ThinkingOptions) { o.WithDynamicBudget() },
			dims: dimsToggleEffort, efforts: ladderNoOff,
			want: wantPlan{enable: true, notes: "dynamic thinking dropped: model has no dynamic setting"},
		}, {
			// Nothing at all to say: no toggle to enable and no dynamic setting.
			name: "dynamic is only a note on a model with neither knob",
			opts: func(o *ThinkingOptions) { o.WithDynamicBudget() },
			dims: dimsEffortOnly, efforts: ladderNoOff,
			want: wantPlan{notes: "dynamic thinking dropped: model has no dynamic setting"},
		}, {
			// A negative budget that is not the dynamic sentinel is nonsense and
			// is discarded by the setter, so nothing is left to send.
			name: "a negative budget is discarded before it can be translated",
			opts: func(o *ThinkingOptions) { o.WithBudget(-7) },
			dims: dimsBudgetOnly, br: window32k,
			want: wantPlan{},
		},

		// --- a model with no depth knob at all --------------------------------
		{
			name: "effort is dropped on a toggle-only model",
			opts: func(o *ThinkingOptions) { o.WithEffort(ThinkingEffortHigh) },
			dims: dimsToggleOnly,
			want: wantPlan{enable: true, notes: "effort high dropped: model takes no depth setting"},
		}, {
			name: "budget is dropped on a toggle-only model",
			opts: func(o *ThinkingOptions) { o.WithBudget(5000) },
			dims: dimsToggleOnly,
			want: wantPlan{enable: true, notes: "budget 5000 dropped: model takes no token budget"},
		}, {
			name: "both are dropped on a model with nothing",
			opts: func(o *ThinkingOptions) { o.WithEffort(ThinkingEffortHigh).WithBudget(5000) },
			dims: dimsNothing,
			want: wantPlan{notes: "effort high dropped: model takes no depth setting; " +
				"budget 5000 dropped: model takes no token budget"},
		},

		// --- both depths named at once ----------------------------------------
		{
			// An explicit budget outranks a derived one, so the effort is not
			// projected. The note reads as though the model took no depth at
			// all, which understates what happened: the budget below is the
			// depth. Pinned here as the behaviour, not endorsed as the wording.
			name: "an explicit budget wins on a budget-only model",
			opts: func(o *ThinkingOptions) { o.WithEffort(ThinkingEffortLow).WithBudget(9000) },
			dims: dimsBudgetOnly, br: window32k,
			want: wantPlan{budget: 9000, notes: "effort low dropped: model takes no depth setting"},
		}, {
			name: "an explicit effort wins on an effort-only model",
			opts: func(o *ThinkingOptions) { o.WithEffort(ThinkingEffortHigh).WithBudget(9000) },
			dims: dimsEffortOnly, efforts: ladderNoOff,
			want: wantPlan{effort: ThinkingEffortHigh, notes: "budget 9000 dropped: model takes no token budget"},
		}, {
			// A model with both knobs sends both, each in its own vocabulary.
			name: "a model with both knobs takes both",
			opts: func(o *ThinkingOptions) { o.WithEffort(ThinkingEffortHigh).WithBudget(9000) },
			dims: dimsEverything, br: window32k, efforts: ladderFull,
			want: wantPlan{enable: true, effort: ThinkingEffortHigh, budget: 9000},
		},

		// --- trace ------------------------------------------------------------
		{
			name: "trace omission reaches a model that can withhold it",
			opts: func(o *ThinkingOptions) { o.WithTrace(ThinkingTraceOmit) },
			dims: dimsEverything, br: window32k, efforts: ladderFull,
			want: wantPlan{hideTrace: true},
		}, {
			name: "trace omission is dropped where the trace always comes back",
			opts: func(o *ThinkingOptions) { o.WithTrace(ThinkingTraceOmit) },
			dims: dimsToggleEffort, efforts: ladderNoOff,
			want: wantPlan{notes: "trace omission dropped: model always returns its trace"},
		}, {
			name: "asking for the trace reaches a model that can be asked",
			opts: func(o *ThinkingOptions) { o.WithTrace(ThinkingTraceInclude) },
			dims: dimsEverything, br: window32k, efforts: ladderFull,
			want: wantPlan{showTrace: true},
		}, {
			// Asymmetric on purpose: a provider that always returns its trace is
			// already doing what was asked, so there is nothing to report.
			name: "asking for the trace is silent where it cannot be asked",
			opts: func(o *ThinkingOptions) { o.WithTrace(ThinkingTraceInclude) },
			dims: dimsToggleEffort, efforts: ladderNoOff,
			want: wantPlan{},
		},

		// --- everything at once ------------------------------------------------
		{
			name: "a full opt-in on a model that honours all of it",
			opts: func(o *ThinkingOptions) {
				o.Enable().WithEffort(ThinkingEffortXHigh).WithDynamicBudget().WithTrace(ThinkingTraceInclude)
			},
			dims: dimsEverything, br: window32k, efforts: ladderFull,
			want: wantPlan{enable: true, dynamic: true, effort: ThinkingEffortXHigh, showTrace: true},
		}, {
			name: "a full opt-in on a model that honours none of it",
			opts: func(o *ThinkingOptions) {
				o.Enable().WithEffort(ThinkingEffortXHigh).WithBudget(9000).WithTrace(ThinkingTraceOmit)
			},
			dims: dimsNothing,
			want: wantPlan{notes: "effort xhigh dropped: model takes no depth setting; " +
				"budget 9000 dropped: model takes no token budget; " +
				"trace omission dropped: model always returns its trace"},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var o ThinkingOptions
			tc.opts(&o)
			checkPlan(t, planThinking(&o, tc.dims, tc.br, tc.efforts...), tc.want)
		})
	}
}

// TestPlanThinkingIsAZeroPlanForUntouchedOptions is the opt-in guarantee at its
// root: every provider builds its request from this function, so a zero plan on
// untouched options is what makes an untouched model byte-identical.
func TestPlanThinkingIsAZeroPlanForUntouchedOptions(t *testing.T) {
	for _, tc := range []struct {
		name string
		o    *ThinkingOptions
	}{
		{"nil options", nil},
		{"zero options", &ThinkingOptions{}},
		// A setter that wrote nothing leaves nothing behind either.
		{"an effort set back to empty", (&ThinkingOptions{}).WithEffort("")},
		{"a budget set to zero", (&ThinkingOptions{}).WithBudget(0)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := planThinking(tc.o, dimsEverything, window32k, ladderFull...)
			checkPlan(t, p, wantPlan{})
			if p.notes != nil {
				t.Errorf("notes = %v, want none", p.notes)
			}
		})
	}
}

// TestPlanThinkingForwardsPinnedDimensionsVerbatim is the backward-compat
// device itself. A dimension a per-model setter wrote, or a constructor seeded,
// is forwarded exactly as given: not clamped into a range, not translated into
// another vocabulary, not dropped for being a word lingo has never seen. The
// same value arriving through the portable surface is adapted.
func TestPlanThinkingForwardsPinnedDimensionsVerbatim(t *testing.T) {
	t.Run("an illegal budget is sent as given", func(t *testing.T) {
		var pinned, portable ThinkingOptions
		pinned.WithBudget(500).pin(ThinkingCanSetBudget)
		portable.WithBudget(500)

		checkPlan(t, planThinking(&pinned, dimsBudgetOnly, window32k), wantPlan{budget: 500})
		checkPlan(t, planThinking(&portable, dimsBudgetOnly, window32k),
			wantPlan{budget: 1024, notes: "budget 500 clamped to 1024"})
	})

	t.Run("an off-ladder effort is sent as given", func(t *testing.T) {
		var pinned, portable ThinkingOptions
		pinned.WithEffort("something-the-endpoint-invented").pin(ThinkingCanSetEffort)
		portable.WithEffort("something-the-endpoint-invented")

		checkPlan(t, planThinking(&pinned, dimsEffortOnly, budgetRange{}, ladderNoOff...),
			wantPlan{effort: "something-the-endpoint-invented"})
		checkPlan(t, planThinking(&portable, dimsEffortOnly, budgetRange{}, ladderNoOff...),
			wantPlan{notes: "effort something-the-endpoint-invented dropped: not on this model's ladder"})
	})

	t.Run("a rung the model rejects is still sent", func(t *testing.T) {
		var pinned ThinkingOptions
		pinned.WithEffort(ThinkingEffortMax).pin(ThinkingCanSetEffort)
		// The caller named this model's own setter, so they get this model's
		// own error rather than a value lingo substituted.
		checkPlan(t, planThinking(&pinned, dimsEffortOnly, budgetRange{}, ladderNoOff...),
			wantPlan{effort: ThinkingEffortMax})
	})

	t.Run("a pin is dropped when the portable surface writes over it", func(t *testing.T) {
		// This is what stops a constructor-seeded effort making every later
		// portable call unclampable.
		var o ThinkingOptions
		o.WithEffort(ThinkingEffortMax).pin(ThinkingCanSetEffort)
		if !o.isPinned(ThinkingCanSetEffort) {
			t.Fatal("pin did not take")
		}
		o.WithEffort(ThinkingEffortMax)
		if o.isPinned(ThinkingCanSetEffort) {
			t.Error("a portable write must clear the pin on the dimension it wrote")
		}
		checkPlan(t, planThinking(&o, dimsEffortOnly, budgetRange{}, ladderNoOff...),
			wantPlan{effort: ThinkingEffortHigh, notes: "effort max clamped to high"})

		// And only on that dimension: a pinned budget beside it is untouched.
		var both ThinkingOptions
		both.WithBudget(500).pin(ThinkingCanSetBudget)
		both.WithEffort(ThinkingEffortHigh)
		if !both.isPinned(ThinkingCanSetBudget) {
			t.Error("writing the effort must not unpin the budget")
		}
	})
}

// TestNoThinkingWithoutAnOffSwitchIsATrueNoOp is the other half of the pin
// contract, on the one path that used to break it. planThinking returned from
// the toggle section the moment the mode was off, so on a model with no off
// switch the caller lost BOTH the off they could not have and the depth they
// already had.
//
// Dropping the depth reads as the conservative choice and is the opposite of
// one. o3 with no reasoning_effort on the wire is not o3 thinking less: it is o3
// at OpenAI's own server-side default, which is more reasoning than the "high"
// the caller pinned, at a price they did not agree to. "I cannot switch it off"
// must never turn into "so I turned it up".
func TestNoThinkingWithoutAnOffSwitchIsATrueNoOp(t *testing.T) {
	const dropped = "thinking off dropped: model has no off switch"

	t.Run("a pinned effort survives", func(t *testing.T) {
		// The shape of NoThinking(NewO3().WithReasoningEffort("high")): a
		// per-model setter pinned the depth, then the portable surface asked for
		// an off this model has no way to express.
		var o ThinkingOptions
		o.WithEffort(ThinkingEffortHigh).pin(ThinkingCanSetEffort)
		o.Disable()
		checkPlan(t, planThinking(&o, dimsEffortOnly, budgetRange{}, ladderNoOff...),
			wantPlan{effort: ThinkingEffortHigh, notes: dropped})
	})

	t.Run("a pinned budget survives", func(t *testing.T) {
		var o ThinkingOptions
		o.WithBudget(500).pin(ThinkingCanSetBudget)
		o.Disable()
		checkPlan(t, planThinking(&o, dimsBudgetOnly, window32k), wantPlan{budget: 500, notes: dropped})
	})

	t.Run("an unpinned depth survives and is still adapted", func(t *testing.T) {
		// The portable rule is unchanged by the no-op: what came from the
		// portable surface is clamped as usual, it is simply not thrown away.
		var o ThinkingOptions
		o.WithEffort(ThinkingEffortMax).Disable()
		checkPlan(t, planThinking(&o, dimsEffortOnly, budgetRange{}, ladderNoOff...),
			wantPlan{effort: ThinkingEffortHigh, notes: dropped + "; effort max clamped to high"})
	})

	t.Run("nothing to keep still just notes the drop", func(t *testing.T) {
		var o ThinkingOptions
		o.Disable()
		checkPlan(t, planThinking(&o, dimsEffortOnly, budgetRange{}, ladderNoOff...), wantPlan{notes: dropped})
	})

	// A model that CAN be switched off is untouched by any of this: the off is
	// real, so the depth really is meaningless and really is dropped.
	t.Run("a real toggle still wins over the depth", func(t *testing.T) {
		var o ThinkingOptions
		o.WithEffort(ThinkingEffortHigh).pin(ThinkingCanSetEffort)
		o.Disable()
		checkPlan(t, planThinking(&o, dimsToggleEffort, budgetRange{}, ladderNoOff...),
			wantPlan{disable: true})
	})

	t.Run("a none rung still wins over the depth", func(t *testing.T) {
		var o ThinkingOptions
		o.WithEffort(ThinkingEffortHigh).pin(ThinkingCanSetEffort)
		o.Disable()
		checkPlan(t, planThinking(&o, dimsEffortOnly, budgetRange{}, ladderWithOff...),
			wantPlan{effort: ThinkingEffortNone, notes: "thinking off sent as effort=none"})
	})

	// And the same contract observed on the wire, one row per provider whose
	// models can reach an off with no off switch.
	t.Run("wire", func(t *testing.T) {
		// OpenAI: a setter-pinned effort, and the effort 26 of the 27 reasoning
		// constructors have seeded since they shipped.
		wantJSON(t, openAIWire(t, NoThinking(NewO3().WithReasoningEffort("high"))),
			"reasoning_effort", `"high"`)
		wantJSON(t, openAIWire(t, NoThinking(NewGPT5())), "reasoning_effort", `"medium"`)
		wantJSON(t, openAIWire(t, NoThinking(NewO1Pro())), "reasoning_effort", `"high"`)
		// The families that DO have a none rung keep spelling off as none.
		wantJSON(t, openAIWire(t, NoThinking(NewGPT51())), "reasoning_effort", `"none"`)
		wantJSON(t, openAIWire(t, NoThinking(NewGPT56Sol())), "reasoning_effort", `"none"`)

		// xAI: grok-4.5 and later dropped the none rung, grok-4.3 still has it.
		wantJSON(t, oaiCompatWire(t, xaiCfg, NoThinking(NewXAIModel("grok-4.5").WithReasoningEffort("high"))),
			"reasoning_effort", `"high"`)
		wantJSON(t, oaiCompatWire(t, xaiCfg, NoThinking(NewGrok43())), "reasoning_effort", `"none"`)

		// A generic endpoint has no ladder at all, so it can never have a none
		// rung and its pinned effort must always survive.
		wantJSON(t, oaiCompatWire(t, oaiCompatCfg,
			NoThinking(NewOpenAICompatibleModel("m").WithReasoningEffort("obsessive"))),
			"reasoning_effort", `"obsessive"`)
	})
}

func TestThinkingPlanTranslationRendering(t *testing.T) {
	var p *thinkingPlan
	if got := p.translation(); got != "" {
		t.Errorf("nil plan translation = %q, want empty", got)
	}
	p = &thinkingPlan{}
	if got := p.translation(); got != "" {
		t.Errorf("empty translation = %q, want empty", got)
	}
	p.note("first %d", 1)
	if got := p.translation(); got != "first 1" {
		t.Errorf("translation = %q", got)
	}
	p.note("second")
	p.note("third")
	if got := p.translation(); got != "first 1; second; third" {
		t.Errorf("translation = %q, want the notes joined in the order they were decided", got)
	}
}

// ============================================================================
// OPT-IN GUARANTEE: STORAGE
// ============================================================================

// thinkingFreshModels lists one freshly constructed model per provider, chosen
// so that every ThinkingOptions storage in the library is represented. Every
// entry must start with thinking neither on nor off, which is what keeps an
// untouched model producing the request it produced before this feature.
func thinkingFreshModels() []Model {
	return []Model{
		NewGPT4o(),
		NewClaudeSonnet5(),
		NewClaude37Sonnet(),
		NewClaudeOpus46(),
		NewAnthropicModel("claude-opus-5"),
		NewGemini25Flash(),
		NewGemini3Pro(),
		NewGoogleModel("gemini-3-pro-preview"),
		NewBedrockClaudeSonnet5(),
		NewBedrockClaude37Sonnet(),
		NewBedrockNovaPro(),
		NewBedrockModel("us.anthropic.claude-opus-5", "claude"),
		NewAzureOpenAIModel("my-deployment"),
		NewAzureOpenAIReasoningModel("my-deployment"),
		NewGrok45(),
		NewDeepSeekV4Pro(),
		NewOpenRouterModel("anthropic/claude-opus-5"),
		NewOpenAICompatibleModel("llama-3.3-70b"),
		NewCommandAPlus(),
		NewCommandAReasoning(),
		NewCohereModel("command-a-reasoning-08-2025"),
		NewSonarDeepResearch(),
		NewPerplexityModel("sonar-deep-research"),
		NewQwen3(),
		NewOllamaModel("gpt-oss:20b"),
	}
}

func TestFreshModelHasThinkingUntouched(t *testing.T) {
	for _, m := range thinkingFreshModels() {
		to := modelThinkingOptions(m)
		if to.Enabled() || to.Disabled() || to.Mode() != ThinkingModeDefault {
			t.Errorf("%s/%s: fresh model carries a mode: %v", m.Provider(), m.ModelName(), to.Mode())
		}
		if to.Effort() != "" {
			t.Errorf("%s/%s: fresh model carries effort %q", m.Provider(), m.ModelName(), to.Effort())
		}
		if to.Budget() != 0 || to.DynamicBudget() {
			t.Errorf("%s/%s: fresh model carries budget %d", m.Provider(), m.ModelName(), to.Budget())
		}
		if to.Trace() != ThinkingTraceDefault {
			t.Errorf("%s/%s: fresh model carries a trace setting: %v", m.Provider(), m.ModelName(), to.Trace())
		}
		if to != nil && to.pinned != 0 {
			t.Errorf("%s/%s: fresh model pins %06b", m.Provider(), m.ModelName(), to.pinned)
		}
	}
}

// TestSeededConstructorDefaultsSurviveAsPinned covers the models the guarantee
// above cannot: 26 OpenAI reasoning constructors and one xAI constructor have
// always seeded a reasoning_effort, so their "untouched" request has always
// carried the field. Each seed must survive with the exact value it has always
// had AND be pinned, because an unpinned seed would be clamped to the model's
// ladder -- invisible for the medium rows, but fatal on o1-mini, whose ladder is
// empty, and on gpt-5-pro, which accepts only high.
func TestSeededConstructorDefaultsSurviveAsPinned(t *testing.T) {
	tests := []struct {
		model Model
		want  ThinkingEffort
	}{
		{NewO1(), ThinkingEffortMedium},
		{NewO1Mini(), ThinkingEffortMedium},
		{NewO1Preview(), ThinkingEffortMedium},
		{NewO1Pro(), ThinkingEffortHigh},
		{NewO3(), ThinkingEffortMedium},
		{NewO3Mini(), ThinkingEffortMedium},
		{NewO3Pro(), ThinkingEffortHigh},
		{NewO4Mini(), ThinkingEffortMedium},
		{NewGPT5(), ThinkingEffortMedium},
		{NewGPT5Mini(), ThinkingEffortMedium},
		{NewGPT5Nano(), ThinkingEffortMedium},
		{NewGPT5Pro(), ThinkingEffortHigh},
		{NewGPT51(), ThinkingEffortMedium},
		{NewGPT51Mini(), ThinkingEffortMedium},
		{NewGPT51Nano(), ThinkingEffortMedium},
		{NewGPT51Codex(), ThinkingEffortMedium},
		{NewGPT51CodexMini(), ThinkingEffortMedium},
		{NewGPT54(), ThinkingEffortMedium},
		{NewGPT54Mini(), ThinkingEffortMedium},
		{NewGPT54Nano(), ThinkingEffortMedium},
		{NewGPT54Pro(), ThinkingEffortHigh},
		{NewGPT55(), ThinkingEffortMedium},
		{NewGPT55Pro(), ThinkingEffortHigh},
		{NewGPT56Sol(), ThinkingEffortMedium},
		{NewGPT56Terra(), ThinkingEffortMedium},
		{NewGPT56Luna(), ThinkingEffortLow},
		{NewGrok43(), ThinkingEffort(XAIEffortLow)},
	}

	for _, tc := range tests {
		t.Run(tc.model.ModelName(), func(t *testing.T) {
			to := modelThinkingOptions(tc.model)
			if to.Effort() != tc.want {
				t.Errorf("seeded effort = %q, want %q", to.Effort(), tc.want)
			}
			if !to.isPinned(ThinkingCanSetEffort) {
				t.Error("a constructor-seeded effort must be pinned, or the portable surface may clamp it")
			}
			// The seed is a depth and nothing else: it must not look like an
			// opt-in, or a model nobody touched would start sending a toggle.
			if to.Mode() != ThinkingModeDefault {
				t.Errorf("mode = %v, want the default: seeding an effort is not opting in", to.Mode())
			}
			if to.Budget() != 0 || to.Trace() != ThinkingTraceDefault {
				t.Errorf("a seeded constructor carries more than an effort: %+v", to)
			}
			if to.pinned != ThinkingCanSetEffort {
				t.Errorf("pinned = %06b, want the effort dimension alone", to.pinned)
			}
		})
	}

	// The generic escape hatches seed nothing, so they must stay silent.
	for _, m := range []Model{NewOpenAIReasoningModel("gpt-9"), NewXAIModel("grok-4.6")} {
		if e := modelThinkingOptions(m).Effort(); e != "" {
			t.Errorf("%s: raw-id model seeded %q", m.ModelName(), e)
		}
	}
}

// ============================================================================
// OPT-IN GUARANTEE: WIRE
// ============================================================================
//
// The storage tests above say the configuration is empty; these say the request
// is. Every provider with an offline seam is here, because the failure this
// guards against -- a field that appears on a model nobody touched -- is the one
// that breaks callers who never asked for any of this.

func TestUntouchedModelSendsNoThinkingField(t *testing.T) {
	t.Run("openai standard", func(t *testing.T) {
		body := openAIWire(t, NewGPT4o())
		wantJSON(t, body, "reasoning_effort", "")
		wantJSON(t, body, "thinking", "")
	})

	t.Run("openai reasoning keeps its seed and gains nothing", func(t *testing.T) {
		// The one provider where the guarantee is not "nothing": the seeded
		// effort has always been on the wire and must stay, alone.
		body := openAIWire(t, NewGPT5())
		wantJSON(t, body, "reasoning_effort", `"medium"`)
		wantJSON(t, body, "thinking", "")
		wantJSON(t, body, "reasoning", "")
	})

	t.Run("anthropic", func(t *testing.T) {
		body := anthropicWire(t, NewClaudeSonnet5().WithSystemPrompt("be terse"))
		wantJSON(t, body, "thinking", "")
		wantJSON(t, body, "output_config", "")
	})

	t.Run("google", func(t *testing.T) {
		wantThinkingConfig(t, geminiThinkingWire(t, NewGemini25Flash()), "")
		wantThinkingConfig(t, geminiThinkingWire(t, NewGemini3Pro()), "")
	})

	t.Run("bedrock claude", func(t *testing.T) {
		// Both eras: the one whose only knob is a fixed budget, where a leak
		// shows up as a budget_tokens nobody asked for, and the one whose only
		// knob is the off switch.
		for _, m := range []Model{NewBedrockClaude37Sonnet(), NewBedrockClaudeSonnet5()} {
			raw, notes := bedrockClaudeBody(t, m)
			if strings.Contains(raw, "thinking") {
				t.Errorf("%s: body = %s, want no thinking field", m.ModelName(), raw)
			}
			if notes != "" {
				t.Errorf("%s: translation = %q, want nothing to have been adapted", m.ModelName(), notes)
			}
		}
	})

	t.Run("cohere", func(t *testing.T) {
		wantJSON(t, cohereWire(t, NewCommandAReasoning()), "thinking", "")
	})

	t.Run("deepseek", func(t *testing.T) {
		body := oaiCompatWire(t, deepSeekCfg, NewDeepSeekV4Pro())
		wantJSON(t, body, "thinking", "")
		wantJSON(t, body, "reasoning_effort", "")
	})

	t.Run("openrouter", func(t *testing.T) {
		wantJSON(t, oaiCompatWire(t, openRouterCfg, NewOpenRouterModel("anthropic/claude-opus-5")),
			"reasoning", "")
	})

	t.Run("xai keeps its seed and gains nothing", func(t *testing.T) {
		body := oaiCompatWire(t, xaiCfg, NewGrok43())
		wantJSON(t, body, "reasoning_effort", `"low"`)
		body = oaiCompatWire(t, xaiCfg, NewGrok45())
		wantJSON(t, body, "reasoning_effort", "")
	})

	t.Run("azure", func(t *testing.T) {
		wantJSON(t, oaiCompatWire(t, func(u string) ProviderConfig {
			return &AzureOpenAIConfig{Endpoint: u, APIKey: "k"}
		}, NewAzureOpenAIReasoningModel("my-deployment")), "reasoning_effort", "")
	})

	t.Run("generic endpoint", func(t *testing.T) {
		wantJSON(t, oaiCompatWire(t, oaiCompatCfg, NewOpenAICompatibleModel("llama-3.3-70b")),
			"reasoning_effort", "")
	})

	t.Run("perplexity", func(t *testing.T) {
		wantJSON(t, perplexityWire(t, NewSonarDeepResearch()), "reasoning_effort", "")
	})

	t.Run("ollama", func(t *testing.T) {
		wantJSON(t, ollamaWire(t, NewQwen3()), "think", "")
	})
}

// TestOptingInOnAModelThatCannotThinkChangesNothing is the never-error rule at
// the wire. Asking a model with no thinking field to think hardest, dynamically,
// with the trace withheld must produce the request it produced before anyone
// asked -- silently, and without an error anywhere.
func TestOptingInOnAModelThatCannotThinkChangesNothing(t *testing.T) {
	maximal := func(m Model) Model {
		return Thinking(m, WithThinkingEffort(ThinkingEffortMax), WithDynamicThinking(),
			WithThinkingTrace(ThinkingTraceOmit))
	}

	t.Run("openai standard", func(t *testing.T) {
		wantJSON(t, openAIWire(t, maximal(NewGPT4o())), "reasoning_effort", "")
	})
	t.Run("claude 3.x", func(t *testing.T) {
		body := anthropicWire(t, maximal(NewClaude35Sonnet()))
		wantJSON(t, body, "thinking", "")
		wantJSON(t, body, "output_config", "")
	})
	t.Run("gemini 1.5", func(t *testing.T) {
		wantThinkingConfig(t, geminiThinkingWire(t, maximal(NewGemini15Pro())), "")
	})
	t.Run("command r", func(t *testing.T) {
		wantJSON(t, cohereWire(t, maximal(NewCommandR())), "thinking", "")
	})
	t.Run("ollama without the capability", func(t *testing.T) {
		wantJSON(t, ollamaWire(t, maximal(NewLlama33())), "think", "")
	})
	t.Run("bedrock claude 3.5", func(t *testing.T) {
		// The storage is there -- every Bedrock Claude embeds it -- so only the
		// era table stops a knob reaching a generation that has none.
		raw, _ := bedrockClaudeBody(t, maximal(NewBedrockClaude35Sonnet()))
		if strings.Contains(raw, "thinking") {
			t.Errorf("body = %s, want no thinking field on a 3.5 model", raw)
		}
	})
	t.Run("perplexity sonar", func(t *testing.T) {
		wantJSON(t, perplexityWire(t, maximal(NewSonar())), "reasoning_effort", "")
	})
}

// ============================================================================
// BACKWARD COMPAT: THE PRE-EXISTING SETTERS
// ============================================================================
//
// Every setter below shipped before the portable surface existed and is public,
// so its behaviour is a contract. These two tests are the contract from both
// sides: what the setter writes into the shared storage, and what that storage
// puts on the wire.

func TestLegacySettersWriteTheSharedStorage(t *testing.T) {
	tests := []struct {
		name       string
		model      Model
		wantMode   ThinkingMode
		wantEffort ThinkingEffort
		wantBudget int
		wantTrace  ThinkingTrace
		wantPinned ThinkingDimension
	}{
		// Anthropic. A budget enables and pins; a non-positive one clears,
		// matching the `if thinkingBudget > 0` wire guard it replaced.
		{"anthropic WithThinkingBudget", NewClaude37Sonnet().WithThinkingBudget(2048),
			ThinkingModeOn, "", 2048, ThinkingTraceDefault, ThinkingCanSetBudget},
		{"anthropic WithThinkingBudget zero", NewClaude37Sonnet().WithThinkingBudget(0),
			ThinkingModeDefault, "", 0, ThinkingTraceDefault, 0},
		{"anthropic WithThinkingBudget negative", NewClaude37Sonnet().WithThinkingBudget(-7),
			ThinkingModeDefault, "", 0, ThinkingTraceDefault, 0},
		// Adaptive is "budget, but you decide", so it is the dynamic budget.
		{"anthropic WithAdaptiveThinking", NewClaudeOpus46().WithAdaptiveThinking(),
			ThinkingModeOn, "", ThinkingBudgetDynamic, ThinkingTraceDefault, ThinkingCanSetBudget},
		{"anthropic WithThinkingDisabled", NewClaudeOpus5().WithThinkingDisabled(),
			ThinkingModeOff, "", 0, ThinkingTraceDefault, ThinkingCanToggle},
		// WithEffort deliberately leaves the mode alone: output_config.effort
		// caps spend, it does not switch reasoning on.
		{"anthropic WithEffort", NewClaudeOpus46().WithEffort(EffortHigh),
			ThinkingModeDefault, ThinkingEffortHigh, 0, ThinkingTraceDefault, ThinkingCanSetEffort},
		{"anthropic WithEffort off-ladder", NewClaudeOpus46().WithEffort("ultra"),
			ThinkingModeDefault, "ultra", 0, ThinkingTraceDefault, ThinkingCanSetEffort},

		// OpenAI and the whole oaicompat family share one shim, which also
		// leaves the mode alone: on these endpoints the effort is a value, not
		// a switch.
		{"openai WithReasoningEffort", NewGPT5().WithReasoningEffort("high"),
			ThinkingModeDefault, ThinkingEffortHigh, 0, ThinkingTraceDefault, ThinkingCanSetEffort},
		{"openai WithReasoningEffort clears the seed", NewGPT5().WithReasoningEffort(""),
			ThinkingModeDefault, "", 0, ThinkingTraceDefault, ThinkingCanSetEffort},
		{"xai WithReasoningEffort", NewGrok43().WithReasoningEffort(XAIEffortNone),
			ThinkingModeDefault, ThinkingEffort(XAIEffortNone), 0, ThinkingTraceDefault, ThinkingCanSetEffort},
		{"azure WithReasoningEffort", NewAzureOpenAIReasoningModel("d").WithReasoningEffort("medium"),
			ThinkingModeDefault, ThinkingEffortMedium, 0, ThinkingTraceDefault, ThinkingCanSetEffort},
		{"generic endpoint WithReasoningEffort", NewOpenAICompatibleModel("m").WithReasoningEffort("obsessive"),
			ThinkingModeDefault, "obsessive", 0, ThinkingTraceDefault, ThinkingCanSetEffort},

		// DeepSeek's toggle is a real toggle.
		{"deepseek WithThinkingDisabled", NewDeepSeekV4Pro().WithThinkingDisabled(),
			ThinkingModeOff, "", 0, ThinkingTraceDefault, ThinkingCanToggle},
		{"deepseek WithThinkingEnabled", NewDeepSeekV4Flash().WithThinkingEnabled(),
			ThinkingModeOn, "", 0, ThinkingTraceDefault, ThinkingCanToggle},
		{"deepseek both setters", NewDeepSeekV4Flash().WithThinkingDisabled().WithReasoningEffort("high"),
			ThinkingModeOff, ThinkingEffortHigh, 0, ThinkingTraceDefault,
			ThinkingCanToggle | ThinkingCanSetEffort},

		// OpenRouter spells all three knobs on one object.
		{"openrouter WithReasoningEffort", NewOpenRouterModel("m").WithReasoningEffort("xhigh"),
			ThinkingModeDefault, ThinkingEffortXHigh, 0, ThinkingTraceDefault, ThinkingCanSetEffort},
		{"openrouter WithReasoningMaxTokens", NewOpenRouterModel("m").WithReasoningMaxTokens(2000),
			ThinkingModeOn, "", 2000, ThinkingTraceDefault, ThinkingCanSetBudget},
		{"openrouter WithReasoningExcluded", NewOpenRouterModel("m").WithReasoningExcluded(),
			ThinkingModeDefault, "", 0, ThinkingTraceOmit, ThinkingCanHideTrace},

		// Cohere: naming a budget has always been how thinking is switched on,
		// and a non-positive one still enables it with no ceiling.
		{"cohere WithThinkingBudget", NewCommandAPlus().WithThinkingBudget(2048),
			ThinkingModeOn, "", 2048, ThinkingTraceDefault, ThinkingCanToggle | ThinkingCanSetBudget},
		{"cohere WithThinkingBudget zero still enables", NewCommandAPlus().WithThinkingBudget(0),
			ThinkingModeOn, "", 0, ThinkingTraceDefault, ThinkingCanToggle | ThinkingCanSetBudget},
		{"cohere WithThinkingDisabled", NewCommandAPlus().WithThinkingDisabled(),
			ThinkingModeOff, "", 0, ThinkingTraceDefault, ThinkingCanToggle},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			to := modelThinkingOptions(tc.model)
			if to == nil {
				t.Fatal("model carries no thinking storage")
			}
			if to.Mode() != tc.wantMode {
				t.Errorf("mode = %v, want %v", to.Mode(), tc.wantMode)
			}
			if to.Effort() != tc.wantEffort {
				t.Errorf("effort = %q, want %q", to.Effort(), tc.wantEffort)
			}
			if to.Budget() != tc.wantBudget {
				t.Errorf("budget = %d, want %d", to.Budget(), tc.wantBudget)
			}
			if to.Trace() != tc.wantTrace {
				t.Errorf("trace = %v, want %v", to.Trace(), tc.wantTrace)
			}
			if to.pinned != tc.wantPinned {
				t.Errorf("pinned = %06b, want %06b: a per-model setter's value must be forwarded verbatim",
					to.pinned, tc.wantPinned)
			}
		})
	}
}

// TestLegacySettersStillReachTheWire is the same contract observed after
// serialization, one row per provider that has an offline seam. The values are
// the ones lingo sent before the portable surface existed.
func TestLegacySettersStillReachTheWire(t *testing.T) {
	t.Run("anthropic", func(t *testing.T) {
		wantJSON(t, anthropicWire(t, NewClaude37Sonnet().WithThinkingBudget(2048)),
			"thinking", `{"type":"enabled","budget_tokens":2048}`)
		// Below the API's 1024 floor, and forwarded anyway: lingo has never
		// validated this, so the caller still sees the provider's own error.
		wantJSON(t, anthropicWire(t, NewClaudeSonnet4().WithThinkingBudget(500)),
			"thinking", `{"type":"enabled","budget_tokens":500}`)
		wantJSON(t, anthropicWire(t, NewClaudeOpus46().WithAdaptiveThinking()),
			"thinking", `{"type":"adaptive"}`)
		wantJSON(t, anthropicWire(t, NewClaudeOpus5().WithThinkingDisabled()),
			"thinking", `{"type":"disabled"}`)
		// An effort is a spend cap, so it goes out alone, with no thinking key.
		body := anthropicWire(t, NewClaudeOpus46().WithEffort(EffortHigh))
		wantJSON(t, body, "output_config", `{"effort":"high"}`)
		wantJSON(t, body, "thinking", "")
		// xhigh is rejected by 4.6 and forwarded regardless, because the caller
		// named this model's own setter.
		wantJSON(t, anthropicWire(t, NewClaudeOpus46().WithEffort(EffortXHigh)),
			"output_config", `{"effort":"xhigh"}`)
	})

	t.Run("openai", func(t *testing.T) {
		wantJSON(t, openAIWire(t, NewGPT5().WithReasoningEffort("high")), "reasoning_effort", `"high"`)
		// A word neither OpenAI nor lingo knows, forwarded byte for byte.
		wantJSON(t, openAIWire(t, NewGPT56Sol().WithReasoningEffort("something-openai-invented")),
			"reasoning_effort", `"something-openai-invented"`)
		// Clearing the seed clears the field.
		wantJSON(t, openAIWire(t, NewGPT5().WithReasoningEffort("")), "reasoning_effort", "")
	})

	t.Run("xai", func(t *testing.T) {
		wantJSON(t, oaiCompatWire(t, xaiCfg, NewGrok43().WithReasoningEffort(XAIEffortHigh)),
			"reasoning_effort", `"high"`)
		wantJSON(t, oaiCompatWire(t, xaiCfg, NewGrok43().WithReasoningEffort(XAIEffortNone)),
			"reasoning_effort", `"none"`)
	})

	t.Run("deepseek", func(t *testing.T) {
		wantJSON(t, oaiCompatWire(t, deepSeekCfg, NewDeepSeekV4Pro().WithThinkingDisabled()),
			"thinking", `{"type":"disabled"}`)
		wantJSON(t, oaiCompatWire(t, deepSeekCfg, NewDeepSeekV4Flash().WithThinkingEnabled()),
			"thinking", `{"type":"enabled"}`)
		// The two knobs are independent fields, so both go out.
		body := oaiCompatWire(t, deepSeekCfg,
			NewDeepSeekV4Flash().WithThinkingDisabled().WithReasoningEffort("high"))
		wantJSON(t, body, "thinking", `{"type":"disabled"}`)
		wantJSON(t, body, "reasoning_effort", `"high"`)
	})

	t.Run("openrouter", func(t *testing.T) {
		wantJSON(t, oaiCompatWire(t, openRouterCfg, NewOpenRouterModel("m").WithReasoningEffort("high")),
			"reasoning", `{"effort":"high"}`)
		wantJSON(t, oaiCompatWire(t, openRouterCfg, NewOpenRouterModel("m").WithReasoningMaxTokens(2000)),
			"reasoning", `{"max_tokens":2000}`)
		wantJSON(t, oaiCompatWire(t, openRouterCfg, NewOpenRouterModel("m").WithReasoningExcluded()),
			"reasoning", `{"exclude":true}`)
		// Two setters, one object: the shape callers already depend on.
		wantJSON(t, oaiCompatWire(t, openRouterCfg,
			NewOpenRouterModel("m").WithReasoningEffort("high").WithReasoningExcluded()),
			"reasoning", `{"effort":"high","exclude":true}`)
		// The flat field is never sent beside the object.
		wantJSON(t, oaiCompatWire(t, openRouterCfg, NewOpenRouterModel("m").WithReasoningEffort("high")),
			"reasoning_effort", "")
	})

	t.Run("azure", func(t *testing.T) {
		wantJSON(t, oaiCompatWire(t, func(u string) ProviderConfig {
			return &AzureOpenAIConfig{Endpoint: u, APIKey: "k"}
		}, NewAzureOpenAIReasoningModel("d").WithReasoningEffort("medium")), "reasoning_effort", `"medium"`)
	})

	t.Run("generic endpoint", func(t *testing.T) {
		wantJSON(t, oaiCompatWire(t, oaiCompatCfg,
			NewOpenAICompatibleModel("m").WithReasoningEffort("obsessive")),
			"reasoning_effort", `"obsessive"`)
	})

	t.Run("cohere", func(t *testing.T) {
		wantJSON(t, cohereWire(t, NewCommandAPlus().WithThinkingBudget(2048)),
			"thinking", `{"type":"enabled","token_budget":2048}`)
		// A non-positive budget still enables thinking, with no ceiling.
		wantJSON(t, cohereWire(t, NewCommandAPlus().WithThinkingBudget(0)),
			"thinking", `{"type":"enabled"}`)
		wantJSON(t, cohereWire(t, NewCommandAPlus().WithThinkingDisabled()),
			"thinking", `{"type":"disabled"}`)
	})
}

// TestOpenRouterLegacySettersForwardEdgeValuesVerbatim covers the values the
// portable surface reads as sentinels rather than as numbers.
//
// OpenRouter's three reasoning setters have always written their argument into
// the reasoning object unexamined, so every one of these rows is a request some
// caller's code was already sending. Routing them through the portable
// ThinkingOptions.WithBudget silently reinterpreted four of them: -1 became
// ThinkingBudgetDynamic and came out as enabled:true, 0 and -4096 were clamped
// to "unset" and dropped the key entirely, and an empty effort read as "no
// effort" rather than as the empty string the caller named. Every row here was
// recorded against `git archive HEAD` and must keep marshalling byte for byte.
func TestOpenRouterLegacySettersForwardEdgeValuesVerbatim(t *testing.T) {
	tests := []struct {
		name      string
		model     Model
		reasoning string
	}{
		// The four that regressed.
		{"max_tokens zero", NewOpenRouterModel("m").WithReasoningMaxTokens(0),
			`{"max_tokens":0}`},
		{"max_tokens minus one is not the dynamic sentinel here",
			NewOpenRouterModel("m").WithReasoningMaxTokens(-1), `{"max_tokens":-1}`},
		{"max_tokens negative is not clamped to zero",
			NewOpenRouterModel("m").WithReasoningMaxTokens(-4096), `{"max_tokens":-4096}`},
		{"an empty effort is a value, not an absence",
			NewOpenRouterModel("m").WithReasoningEffort(""), `{"effort":""}`},

		// The rows that never regressed, so the fix cannot be bought at their
		// expense. A pinned budget is still never clamped into the 1024-128000
		// window OpenRouter documents for its Anthropic upstreams.
		{"an ordinary max_tokens", NewOpenRouterModel("m").WithReasoningMaxTokens(2000),
			`{"max_tokens":2000}`},
		{"a max_tokens under the documented floor", NewOpenRouterModel("m").WithReasoningMaxTokens(1),
			`{"max_tokens":1}`},
		{"a max_tokens over the documented ceiling", NewOpenRouterModel("m").WithReasoningMaxTokens(500000),
			`{"max_tokens":500000}`},
		{"an ordinary effort", NewOpenRouterModel("m").WithReasoningEffort("high"),
			`{"effort":"high"}`},
		{"effort none", NewOpenRouterModel("m").WithReasoningEffort("none"),
			`{"effort":"none"}`},
		{"two setters, one object",
			NewOpenRouterModel("m").WithReasoningEffort("high").WithReasoningMaxTokens(0),
			`{"effort":"high","max_tokens":0}`},
		{"last write wins", NewOpenRouterModel("m").WithReasoningMaxTokens(2048).WithReasoningMaxTokens(0),
			`{"max_tokens":0}`},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			wantJSON(t, oaiCompatWire(t, openRouterCfg, tc.model), "reasoning", tc.reasoning)
		})
	}

	// The portable API keeps its own reading of the same numbers: -1 is
	// ThinkingBudgetDynamic, which OpenRouter has no field for and which
	// degrades to a plain enable, and a negative that is not the sentinel is
	// clamped away. Only the per-model setters bypass that.
	t.Run("the portable surface still reads its sentinels", func(t *testing.T) {
		wantJSON(t, oaiCompatWire(t, openRouterCfg, Thinking(NewOpenRouterModel("m"), WithDynamicThinking())),
			"reasoning", `{"enabled":true}`)
		wantJSON(t, oaiCompatWire(t, openRouterCfg, Thinking(NewOpenRouterModel("m"), WithThinkingBudget(-4096))),
			"reasoning", `{"enabled":true}`)
		// And a portable budget is clamped where a pinned one is not.
		wantJSON(t, oaiCompatWire(t, openRouterCfg, Thinking(NewOpenRouterModel("m"), WithThinkingBudget(1))),
			"reasoning", `{"max_tokens":1024}`)
	})

	// A pin only outranks the plan where the plan could not carry it. An off is
	// the plan deliberately replacing the pin, so no depth goes out beside it.
	t.Run("a disable still replaces the pinned depth", func(t *testing.T) {
		wantJSON(t, oaiCompatWire(t, openRouterCfg,
			NoThinking(NewOpenRouterModel("m").WithReasoningMaxTokens(0))),
			"reasoning", `{"enabled":false}`)
		wantJSON(t, oaiCompatWire(t, openRouterCfg,
			NoThinking(NewOpenRouterModel("m").WithReasoningEffort(""))),
			"reasoning", `{"enabled":false}`)
	})
}

// TestPinnedValuesBeatThePortableOnesOnTheWire is the two surfaces side by side
// on one model: the same number, named through the old setter and through the
// new one, and only the new one is adapted.
func TestPinnedValuesBeatThePortableOnesOnTheWire(t *testing.T) {
	// Anthropic 4.6 accepts budgets from 1024; 500 is illegal either way, but
	// only the portable path is lingo's to correct.
	wantJSON(t, anthropicWire(t, NewClaudeSonnet46().WithThinkingBudget(500)),
		"thinking", `{"type":"enabled","budget_tokens":500}`)
	wantJSON(t, anthropicWire(t, Thinking(NewClaudeSonnet46(), WithThinkingBudget(500))),
		"thinking", `{"type":"enabled","budget_tokens":1024}`)

	// OpenAI over chat completions cannot send max at all, so the portable
	// request is clamped to the deepest rung the dialect has.
	wantJSON(t, openAIWire(t, NewGPT56Sol().WithReasoningEffort("max")), "reasoning_effort", `"max"`)
	wantJSON(t, openAIWire(t, Thinking(NewGPT56Sol(), WithThinkingEffort(ThinkingEffortMax))),
		"reasoning_effort", `"xhigh"`)
}

// ============================================================================
// Thinking / NoThinking
// ============================================================================

func TestThinkingPreservesConcreteTypeAndChains(t *testing.T) {
	// The point of the generic signature: Thinking returns *ClaudeSonnet5, not
	// Model, so the builder chain survives on both sides of it. This has to
	// compile as written.
	m := Thinking(NewClaudeSonnet5().WithSystemPrompt("be terse"),
		WithThinkingEffort(ThinkingEffortHigh), WithThinkingTrace(ThinkingTraceOmit)).
		WithMaxTokens(8192)

	var _ *ClaudeSonnet5 = m

	to := m.ThinkingOptions()
	if !to.Enabled() || to.Disabled() || to.Mode() != ThinkingModeOn {
		t.Errorf("mode = %v, want ThinkingModeOn", to.Mode())
	}
	if to.Effort() != ThinkingEffortHigh {
		t.Errorf("effort = %q", to.Effort())
	}
	if to.Trace() != ThinkingTraceOmit {
		t.Errorf("trace = %v", to.Trace())
	}
	if m.maxTokens != 8192 || m.systemPrompt != "be terse" {
		t.Errorf("builder options lost: %+v", m.anthropicThinkingOptions)
	}

	// NoThinking is the same shape, and the last call wins on the portable
	// surface -- unlike the Anthropic setters, whose fixed precedence is their
	// own backward-compat concern.
	g := NoThinking(Thinking(NewGemini25Flash(), WithThinkingBudget(4096))).WithMaxTokens(512)
	var _ *Gemini25Flash = g
	if !g.ThinkingOptions().Disabled() {
		t.Errorf("mode = %v, want ThinkingModeOff", g.ThinkingOptions().Mode())
	}
	if g.ThinkingOptions().Budget() != 4096 {
		t.Error("NoThinking must switch thinking off without erasing the rest of the configuration")
	}
}

func TestThinkingMutatesInPlace(t *testing.T) {
	m := NewGPT5()
	if got := Thinking(m); got != m {
		t.Error("Thinking returned a different pointer; the model must be mutated in place")
	}
	if got := NoThinking(m); got != m {
		t.Error("NoThinking returned a different pointer")
	}
	// A bare Thinking sets the mode and nothing else, so a provider that
	// reasons by default sends exactly what it sent before.
	fresh := Thinking(NewClaudeOpus5())
	to := fresh.ThinkingOptions()
	if !to.Enabled() || to.Effort() != "" || to.Budget() != 0 || to.Trace() != ThinkingTraceDefault {
		t.Errorf("bare Thinking set more than the mode: %+v", to)
	}
}

func TestThinkingOnAModelThatCarriesNoConfigurationIsSilent(t *testing.T) {
	// GPT-4o, Claude 3.x and the non-Claude Bedrock families carry no thinking
	// storage at all, so the call has to fall through the type assertion rather
	// than panic -- and still return the concrete type it was given.
	for _, m := range []Model{
		NewGPT4o(),
		NewClaude3Opus(),
		NewClaude35Sonnet(),
		NewBedrockTitanTextPremier(),
		NewBedrockLlama33Instruct70B(),
		NewBedrockMistralLarge(),
	} {
		if _, ok := m.(ThinkingModel); ok {
			t.Errorf("%s/%s unexpectedly satisfies ThinkingModel", m.Provider(), m.ModelName())
		}
		if to := modelThinkingOptions(m); to != nil {
			t.Errorf("%s/%s: modelThinkingOptions = %+v, want nil", m.Provider(), m.ModelName(), to)
		}
	}

	gpt4o := NewGPT4o()
	if got := Thinking(gpt4o, WithThinkingEffort(ThinkingEffortMax)); got != gpt4o {
		t.Error("Thinking must return the same model")
	}
	if got := NoThinking(gpt4o).WithMaxTokens(128); got != gpt4o {
		t.Error("NoThinking must return the same model and keep its concrete type")
	}

	claude := NewClaude35Sonnet()
	if got := Thinking(claude).WithTemperature(0.2); got != claude {
		t.Error("Thinking must return the same Claude 3.5 model and keep its concrete type")
	}
}

func TestThinkingOptionsSettersRoundTrip(t *testing.T) {
	m := NewGemini25Flash()
	m.ThinkingOptions().Enable().WithBudget(4096).WithTrace(ThinkingTraceInclude)

	to := modelThinkingOptions(m)
	if !to.Enabled() || to.Budget() != 4096 || to.Trace() != ThinkingTraceInclude {
		t.Errorf("options = %+v", to)
	}
	// The statement form and the functional form reach the same struct.
	if to != m.ThinkingOptions() {
		t.Error("modelThinkingOptions and ThinkingOptions returned different pointers")
	}
	// And the portable option functions are the same writes by another name.
	var direct, viaOptions ThinkingOptions
	direct.WithEffort(ThinkingEffortLow).WithBudget(2048).WithTrace(ThinkingTraceOmit)
	for _, opt := range []ThinkingOption{
		WithThinkingEffort(ThinkingEffortLow), WithThinkingBudget(2048),
		WithThinkingTrace(ThinkingTraceOmit),
	} {
		opt(&viaOptions)
	}
	if direct != viaOptions {
		t.Errorf("option functions wrote %+v, the methods wrote %+v", viaOptions, direct)
	}

	// WithDynamicThinking is WithBudget(ThinkingBudgetDynamic).
	var dyn ThinkingOptions
	WithDynamicThinking()(&dyn)
	if !dyn.DynamicBudget() || dyn.Budget() != ThinkingBudgetDynamic || !dyn.Enabled() {
		t.Errorf("dynamic options = %+v", dyn)
	}
}

// ============================================================================
// USAGE NORMALIZATION
// ============================================================================

func TestWithThinkingNormalizesBothReportingStyles(t *testing.T) {
	tests := []struct {
		name       string
		in         TokenUsage
		thinking   int
		subset     bool
		want       TokenUsage
		wantAnswer int
	}{{
		// Everyone but Google: the count is a breakdown of the completion
		// total, so folding it in would double count.
		name:     "subset leaves the completion and total alone",
		in:       TokenUsage{PromptTokens: 10, CompletionTokens: 50, TotalTokens: 60},
		thinking: 20, subset: true,
		want:       TokenUsage{PromptTokens: 10, CompletionTokens: 50, TotalTokens: 60, ThinkingTokens: 20},
		wantAnswer: 30,
	}, {
		// Google counts thoughts inside totalTokenCount but outside
		// candidatesTokenCount, so the completion grows and the total does not.
		name:     "outside grows the completion but not a total that already covers it",
		in:       TokenUsage{PromptTokens: 100, CompletionTokens: 7, TotalTokens: 150},
		thinking: 43, subset: false,
		want:       TokenUsage{PromptTokens: 100, CompletionTokens: 50, TotalTokens: 150, ThinkingTokens: 43},
		wantAnswer: 7,
	}, {
		// The guard only raises a total that is demonstrably short.
		name:     "outside raises a total that is short",
		in:       TokenUsage{PromptTokens: 100, CompletionTokens: 7, TotalTokens: 107},
		thinking: 43, subset: false,
		want:       TokenUsage{PromptTokens: 100, CompletionTokens: 50, TotalTokens: 150, ThinkingTokens: 43},
		wantAnswer: 7,
	}, {
		name:     "no thinking activity is a no-op, subset",
		in:       TokenUsage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
		thinking: 0, subset: true,
		want:       TokenUsage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
		wantAnswer: 5,
	}, {
		name:     "no thinking activity is a no-op, outside",
		in:       TokenUsage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
		thinking: 0, subset: false,
		want:       TokenUsage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
		wantAnswer: 5,
	}, {
		// A provider reporting nonsense must not produce a negative counter or
		// shrink the completion total.
		name:     "a negative count clamps to zero, subset",
		in:       TokenUsage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
		thinking: -20, subset: true,
		want:       TokenUsage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
		wantAnswer: 5,
	}, {
		name:     "a negative count clamps to zero, outside",
		in:       TokenUsage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
		thinking: -20, subset: false,
		want:       TokenUsage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
		wantAnswer: 5,
	}, {
		// Larger than the completion can only be a provider bug; AnswerTokens
		// floors at zero rather than going negative.
		name:     "an over-large subset count floors the answer",
		in:       TokenUsage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
		thinking: 40, subset: true,
		want:       TokenUsage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15, ThinkingTokens: 40},
		wantAnswer: 0,
	}}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := tc.in.withThinking(tc.thinking, tc.subset)
			if got != tc.want {
				t.Errorf("withThinking(%d, %t) = %+v, want %+v", tc.thinking, tc.subset, got, tc.want)
			}
			if got.AnswerTokens() != tc.wantAnswer {
				t.Errorf("AnswerTokens() = %d, want %d", got.AnswerTokens(), tc.wantAnswer)
			}
		})
	}
}

func TestWithThinkingDoesNotMutateTheReceiver(t *testing.T) {
	// Providers chain withCache and withThinking off the usage they parsed, so
	// a value receiver that mutated would corrupt whichever came first.
	u := TokenUsage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15}
	_ = u.withThinking(20, false)
	if (u != TokenUsage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15}) {
		t.Errorf("receiver mutated: %+v", u)
	}
	// The two normalizations compose in either order.
	a := u.withCache(64, 0, false).withThinking(2, true)
	b := u.withThinking(2, true).withCache(64, 0, false)
	if a != b {
		t.Errorf("withCache then withThinking = %+v, the other order = %+v", a, b)
	}
	if a.PromptTokens != 74 || a.CompletionTokens != 5 || a.ThinkingTokens != 2 {
		t.Errorf("composed usage = %+v", a)
	}
}

// ============================================================================
// RESPONSE EXTRACTION: THE TWO BUGS THIS FEATURE FIXED
// ============================================================================

// TestAnthropicMultiBlockAnswerSurvivesExtraction guards a live bug the thinking
// work uncovered: the extraction loop assigned where it had to accumulate, so a
// response split across several blocks -- the normal shape once thinking is on --
// came back holding only its last text block. The answer was silently truncated,
// with no error and no clue in the metadata.
// multiBlockAnthropicStub returns an answer split across three text blocks with
// the trace interleaved, which is what a real extended-thinking response looks
// like once more than one block comes back.
func multiBlockAnthropicStub(t *testing.T) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"msg_1","type":"message","role":"assistant",
			"model":"claude-opus-5","stop_reason":"end_turn",
			"content":[
				{"type":"thinking","thinking":"first the premise, ","signature":"sig-a"},
				{"type":"text","text":"the answer, "},
				{"type":"thinking","thinking":"then the arithmetic","signature":"sig-b"},
				{"type":"text","text":"in three "},
				{"type":"text","text":"parts"}],
			"usage":{"input_tokens":10,"output_tokens":30,
				"output_tokens_details":{"thinking_tokens":12}}}`)
	}))
	t.Setenv("ANTHROPIC_BASE_URL", srv.URL)
	return srv
}

func TestAnthropicMultiBlockAnswerSurvivesExtraction(t *testing.T) {
	srv := multiBlockAnthropicStub(t)
	defer srv.Close()

	resp := generate(t, &AnthropicConfig{APIKey: "k"}, NewClaudeOpus5())

	if resp.Text != "the answer, in three parts" {
		t.Errorf("Text = %q, want every text block concatenated in order", resp.Text)
	}
	if resp.Thinking != "first the premise, then the arithmetic" {
		t.Errorf("Thinking = %q, want every thinking block concatenated in order", resp.Thinking)
	}
	// The trace is never part of the answer.
	if strings.Contains(resp.Text, "premise") || strings.Contains(resp.Text, "arithmetic") {
		t.Errorf("the trace leaked into the answer: %q", resp.Text)
	}
	// The deprecated key still says exactly what the typed field says.
	if resp.Metadata["thinking"] != resp.Thinking {
		t.Errorf("Metadata[thinking] = %q, Thinking = %q", resp.Metadata["thinking"], resp.Thinking)
	}
}

// TestGoogleThoughtPartsStayOutOfTheAnswer guards the second live bug: Google
// returns its reasoning as ordinary text parts flagged Thought, and the response
// loop concatenated every part with non-empty text, so the trace was handed back
// to callers as part of the answer.
func TestGoogleThoughtPartsStayOutOfTheAnswer(t *testing.T) {
	var c capture
	srv := geminiThinkingStub(t, &c, geminiThinkingResponse)
	defer srv.Close()

	resp := generate(t, &GoogleConfig{APIKey: "k"}, NewGemini3Pro())

	if resp.Text != "the answer continues" {
		t.Errorf("Text = %q, want the answer parts only", resp.Text)
	}
	if strings.Contains(resp.Text, "weighing the options") {
		t.Errorf("a thought part leaked into the answer: %q", resp.Text)
	}
	if resp.Thinking != "weighing the options" {
		t.Errorf("Thinking = %q, want the thought part", resp.Thinking)
	}
	// Google is the provider whose thoughts sit outside the candidate total, so
	// the completion count is normalized and the answer share stays honest.
	if resp.Usage.ThinkingTokens != 43 || resp.Usage.CompletionTokens != 50 {
		t.Errorf("usage = %+v, want the thoughts folded into the completion total", resp.Usage)
	}
	if resp.Usage.AnswerTokens() != 7 {
		t.Errorf("AnswerTokens() = %d, want 7", resp.Usage.AnswerTokens())
	}
}

// ============================================================================
// NIL SAFETY
// ============================================================================

// TestNilThinkingOptionsAccessorsAreSafe covers the read path providers take on
// a model that carries no thinking configuration: modelThinkingOptions returns
// nil and every reader is called straight off it. The mutators are deliberately
// not nil-safe -- they are only ever reached through a model's own accessor,
// which returns the address of a field -- so a nil here must read as "nothing
// was configured", never panic.
func TestNilThinkingOptionsAccessorsAreSafe(t *testing.T) {
	var to *ThinkingOptions

	if to.Mode() != ThinkingModeDefault {
		t.Errorf("Mode() = %v", to.Mode())
	}
	if to.Enabled() {
		t.Error("Enabled() = true")
	}
	if to.Disabled() {
		t.Error("Disabled() = true")
	}
	if to.Effort() != "" {
		t.Errorf("Effort() = %q", to.Effort())
	}
	if to.Budget() != 0 {
		t.Errorf("Budget() = %d", to.Budget())
	}
	if to.DynamicBudget() {
		t.Error("DynamicBudget() = true")
	}
	if to.Trace() != ThinkingTraceDefault {
		t.Errorf("Trace() = %v", to.Trace())
	}
	if to.isPinned(ThinkingCanSetEffort) || to.isPinned(0) {
		t.Error("isPinned() = true on a nil receiver")
	}

	// And the two functions that consume one.
	checkPlan(t, planThinking(nil, dimsEverything, window32k, ladderFull...), wantPlan{})
	if got := modelThinkingOptions(NewGPT4o()); got != nil {
		t.Errorf("modelThinkingOptions of a model without storage = %+v, want nil", got)
	}
}
