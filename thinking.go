package lingo

import "fmt"

// ============================================================================
// THINKING CONTROL
// ============================================================================
//
// Reasoning models spend tokens working a problem out before they answer.
// Providers let you steer that spend in three incompatible vocabularies:
//
//   - budget   a ceiling in thinking tokens (Anthropic's legacy
//              thinking.budget_tokens, Google's thinkingBudget, Cohere's
//              thinking.token_budget, OpenRouter's reasoning.max_tokens)
//   - effort   an ordinal ladder (OpenAI reasoning_effort, Anthropic
//              output_config.effort on 4.6+, xAI, Google's 3.x thinkingLevel,
//              OpenRouter reasoning.effort, Ollama's think levels)
//   - toggle   on or off, with the depth left to the model (DeepSeek's
//              thinking object, Cohere's thinking.type, Ollama's think bool,
//              Anthropic's adaptive and disabled thinking configs)
//
// ThinkingOptions carries all three plus a trace-visibility knob, and each
// provider projects it onto whatever its API models. Use ThinkingSupport to ask
// how far a provider goes and ModelThinkingDimensions to ask what one model
// actually honours -- unlike caching, thinking support varies per model inside
// Anthropic, xAI, Cohere, Google and Ollama, not just per provider.
//
// Thinking is opt-in on the request side. A model whose ThinkingOptions were
// never touched produces byte-for-byte the request it produced before this
// feature existed -- including the reasoning_effort several OpenAI and xAI
// constructors have always seeded, which is preserved as a pinned default (see
// ThinkingOptions.pinned).
//
// Asking for something a provider cannot do is never an error. A dimension a
// provider does not model is either translated to the nearest one it does
// (deterministically, by the published tables below, with a breadcrumb in
// GenerationResponse.Metadata["thinking_translation"]) or dropped. It is never
// forwarded blind and never returned as an error.
//
// Read-side reporting is unconditional: TokenUsage.ThinkingTokens and
// GenerationResponse.Thinking are filled in whenever the provider reports them,
// whether or not the caller opted in.

// ThinkingMode is the tri-state opt-in for thinking. The zero value,
// ThinkingModeDefault, leaves the provider's own behaviour untouched and is
// what every model carries until a caller changes it.
//
// The default is not "off": DeepSeek V4, Cohere's reasoning models, Claude 5,
// Gemini 3.x and Ollama's thinking models all reason by default. Default means
// lingo sends no toggle at all.
type ThinkingMode int

const (
	// ThinkingModeDefault leaves thinking to the provider. lingo sends no
	// enable/disable field, so models that reason by default keep doing so.
	ThinkingModeDefault ThinkingMode = iota
	// ThinkingModeOn asks the model to reason before answering. On providers
	// that already reason by default it is a no-op that changes no bytes.
	ThinkingModeOn
	// ThinkingModeOff asks the model not to reason. Only providers with a real
	// off switch honour it -- Anthropic Claude 5, Cohere, DeepSeek, Google
	// Gemini 2.5 Flash, Ollama, and the providers whose effort ladder includes
	// "none". Elsewhere it is a no-op that changes no bytes: a depth already
	// configured stays on the wire rather than being dropped, because dropping it
	// would hand the model back to a server-side default that reasons harder than
	// the caller asked for.
	ThinkingModeOff
)

// ThinkingEffort is an ordinal thinking-depth level. It is a string type with
// an open vocabulary on purpose: the constants below are the neutral ladder,
// but any provider-specific value a caller passes through the older
// WithReasoningEffort setters round-trips unchanged.
//
// No provider accepts the whole ladder. Each maps it down; see the table on
// ThinkingDimensions and the README for what each one does with each rung.
type ThinkingEffort string

const (
	// ThinkingEffortNone asks for no reasoning at all. Equivalent to
	// ThinkingModeOff on providers that spell "off" as an effort level
	// (OpenAI gpt-5.1+, xAI grok-4.3, OpenRouter).
	ThinkingEffortNone ThinkingEffort = "none"
	// ThinkingEffortMinimal is the shallowest level that still reasons.
	// OpenAI's original GPT-5 family and Google's Gemini 3.x only.
	ThinkingEffortMinimal ThinkingEffort = "minimal"
	// ThinkingEffortLow suits short, scoped, latency-sensitive tasks.
	ThinkingEffortLow ThinkingEffort = "low"
	// ThinkingEffortMedium trades some depth for reduced token usage.
	ThinkingEffortMedium ThinkingEffort = "medium"
	// ThinkingEffortHigh is the recommended minimum for intelligence-sensitive
	// work and is the API default on Anthropic and DeepSeek.
	ThinkingEffortHigh ThinkingEffort = "high"
	// ThinkingEffortXHigh sits between high and max. Anthropic 4.7+, OpenAI
	// gpt-5.4+, xAI grok-4.6, OpenRouter.
	ThinkingEffortXHigh ThinkingEffort = "xhigh"
	// ThinkingEffortMax is the deepest setting. Anthropic, DeepSeek and Ollama;
	// on OpenAI it exists only on the Responses API, which lingo does not use,
	// so lingo maps it down to xhigh there.
	ThinkingEffortMax ThinkingEffort = "max"
)

// rank orders the neutral ladder so a provider can clamp to its own top and
// bottom rung. Values outside the ladder rank -1 and are never translated, only
// forwarded to providers that were given them explicitly.
func (e ThinkingEffort) rank() int {
	switch e {
	case ThinkingEffortNone:
		return 0
	case ThinkingEffortMinimal:
		return 1
	case ThinkingEffortLow:
		return 2
	case ThinkingEffortMedium:
		return 3
	case ThinkingEffortHigh:
		return 4
	case ThinkingEffortXHigh:
		return 5
	case ThinkingEffortMax:
		return 6
	}
	return -1
}

// clampEffort returns the highest rung in allowed that does not exceed e, or
// the lowest rung above e when nothing sits below it. It returns ok=false when
// e is off-ladder or allowed is empty, in which case the caller drops the field
// rather than guessing.
func clampEffort(e ThinkingEffort, allowed ...ThinkingEffort) (ThinkingEffort, bool) {
	if len(allowed) == 0 {
		return "", false
	}
	for _, a := range allowed {
		if a == e {
			return e, true
		}
	}
	r := e.rank()
	if r < 0 {
		return "", false
	}
	// A request to think a little must never clamp down to not thinking at all,
	// so "none" is a candidate only when it is what was asked for. Without this
	// a minimal on a ladder that has none but no minimal -- OpenAI gpt-5.1,
	// xAI grok-4.3 -- would silently switch reasoning off.
	floor := 0
	if r > 0 {
		floor = 1
	}
	best, bestRank := ThinkingEffort(""), -1
	for _, a := range allowed {
		if ar := a.rank(); ar >= floor && ar <= r && ar > bestRank {
			best, bestRank = a, ar
		}
	}
	if bestRank >= 0 {
		return best, true
	}
	// Nothing at or below e; take the lowest rung the provider does accept.
	best, bestRank = "", 1<<30
	for _, a := range allowed {
		if ar := a.rank(); ar >= floor && ar < bestRank {
			best, bestRank = a, ar
		}
	}
	return best, best != ""
}

// ThinkingBudgetDynamic asks the model to decide for itself how much to think,
// rather than naming a ceiling. It is Anthropic's adaptive thinking config and
// Google's thinkingBudget of -1; on providers with no dynamic setting it
// degrades to plain ThinkingModeOn.
const ThinkingBudgetDynamic = -1

// ThinkingTrace controls whether the model's reasoning comes back with the
// answer. It is orthogonal to how much the model thinks.
type ThinkingTrace int

const (
	// ThinkingTraceDefault leaves visibility to the provider.
	ThinkingTraceDefault ThinkingTrace = iota
	// ThinkingTraceInclude asks for the reasoning to be returned. It arrives in
	// GenerationResponse.Thinking. On Anthropic it sets thinking display to
	// summarized; on Google it sets includeThoughts. Providers that always
	// return their trace, and those that cannot be asked over the API dialect
	// lingo speaks (OpenAI and Azure over chat completions), ignore it.
	ThinkingTraceInclude
	// ThinkingTraceOmit asks the provider to reason but withhold the trace,
	// saving the tokens it would cost to return. Anthropic (display omitted),
	// Google (includeThoughts false) and OpenRouter (reasoning.exclude) honour
	// it; elsewhere it is a no-op, and a trace lingo receives anyway is still
	// reported.
	ThinkingTraceOmit
)

// ThinkingDimension is a bitmask of the thinking knobs an API models. It is how
// lingo answers "will this actually do anything" precisely, where the coarse
// ThinkingSupport ladder cannot.
type ThinkingDimension uint8

const (
	// ThinkingCanToggle means thinking can be switched on or off.
	ThinkingCanToggle ThinkingDimension = 1 << iota
	// ThinkingCanSetEffort means the API takes an ordinal depth level.
	ThinkingCanSetEffort
	// ThinkingCanSetBudget means the API takes a ceiling in thinking tokens.
	ThinkingCanSetBudget
	// ThinkingCanHideTrace means the API can be asked to withhold the trace.
	ThinkingCanHideTrace
	// ThinkingCanReportTokens means the API reports a thinking token count.
	ThinkingCanReportTokens
	// ThinkingCanReportTrace means the API returns the reasoning text.
	ThinkingCanReportTrace
)

// Has reports whether every dimension in want is present in d.
func (d ThinkingDimension) Has(want ThinkingDimension) bool { return d&want == want }

// ThinkSupport describes how far a provider goes with thinking control. It is
// deliberately coarse: the ladder answers "is there anything to ask for", and
// ThinkingDimensions answers what.
type ThinkSupport int

const (
	// ThinkSupportNone means the provider neither takes a thinking instruction
	// nor reports anything about thinking.
	ThinkSupportNone ThinkSupport = iota
	// ThinkSupportUsageOnly means the model reasons on its own terms and lingo
	// can only report what came back -- a thinking token count, a trace, or
	// both. There is nothing to ask for.
	ThinkSupportUsageOnly
	// ThinkSupportControl means the request carries a thinking instruction that
	// lingo builds for you. Which of the four request-side dimensions are real
	// varies by model; call ThinkingDimensions or ModelThinkingDimensions.
	ThinkSupportControl
)

// String returns the support level as a short lowercase label.
func (s ThinkSupport) String() string {
	switch s {
	case ThinkSupportUsageOnly:
		return "usage-only"
	case ThinkSupportControl:
		return "control"
	default:
		return "none"
	}
}

// ThinkingSupport reports how much of the thinking surface a provider honours.
// It is advisory: calling a thinking setter on a provider below
// ThinkSupportControl is a no-op, not an error.
//
// It answers for the bulk of a provider's catalogue. Every provider here has
// models that go less far than their provider's level -- Claude 3.5, GPT-4o,
// Command R and Llama on Bedrock take no thinking instruction at all -- so use
// ModelThinkingDimensions when the answer has to be right for one model.
func ThinkingSupport(provider ProviderType) ThinkSupport {
	switch provider {
	case ProviderAnthropic, ProviderOpenAI, ProviderGoogle, ProviderBedrock,
		ProviderAzure, ProviderXAI, ProviderDeepSeek, ProviderOpenRouter,
		ProviderCohere, ProviderOllama, ProviderPerplexity:
		return ThinkSupportControl
	case ProviderOpenAICompatible:
		// The endpoint behind BaseURL decides. lingo forwards only the effort
		// the caller set explicitly and synthesizes nothing.
		return ThinkSupportUsageOnly
	default:
		return ThinkSupportNone
	}
}

// ThinkingDimensions reports which thinking knobs a provider's reasoning models
// accept, for the bulk of its catalogue.
//
//	Anthropic   toggle | effort | budget | hide-trace | tokens | trace
//	Google      toggle | effort | budget | hide-trace | tokens | trace
//	OpenRouter  toggle | effort | budget | hide-trace | tokens | trace
//	Bedrock     toggle | effort | budget                       | trace
//	Cohere      toggle |          budget                       | trace
//	DeepSeek    toggle | effort                      | tokens  | trace
//	Ollama      toggle | effort                                | trace
//	OpenAI               effort                      | tokens
//	Azure                effort                      | tokens
//	xAI                  effort                      | tokens  | trace
//	Perplexity           effort                      | tokens
//	OAI-compat                                       | tokens  | trace
//
// Bedrock reports no thinking token count on either of its two request paths:
// bedrockruntime's TokenUsage has no reasoning field, and lingo's own Claude
// InvokeModel body would have to read Anthropic's
// usage.output_tokens_details.thinking_tokens, which Bedrock does not return.
func ThinkingDimensions(provider ProviderType) ThinkingDimension {
	const report = ThinkingCanReportTokens | ThinkingCanReportTrace
	switch provider {
	case ProviderAnthropic, ProviderGoogle, ProviderOpenRouter:
		return ThinkingCanToggle | ThinkingCanSetEffort | ThinkingCanSetBudget |
			ThinkingCanHideTrace | report
	case ProviderBedrock:
		return ThinkingCanToggle | ThinkingCanSetEffort | ThinkingCanSetBudget |
			ThinkingCanReportTrace
	case ProviderCohere:
		return ThinkingCanToggle | ThinkingCanSetBudget | ThinkingCanReportTrace
	case ProviderDeepSeek:
		return ThinkingCanToggle | ThinkingCanSetEffort | report
	case ProviderOllama:
		return ThinkingCanToggle | ThinkingCanSetEffort | ThinkingCanReportTrace
	case ProviderXAI:
		return ThinkingCanSetEffort | report
	case ProviderOpenAI, ProviderAzure, ProviderPerplexity:
		return ThinkingCanSetEffort | ThinkingCanReportTokens
	case ProviderOpenAICompatible:
		return report
	default:
		return 0
	}
}

// thinkingCapable is the unexported hook a provider's model types implement to
// answer for themselves, rather than inheriting their provider's answer. It is
// declared in the same idiom as openAIExplicitCacheModel: the value is
// consulted, not merely the type, so a generic escape-hatch model
// (AnthropicModel, GoogleModel, OllamaModel, OpenRouterModel) can resolve its
// own capabilities from the model id it was handed.
type thinkingCapable interface {
	thinkingDimensions() ThinkingDimension
}

// ModelThinkingDimensions reports which thinking knobs one model honours,
// falling back to its provider's answer when the model does not declare its
// own. This is the precise question; ThinkingSupport is the coarse one.
//
//	if lingo.ModelThinkingDimensions(m).Has(lingo.ThinkingCanSetBudget) {
//		m = lingo.Thinking(m, lingo.WithThinkingBudget(16000))
//	}
func ModelThinkingDimensions(model Model) ThinkingDimension {
	if model == nil {
		return 0
	}
	if c, ok := model.(thinkingCapable); ok {
		return c.thinkingDimensions()
	}
	return ThinkingDimensions(model.Provider())
}

// ============================================================================
// OPTIONS
// ============================================================================

// ThinkingOptions is the provider-neutral thinking configuration carried by a
// model. Obtain it from a model with ThinkingOptions(), or configure it in a
// fluent chain with Thinking.
//
//	m := lingo.Thinking(lingo.NewClaudeOpus5().WithMaxTokens(8192),
//		lingo.WithThinkingEffort(lingo.ThinkingEffortXHigh))
//
//	// equivalent, statement form
//	m := lingo.NewClaudeOpus5().WithMaxTokens(8192)
//	m.ThinkingOptions().Enable().WithEffort(lingo.ThinkingEffortXHigh)
//
// It is also the single storage behind every per-model thinking setter lingo
// already shipped. WithThinkingBudget, WithEffort, WithAdaptiveThinking,
// WithThinkingDisabled, WithThinkingEnabled, WithReasoningEffort,
// WithReasoningMaxTokens and WithReasoningExcluded all read and write these
// same fields, so the two views can never disagree.
type ThinkingOptions struct {
	mode   ThinkingMode
	effort ThinkingEffort
	// budget is a ceiling in thinking tokens, 0 when unset and
	// ThinkingBudgetDynamic when the model should decide.
	budget int
	trace  ThinkingTrace
	// pinned records which dimensions were set through a model-specific setter
	// (or seeded by a constructor) rather than through the neutral surface.
	//
	// A pinned dimension is forwarded verbatim: not clamped, not translated,
	// not dropped. The caller named a knob on a concrete model type, so they
	// meant that model's dialect, and changing what goes on the wire would be a
	// behaviour change for code written before this feature existed. An
	// unpinned dimension came from the portable API, which promises adaptation
	// rather than fidelity, so lingo is free to clamp it into the model's legal
	// range or drop it.
	pinned ThinkingDimension
}

// Enable asks the model to think. It sets no depth, so a provider that reasons
// by default sends exactly what it sent before.
func (o *ThinkingOptions) Enable() *ThinkingOptions {
	o.mode = ThinkingModeOn
	return o.unpin(ThinkingCanToggle)
}

// Disable asks the model not to think. Providers with no off switch ignore it.
func (o *ThinkingOptions) Disable() *ThinkingOptions {
	o.mode = ThinkingModeOff
	return o.unpin(ThinkingCanToggle)
}

// WithEffort sets the thinking depth and enables thinking. Providers clamp the
// level to their own ladder; those with no effort knob derive a token budget
// from it when they take one, and otherwise ignore it.
func (o *ThinkingOptions) WithEffort(e ThinkingEffort) *ThinkingOptions {
	o.effort = e
	if e == ThinkingEffortNone {
		o.mode = ThinkingModeOff
	} else if e != "" {
		o.mode = ThinkingModeOn
	}
	return o.unpin(ThinkingCanSetEffort)
}

// WithBudget caps thinking at tokens and enables thinking. Pass
// ThinkingBudgetDynamic to let the model decide. Providers clamp the value into
// the range their model accepts; those with no budget knob derive an effort
// level from it when they take one, and otherwise ignore it.
func (o *ThinkingOptions) WithBudget(tokens int) *ThinkingOptions {
	if tokens < 0 && tokens != ThinkingBudgetDynamic {
		tokens = 0
	}
	o.budget = tokens
	if tokens != 0 {
		o.mode = ThinkingModeOn
	}
	return o.unpin(ThinkingCanSetBudget)
}

// WithDynamicBudget asks the model to decide how much to think.
func (o *ThinkingOptions) WithDynamicBudget() *ThinkingOptions {
	return o.WithBudget(ThinkingBudgetDynamic)
}

// WithTrace controls whether the reasoning comes back with the answer.
func (o *ThinkingOptions) WithTrace(t ThinkingTrace) *ThinkingOptions {
	o.trace = t
	return o.unpin(ThinkingCanHideTrace)
}

// Mode reports the configured mode.
func (o *ThinkingOptions) Mode() ThinkingMode {
	if o == nil {
		return ThinkingModeDefault
	}
	return o.mode
}

// Enabled reports whether thinking was explicitly turned on.
func (o *ThinkingOptions) Enabled() bool { return o != nil && o.mode == ThinkingModeOn }

// Disabled reports whether thinking was explicitly turned off.
func (o *ThinkingOptions) Disabled() bool { return o != nil && o.mode == ThinkingModeOff }

// Effort reports the requested depth, "" when unset.
func (o *ThinkingOptions) Effort() ThinkingEffort {
	if o == nil {
		return ""
	}
	return o.effort
}

// Budget reports the requested thinking token ceiling: 0 when unset,
// ThinkingBudgetDynamic when the model decides.
func (o *ThinkingOptions) Budget() int {
	if o == nil {
		return 0
	}
	return o.budget
}

// DynamicBudget reports whether the model was asked to decide for itself.
func (o *ThinkingOptions) DynamicBudget() bool { return o != nil && o.budget == ThinkingBudgetDynamic }

// Trace reports the requested trace visibility.
func (o *ThinkingOptions) Trace() ThinkingTrace {
	if o == nil {
		return ThinkingTraceDefault
	}
	return o.trace
}

// setBudgetVerbatim writes a thinking token ceiling exactly as a per-model
// setter was handed it, pins it, and asks for thinking.
//
// It is the budget counterpart of setReasoningEffort: WithBudget belongs to the
// portable surface and so reads the value first -- ThinkingBudgetDynamic out of
// -1, zero out of every other negative -- while a setter on a concrete model
// type has always put its argument into that model's request object unexamined.
// The two edge values that costs are 0, which the portable field spells "unset",
// and a negative that is not the dynamic sentinel, which the portable field
// clamps away; both are values callers' code has been sending since before this
// package modelled thinking at all.
//
// Pinning alone is not enough to carry them: pinned means "do not clamp or
// translate", and these two are lost before any clamp, in the reading. The
// provider that consumes a pinned budget reads it back off the options rather
// than off the plan for exactly that reason -- see oaiCompatClient.thinkingBody.
func (o *ThinkingOptions) setBudgetVerbatim(tokens int) *ThinkingOptions {
	o.budget = tokens
	o.mode = ThinkingModeOn
	return o.pin(ThinkingCanSetBudget)
}

// pin marks dimensions as set by a model-specific setter, or seeded by a
// constructor. Providers' own builders call it; the neutral surface never does.
func (o *ThinkingOptions) pin(d ThinkingDimension) *ThinkingOptions { o.pinned |= d; return o }

// unpin clears the pin on a dimension the portable surface has just written.
//
// The pin says "this value came from a model-specific setter, so forward it
// verbatim". Once the portable surface writes the same dimension that is no
// longer true, and leaving the pin in place would make the portable call
// inherit fidelity it never asked for -- on OpenAI, where every named reasoning
// constructor seeds a pinned effort, it would mean WithThinkingEffort could
// never be clamped to the model's ladder.
//
// Every per-model setter in the library mutates and then pins, so a pin applied
// after the mutation always wins.
func (o *ThinkingOptions) unpin(d ThinkingDimension) *ThinkingOptions { o.pinned &^= d; return o }

// isPinned reports whether a dimension was set through a model-specific setter.
func (o *ThinkingOptions) isPinned(d ThinkingDimension) bool {
	return o != nil && o.pinned&d != 0
}

// ThinkingOption configures a ThinkingOptions in a fluent chain. See Thinking.
type ThinkingOption func(*ThinkingOptions)

// WithThinkingEffort requests a thinking depth.
func WithThinkingEffort(e ThinkingEffort) ThinkingOption {
	return func(o *ThinkingOptions) { o.WithEffort(e) }
}

// WithThinkingBudget caps thinking at tokens.
func WithThinkingBudget(tokens int) ThinkingOption {
	return func(o *ThinkingOptions) { o.WithBudget(tokens) }
}

// WithDynamicThinking lets the model decide how much to think.
func WithDynamicThinking() ThinkingOption {
	return func(o *ThinkingOptions) { o.WithDynamicBudget() }
}

// WithThinkingTrace controls whether the reasoning is returned.
func WithThinkingTrace(t ThinkingTrace) ThinkingOption {
	return func(o *ThinkingOptions) { o.WithTrace(t) }
}

// ThinkingModel is the optional capability interface a model implements when it
// can carry thinking configuration. Providers type-assert it; models that do
// not implement it are generated exactly as before. Model itself is unchanged,
// so external implementations keep compiling.
//
// Carrying ThinkingOptions is not a promise that any of it reaches the wire.
// Every OpenAI-compatible model carries one because they share an option
// struct, yet a Groq endpoint may ignore every field. The reverse holds too:
// Claude 3.5, GPT-4o and Command R deliberately do not implement it, because
// their APIs have no thinking field and a setter on them would be a lie. Use
// ModelThinkingDimensions to ask what a model does; this interface only says
// where configuration can be stored.
type ThinkingModel interface {
	Model
	// ThinkingOptions returns the model's thinking configuration, never nil.
	ThinkingOptions() *ThinkingOptions
}

// Thinking turns thinking on for a model and returns the same model, so it
// slots into the existing builder chain and keeps the concrete type:
//
//	m := lingo.Thinking(lingo.NewClaudeOpus5(),
//		lingo.WithThinkingEffort(lingo.ThinkingEffortXHigh)).
//		WithMaxTokens(8192)
//
// Models whose provider takes no thinking instruction are returned untouched,
// so the call is safe to make generically across providers.
func Thinking[M Model](model M, opts ...ThinkingOption) M {
	if t, ok := any(model).(ThinkingModel); ok {
		to := t.ThinkingOptions()
		to.Enable()
		for _, opt := range opts {
			opt(to)
		}
	}
	return model
}

// NoThinking asks the model not to think. Only providers with a real off switch
// honour it; elsewhere it is a no-op, and a genuine one -- a depth the model was
// already carrying, whether pinned by a per-model setter or set through this
// package's own portable options, still reaches the wire unchanged. lingo will
// not answer "I cannot switch thinking off" by dropping the ceiling the caller
// had put on it.
func NoThinking[M Model](model M) M {
	if t, ok := any(model).(ThinkingModel); ok {
		t.ThinkingOptions().Disable()
	}
	return model
}

// modelThinkingOptions returns the model's thinking configuration, or nil when
// the model does not carry one. Providers call this instead of type-asserting.
func modelThinkingOptions(model Model) *ThinkingOptions {
	if t, ok := model.(ThinkingModel); ok {
		return t.ThinkingOptions()
	}
	return nil
}

// ============================================================================
// CROSS-VOCABULARY TRANSLATION
// ============================================================================
//
// The three vocabularies do not convert cleanly. lingo translates anyway --
// dropping the caller's intent entirely is worse -- but the translation is
// deterministic, published here, applied only to unpinned dimensions, and
// reported back in GenerationResponse.Metadata["thinking_translation"] so it is
// never silently wrong.

// thinkingBudgetShare is the fraction of a model's usable thinking-token range
// that each effort level asks for, used when a caller sets an effort on a
// provider that only budgets in tokens (Cohere, Gemini 2.5, Claude 3.7-4.5).
//
// The ladder is geometric rather than linear because thinking quality is
// roughly logarithmic in budget: the step from low to medium buys much more
// than the step from xhigh to max.
var thinkingBudgetShare = map[ThinkingEffort]float64{
	ThinkingEffortMinimal: 0.05,
	ThinkingEffortLow:     0.12,
	ThinkingEffortMedium:  0.30,
	ThinkingEffortHigh:    0.60,
	ThinkingEffortXHigh:   0.85,
	ThinkingEffortMax:     1.00,
}

// ThinkingBudgetForEffort returns the thinking token budget lingo derives from
// an effort level for a model whose API budgets in tokens, clamped into
// [min, max]. It returns 0 when the effort has no budget equivalent
// (ThinkingEffortNone, or any off-ladder string), meaning the field is dropped
// rather than guessed.
//
// It is exported so the mapping can be inspected and tested: a caller who
// dislikes the derived number sets an explicit budget instead.
func ThinkingBudgetForEffort(e ThinkingEffort, min, max int) int {
	share, ok := thinkingBudgetShare[e]
	if !ok || max <= 0 || min > max {
		return 0
	}
	n := int(float64(max) * share)
	if n < min {
		n = min
	}
	if n > max {
		n = max
	}
	return n
}

// ThinkingEffortForBudget returns the effort level lingo derives from a token
// budget for a model whose API has no budget knob (OpenAI, Azure, xAI,
// DeepSeek, Ollama, Gemini 3.x).
//
// The thresholds are absolute rather than relative to max_tokens, because an
// effort-only API has no budget to be relative to.
//
//	    budget <= 0            (unset)      ""
//	 1 - 2048                  minimal
//	 2049 - 8192               low
//	 8193 - 24576              medium
//	24577 - 65536              high
//	65537 and above            xhigh
//	ThinkingBudgetDynamic                   "" (falls back to plain enable)
func ThinkingEffortForBudget(tokens int) ThinkingEffort {
	switch {
	case tokens <= 0:
		return ""
	case tokens <= 2048:
		return ThinkingEffortMinimal
	case tokens <= 8192:
		return ThinkingEffortLow
	case tokens <= 24576:
		return ThinkingEffortMedium
	case tokens <= 65536:
		return ThinkingEffortHigh
	default:
		return ThinkingEffortXHigh
	}
}

// thinkingPlan is what one provider decided to put on the wire for one request.
// Providers build it from ThinkingOptions and their own model capabilities,
// then apply it; it exists so the translation is computed in one place and
// reported the same way everywhere.
type thinkingPlan struct {
	// enable and disable are the toggle actually sent, if any.
	enable  bool
	disable bool
	// dynamic asks the model to decide (Anthropic adaptive, Google -1).
	dynamic bool
	// effort and budget are the values actually sent, "" and 0 when the field
	// is not on the wire.
	effort ThinkingEffort
	budget int
	// hideTrace and showTrace request trace visibility.
	hideTrace bool
	showTrace bool
	// notes records every dimension lingo translated or dropped, in the order
	// it decided. It lands in Metadata["thinking_translation"].
	notes []string
}

// note records one translation or drop.
func (p *thinkingPlan) note(format string, args ...any) {
	p.notes = append(p.notes, fmt.Sprintf(format, args...))
}

// translation renders the plan's notes for Metadata, "" when nothing was
// translated or dropped.
func (p *thinkingPlan) translation() string {
	if p == nil || len(p.notes) == 0 {
		return ""
	}
	s := p.notes[0]
	for _, n := range p.notes[1:] {
		s += "; " + n
	}
	return s
}

// budgetRange is a model's legal thinking-token window, used to clamp both
// caller-set and derived budgets. A zero max means the model takes no budget.
type budgetRange struct{ min, max int }

// planThinking projects neutral options onto one model's capabilities. It is
// the single place the three vocabularies meet, so every provider gets the same
// answers and the same breadcrumbs.
//
// dims is what the model honours. br is its legal budget window, consulted only
// when dims has ThinkingCanSetBudget. efforts is the ladder the model accepts,
// consulted only when dims has ThinkingCanSetEffort.
//
// A nil o, or one nobody touched, returns a zero plan: nothing goes on the wire
// and the request is byte-identical to what it was before this feature existed.
//
// ThinkingModeOff on a model with no off switch -- no ThinkingCanToggle and no
// "none" rung on its ladder -- is a TRUE no-op, not a partial one: the depth is
// planned exactly as it would have been without the Disable, and only a note
// records that the off was dropped. That holds for a pinned depth, which must
// reach the wire verbatim whatever else was asked, and for an unpinned one,
// which is still clamped and translated as usual. The alternative -- dropping
// the depth along with the off -- would leave the model at the provider's own
// server-side default, which on OpenAI, xAI and the OpenAI-compatible endpoints
// is deeper reasoning at a different price than the caller had already asked
// for. "I cannot switch it off" must not silently turn into "so I turned it up".
func planThinking(o *ThinkingOptions, dims ThinkingDimension, br budgetRange, efforts ...ThinkingEffort) thinkingPlan {
	var p thinkingPlan
	if o == nil || (o.mode == ThinkingModeDefault && o.effort == "" && o.budget == 0 && o.trace == ThinkingTraceDefault) {
		return p
	}

	// offDropped records that an off had nowhere to go, so the depth section
	// below runs as though it had never been asked for.
	var offDropped bool

	// --- toggle -------------------------------------------------------------
	switch o.mode {
	case ThinkingModeOn:
		if dims.Has(ThinkingCanToggle) {
			p.enable = true
		}
	case ThinkingModeOff:
		var off bool
		if dims.Has(ThinkingCanToggle) {
			p.disable, off = true, true
		} else if dims.Has(ThinkingCanSetEffort) {
			if e, ok := clampEffort(ThinkingEffortNone, efforts...); ok && e == ThinkingEffortNone {
				p.effort, off = ThinkingEffortNone, true
				p.note("thinking off sent as effort=none")
			}
		}
		if off {
			// Thinking really is off, so a depth has nothing left to describe.
			return p
		}
		// No off switch, so off is a no-op rather than a half-measure: the depth
		// section below runs exactly as it would have without the Disable. Cutting
		// it short here instead would drop the depth and leave the model at the
		// provider's own default, which on OpenAI, xAI and every OpenAI-compatible
		// endpoint means MORE thinking than the caller had asked for, not less.
		p.note("thinking off dropped: model has no off switch")
		offDropped = true
	}

	// --- depth --------------------------------------------------------------
	wantEffort, wantBudget := o.effort, o.budget

	// "none" is the off request spelled as a level, not a depth of its own, so
	// where the off was just dropped an unpinned none goes with it. Leaving it
	// for the ladder below would hand it to clampEffort, which raises it to the
	// lowest real rung -- turning "do not think" into "think a little" one line
	// after this function decided that thinking less is not the same as off.
	// A pinned none is a literal the caller named through this model's own
	// setter and is still forwarded verbatim.
	if offDropped && wantEffort == ThinkingEffortNone && !o.isPinned(ThinkingCanSetEffort) {
		wantEffort = ""
	}
	canEffort, canBudget := dims.Has(ThinkingCanSetEffort), dims.Has(ThinkingCanSetBudget)

	if wantBudget == ThinkingBudgetDynamic {
		if canBudget {
			p.dynamic = true
		} else {
			p.note("dynamic thinking dropped: model has no dynamic setting")
		}
		wantBudget = 0
	}

	switch {
	case wantEffort != "" && canEffort:
		if o.isPinned(ThinkingCanSetEffort) {
			// The caller named this model's own setter; forward verbatim.
			p.effort = wantEffort
		} else if e, ok := clampEffort(wantEffort, efforts...); ok {
			if e != wantEffort {
				p.note("effort %s clamped to %s", wantEffort, e)
			}
			p.effort = e
		} else {
			p.note("effort %s dropped: not on this model's ladder", wantEffort)
		}
	case wantEffort != "" && canBudget && !p.dynamic && wantBudget == 0:
		if n := ThinkingBudgetForEffort(wantEffort, br.min, br.max); n > 0 {
			p.budget = n
			p.note("effort %s mapped to budget %d tokens", wantEffort, n)
		} else {
			p.note("effort %s dropped: no budget equivalent", wantEffort)
		}
	case wantEffort != "":
		p.note("effort %s dropped: model takes no depth setting", wantEffort)
	}

	switch {
	case wantBudget > 0 && canBudget:
		n := wantBudget
		if !o.isPinned(ThinkingCanSetBudget) {
			if br.max > 0 && n > br.max {
				n = br.max
			}
			if n < br.min {
				n = br.min
			}
			if n != wantBudget {
				p.note("budget %d clamped to %d", wantBudget, n)
			}
		}
		p.budget = n
	case wantBudget > 0 && canEffort && p.effort == "":
		if e := ThinkingEffortForBudget(wantBudget); e != "" {
			if c, ok := clampEffort(e, efforts...); ok {
				p.effort = c
				p.note("budget %d mapped to effort %s", wantBudget, c)
			} else {
				p.note("budget %d dropped: no effort equivalent", wantBudget)
			}
		}
	case wantBudget > 0:
		p.note("budget %d dropped: model takes no token budget", wantBudget)
	}

	// --- trace --------------------------------------------------------------
	switch o.trace {
	case ThinkingTraceInclude:
		if dims.Has(ThinkingCanHideTrace) {
			p.showTrace = true
		}
	case ThinkingTraceOmit:
		if dims.Has(ThinkingCanHideTrace) {
			p.hideTrace = true
		} else {
			p.note("trace omission dropped: model always returns its trace")
		}
	}
	return p
}
