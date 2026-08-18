package lingo

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"time"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/shared"
)

// ============================================================================
// SHARED OPENAI-COMPATIBLE CORE
// ============================================================================
//
// Most providers speak the OpenAI /chat/completions dialect, so xAI, DeepSeek,
// OpenRouter, Azure OpenAI and the generic OpenAI-compatible provider all share
// the client below.
//
// Unlike openai.go, which applies options through a per-model type switch,
// these providers dispatch through the oaiCompatibleModel interface: a new
// model is a struct embedding oaiOptions and needs no edit to the client.

// oaiOptions is the generation option set shared by models served over the
// OpenAI chat completions API. Providers embed it and expose only the setters
// their own API actually honours.
type oaiOptions struct {
	modelVersion        string // Optional: override model name with specific version
	maxTokens           int
	maxCompletionTokens int
	temperature         float64
	topP                float64
	thinking            ThinkingOptions
	systemPrompt        string
	reasoning           bool           // Model reasons before answering; reported in metadata
	extraFields         map[string]any // Provider-specific body fields (e.g. DeepSeek "thinking")
	cache               CacheOptions
}

// SystemPrompt satisfies Model for every type embedding oaiOptions.
func (o *oaiOptions) SystemPrompt() string { return o.systemPrompt }

// CacheOptions satisfies CacheableModel for every type embedding oaiOptions.
// What actually reaches the wire depends on the endpoint's oaiCacheCaps.
func (o *oaiOptions) CacheOptions() *CacheOptions { return &o.cache }

// ThinkingOptions satisfies ThinkingModel for every type embedding oaiOptions,
// which is every Azure, xAI, DeepSeek, OpenRouter and generic OpenAI-compatible
// model in one declaration. What actually reaches the wire depends on the
// endpoint's oaiThinkingCaps and on the model's own thinkingDimensions.
//
// It is the single storage behind WithReasoningEffort, WithThinkingEnabled,
// WithThinkingDisabled, WithReasoningMaxTokens and WithReasoningExcluded, so
// the portable surface and the per-model setters can never disagree.
func (o *oaiOptions) ThinkingOptions() *ThinkingOptions { return &o.thinking }

// setReasoningEffort backs the per-model WithReasoningEffort setters.
//
// It writes the level and pins it, so a caller's own string reaches the wire
// byte for byte: pinned means never clamped to the endpoint's ladder, never
// translated, never dropped for being a value this library has not heard of.
// These setters have always taken a bare string and forwarded it unexamined --
// xAI's "none", DeepSeek's "max", an endpoint-specific word lingo has never
// seen -- and that is preserved exactly.
//
// It leaves the mode alone. On these endpoints the effort is a value, not a
// switch: reading WithReasoningEffort("none") as ThinkingModeOff would send the
// portable surface's idea of off instead of the literal the caller named.
func (o *oaiOptions) setReasoningEffort(e string) {
	o.thinking.effort = ThinkingEffort(e)
	o.thinking.pin(ThinkingCanSetEffort)
}

// reasoningModel reports what Metadata["is_reasoning_model"] says: the flag a
// constructor or setter recorded, or thinking asked for through the portable
// surface, so the metadata cannot contradict the request lingo just built.
//
// It takes the plan because the constructor flag alone contradicts it. Every
// DeepSeek V4 model is built with reasoning true, and a request that carries
// thinking={"type":"disabled"} is not a reasoning request however the model was
// built or which surface switched it off. A plan that switched thinking off is
// therefore the last word, whatever the other two say.
func (o *oaiOptions) reasoningModel(plan thinkingPlan) bool {
	// Off reaches the wire two ways: as a toggle, and as the "none" rung of an
	// effort ladder. Both mean this request does not reason, so both answer
	// false -- otherwise the portable NoThinking would disagree with the
	// per-model WithReasoningEffort(XAIEffortNone), which has always reported
	// false for the identical body.
	if plan.disable || plan.effort == ThinkingEffortNone {
		return false
	}
	return o.reasoning || o.thinking.Enabled()
}

// chatOptions exposes the embedded option set to the shared client.
func (o *oaiOptions) chatOptions() *oaiOptions { return o }

// setExtra records a provider-specific field to merge into the request body.
func (o *oaiOptions) setExtra(key string, value any) {
	if o.extraFields == nil {
		o.extraFields = make(map[string]any)
	}
	o.extraFields[key] = value
}

// oaiCompatibleModel is implemented by every model routed through the shared
// OpenAI-compatible client.
type oaiCompatibleModel interface {
	Model
	chatOptions() *oaiOptions
}

// oaiThinkingModel is the per-model half of thinking capability on the shared
// client. Unlike caching, thinking support varies per model inside a single
// endpoint -- grok-4.3 takes an effort and grok-4.20-non-reasoning takes none --
// so the model answers for itself and the endpoint's caps say only which body
// shape can carry the answer.
//
// It is declared in the same idiom as openAIExplicitCacheModel: the value is
// consulted, not merely the type, so the raw-id escape hatches (XAIModel,
// DeepSeekModel, OpenRouterModel, OpenAICompatibleModel) resolve their own
// capabilities rather than inheriting a named model's.
type oaiThinkingModel interface {
	// thinkingDimensions reports which knobs this model honours.
	thinkingDimensions() ThinkingDimension
	// thinkingEfforts is the effort ladder it accepts. A nil ladder means the
	// portable surface has nothing to clamp to and drops the field; a value a
	// per-model setter pinned is still forwarded.
	thinkingEfforts() []ThinkingEffort
}

// oaiModelThinking resolves one model's thinking capabilities, answering "no
// knobs at all" for a model that declares none.
func oaiModelThinking(model Model) (ThinkingDimension, []ThinkingEffort) {
	m, ok := model.(oaiThinkingModel)
	if !ok {
		return 0, nil
	}
	return m.thinkingDimensions(), m.thinkingEfforts()
}

// oaiPinnedThinking reports which dimensions a per-model setter pinned, nil-safe
// for the models that carry no thinking configuration.
func oaiPinnedThinking(o *ThinkingOptions) ThinkingDimension {
	if o == nil {
		return 0
	}
	return o.pinned
}

// resolveModelName returns the version override when one is set.
func resolveModelName(o *oaiOptions, defaultName string) string {
	if o.modelVersion != "" {
		return o.modelVersion
	}
	return defaultName
}

// ============================================================================
// SHARED CLIENT
// ============================================================================

// oaiCacheCaps records which prompt caching fields an endpoint is known to
// accept. Every capability defaults to false, so a provider that declares
// nothing keeps sending the exact request body it sent before caching existed.
type oaiCacheCaps struct {
	// promptCacheKey allows the OpenAI-dialect prompt_cache_key field
	promptCacheKey bool
	// contentCacheControl allows Anthropic-dialect cache_control markers on
	// message content parts, which OpenRouter forwards to Anthropic upstreams
	contentCacheControl bool
}

// oaiThinkingCaps records which shape an endpoint's thinking instruction takes.
// Every capability defaults to false, so a provider that declares nothing keeps
// sending the exact request body it sent before thinking existed.
//
// The three shapes are alternatives in the sense that they are different fields,
// not different spellings of one: DeepSeek genuinely takes both the flat effort
// and its own thinking object, while OpenRouter folds the effort into its
// reasoning object and must never also send the flat one.
type oaiThinkingCaps struct {
	// flatEffort allows the OpenAI-dialect top-level reasoning_effort field
	flatEffort bool
	// reasoningObject builds OpenRouter's reasoning{effort,max_tokens,exclude,
	// enabled} object, the one request shape in this family that models every
	// dimension lingo has
	reasoningObject bool
	// thinkingObject builds DeepSeek's thinking{type} toggle
	thinkingObject bool
	// budget is the reasoning-token window this endpoint's models accept, used
	// to clamp an unpinned budget. A zero max means there is no budget field.
	budget budgetRange
}

// dimensions reports which knobs the endpoint's body shape can carry at all.
// A model may declare more than its endpoint can express -- an OpenRouter model
// id pointing at Claude models a token budget the flat dialect has no field for
// -- and this is where that is resolved.
func (c oaiThinkingCaps) dimensions() ThinkingDimension {
	var d ThinkingDimension
	if c.flatEffort {
		d |= ThinkingCanSetEffort
	}
	if c.reasoningObject {
		d |= ThinkingCanToggle | ThinkingCanSetEffort | ThinkingCanSetBudget | ThinkingCanHideTrace
	}
	if c.thinkingObject {
		d |= ThinkingCanToggle
	}
	// Reporting is unconditional on every endpoint in this family: the usage
	// breakdown and the out-of-band trace are read whether or not the caller
	// asked for anything.
	return d | ThinkingCanReportTokens | ThinkingCanReportTrace
}

// oaiCompatClient implements Provider for any endpoint speaking the OpenAI
// chat completions dialect.
type oaiCompatClient struct {
	client   openai.Client
	provider ProviderType
	// label is the human-readable provider name used in errors and logs
	label string
	// healthModel is generated against by Health; when empty Health lists
	// models instead, which every compliant endpoint supports
	healthModel string
	// cacheCaps limits which cache fields Generate may put on the wire
	cacheCaps oaiCacheCaps
	// thinkingCaps limits which thinking fields Generate may put on the wire
	thinkingCaps oaiThinkingCaps
	timeout      time.Duration
	logger       Logger
	rateLimiter  *rateLimiter
}

// newOAICompatClient builds a shared client from a base set of request options.
func newOAICompatClient(provider ProviderType, label string, healthModel string, timeout time.Duration, rl *RateLimitConfig, logger Logger, caps oaiCacheCaps, thinkCaps oaiThinkingCaps, opts ...option.RequestOption) *oaiCompatClient {
	if timeout == 0 {
		timeout = defaultTimeout()
	}
	return &oaiCompatClient{
		client:       openai.NewClient(opts...),
		provider:     provider,
		label:        label,
		healthModel:  healthModel,
		cacheCaps:    caps,
		thinkingCaps: thinkCaps,
		timeout:      timeout,
		logger:       logger,
		rateLimiter:  newRateLimiter(rl, logger),
	}
}

// Generate generates text over the OpenAI chat completions API
func (c *oaiCompatClient) Generate(ctx context.Context, model Model, prompt string) (*GenerationResponse, error) {
	// Verify model belongs to this provider
	if model.Provider() != c.provider {
		return nil, fmt.Errorf("model %s is not a %s model", model.ModelName(), c.label)
	}

	m, ok := model.(oaiCompatibleModel)
	if !ok {
		return nil, fmt.Errorf("model %s does not carry %s generation options", model.ModelName(), c.label)
	}
	opts := m.chatOptions()

	// Set timeout
	ctx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()

	// Caching is opt-in and capability-gated: with either half missing the
	// messages below stay in the plain string form they have always had.
	// cacheBreakpoint records the marker that was placed, which is not the same
	// as the one that was asked for -- a model with no system prompt gets none.
	co := modelCacheOptions(model)
	breakpoints := c.cacheCaps.contentCacheControl
	var cacheBreakpoint bool

	// Build messages with optional system prompt. Third-party endpoints
	// accept "system" universally; "developer" is OpenAI-only.
	var messages []openai.ChatCompletionMessageParamUnion
	if s := model.SystemPrompt(); s != "" {
		if breakpoints && co.SystemPromptCached() {
			messages = append(messages, openai.SystemMessage([]openai.ChatCompletionContentPartTextParam{oaiCachedTextPart(s, co)}))
			cacheBreakpoint = true
		} else {
			messages = append(messages, openai.SystemMessage(s))
		}
	}
	if breakpoints && co.PromptCached() {
		part := oaiCachedTextPart(prompt, co)
		messages = append(messages, openai.UserMessage([]openai.ChatCompletionContentPartUnionParam{{OfText: &part}}))
		cacheBreakpoint = true
	} else {
		messages = append(messages, openai.UserMessage(prompt))
	}

	params := openai.ChatCompletionNewParams{
		Model:    openai.ChatModel(model.ModelName()),
		Messages: messages,
	}
	if opts.maxTokens > 0 {
		params.MaxTokens = openai.Int(int64(opts.maxTokens))
	}
	if opts.maxCompletionTokens > 0 {
		params.MaxCompletionTokens = openai.Int(int64(opts.maxCompletionTokens))
	}
	if opts.temperature > 0 {
		params.Temperature = openai.Float(opts.temperature)
	}
	if opts.topP > 0 {
		params.TopP = openai.Float(opts.topP)
	}
	// A cache key partitions the provider's cache; it does not enable caching,
	// so it is sent whenever one is set and not explicitly suppressed
	if key := co.Key(); key != "" && c.cacheCaps.promptCacheKey && !co.Disabled() {
		params.PromptCacheKey = openai.String(key)
	}

	// Thinking is opt-in and doubly gated: the model says which knobs it
	// honours and the endpoint's caps say which body shape can carry them, so
	// a knob neither half models is dropped rather than sent to be rejected.
	//
	// A dimension a per-model setter pinned is added back, whatever the model
	// declares. The caller reached for that endpoint's own setter -- including
	// the effort NewGrok43 has always seeded -- so lingo forwards it and lets
	// the API answer, exactly as it did before the portable surface existed.
	//
	// A model whose ThinkingOptions were never touched produces a zero plan and
	// leaves both params and the body fields below exactly as built above.
	modelDims, efforts := oaiModelThinking(model)
	pinnedThinking := oaiPinnedThinking(&opts.thinking)
	dims := (modelDims | pinnedThinking) & c.thinkingCaps.dimensions()
	plan := planThinking(&opts.thinking, dims, c.thinkingCaps.budget, efforts...)

	// A pinned effort survives a disable when the off switch is a field of its
	// own, because there the two are independent: DeepSeek's thinking object
	// says whether to reason and reasoning_effort says how hard, and lingo has
	// always sent both when both were asked for. The plan drops the depth
	// because for most providers depth is meaningless once thinking is off,
	// which is true where the effort IS the off switch (OpenAI's dialect, where
	// off is the rung "none") and where one object carries both (OpenRouter's
	// reasoning), but not here.
	if plan.disable && plan.effort == "" && c.thinkingCaps.thinkingObject &&
		pinnedThinking.Has(ThinkingCanSetEffort) {
		plan.effort = opts.thinking.Effort()
	}

	// The flat field and the reasoning object are two spellings of one knob;
	// sending both would be an undocumented combination, so an endpoint that
	// takes the object never also takes the flat field.
	if c.thinkingCaps.flatEffort && plan.effort != "" {
		params.ReasoningEffort = shared.ReasoningEffort(plan.effort)
	}

	// Provider-specific body fields the typed params don't model. lingo's
	// derived thinking fields go in first and the caller's own on top, so an
	// explicit WithExtraField always wins over a field lingo inferred.
	body := c.thinkingBody(plan, &opts.thinking)
	for k, v := range opts.extraFields {
		body[k] = v
	}
	var reqOpts []option.RequestOption
	for k, v := range body {
		reqOpts = append(reqOpts, option.WithJSONSet(k, v))
	}

	c.logger.Debug().
		Str("provider", string(c.provider)).
		Str("model", model.ModelName()).
		Bool("is_reasoning_model", opts.reasoningModel(plan)).
		Str("reasoning_effort", string(plan.effort)).
		Str("thinking_translation", plan.translation()).
		Bool("cache_breakpoint", cacheBreakpoint).
		Msg("Making " + c.label + " API request")

	// Make request with rate limit handling
	var resp *openai.ChatCompletion
	err := c.rateLimiter.Execute(ctx, func() error {
		var reqErr error
		resp, reqErr = c.client.Chat.Completions.New(ctx, params, reqOpts...)
		return reqErr
	})
	if err != nil {
		c.logger.Error().
			Err(err).
			Str("provider", string(c.provider)).
			Str("model", model.ModelName()).
			Str("prompt_preview", truncateString(prompt, 100)).
			Msg(c.label + " generation failed")
		return nil, fmt.Errorf("%s generation failed: %w", c.label, err)
	}

	if len(resp.Choices) == 0 {
		return nil, fmt.Errorf("no response choices returned from %s", c.label)
	}

	choice := resp.Choices[0]

	// Model echoed by the API can differ from the one requested (OpenRouter
	// routes to a backing provider, Azure echoes the deployment name)
	modelName := resp.Model
	if modelName == "" {
		modelName = model.ModelName()
	}

	// Cache accounting is reported whether or not the caller opted in. In this
	// dialect cached tokens are a breakdown of prompt_tokens, never additive.
	cacheRead := int(resp.Usage.PromptTokensDetails.CachedTokens)
	cacheWrite := int(resp.Usage.PromptTokensDetails.CacheWriteTokens)
	if cacheRead == 0 {
		// DeepSeek reports hits under its own key instead of the OpenAI one
		cacheRead = oaiExtraUsageTokens(resp.Usage, "prompt_cache_hit_tokens")
	}

	response := &GenerationResponse{
		Text: choice.Message.Content,
		// Reasoning traces are returned out of band by most of these providers,
		// under a field name the OpenAI schema does not model. Reporting is
		// unconditional: the trace and its token count are surfaced whether or
		// not the caller opted in.
		Thinking:     extractReasoningContent(choice.Message),
		Model:        modelName,
		FinishReason: string(choice.FinishReason),
		// In this dialect both breakdowns are subsets of the total they belong
		// to: cached tokens are part of prompt_tokens and reasoning tokens are
		// part of completion_tokens, so neither is folded back in.
		Usage: TokenUsage{
			PromptTokens:     int(resp.Usage.PromptTokens),
			CompletionTokens: int(resp.Usage.CompletionTokens),
			TotalTokens:      int(resp.Usage.TotalTokens),
		}.withCache(cacheRead, cacheWrite, true).
			withThinking(int(resp.Usage.CompletionTokensDetails.ReasoningTokens), true),
		Metadata: map[string]string{
			"provider":           string(c.provider),
			"model":              modelName,
			"is_reasoning_model": fmt.Sprintf("%t", opts.reasoningModel(plan)),
		},
	}

	// Requested model is worth keeping when the API rewrote it
	if modelName != model.ModelName() {
		response.Metadata["requested_model"] = model.ModelName()
	}

	// The count and the trace now have typed homes in TokenUsage.ThinkingTokens
	// and GenerationResponse.Thinking; the metadata keys they used to live under
	// are kept for one release so existing readers keep working. Note the wire
	// spelling differs from lingo's: these dialects call them reasoning tokens
	// and reasoning content, lingo calls the concept thinking.
	//
	// Deprecated: read Usage.ThinkingTokens instead of Metadata["reasoning_tokens"],
	// and Thinking instead of Metadata["reasoning_content"].
	if resp.Usage.CompletionTokensDetails.ReasoningTokens > 0 {
		response.Metadata["reasoning_tokens"] = fmt.Sprintf("%d", resp.Usage.CompletionTokensDetails.ReasoningTokens)
	}
	if response.Thinking != "" {
		response.Metadata["reasoning_content"] = response.Thinking
	}

	// Whatever lingo had to translate or drop to fit the caller's request onto
	// this model and this endpoint, so a silent adaptation is never invisible.
	if s := plan.translation(); s != "" {
		response.Metadata["thinking_translation"] = s
	}
	// DeepSeek splits the prompt into hit and miss; the miss half has no home
	// in TokenUsage, which models cached tokens rather than uncached ones
	if miss := oaiExtraUsageTokens(resp.Usage, "prompt_cache_miss_tokens"); miss > 0 {
		response.Metadata["prompt_cache_miss_tokens"] = fmt.Sprintf("%d", miss)
	}

	c.logger.Debug().
		Str("provider", string(c.provider)).
		Str("model", modelName).
		Bool("is_reasoning_model", opts.reasoningModel(plan)).
		Int64("prompt_tokens", resp.Usage.PromptTokens).
		Int64("completion_tokens", resp.Usage.CompletionTokens).
		Int64("total_tokens", resp.Usage.TotalTokens).
		Int("cache_read_tokens", response.Usage.CacheReadTokens).
		Int("cache_write_tokens", response.Usage.CacheWriteTokens).
		Int("thinking_tokens", response.Usage.ThinkingTokens).
		Msg(c.label + " generation completed")

	return response, nil
}

// thinkingBody builds the body fields the typed chat-completions params cannot
// carry: DeepSeek's thinking object and OpenRouter's reasoning object. It
// returns an empty map for a zero plan and for every endpoint whose whole
// thinking surface is the flat reasoning_effort.
//
// to is the model's own thinking storage, consulted only to honour a pin the
// plan could not carry; it may be nil.
func (c *oaiCompatClient) thinkingBody(plan thinkingPlan, to *ThinkingOptions) map[string]any {
	body := map[string]any{}

	// DeepSeek reasons by default on every V4 model, so the object exists to
	// say "stop" as often as to say "start".
	if c.thinkingCaps.thinkingObject {
		switch {
		case plan.disable:
			body["thinking"] = thinkingDisabled()
		case plan.enable:
			body["thinking"] = thinkingEnabled()
		}
	}

	// OpenRouter normalizes one object across every upstream vendor, so unlike
	// everywhere else lingo does not choose between an effort and a budget: it
	// passes through whichever the caller expressed and lets OpenRouter map it
	// onto whatever the model behind the id actually speaks.
	if c.thinkingCaps.reasoningObject {
		reasoning := map[string]any{}

		// A dimension one of this model's own setters pinned is read back off the
		// options rather than off the plan, because the plan's fields carry the
		// portable surface's sentinels: "" is an absent effort and 0 is an absent
		// budget, so a caller who named reasoning.effort="" or
		// reasoning.max_tokens=0 has no way to say so through them. These setters
		// have written whatever they were given straight into this object since
		// before lingo modelled thinking, and OpenRouter is the one that gets to
		// answer for it.
		//
		// A disable is not overridden this way: there the plan is not failing to
		// carry the pin, it is deliberately replacing it, and sending a depth
		// beside enabled:false would contradict the request.
		verbatim := !plan.disable
		var depth bool
		switch {
		case verbatim && to.isPinned(ThinkingCanSetEffort):
			reasoning["effort"] = string(to.Effort())
			depth = true
		case plan.effort != "":
			reasoning["effort"] = string(plan.effort)
			depth = true
		}
		switch {
		case verbatim && to.isPinned(ThinkingCanSetBudget):
			reasoning["max_tokens"] = to.Budget()
			depth = true
		case plan.budget > 0:
			reasoning["max_tokens"] = plan.budget
			depth = true
		}
		if plan.hideTrace {
			reasoning["exclude"] = true
		}
		// enabled is redundant once a depth is named -- OpenRouter infers it
		// from effort or max_tokens -- so it is sent only when thinking was
		// asked for without one, and a disable is spelled as effort none.
		switch {
		case plan.enable && !depth:
			reasoning["enabled"] = true
		case plan.disable:
			reasoning["enabled"] = false
		}
		if len(reasoning) > 0 {
			body["reasoning"] = reasoning
		}
	}
	return body
}

// oaiCachedTextPart builds a text content part carrying an Anthropic-dialect
// cache breakpoint. The marker is not part of the OpenAI schema, so it rides
// along as an extra field, which is exactly how OpenRouter expects it. The TTL
// is passed through only when the caller asked for one: OpenRouter's accepted
// vocabulary is upstream-dependent and undocumented in any SDK.
func oaiCachedTextPart(text string, co *CacheOptions) openai.ChatCompletionContentPartTextParam {
	control := map[string]any{"type": "ephemeral"}
	if ttl := co.TTL(); ttl != CacheTTLDefault {
		control["ttl"] = string(ttl)
	}
	part := openai.ChatCompletionContentPartTextParam{Text: text}
	part.SetExtraFields(map[string]any{"cache_control": control})
	return part
}

// oaiExtraUsageTokens reads a token counter the OpenAI usage schema does not
// model, such as DeepSeek's prompt_cache_hit_tokens. Absent or unparseable
// values report zero rather than failing an otherwise successful generation.
func oaiExtraUsageTokens(usage openai.CompletionUsage, key string) int {
	// Extra fields never reach respjson's valid state, so Raw is the only test
	raw := usage.JSON.ExtraFields[key].Raw()
	if raw == "" {
		return 0
	}
	n, err := strconv.Atoi(raw)
	if err != nil || n < 0 {
		return 0
	}
	return n
}

// extractReasoningContent pulls the non-standard reasoning field that DeepSeek
// and several OpenRouter backends return alongside the answer.
func extractReasoningContent(msg openai.ChatCompletionMessage) string {
	for _, key := range []string{"reasoning_content", "reasoning"} {
		raw, ok := msg.JSON.ExtraFields[key]
		if !ok {
			continue
		}

		value := raw.Raw()
		if value == "" || value == "null" {
			continue
		}

		// The field arrives as raw JSON; unwrap the common string case so
		// callers get the text rather than a quoted literal
		var text string
		if err := json.Unmarshal([]byte(value), &text); err == nil {
			if text == "" {
				continue
			}
			return text
		}
		return value
	}
	return ""
}

// Health checks the health of the endpoint
func (c *oaiCompatClient) Health(ctx context.Context) error {
	ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	// Without a known-good model, listing models is the cheapest call that
	// still proves credentials and connectivity
	if c.healthModel == "" {
		if _, err := c.client.Models.List(ctx); err != nil {
			return fmt.Errorf("%s health check failed: %w", c.label, err)
		}
		return nil
	}

	params := openai.ChatCompletionNewParams{
		Model:     openai.ChatModel(c.healthModel),
		Messages:  []openai.ChatCompletionMessageParamUnion{openai.UserMessage("Hello")},
		MaxTokens: openai.Int(5),
	}

	if _, err := c.client.Chat.Completions.New(ctx, params); err != nil {
		return fmt.Errorf("%s health check failed: %w", c.label, err)
	}

	return nil
}

// Close closes the client (no-op for HTTP-backed providers)
func (c *oaiCompatClient) Close() error {
	return nil
}
