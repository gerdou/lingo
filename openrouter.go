package lingo

import (
	"fmt"
	"time"

	"github.com/openai/openai-go/v3/option"
)

func init() {
	RegisterProvider(ProviderOpenRouter, func(config ProviderConfig, logger Logger) (Provider, error) {
		cfg, ok := config.(*OpenRouterConfig)
		if !ok {
			return nil, fmt.Errorf("invalid config type for OpenRouter provider")
		}
		return newOpenRouterClient(cfg, logger)
	})
}

// BaseURLOpenRouter is the OpenRouter API base URL
const BaseURLOpenRouter = "https://openrouter.ai/api/v1"

// ============================================================================
// OPENROUTER PROVIDER CONFIG
// ============================================================================

// OpenRouterConfig contains configuration for the OpenRouter provider.
//
// OpenRouter is an aggregator: one key reaches hundreds of models from many
// vendors. Its catalogue changes constantly, so lingo exposes models by ID
// rather than as typed structs — use NewOpenRouterModel("vendor/model").
type OpenRouterConfig struct {
	// APIKey is the OpenRouter API key (required)
	APIKey string
	// Timeout is the request timeout (default: 60s)
	Timeout time.Duration
	// RateLimiter is the optional rate limit configuration
	RateLimiter *RateLimitConfig
	// BaseURL is an optional custom base URL (default: https://openrouter.ai/api/v1)
	BaseURL string
	// SiteURL sets the HTTP-Referer header, used for openrouter.ai rankings
	SiteURL string
	// AppName sets the X-OpenRouter-Title header, used for openrouter.ai rankings
	AppName string
	// Headers are extra headers sent with every request
	Headers map[string]string
}

// Implement ProviderConfig interface
func (c *OpenRouterConfig) providerType() ProviderType        { return ProviderOpenRouter }
func (c *OpenRouterConfig) apiKey() string                    { return c.APIKey }
func (c *OpenRouterConfig) timeout() time.Duration            { return c.Timeout }
func (c *OpenRouterConfig) rateLimitConfig() *RateLimitConfig { return c.RateLimiter }

// ============================================================================
// MODELS
// ============================================================================

// OpenRouterModel is a model addressed by its OpenRouter ID, in the form
// "vendor/model" — for example "anthropic/claude-opus-5" or "deepseek/deepseek-v4-pro".
type OpenRouterModel struct {
	oaiOptions
	modelID string
	// provider is OpenRouter's own routing object, built up across setter calls
	// and merged into the body as a whole. The reasoning object next to it in
	// the body is no longer built here: it is derived at request time from the
	// model's ThinkingOptions, so the portable surface and the three reasoning
	// setters share one storage and a caller's own WithExtraField("reasoning",
	// ...) always wins over what lingo derived.
	providerOpts map[string]any
}

func (m *OpenRouterModel) ModelName() string      { return m.modelID }
func (m *OpenRouterModel) Provider() ProviderType { return ProviderOpenRouter }

// OpenRouter is the one endpoint whose native request shape is already the
// portable one: a single reasoning object carrying a toggle, an effort, a token
// budget and a trace switch, which OpenRouter normalizes onto whatever the
// model behind the id actually speaks. So every dimension lingo has is real
// here, and none of it has to be translated away.
func (m *OpenRouterModel) thinkingDimensions() ThinkingDimension {
	return ThinkingCanToggle | ThinkingCanSetEffort | ThinkingCanSetBudget |
		ThinkingCanHideTrace | ThinkingCanReportTokens | ThinkingCanReportTrace
}

// thinkingEfforts is OpenRouter's normalized ladder, the widest in the library.
func (m *OpenRouterModel) thinkingEfforts() []ThinkingEffort {
	return []ThinkingEffort{
		ThinkingEffortNone, ThinkingEffortMinimal, ThinkingEffortLow,
		ThinkingEffortMedium, ThinkingEffortHigh, ThinkingEffortXHigh, ThinkingEffortMax,
	}
}

func (m *OpenRouterModel) WithMaxTokens(n int) *OpenRouterModel       { m.maxTokens = n; return m }
func (m *OpenRouterModel) WithTemperature(t float64) *OpenRouterModel { m.temperature = t; return m }
func (m *OpenRouterModel) WithTopP(p float64) *OpenRouterModel        { m.topP = p; return m }
func (m *OpenRouterModel) WithSystemPrompt(s string) *OpenRouterModel { m.systemPrompt = s; return m }
func (m *OpenRouterModel) WithExtraField(k string, v any) *OpenRouterModel {
	m.setExtra(k, v)
	return m
}

// setProvider merges a key into the "provider" routing object
func (m *OpenRouterModel) setProvider(key string, value any) {
	if m.providerOpts == nil {
		m.providerOpts = make(map[string]any)
	}
	m.providerOpts[key] = value
	m.setExtra("provider", m.providerOpts)
}

// WithReasoningEffort sets reasoning.effort. OpenRouter normalises this
// across vendors and accepts none, minimal, low, medium, high and xhigh.
//
// The value is pinned, so it reaches reasoning.effort exactly as given rather
// than being clamped to the ladder above.
func (m *OpenRouterModel) WithReasoningEffort(e string) *OpenRouterModel {
	m.setReasoningEffort(e)
	m.reasoning = e != "none"
	return m
}

// WithReasoningMaxTokens caps reasoning.max_tokens for models that budget
// thinking in tokens rather than effort levels.
//
// The value is pinned, so it lands in reasoning.max_tokens exactly as given:
// never clamped into the 1024-128000 window OpenRouter documents for its
// Anthropic upstreams, and never read as one of the portable surface's
// sentinels. This setter has always written its argument into the reasoning
// object unexamined -- 0, -1 and -4096 included -- and OpenRouter, not lingo,
// is the one that gets to reject them.
func (m *OpenRouterModel) WithReasoningMaxTokens(n int) *OpenRouterModel {
	m.thinking.setBudgetVerbatim(n)
	m.reasoning = true
	return m
}

// WithReasoningExcluded keeps reasoning on but drops the trace from the response
func (m *OpenRouterModel) WithReasoningExcluded() *OpenRouterModel {
	m.thinking.WithTrace(ThinkingTraceOmit).pin(ThinkingCanHideTrace)
	m.reasoning = true
	return m
}

// WithFallbackModels sets the "models" array. OpenRouter tries them in order
// when the primary model is unavailable.
func (m *OpenRouterModel) WithFallbackModels(models []string) *OpenRouterModel {
	m.setExtra("models", models)
	return m
}

// WithProviderOrder sets provider.order, the upstream providers to try in order
func (m *OpenRouterModel) WithProviderOrder(providers []string) *OpenRouterModel {
	m.setProvider("order", providers)
	return m
}

// WithProviderOnly restricts routing to provider.only
func (m *OpenRouterModel) WithProviderOnly(providers []string) *OpenRouterModel {
	m.setProvider("only", providers)
	return m
}

// WithProviderIgnore excludes the providers in provider.ignore
func (m *OpenRouterModel) WithProviderIgnore(providers []string) *OpenRouterModel {
	m.setProvider("ignore", providers)
	return m
}

// WithAllowFallbacks sets provider.allow_fallbacks. Set false to fail rather
// than silently route to a provider outside the requested order.
func (m *OpenRouterModel) WithAllowFallbacks(allow bool) *OpenRouterModel {
	m.setProvider("allow_fallbacks", allow)
	return m
}

// WithProviderSort sets provider.sort, e.g. "price", "throughput" or "latency"
func (m *OpenRouterModel) WithProviderSort(sort string) *OpenRouterModel {
	m.setProvider("sort", sort)
	return m
}

// WithDataCollection sets provider.data_collection to "allow" or "deny".
// "deny" routes only to providers that do not store prompts.
func (m *OpenRouterModel) WithDataCollection(policy string) *OpenRouterModel {
	m.setProvider("data_collection", policy)
	return m
}

// WithProviderPreferences replaces the whole provider routing object, for
// fields the typed setters above do not cover.
func (m *OpenRouterModel) WithProviderPreferences(prefs map[string]any) *OpenRouterModel {
	m.providerOpts = prefs
	m.setExtra("provider", prefs)
	return m
}

// WithTransforms sets the "transforms" array, e.g. []string{"middle-out"}
// to compress prompts that exceed the model's context window.
func (m *OpenRouterModel) WithTransforms(transforms []string) *OpenRouterModel {
	m.setExtra("transforms", transforms)
	return m
}

// NewOpenRouterModel creates a model by OpenRouter ID, e.g. "openai/gpt-5.6-sol".
// Append ":free" or a provider suffix exactly as OpenRouter documents it.
func NewOpenRouterModel(modelID string) *OpenRouterModel {
	return &OpenRouterModel{modelID: modelID}
}

// Compile-time check that the model routes through the shared client
var _ oaiCompatibleModel = (*OpenRouterModel)(nil)

// ============================================================================
// CLIENT
// ============================================================================

// newOpenRouterClient creates a new OpenRouter client
func newOpenRouterClient(config *OpenRouterConfig, logger Logger) (Provider, error) {
	if config.APIKey == "" {
		return nil, fmt.Errorf("OpenRouter API key is required")
	}

	baseURL := config.BaseURL
	if baseURL == "" {
		baseURL = BaseURLOpenRouter
	}

	opts := []option.RequestOption{
		option.WithAPIKey(config.APIKey),
		option.WithBaseURL(baseURL),
	}
	if config.SiteURL != "" {
		opts = append(opts, option.WithHeader("HTTP-Referer", config.SiteURL))
	}
	if config.AppName != "" {
		opts = append(opts, option.WithHeader("X-OpenRouter-Title", config.AppName))
	}
	for k, v := range config.Headers {
		opts = append(opts, option.WithHeader(k, v))
	}

	return newOAICompatClient(
		ProviderOpenRouter,
		"OpenRouter",
		"", // Health lists models rather than paying for a generation
		config.Timeout,
		config.RateLimiter,
		logger,
		// OpenRouter forwards Anthropic-dialect breakpoints to upstreams that
		// need them, and accepts prompt_cache_key for the OpenAI-shaped ones
		oaiCacheCaps{promptCacheKey: true, contentCacheControl: true},
		// The provider that gets every flag on: its reasoning object carries
		// the effort, the budget and the trace switch in one place, so the flat
		// OpenAI-dialect field must stay off to avoid sending both spellings.
		// The window is the one OpenRouter documents for Anthropic upstreams,
		// the narrowest of the vendors it normalizes.
		oaiThinkingCaps{reasoningObject: true, budget: budgetRange{min: 1024, max: 128000}},
		opts...,
	), nil
}
