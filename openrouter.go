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
	// reasoning and provider are OpenRouter's own request objects, built up
	// across setter calls and merged into the body as a whole
	reasoningOpts map[string]any
	providerOpts  map[string]any
}

func (m *OpenRouterModel) ModelName() string      { return m.modelID }
func (m *OpenRouterModel) Provider() ProviderType { return ProviderOpenRouter }

func (m *OpenRouterModel) WithMaxTokens(n int) *OpenRouterModel       { m.maxTokens = n; return m }
func (m *OpenRouterModel) WithTemperature(t float64) *OpenRouterModel { m.temperature = t; return m }
func (m *OpenRouterModel) WithTopP(p float64) *OpenRouterModel        { m.topP = p; return m }
func (m *OpenRouterModel) WithSystemPrompt(s string) *OpenRouterModel { m.systemPrompt = s; return m }
func (m *OpenRouterModel) WithExtraField(k string, v any) *OpenRouterModel {
	m.setExtra(k, v)
	return m
}

// setReasoning merges a key into the "reasoning" request object
func (m *OpenRouterModel) setReasoning(key string, value any) {
	if m.reasoningOpts == nil {
		m.reasoningOpts = make(map[string]any)
	}
	m.reasoningOpts[key] = value
	m.setExtra("reasoning", m.reasoningOpts)
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
func (m *OpenRouterModel) WithReasoningEffort(e string) *OpenRouterModel {
	m.setReasoning("effort", e)
	m.reasoning = e != "none"
	return m
}

// WithReasoningMaxTokens caps reasoning.max_tokens for models that budget
// thinking in tokens rather than effort levels.
func (m *OpenRouterModel) WithReasoningMaxTokens(n int) *OpenRouterModel {
	m.setReasoning("max_tokens", n)
	m.reasoning = true
	return m
}

// WithReasoningExcluded keeps reasoning on but drops the trace from the response
func (m *OpenRouterModel) WithReasoningExcluded() *OpenRouterModel {
	m.setReasoning("exclude", true)
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
		opts...,
	), nil
}
