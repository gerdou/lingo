package lingo

import (
	"fmt"
	"time"

	"github.com/openai/openai-go/v3/option"
)

func init() {
	RegisterProvider(ProviderOpenAICompatible, func(config ProviderConfig, logger Logger) (Provider, error) {
		cfg, ok := config.(*OpenAICompatibleConfig)
		if !ok {
			return nil, fmt.Errorf("invalid config type for OpenAI-compatible provider")
		}
		return newOpenAICompatibleClient(cfg, logger)
	})
}

// ============================================================================
// KNOWN OPENAI-COMPATIBLE ENDPOINTS
// ============================================================================
//
// Any endpoint speaking the OpenAI chat completions dialect works — these are
// only shortcuts for the common ones. Pass any other URL to BaseURL directly.

const (
	// Hosted inference services
	BaseURLGroq        = "https://api.groq.com/openai/v1"
	BaseURLTogether    = "https://api.together.xyz/v1"
	BaseURLFireworks   = "https://api.fireworks.ai/inference/v1"
	BaseURLCerebras    = "https://api.cerebras.ai/v1"
	BaseURLDeepInfra   = "https://api.deepinfra.com/v1/openai"
	BaseURLSambaNova   = "https://api.sambanova.ai/v1"
	BaseURLHyperbolic  = "https://api.hyperbolic.xyz/v1"
	BaseURLNebius      = "https://api.studio.nebius.ai/v1"
	BaseURLNvidiaNIM   = "https://integrate.api.nvidia.com/v1"
	BaseURLHuggingFace = "https://router.huggingface.co/v1"

	// Model vendors with an OpenAI-compatible surface
	BaseURLMistral = "https://api.mistral.ai/v1"
	BaseURLZAI     = "https://api.z.ai/api/paas/v4"

	// Local servers
	BaseURLVLLM      = "http://localhost:8000/v1"
	BaseURLLMStudio  = "http://localhost:1234/v1"
	BaseURLLlamaCPP  = "http://localhost:8080/v1"
	BaseURLLocalAI   = "http://localhost:8080/v1"
	BaseURLOllamaOAI = "http://localhost:11434/v1"
)

// ============================================================================
// OPENAI-COMPATIBLE PROVIDER CONFIG
// ============================================================================

// OpenAICompatibleConfig configures any endpoint that speaks the OpenAI chat
// completions dialect — Groq, Together, Fireworks, Cerebras, DeepInfra,
// SambaNova, vLLM, LM Studio, llama.cpp, LocalAI and others.
//
// Use it when lingo has no dedicated provider for the service. Providers with
// a dedicated implementation (OpenAI, Azure, xAI, DeepSeek, OpenRouter) should
// use theirs instead: they ship typed models and provider-specific options.
type OpenAICompatibleConfig struct {
	// BaseURL is the endpoint root, including the version path (required).
	// Example: lingo.BaseURLGroq, or "http://localhost:8000/v1".
	BaseURL string
	// APIKey is the bearer token. Optional: local servers rarely need one.
	APIKey string
	// Timeout is the request timeout (default: 60s)
	Timeout time.Duration
	// RateLimiter is the optional rate limit configuration
	RateLimiter *RateLimitConfig
	// Headers are extra headers sent with every request
	Headers map[string]string
	// HealthCheckModel is generated against by Health. When empty, Health
	// lists models instead.
	HealthCheckModel string
}

// Implement ProviderConfig interface
func (c *OpenAICompatibleConfig) providerType() ProviderType        { return ProviderOpenAICompatible }
func (c *OpenAICompatibleConfig) apiKey() string                    { return c.APIKey }
func (c *OpenAICompatibleConfig) timeout() time.Duration            { return c.Timeout }
func (c *OpenAICompatibleConfig) rateLimitConfig() *RateLimitConfig { return c.RateLimiter }

// ============================================================================
// MODELS
// ============================================================================

// OpenAICompatibleModel is any model on an OpenAI-compatible endpoint,
// addressed by its raw model ID.
type OpenAICompatibleModel struct {
	oaiOptions
	modelID string
}

func (m *OpenAICompatibleModel) ModelName() string      { return m.modelID }
func (m *OpenAICompatibleModel) Provider() ProviderType { return ProviderOpenAICompatible }

func (m *OpenAICompatibleModel) WithMaxTokens(n int) *OpenAICompatibleModel {
	m.maxTokens = n
	return m
}
func (m *OpenAICompatibleModel) WithMaxCompletionTokens(n int) *OpenAICompatibleModel {
	m.maxCompletionTokens = n
	return m
}
func (m *OpenAICompatibleModel) WithTemperature(t float64) *OpenAICompatibleModel {
	m.temperature = t
	return m
}
func (m *OpenAICompatibleModel) WithTopP(p float64) *OpenAICompatibleModel { m.topP = p; return m }
func (m *OpenAICompatibleModel) WithReasoningEffort(e string) *OpenAICompatibleModel {
	m.reasoningEffort = e
	m.reasoning = true
	return m
}
func (m *OpenAICompatibleModel) WithSystemPrompt(s string) *OpenAICompatibleModel {
	m.systemPrompt = s
	return m
}

// WithExtraField sets a provider-specific field on the request body for
// options this API exposes that the OpenAI schema does not model.
func (m *OpenAICompatibleModel) WithExtraField(key string, value any) *OpenAICompatibleModel {
	m.setExtra(key, value)
	return m
}

// NewOpenAICompatibleModel creates a model addressed by its raw ID, e.g.
// "llama-3.3-70b-versatile" on Groq or "qwen3-coder" on a local vLLM server.
func NewOpenAICompatibleModel(modelID string) *OpenAICompatibleModel {
	return &OpenAICompatibleModel{modelID: modelID}
}

// Compile-time check that the model routes through the shared client
var _ oaiCompatibleModel = (*OpenAICompatibleModel)(nil)

// ============================================================================
// CLIENT
// ============================================================================

// newOpenAICompatibleClient creates a client for a generic OpenAI-compatible endpoint
func newOpenAICompatibleClient(config *OpenAICompatibleConfig, logger Logger) (Provider, error) {
	if config.BaseURL == "" {
		return nil, fmt.Errorf("BaseURL is required for the OpenAI-compatible provider")
	}

	opts := []option.RequestOption{option.WithBaseURL(config.BaseURL)}
	if config.APIKey != "" {
		opts = append(opts, option.WithAPIKey(config.APIKey))
	}
	for k, v := range config.Headers {
		opts = append(opts, option.WithHeader(k, v))
	}

	return newOAICompatClient(
		ProviderOpenAICompatible,
		"OpenAI-compatible endpoint",
		config.HealthCheckModel,
		config.Timeout,
		config.RateLimiter,
		logger,
		opts...,
	), nil
}
