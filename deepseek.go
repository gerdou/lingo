package lingo

import (
	"fmt"
	"time"

	"github.com/openai/openai-go/v3/option"
)

func init() {
	RegisterProvider(ProviderDeepSeek, func(config ProviderConfig, logger Logger) (Provider, error) {
		cfg, ok := config.(*DeepSeekConfig)
		if !ok {
			return nil, fmt.Errorf("invalid config type for DeepSeek provider")
		}
		return newDeepSeekClient(cfg, logger)
	})
}

// BaseURLDeepSeek is the DeepSeek OpenAI-compatible API base URL
const BaseURLDeepSeek = "https://api.deepseek.com/v1"

// ============================================================================
// DEEPSEEK PROVIDER CONFIG
// ============================================================================

// DeepSeekConfig contains configuration for the DeepSeek provider
type DeepSeekConfig struct {
	// APIKey is the DeepSeek API key (required)
	APIKey string
	// Timeout is the request timeout (default: 60s). DeepSeek's thinking
	// modes can run long; raise this for deep reasoning workloads.
	Timeout time.Duration
	// RateLimiter is the optional rate limit configuration
	RateLimiter *RateLimitConfig
	// BaseURL is an optional custom base URL (default: https://api.deepseek.com/v1)
	BaseURL string
}

// Implement ProviderConfig interface
func (c *DeepSeekConfig) providerType() ProviderType        { return ProviderDeepSeek }
func (c *DeepSeekConfig) apiKey() string                    { return c.APIKey }
func (c *DeepSeekConfig) timeout() time.Duration            { return c.Timeout }
func (c *DeepSeekConfig) rateLimitConfig() *RateLimitConfig { return c.RateLimiter }

// thinkingDisabled is the request body value that turns DeepSeek's thinking
// mode off. Thinking is on by default on every V4 model, so lingo only sends
// the field when the caller asks for a change.
func thinkingDisabled() map[string]any { return map[string]any{"type": "disabled"} }

// thinkingEnabled restores thinking mode explicitly.
func thinkingEnabled() map[string]any { return map[string]any{"type": "enabled"} }

// ============================================================================
// MODELS
// ============================================================================

// DeepSeekV4Flash represents deepseek-v4-flash (DeepSeek-V4-Flash), the
// cost-efficient V4 model. 1M token context window, 384K max output tokens.
// Thinking is on by default; disable it with WithThinkingDisabled.
type DeepSeekV4Flash struct{ oaiOptions }

func (m *DeepSeekV4Flash) ModelName() string {
	return resolveModelName(&m.oaiOptions, "deepseek-v4-flash")
}
func (m *DeepSeekV4Flash) Provider() ProviderType { return ProviderDeepSeek }

func (m *DeepSeekV4Flash) WithVersion(v string) *DeepSeekV4Flash      { m.modelVersion = v; return m }
func (m *DeepSeekV4Flash) WithMaxTokens(n int) *DeepSeekV4Flash       { m.maxTokens = n; return m }
func (m *DeepSeekV4Flash) WithTemperature(t float64) *DeepSeekV4Flash { m.temperature = t; return m }
func (m *DeepSeekV4Flash) WithTopP(p float64) *DeepSeekV4Flash        { m.topP = p; return m }
func (m *DeepSeekV4Flash) WithSystemPrompt(s string) *DeepSeekV4Flash {
	m.systemPrompt = s
	return m
}
func (m *DeepSeekV4Flash) WithExtraField(k string, v any) *DeepSeekV4Flash {
	m.setExtra(k, v)
	return m
}

// WithReasoningEffort sets reasoning_effort (low, medium, high) for thinking mode
func (m *DeepSeekV4Flash) WithReasoningEffort(e string) *DeepSeekV4Flash {
	m.reasoningEffort = e
	return m
}

// WithThinkingDisabled turns thinking mode off, trading depth for latency
func (m *DeepSeekV4Flash) WithThinkingDisabled() *DeepSeekV4Flash {
	m.setExtra("thinking", thinkingDisabled())
	m.reasoning = false
	return m
}

// WithThinkingEnabled turns thinking mode back on explicitly
func (m *DeepSeekV4Flash) WithThinkingEnabled() *DeepSeekV4Flash {
	m.setExtra("thinking", thinkingEnabled())
	m.reasoning = true
	return m
}

// NewDeepSeekV4Flash creates a new DeepSeek V4 Flash model with default options
func NewDeepSeekV4Flash() *DeepSeekV4Flash {
	return &DeepSeekV4Flash{oaiOptions{maxTokens: 8192, reasoning: true}}
}

// DeepSeekV4Pro represents deepseek-v4-pro (DeepSeek-V4-Pro), the most capable
// V4 model. 1M token context window, 384K max output tokens. Thinking is on by
// default; disable it with WithThinkingDisabled.
type DeepSeekV4Pro struct{ oaiOptions }

func (m *DeepSeekV4Pro) ModelName() string {
	return resolveModelName(&m.oaiOptions, "deepseek-v4-pro")
}
func (m *DeepSeekV4Pro) Provider() ProviderType { return ProviderDeepSeek }

func (m *DeepSeekV4Pro) WithVersion(v string) *DeepSeekV4Pro      { m.modelVersion = v; return m }
func (m *DeepSeekV4Pro) WithMaxTokens(n int) *DeepSeekV4Pro       { m.maxTokens = n; return m }
func (m *DeepSeekV4Pro) WithTemperature(t float64) *DeepSeekV4Pro { m.temperature = t; return m }
func (m *DeepSeekV4Pro) WithTopP(p float64) *DeepSeekV4Pro        { m.topP = p; return m }
func (m *DeepSeekV4Pro) WithSystemPrompt(s string) *DeepSeekV4Pro { m.systemPrompt = s; return m }
func (m *DeepSeekV4Pro) WithExtraField(k string, v any) *DeepSeekV4Pro {
	m.setExtra(k, v)
	return m
}

// WithReasoningEffort sets reasoning_effort (low, medium, high) for thinking mode
func (m *DeepSeekV4Pro) WithReasoningEffort(e string) *DeepSeekV4Pro {
	m.reasoningEffort = e
	return m
}

// WithThinkingDisabled turns thinking mode off, trading depth for latency
func (m *DeepSeekV4Pro) WithThinkingDisabled() *DeepSeekV4Pro {
	m.setExtra("thinking", thinkingDisabled())
	m.reasoning = false
	return m
}

// WithThinkingEnabled turns thinking mode back on explicitly
func (m *DeepSeekV4Pro) WithThinkingEnabled() *DeepSeekV4Pro {
	m.setExtra("thinking", thinkingEnabled())
	m.reasoning = true
	return m
}

// NewDeepSeekV4Pro creates a new DeepSeek V4 Pro model with default options
func NewDeepSeekV4Pro() *DeepSeekV4Pro {
	return &DeepSeekV4Pro{oaiOptions{maxTokens: 8192, reasoning: true}}
}

// DeepSeekModel is any DeepSeek model addressed by its raw ID, for models
// newer than this package or for legacy IDs.
type DeepSeekModel struct {
	oaiOptions
	modelID string
}

func (m *DeepSeekModel) ModelName() string      { return m.modelID }
func (m *DeepSeekModel) Provider() ProviderType { return ProviderDeepSeek }

func (m *DeepSeekModel) WithMaxTokens(n int) *DeepSeekModel       { m.maxTokens = n; return m }
func (m *DeepSeekModel) WithTemperature(t float64) *DeepSeekModel { m.temperature = t; return m }
func (m *DeepSeekModel) WithTopP(p float64) *DeepSeekModel        { m.topP = p; return m }
func (m *DeepSeekModel) WithSystemPrompt(s string) *DeepSeekModel { m.systemPrompt = s; return m }
func (m *DeepSeekModel) WithReasoningEffort(e string) *DeepSeekModel {
	m.reasoningEffort = e
	return m
}
func (m *DeepSeekModel) WithExtraField(k string, v any) *DeepSeekModel {
	m.setExtra(k, v)
	return m
}
func (m *DeepSeekModel) WithThinkingDisabled() *DeepSeekModel {
	m.setExtra("thinking", thinkingDisabled())
	m.reasoning = false
	return m
}
func (m *DeepSeekModel) WithThinkingEnabled() *DeepSeekModel {
	m.setExtra("thinking", thinkingEnabled())
	m.reasoning = true
	return m
}

// NewDeepSeekModel creates a DeepSeek model by ID, e.g. "deepseek-v4-flash"
func NewDeepSeekModel(modelID string) *DeepSeekModel {
	return &DeepSeekModel{modelID: modelID}
}

// Compile-time check that every DeepSeek model routes through the shared client
var (
	_ oaiCompatibleModel = (*DeepSeekV4Flash)(nil)
	_ oaiCompatibleModel = (*DeepSeekV4Pro)(nil)
	_ oaiCompatibleModel = (*DeepSeekModel)(nil)
)

// ============================================================================
// CLIENT
// ============================================================================

// newDeepSeekClient creates a new DeepSeek client over the OpenAI-compatible endpoint
func newDeepSeekClient(config *DeepSeekConfig, logger Logger) (Provider, error) {
	if config.APIKey == "" {
		return nil, fmt.Errorf("DeepSeek API key is required")
	}

	baseURL := config.BaseURL
	if baseURL == "" {
		baseURL = BaseURLDeepSeek
	}

	return newOAICompatClient(
		ProviderDeepSeek,
		"DeepSeek",
		"", // Health lists models rather than paying for a generation
		config.Timeout,
		config.RateLimiter,
		logger,
		option.WithAPIKey(config.APIKey),
		option.WithBaseURL(baseURL),
	), nil
}
