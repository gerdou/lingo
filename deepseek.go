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
//
// The setters no longer write this into extraFields themselves: they record the
// intent in ThinkingOptions and the shared client builds the body from it, so a
// caller's own WithExtraField("thinking", ...) can no longer be clobbered by
// call order and always wins.
func thinkingDisabled() map[string]any { return map[string]any{"type": "disabled"} }

// thinkingEnabled restores thinking mode explicitly.
func thinkingEnabled() map[string]any { return map[string]any{"type": "enabled"} }

// deepSeekThinkingDimensions is what every V4 model honours: a real on/off
// switch plus a depth, and both a trace and a token count on the way back.
// DeepSeek has no reasoning-token budget and no way to suppress the trace.
const deepSeekThinkingDimensions = ThinkingCanToggle | ThinkingCanSetEffort |
	ThinkingCanReportTokens | ThinkingCanReportTrace

// deepSeekEfforts is DeepSeek's own ladder, and it is neither OpenAI's nor
// Anthropic's: low, high and max, defaulting to high. Medium and xhigh are
// accepted but silently folded up to high, so lingo clamps them down to low and
// high itself -- a downgrade the caller can see in
// Metadata["thinking_translation"] beats an upgrade nobody was told about.
// A value WithReasoningEffort pinned is still forwarded exactly as given.
var deepSeekEfforts = []ThinkingEffort{ThinkingEffortLow, ThinkingEffortHigh, ThinkingEffortMax}

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

func (m *DeepSeekV4Flash) thinkingDimensions() ThinkingDimension { return deepSeekThinkingDimensions }
func (m *DeepSeekV4Flash) thinkingEfforts() []ThinkingEffort     { return deepSeekEfforts }

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

// WithReasoningEffort sets reasoning_effort for thinking mode. DeepSeek's own
// vocabulary is low, high and max, defaulting to high; medium and xhigh are
// accepted but folded up to high. The value is forwarded exactly as given.
func (m *DeepSeekV4Flash) WithReasoningEffort(e string) *DeepSeekV4Flash {
	m.setReasoningEffort(e)
	return m
}

// WithThinkingDisabled turns thinking mode off, trading depth for latency
func (m *DeepSeekV4Flash) WithThinkingDisabled() *DeepSeekV4Flash {
	m.thinking.Disable().pin(ThinkingCanToggle)
	m.reasoning = false
	return m
}

// WithThinkingEnabled turns thinking mode back on explicitly
func (m *DeepSeekV4Flash) WithThinkingEnabled() *DeepSeekV4Flash {
	m.thinking.Enable().pin(ThinkingCanToggle)
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

func (m *DeepSeekV4Pro) thinkingDimensions() ThinkingDimension { return deepSeekThinkingDimensions }
func (m *DeepSeekV4Pro) thinkingEfforts() []ThinkingEffort     { return deepSeekEfforts }

func (m *DeepSeekV4Pro) WithVersion(v string) *DeepSeekV4Pro      { m.modelVersion = v; return m }
func (m *DeepSeekV4Pro) WithMaxTokens(n int) *DeepSeekV4Pro       { m.maxTokens = n; return m }
func (m *DeepSeekV4Pro) WithTemperature(t float64) *DeepSeekV4Pro { m.temperature = t; return m }
func (m *DeepSeekV4Pro) WithTopP(p float64) *DeepSeekV4Pro        { m.topP = p; return m }
func (m *DeepSeekV4Pro) WithSystemPrompt(s string) *DeepSeekV4Pro { m.systemPrompt = s; return m }
func (m *DeepSeekV4Pro) WithExtraField(k string, v any) *DeepSeekV4Pro {
	m.setExtra(k, v)
	return m
}

// WithReasoningEffort sets reasoning_effort for thinking mode. DeepSeek's own
// vocabulary is low, high and max, defaulting to high; medium and xhigh are
// accepted but folded up to high. The value is forwarded exactly as given.
func (m *DeepSeekV4Pro) WithReasoningEffort(e string) *DeepSeekV4Pro {
	m.setReasoningEffort(e)
	return m
}

// WithThinkingDisabled turns thinking mode off, trading depth for latency
func (m *DeepSeekV4Pro) WithThinkingDisabled() *DeepSeekV4Pro {
	m.thinking.Disable().pin(ThinkingCanToggle)
	m.reasoning = false
	return m
}

// WithThinkingEnabled turns thinking mode back on explicitly
func (m *DeepSeekV4Pro) WithThinkingEnabled() *DeepSeekV4Pro {
	m.thinking.Enable().pin(ThinkingCanToggle)
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

// The raw-id escape hatch answers as a V4 model. An older DeepSeek id that
// never had a thinking mode ignores the object rather than rejecting it, and a
// value a setter pinned is forwarded whatever the id turns out to be.
func (m *DeepSeekModel) thinkingDimensions() ThinkingDimension { return deepSeekThinkingDimensions }
func (m *DeepSeekModel) thinkingEfforts() []ThinkingEffort     { return deepSeekEfforts }

func (m *DeepSeekModel) WithMaxTokens(n int) *DeepSeekModel       { m.maxTokens = n; return m }
func (m *DeepSeekModel) WithTemperature(t float64) *DeepSeekModel { m.temperature = t; return m }
func (m *DeepSeekModel) WithTopP(p float64) *DeepSeekModel        { m.topP = p; return m }
func (m *DeepSeekModel) WithSystemPrompt(s string) *DeepSeekModel { m.systemPrompt = s; return m }
func (m *DeepSeekModel) WithReasoningEffort(e string) *DeepSeekModel {
	m.setReasoningEffort(e)
	return m
}
func (m *DeepSeekModel) WithExtraField(k string, v any) *DeepSeekModel {
	m.setExtra(k, v)
	return m
}
func (m *DeepSeekModel) WithThinkingDisabled() *DeepSeekModel {
	m.thinking.Disable().pin(ThinkingCanToggle)
	m.reasoning = false
	return m
}
func (m *DeepSeekModel) WithThinkingEnabled() *DeepSeekModel {
	m.thinking.Enable().pin(ThinkingCanToggle)
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
		oaiCacheCaps{}, // DeepSeek caches server-side; there is nothing to request
		// DeepSeek is the one endpoint in this family that takes both the flat
		// OpenAI effort and an Anthropic-shaped thinking object of its own.
		oaiThinkingCaps{flatEffort: true, thinkingObject: true},
		option.WithAPIKey(config.APIKey),
		option.WithBaseURL(baseURL),
	), nil
}
