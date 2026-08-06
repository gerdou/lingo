package lingo

import (
	"fmt"
	"time"

	"github.com/openai/openai-go/v3/option"
)

func init() {
	RegisterProvider(ProviderXAI, func(config ProviderConfig, logger Logger) (Provider, error) {
		cfg, ok := config.(*XAIConfig)
		if !ok {
			return nil, fmt.Errorf("invalid config type for xAI provider")
		}
		return newXAIClient(cfg, logger)
	})
}

// BaseURLXAI is the xAI API base URL
const BaseURLXAI = "https://api.x.ai/v1"

// ============================================================================
// XAI PROVIDER CONFIG
// ============================================================================

// XAIConfig contains configuration for the xAI (Grok) provider
type XAIConfig struct {
	// APIKey is the xAI API key (required)
	APIKey string
	// Timeout is the request timeout (default: 60s)
	Timeout time.Duration
	// RateLimiter is the optional rate limit configuration
	RateLimiter *RateLimitConfig
	// BaseURL is an optional custom base URL (default: https://api.x.ai/v1)
	BaseURL string
}

// Implement ProviderConfig interface
func (c *XAIConfig) providerType() ProviderType        { return ProviderXAI }
func (c *XAIConfig) apiKey() string                    { return c.APIKey }
func (c *XAIConfig) timeout() time.Duration            { return c.Timeout }
func (c *XAIConfig) rateLimitConfig() *RateLimitConfig { return c.RateLimiter }

// Reasoning effort levels accepted by Grok reasoning models. Unlike OpenAI,
// xAI accepts "none" to switch reasoning off entirely.
const (
	XAIEffortNone   = "none"
	XAIEffortLow    = "low"
	XAIEffortMedium = "medium"
	XAIEffortHigh   = "high"
)

// ============================================================================
// MODELS
// ============================================================================

// Grok45 represents grok-4.5, xAI's latest and fastest model and the one
// recommended for code and chat. 500K token context window.
type Grok45 struct{ oaiOptions }

func (m *Grok45) ModelName() string      { return resolveModelName(&m.oaiOptions, "grok-4.5") }
func (m *Grok45) Provider() ProviderType { return ProviderXAI }

func (m *Grok45) WithVersion(v string) *Grok45           { m.modelVersion = v; return m }
func (m *Grok45) WithMaxCompletionTokens(n int) *Grok45  { m.maxCompletionTokens = n; return m }
func (m *Grok45) WithTemperature(t float64) *Grok45      { m.temperature = t; return m }
func (m *Grok45) WithTopP(p float64) *Grok45             { m.topP = p; return m }
func (m *Grok45) WithSystemPrompt(s string) *Grok45      { m.systemPrompt = s; return m }
func (m *Grok45) WithExtraField(k string, v any) *Grok45 { m.setExtra(k, v); return m }

// NewGrok45 creates a new Grok 4.5 model with default options
func NewGrok45() *Grok45 {
	return &Grok45{oaiOptions{maxCompletionTokens: 8192}}
}

// Grok43 represents grok-4.3. 1M token context window, and the current model
// that accepts reasoning_effort (none, low, medium, high; default low).
type Grok43 struct{ oaiOptions }

func (m *Grok43) ModelName() string      { return resolveModelName(&m.oaiOptions, "grok-4.3") }
func (m *Grok43) Provider() ProviderType { return ProviderXAI }

func (m *Grok43) WithVersion(v string) *Grok43           { m.modelVersion = v; return m }
func (m *Grok43) WithMaxCompletionTokens(n int) *Grok43  { m.maxCompletionTokens = n; return m }
func (m *Grok43) WithTemperature(t float64) *Grok43      { m.temperature = t; return m }
func (m *Grok43) WithTopP(p float64) *Grok43             { m.topP = p; return m }
func (m *Grok43) WithSystemPrompt(s string) *Grok43      { m.systemPrompt = s; return m }
func (m *Grok43) WithExtraField(k string, v any) *Grok43 { m.setExtra(k, v); return m }

// WithReasoningEffort sets reasoning_effort. Accepts XAIEffortNone through
// XAIEffortHigh; XAIEffortNone disables reasoning.
func (m *Grok43) WithReasoningEffort(e string) *Grok43 {
	m.reasoningEffort = e
	m.reasoning = e != XAIEffortNone
	return m
}

// NewGrok43 creates a new Grok 4.3 model with default options
func NewGrok43() *Grok43 {
	return &Grok43{oaiOptions{maxCompletionTokens: 8192, reasoningEffort: XAIEffortLow, reasoning: true}}
}

// Grok420Reasoning represents grok-4.20-0309-reasoning, the reasoning variant
// of the 4.20 line. 1M token context window. Rejects frequency_penalty,
// presence_penalty and stop, and ignores logprobs.
type Grok420Reasoning struct{ oaiOptions }

func (m *Grok420Reasoning) ModelName() string {
	return resolveModelName(&m.oaiOptions, "grok-4.20-0309-reasoning")
}
func (m *Grok420Reasoning) Provider() ProviderType { return ProviderXAI }

func (m *Grok420Reasoning) WithVersion(v string) *Grok420Reasoning { m.modelVersion = v; return m }
func (m *Grok420Reasoning) WithMaxCompletionTokens(n int) *Grok420Reasoning {
	m.maxCompletionTokens = n
	return m
}
func (m *Grok420Reasoning) WithTemperature(t float64) *Grok420Reasoning { m.temperature = t; return m }
func (m *Grok420Reasoning) WithTopP(p float64) *Grok420Reasoning        { m.topP = p; return m }
func (m *Grok420Reasoning) WithSystemPrompt(s string) *Grok420Reasoning {
	m.systemPrompt = s
	return m
}
func (m *Grok420Reasoning) WithExtraField(k string, v any) *Grok420Reasoning {
	m.setExtra(k, v)
	return m
}

// NewGrok420Reasoning creates a new Grok 4.20 reasoning model with default options
func NewGrok420Reasoning() *Grok420Reasoning {
	return &Grok420Reasoning{oaiOptions{maxCompletionTokens: 8192, reasoning: true}}
}

// Grok420NonReasoning represents grok-4.20-0309-non-reasoning, which answers
// without a reasoning pass. 1M token context window.
type Grok420NonReasoning struct{ oaiOptions }

func (m *Grok420NonReasoning) ModelName() string {
	return resolveModelName(&m.oaiOptions, "grok-4.20-0309-non-reasoning")
}
func (m *Grok420NonReasoning) Provider() ProviderType { return ProviderXAI }

func (m *Grok420NonReasoning) WithVersion(v string) *Grok420NonReasoning {
	m.modelVersion = v
	return m
}
func (m *Grok420NonReasoning) WithMaxCompletionTokens(n int) *Grok420NonReasoning {
	m.maxCompletionTokens = n
	return m
}
func (m *Grok420NonReasoning) WithTemperature(t float64) *Grok420NonReasoning {
	m.temperature = t
	return m
}
func (m *Grok420NonReasoning) WithTopP(p float64) *Grok420NonReasoning { m.topP = p; return m }
func (m *Grok420NonReasoning) WithSystemPrompt(s string) *Grok420NonReasoning {
	m.systemPrompt = s
	return m
}
func (m *Grok420NonReasoning) WithExtraField(k string, v any) *Grok420NonReasoning {
	m.setExtra(k, v)
	return m
}

// NewGrok420NonReasoning creates a new Grok 4.20 non-reasoning model with default options
func NewGrok420NonReasoning() *Grok420NonReasoning {
	return &Grok420NonReasoning{oaiOptions{maxCompletionTokens: 8192}}
}

// Grok420MultiAgent represents grok-4.20-multi-agent-0309, which runs several
// agents over a request before answering. 1M token context window.
type Grok420MultiAgent struct{ oaiOptions }

func (m *Grok420MultiAgent) ModelName() string {
	return resolveModelName(&m.oaiOptions, "grok-4.20-multi-agent-0309")
}
func (m *Grok420MultiAgent) Provider() ProviderType { return ProviderXAI }

func (m *Grok420MultiAgent) WithVersion(v string) *Grok420MultiAgent { m.modelVersion = v; return m }
func (m *Grok420MultiAgent) WithMaxCompletionTokens(n int) *Grok420MultiAgent {
	m.maxCompletionTokens = n
	return m
}
func (m *Grok420MultiAgent) WithTemperature(t float64) *Grok420MultiAgent {
	m.temperature = t
	return m
}
func (m *Grok420MultiAgent) WithTopP(p float64) *Grok420MultiAgent { m.topP = p; return m }
func (m *Grok420MultiAgent) WithSystemPrompt(s string) *Grok420MultiAgent {
	m.systemPrompt = s
	return m
}
func (m *Grok420MultiAgent) WithExtraField(k string, v any) *Grok420MultiAgent {
	m.setExtra(k, v)
	return m
}

// NewGrok420MultiAgent creates a new Grok 4.20 multi-agent model with default options
func NewGrok420MultiAgent() *Grok420MultiAgent {
	return &Grok420MultiAgent{oaiOptions{maxCompletionTokens: 8192, reasoning: true}}
}

// GrokBuild01 represents grok-build-0.1, xAI's agentic build model.
// 256K token context window.
type GrokBuild01 struct{ oaiOptions }

func (m *GrokBuild01) ModelName() string {
	return resolveModelName(&m.oaiOptions, "grok-build-0.1")
}
func (m *GrokBuild01) Provider() ProviderType { return ProviderXAI }

func (m *GrokBuild01) WithVersion(v string) *GrokBuild01 { m.modelVersion = v; return m }
func (m *GrokBuild01) WithMaxCompletionTokens(n int) *GrokBuild01 {
	m.maxCompletionTokens = n
	return m
}
func (m *GrokBuild01) WithTemperature(t float64) *GrokBuild01 { m.temperature = t; return m }
func (m *GrokBuild01) WithTopP(p float64) *GrokBuild01        { m.topP = p; return m }
func (m *GrokBuild01) WithSystemPrompt(s string) *GrokBuild01 { m.systemPrompt = s; return m }
func (m *GrokBuild01) WithExtraField(k string, v any) *GrokBuild01 {
	m.setExtra(k, v)
	return m
}

// NewGrokBuild01 creates a new grok-build-0.1 model with default options
func NewGrokBuild01() *GrokBuild01 {
	return &GrokBuild01{oaiOptions{maxCompletionTokens: 8192}}
}

// XAIModel is any Grok model addressed by its raw ID, for models newer than
// this package or for date-pinned and "-latest" aliases.
type XAIModel struct {
	oaiOptions
	modelID string
}

func (m *XAIModel) ModelName() string      { return m.modelID }
func (m *XAIModel) Provider() ProviderType { return ProviderXAI }

func (m *XAIModel) WithMaxCompletionTokens(n int) *XAIModel  { m.maxCompletionTokens = n; return m }
func (m *XAIModel) WithTemperature(t float64) *XAIModel      { m.temperature = t; return m }
func (m *XAIModel) WithTopP(p float64) *XAIModel             { m.topP = p; return m }
func (m *XAIModel) WithSystemPrompt(s string) *XAIModel      { m.systemPrompt = s; return m }
func (m *XAIModel) WithExtraField(k string, v any) *XAIModel { m.setExtra(k, v); return m }
func (m *XAIModel) WithReasoningEffort(e string) *XAIModel {
	m.reasoningEffort = e
	m.reasoning = e != XAIEffortNone
	return m
}

// NewXAIModel creates a Grok model by ID, e.g. "grok-4.5-latest"
func NewXAIModel(modelID string) *XAIModel {
	return &XAIModel{modelID: modelID}
}

// Compile-time check that every xAI model routes through the shared client
var (
	_ oaiCompatibleModel = (*Grok45)(nil)
	_ oaiCompatibleModel = (*Grok43)(nil)
	_ oaiCompatibleModel = (*Grok420Reasoning)(nil)
	_ oaiCompatibleModel = (*Grok420NonReasoning)(nil)
	_ oaiCompatibleModel = (*Grok420MultiAgent)(nil)
	_ oaiCompatibleModel = (*GrokBuild01)(nil)
	_ oaiCompatibleModel = (*XAIModel)(nil)
)

// ============================================================================
// CLIENT
// ============================================================================

// newXAIClient creates a new xAI client over the OpenAI-compatible endpoint
func newXAIClient(config *XAIConfig, logger Logger) (Provider, error) {
	if config.APIKey == "" {
		return nil, fmt.Errorf("xAI API key is required")
	}

	baseURL := config.BaseURL
	if baseURL == "" {
		baseURL = BaseURLXAI
	}

	return newOAICompatClient(
		ProviderXAI,
		"xAI",
		"", // Health lists models rather than paying for a generation
		config.Timeout,
		config.RateLimiter,
		logger,
		option.WithAPIKey(config.APIKey),
		option.WithBaseURL(baseURL),
	), nil
}
