package lingo

import (
	"context"
	"fmt"
	"time"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/shared"
)

func init() {
	RegisterProvider(ProviderOpenAI, func(config ProviderConfig, logger Logger) (Provider, error) {
		cfg, ok := config.(*OpenAIConfig)
		if !ok {
			return nil, fmt.Errorf("invalid config type for OpenAI provider")
		}
		return newOpenAIClient(cfg, logger)
	})
}

// ============================================================================
// OPENAI PROVIDER CONFIG
// ============================================================================

// OpenAIConfig contains configuration for the OpenAI provider
type OpenAIConfig struct {
	// APIKey is the OpenAI API key (required)
	APIKey string
	// Timeout is the request timeout (default: 60s)
	Timeout time.Duration
	// RateLimiter is the optional rate limit configuration
	RateLimiter *RateLimitConfig
	// BaseURL is an optional custom base URL (for Azure OpenAI or proxies)
	BaseURL string
}

// Implement ProviderConfig interface
func (c *OpenAIConfig) providerType() ProviderType        { return ProviderOpenAI }
func (c *OpenAIConfig) apiKey() string                    { return c.APIKey }
func (c *OpenAIConfig) timeout() time.Duration            { return c.Timeout }
func (c *OpenAIConfig) rateLimitConfig() *RateLimitConfig { return c.RateLimiter }

// ============================================================================
// SHARED OPTIONS (embedded in model structs)
// ============================================================================

// openAIStandardOptions contains options for standard OpenAI models (GPT-4o, GPT-4, etc.)
type openAIStandardOptions struct {
	modelVersion string // Optional: override model name with specific version
	maxTokens    int
	temperature  float64
	topP         float64
	systemPrompt string
}

// openAIReasoningOptions contains options for reasoning models (o1, o3, o4, GPT-5)
type openAIReasoningOptions struct {
	modelVersion        string // Optional: override model name with specific version
	maxCompletionTokens int
	reasoningEffort     string // "low", "medium", "high"
	systemPrompt        string
}

// ============================================================================
// STANDARD MODELS (GPT-4o, GPT-4, GPT-3.5, GPT-4.1)
// ============================================================================

// GPT4o represents the GPT-4o model
// Versions: gpt-4o, gpt-4o-2024-11-20, gpt-4o-2024-08-06, gpt-4o-2024-05-13
type GPT4o struct{ openAIStandardOptions }

func (m *GPT4o) ModelName() string {
	if m.modelVersion != "" {
		return m.modelVersion
	}
	return "gpt-4o"
}
func (m *GPT4o) Provider() ProviderType { return ProviderOpenAI }
func (m *GPT4o) SystemPrompt() string   { return m.systemPrompt }
func (m *GPT4o) isStandard() bool       { return true }

func (m *GPT4o) WithVersion(v string) *GPT4o      { m.modelVersion = v; return m }
func (m *GPT4o) WithMaxTokens(n int) *GPT4o       { m.maxTokens = n; return m }
func (m *GPT4o) WithTemperature(t float64) *GPT4o { m.temperature = t; return m }
func (m *GPT4o) WithTopP(p float64) *GPT4o        { m.topP = p; return m }
func (m *GPT4o) WithSystemPrompt(s string) *GPT4o { m.systemPrompt = s; return m }

// NewGPT4o creates a new GPT-4o model with default options
func NewGPT4o() *GPT4o {
	return &GPT4o{openAIStandardOptions{maxTokens: 4096, temperature: 1.0}}
}

// GPT4oMini represents the GPT-4o-mini model
// Versions: gpt-4o-mini, gpt-4o-mini-2024-07-18
type GPT4oMini struct{ openAIStandardOptions }

func (m *GPT4oMini) ModelName() string {
	if m.modelVersion != "" {
		return m.modelVersion
	}
	return "gpt-4o-mini"
}
func (m *GPT4oMini) Provider() ProviderType { return ProviderOpenAI }
func (m *GPT4oMini) SystemPrompt() string   { return m.systemPrompt }
func (m *GPT4oMini) isStandard() bool       { return true }

func (m *GPT4oMini) WithVersion(v string) *GPT4oMini      { m.modelVersion = v; return m }
func (m *GPT4oMini) WithMaxTokens(n int) *GPT4oMini       { m.maxTokens = n; return m }
func (m *GPT4oMini) WithTemperature(t float64) *GPT4oMini { m.temperature = t; return m }
func (m *GPT4oMini) WithTopP(p float64) *GPT4oMini        { m.topP = p; return m }
func (m *GPT4oMini) WithSystemPrompt(s string) *GPT4oMini { m.systemPrompt = s; return m }

// NewGPT4oMini creates a new GPT-4o-mini model with default options
func NewGPT4oMini() *GPT4oMini {
	return &GPT4oMini{openAIStandardOptions{maxTokens: 4096, temperature: 1.0}}
}

// GPT4Turbo represents the GPT-4-turbo model
// Versions: gpt-4-turbo, gpt-4-turbo-2024-04-09, gpt-4-turbo-preview
type GPT4Turbo struct{ openAIStandardOptions }

func (m *GPT4Turbo) ModelName() string {
	if m.modelVersion != "" {
		return m.modelVersion
	}
	return "gpt-4-turbo"
}
func (m *GPT4Turbo) Provider() ProviderType { return ProviderOpenAI }
func (m *GPT4Turbo) SystemPrompt() string   { return m.systemPrompt }
func (m *GPT4Turbo) isStandard() bool       { return true }

func (m *GPT4Turbo) WithVersion(v string) *GPT4Turbo      { m.modelVersion = v; return m }
func (m *GPT4Turbo) WithMaxTokens(n int) *GPT4Turbo       { m.maxTokens = n; return m }
func (m *GPT4Turbo) WithTemperature(t float64) *GPT4Turbo { m.temperature = t; return m }
func (m *GPT4Turbo) WithTopP(p float64) *GPT4Turbo        { m.topP = p; return m }
func (m *GPT4Turbo) WithSystemPrompt(s string) *GPT4Turbo { m.systemPrompt = s; return m }

// NewGPT4Turbo creates a new GPT-4-turbo model with default options
func NewGPT4Turbo() *GPT4Turbo {
	return &GPT4Turbo{openAIStandardOptions{maxTokens: 4096, temperature: 1.0}}
}

// GPT4 represents the GPT-4 model
// Versions: gpt-4, gpt-4-0613
type GPT4 struct{ openAIStandardOptions }

func (m *GPT4) ModelName() string {
	if m.modelVersion != "" {
		return m.modelVersion
	}
	return "gpt-4"
}
func (m *GPT4) Provider() ProviderType { return ProviderOpenAI }
func (m *GPT4) SystemPrompt() string   { return m.systemPrompt }
func (m *GPT4) isStandard() bool       { return true }

func (m *GPT4) WithVersion(v string) *GPT4      { m.modelVersion = v; return m }
func (m *GPT4) WithMaxTokens(n int) *GPT4       { m.maxTokens = n; return m }
func (m *GPT4) WithTemperature(t float64) *GPT4 { m.temperature = t; return m }
func (m *GPT4) WithTopP(p float64) *GPT4        { m.topP = p; return m }
func (m *GPT4) WithSystemPrompt(s string) *GPT4 { m.systemPrompt = s; return m }

// NewGPT4 creates a new GPT-4 model with default options
func NewGPT4() *GPT4 {
	return &GPT4{openAIStandardOptions{maxTokens: 4096, temperature: 1.0}}
}

// GPT41 represents the GPT-4.1 model
// Versions: gpt-4.1, gpt-4.1-2025-04-14
type GPT41 struct{ openAIStandardOptions }

func (m *GPT41) ModelName() string {
	if m.modelVersion != "" {
		return m.modelVersion
	}
	return "gpt-4.1"
}
func (m *GPT41) Provider() ProviderType { return ProviderOpenAI }
func (m *GPT41) SystemPrompt() string   { return m.systemPrompt }
func (m *GPT41) isStandard() bool       { return true }

func (m *GPT41) WithVersion(v string) *GPT41      { m.modelVersion = v; return m }
func (m *GPT41) WithMaxTokens(n int) *GPT41       { m.maxTokens = n; return m }
func (m *GPT41) WithTemperature(t float64) *GPT41 { m.temperature = t; return m }
func (m *GPT41) WithTopP(p float64) *GPT41        { m.topP = p; return m }
func (m *GPT41) WithSystemPrompt(s string) *GPT41 { m.systemPrompt = s; return m }

// NewGPT41 creates a new GPT-4.1 model with default options
func NewGPT41() *GPT41 {
	return &GPT41{openAIStandardOptions{maxTokens: 4096, temperature: 1.0}}
}

// GPT41Mini represents the GPT-4.1-mini model
type GPT41Mini struct{ openAIStandardOptions }

func (m *GPT41Mini) ModelName() string      { return "gpt-4.1-mini" }
func (m *GPT41Mini) Provider() ProviderType { return ProviderOpenAI }
func (m *GPT41Mini) SystemPrompt() string   { return m.systemPrompt }
func (m *GPT41Mini) isStandard() bool       { return true }

func (m *GPT41Mini) WithMaxTokens(n int) *GPT41Mini       { m.maxTokens = n; return m }
func (m *GPT41Mini) WithTemperature(t float64) *GPT41Mini { m.temperature = t; return m }
func (m *GPT41Mini) WithTopP(p float64) *GPT41Mini        { m.topP = p; return m }
func (m *GPT41Mini) WithSystemPrompt(s string) *GPT41Mini { m.systemPrompt = s; return m }

// NewGPT41Mini creates a new GPT-4.1-mini model with default options
func NewGPT41Mini() *GPT41Mini {
	return &GPT41Mini{openAIStandardOptions{maxTokens: 4096, temperature: 1.0}}
}

// GPT41Nano represents the GPT-4.1-nano model
type GPT41Nano struct{ openAIStandardOptions }

func (m *GPT41Nano) ModelName() string      { return "gpt-4.1-nano" }
func (m *GPT41Nano) Provider() ProviderType { return ProviderOpenAI }
func (m *GPT41Nano) SystemPrompt() string   { return m.systemPrompt }
func (m *GPT41Nano) isStandard() bool       { return true }

func (m *GPT41Nano) WithMaxTokens(n int) *GPT41Nano       { m.maxTokens = n; return m }
func (m *GPT41Nano) WithTemperature(t float64) *GPT41Nano { m.temperature = t; return m }
func (m *GPT41Nano) WithTopP(p float64) *GPT41Nano        { m.topP = p; return m }
func (m *GPT41Nano) WithSystemPrompt(s string) *GPT41Nano { m.systemPrompt = s; return m }

// NewGPT41Nano creates a new GPT-4.1-nano model with default options
func NewGPT41Nano() *GPT41Nano {
	return &GPT41Nano{openAIStandardOptions{maxTokens: 4096, temperature: 1.0}}
}

// GPT35Turbo represents the GPT-3.5-turbo model
// Versions: gpt-3.5-turbo, gpt-3.5-turbo-0125
type GPT35Turbo struct{ openAIStandardOptions }

func (m *GPT35Turbo) ModelName() string {
	if m.modelVersion != "" {
		return m.modelVersion
	}
	return "gpt-3.5-turbo"
}
func (m *GPT35Turbo) Provider() ProviderType { return ProviderOpenAI }
func (m *GPT35Turbo) SystemPrompt() string   { return m.systemPrompt }
func (m *GPT35Turbo) isStandard() bool       { return true }

func (m *GPT35Turbo) WithVersion(v string) *GPT35Turbo      { m.modelVersion = v; return m }
func (m *GPT35Turbo) WithMaxTokens(n int) *GPT35Turbo       { m.maxTokens = n; return m }
func (m *GPT35Turbo) WithTemperature(t float64) *GPT35Turbo { m.temperature = t; return m }
func (m *GPT35Turbo) WithTopP(p float64) *GPT35Turbo        { m.topP = p; return m }
func (m *GPT35Turbo) WithSystemPrompt(s string) *GPT35Turbo { m.systemPrompt = s; return m }

// NewGPT35Turbo creates a new GPT-3.5-turbo model with default options
func NewGPT35Turbo() *GPT35Turbo {
	return &GPT35Turbo{openAIStandardOptions{maxTokens: 4096, temperature: 1.0}}
}

// ============================================================================
// REASONING MODELS (O1, O3, O4, GPT-5 series)
// ============================================================================

// O1 represents the O1 reasoning model
// Versions: o1, o1-2024-12-17
type O1 struct{ openAIReasoningOptions }

func (m *O1) ModelName() string {
	if m.modelVersion != "" {
		return m.modelVersion
	}
	return "o1"
}
func (m *O1) Provider() ProviderType { return ProviderOpenAI }
func (m *O1) SystemPrompt() string   { return m.systemPrompt }
func (m *O1) isReasoning() bool      { return true }

func (m *O1) WithVersion(v string) *O1          { m.modelVersion = v; return m }
func (m *O1) WithMaxCompletionTokens(n int) *O1 { m.maxCompletionTokens = n; return m }
func (m *O1) WithReasoningEffort(e string) *O1  { m.reasoningEffort = e; return m }
func (m *O1) WithSystemPrompt(s string) *O1     { m.systemPrompt = s; return m }

// NewO1 creates a new O1 model with default options
func NewO1() *O1 {
	return &O1{openAIReasoningOptions{maxCompletionTokens: 4096, reasoningEffort: "medium"}}
}

// O1Mini represents the O1-mini reasoning model
// Versions: o1-mini, o1-mini-2024-09-12
type O1Mini struct{ openAIReasoningOptions }

func (m *O1Mini) ModelName() string {
	if m.modelVersion != "" {
		return m.modelVersion
	}
	return "o1-mini"
}
func (m *O1Mini) Provider() ProviderType { return ProviderOpenAI }
func (m *O1Mini) SystemPrompt() string   { return m.systemPrompt }
func (m *O1Mini) isReasoning() bool      { return true }

func (m *O1Mini) WithVersion(v string) *O1Mini          { m.modelVersion = v; return m }
func (m *O1Mini) WithMaxCompletionTokens(n int) *O1Mini { m.maxCompletionTokens = n; return m }
func (m *O1Mini) WithReasoningEffort(e string) *O1Mini  { m.reasoningEffort = e; return m }
func (m *O1Mini) WithSystemPrompt(s string) *O1Mini     { m.systemPrompt = s; return m }

// NewO1Mini creates a new O1-mini model with default options
func NewO1Mini() *O1Mini {
	return &O1Mini{openAIReasoningOptions{maxCompletionTokens: 4096, reasoningEffort: "medium"}}
}

// O1Pro represents the O1-pro reasoning model
// Versions: o1-pro, o1-pro-2025-03-19
type O1Pro struct{ openAIReasoningOptions }

func (m *O1Pro) ModelName() string {
	if m.modelVersion != "" {
		return m.modelVersion
	}
	return "o1-pro"
}
func (m *O1Pro) Provider() ProviderType { return ProviderOpenAI }
func (m *O1Pro) SystemPrompt() string   { return m.systemPrompt }
func (m *O1Pro) isReasoning() bool      { return true }

func (m *O1Pro) WithVersion(v string) *O1Pro          { m.modelVersion = v; return m }
func (m *O1Pro) WithMaxCompletionTokens(n int) *O1Pro { m.maxCompletionTokens = n; return m }
func (m *O1Pro) WithReasoningEffort(e string) *O1Pro  { m.reasoningEffort = e; return m }
func (m *O1Pro) WithSystemPrompt(s string) *O1Pro     { m.systemPrompt = s; return m }

// NewO1Pro creates a new O1-pro model with default options
func NewO1Pro() *O1Pro {
	return &O1Pro{openAIReasoningOptions{maxCompletionTokens: 8192, reasoningEffort: "high"}}
}

// O3 represents the O3 reasoning model
// Versions: o3, o3-2025-04-16
type O3 struct{ openAIReasoningOptions }

func (m *O3) ModelName() string {
	if m.modelVersion != "" {
		return m.modelVersion
	}
	return "o3"
}
func (m *O3) Provider() ProviderType { return ProviderOpenAI }
func (m *O3) SystemPrompt() string   { return m.systemPrompt }
func (m *O3) isReasoning() bool      { return true }

func (m *O3) WithVersion(v string) *O3          { m.modelVersion = v; return m }
func (m *O3) WithMaxCompletionTokens(n int) *O3 { m.maxCompletionTokens = n; return m }
func (m *O3) WithReasoningEffort(e string) *O3  { m.reasoningEffort = e; return m }
func (m *O3) WithSystemPrompt(s string) *O3     { m.systemPrompt = s; return m }

// NewO3 creates a new O3 model with default options
func NewO3() *O3 {
	return &O3{openAIReasoningOptions{maxCompletionTokens: 8192, reasoningEffort: "medium"}}
}

// O3Mini represents the O3-mini reasoning model
// Versions: o3-mini, o3-mini-2025-01-31
type O3Mini struct{ openAIReasoningOptions }

func (m *O3Mini) ModelName() string {
	if m.modelVersion != "" {
		return m.modelVersion
	}
	return "o3-mini"
}
func (m *O3Mini) Provider() ProviderType { return ProviderOpenAI }
func (m *O3Mini) SystemPrompt() string   { return m.systemPrompt }
func (m *O3Mini) isReasoning() bool      { return true }

func (m *O3Mini) WithVersion(v string) *O3Mini          { m.modelVersion = v; return m }
func (m *O3Mini) WithMaxCompletionTokens(n int) *O3Mini { m.maxCompletionTokens = n; return m }
func (m *O3Mini) WithReasoningEffort(e string) *O3Mini  { m.reasoningEffort = e; return m }
func (m *O3Mini) WithSystemPrompt(s string) *O3Mini     { m.systemPrompt = s; return m }

// NewO3Mini creates a new O3-mini model with default options
func NewO3Mini() *O3Mini {
	return &O3Mini{openAIReasoningOptions{maxCompletionTokens: 4096, reasoningEffort: "medium"}}
}

// O4Mini represents the O4-mini reasoning model
// Versions: o4-mini, o4-mini-2025-04-16
type O4Mini struct{ openAIReasoningOptions }

func (m *O4Mini) ModelName() string {
	if m.modelVersion != "" {
		return m.modelVersion
	}
	return "o4-mini"
}
func (m *O4Mini) Provider() ProviderType { return ProviderOpenAI }
func (m *O4Mini) SystemPrompt() string   { return m.systemPrompt }
func (m *O4Mini) isReasoning() bool      { return true }

func (m *O4Mini) WithVersion(v string) *O4Mini          { m.modelVersion = v; return m }
func (m *O4Mini) WithMaxCompletionTokens(n int) *O4Mini { m.maxCompletionTokens = n; return m }
func (m *O4Mini) WithReasoningEffort(e string) *O4Mini  { m.reasoningEffort = e; return m }
func (m *O4Mini) WithSystemPrompt(s string) *O4Mini     { m.systemPrompt = s; return m }

// NewO4Mini creates a new O4-mini model with default options
func NewO4Mini() *O4Mini {
	return &O4Mini{openAIReasoningOptions{maxCompletionTokens: 4096, reasoningEffort: "medium"}}
}

// GPT5 represents the GPT-5 reasoning model
type GPT5 struct{ openAIReasoningOptions }

func (m *GPT5) ModelName() string      { return "gpt-5" }
func (m *GPT5) Provider() ProviderType { return ProviderOpenAI }
func (m *GPT5) SystemPrompt() string   { return m.systemPrompt }
func (m *GPT5) isReasoning() bool      { return true }

func (m *GPT5) WithMaxCompletionTokens(n int) *GPT5 { m.maxCompletionTokens = n; return m }
func (m *GPT5) WithReasoningEffort(e string) *GPT5  { m.reasoningEffort = e; return m }
func (m *GPT5) WithSystemPrompt(s string) *GPT5     { m.systemPrompt = s; return m }

// NewGPT5 creates a new GPT-5 model with default options
func NewGPT5() *GPT5 {
	return &GPT5{openAIReasoningOptions{maxCompletionTokens: 8192, reasoningEffort: "medium"}}
}

// GPT5Mini represents the GPT-5-mini reasoning model
type GPT5Mini struct{ openAIReasoningOptions }

func (m *GPT5Mini) ModelName() string      { return "gpt-5-mini" }
func (m *GPT5Mini) Provider() ProviderType { return ProviderOpenAI }
func (m *GPT5Mini) SystemPrompt() string   { return m.systemPrompt }
func (m *GPT5Mini) isReasoning() bool      { return true }

func (m *GPT5Mini) WithMaxCompletionTokens(n int) *GPT5Mini { m.maxCompletionTokens = n; return m }
func (m *GPT5Mini) WithReasoningEffort(e string) *GPT5Mini  { m.reasoningEffort = e; return m }
func (m *GPT5Mini) WithSystemPrompt(s string) *GPT5Mini     { m.systemPrompt = s; return m }

// NewGPT5Mini creates a new GPT-5-mini model with default options
func NewGPT5Mini() *GPT5Mini {
	return &GPT5Mini{openAIReasoningOptions{maxCompletionTokens: 4096, reasoningEffort: "medium"}}
}

// GPT5Nano represents the GPT-5-nano reasoning model
type GPT5Nano struct{ openAIReasoningOptions }

func (m *GPT5Nano) ModelName() string      { return "gpt-5-nano" }
func (m *GPT5Nano) Provider() ProviderType { return ProviderOpenAI }
func (m *GPT5Nano) SystemPrompt() string   { return m.systemPrompt }
func (m *GPT5Nano) isReasoning() bool      { return true }

func (m *GPT5Nano) WithMaxCompletionTokens(n int) *GPT5Nano { m.maxCompletionTokens = n; return m }
func (m *GPT5Nano) WithReasoningEffort(e string) *GPT5Nano  { m.reasoningEffort = e; return m }
func (m *GPT5Nano) WithSystemPrompt(s string) *GPT5Nano     { m.systemPrompt = s; return m }

// NewGPT5Nano creates a new GPT-5-nano model with default options
func NewGPT5Nano() *GPT5Nano {
	return &GPT5Nano{openAIReasoningOptions{maxCompletionTokens: 4096, reasoningEffort: "medium"}}
}

// GPT5Pro represents the GPT-5-pro reasoning model
type GPT5Pro struct{ openAIReasoningOptions }

func (m *GPT5Pro) ModelName() string      { return "gpt-5-pro" }
func (m *GPT5Pro) Provider() ProviderType { return ProviderOpenAI }
func (m *GPT5Pro) SystemPrompt() string   { return m.systemPrompt }
func (m *GPT5Pro) isReasoning() bool      { return true }

func (m *GPT5Pro) WithMaxCompletionTokens(n int) *GPT5Pro { m.maxCompletionTokens = n; return m }
func (m *GPT5Pro) WithReasoningEffort(e string) *GPT5Pro  { m.reasoningEffort = e; return m }
func (m *GPT5Pro) WithSystemPrompt(s string) *GPT5Pro     { m.systemPrompt = s; return m }

// NewGPT5Pro creates a new GPT-5-pro model with default options
func NewGPT5Pro() *GPT5Pro {
	return &GPT5Pro{openAIReasoningOptions{maxCompletionTokens: 8192, reasoningEffort: "high"}}
}

// GPT5Turbo represents the GPT-5-turbo reasoning model
type GPT5Turbo struct{ openAIReasoningOptions }

func (m *GPT5Turbo) ModelName() string      { return "gpt-5-turbo" }
func (m *GPT5Turbo) Provider() ProviderType { return ProviderOpenAI }
func (m *GPT5Turbo) SystemPrompt() string   { return m.systemPrompt }
func (m *GPT5Turbo) isReasoning() bool      { return true }

func (m *GPT5Turbo) WithMaxCompletionTokens(n int) *GPT5Turbo { m.maxCompletionTokens = n; return m }
func (m *GPT5Turbo) WithReasoningEffort(e string) *GPT5Turbo  { m.reasoningEffort = e; return m }
func (m *GPT5Turbo) WithSystemPrompt(s string) *GPT5Turbo     { m.systemPrompt = s; return m }

// NewGPT5Turbo creates a new GPT-5-turbo model with default options
func NewGPT5Turbo() *GPT5Turbo {
	return &GPT5Turbo{openAIReasoningOptions{maxCompletionTokens: 8192, reasoningEffort: "medium"}}
}

// GPT51 represents the GPT-5.1 reasoning model
type GPT51 struct{ openAIReasoningOptions }

func (m *GPT51) ModelName() string      { return "gpt-5.1" }
func (m *GPT51) Provider() ProviderType { return ProviderOpenAI }
func (m *GPT51) SystemPrompt() string   { return m.systemPrompt }
func (m *GPT51) isReasoning() bool      { return true }

func (m *GPT51) WithMaxCompletionTokens(n int) *GPT51 { m.maxCompletionTokens = n; return m }
func (m *GPT51) WithReasoningEffort(e string) *GPT51  { m.reasoningEffort = e; return m }
func (m *GPT51) WithSystemPrompt(s string) *GPT51     { m.systemPrompt = s; return m }

// NewGPT51 creates a new GPT-5.1 model with default options
func NewGPT51() *GPT51 {
	return &GPT51{openAIReasoningOptions{maxCompletionTokens: 8192, reasoningEffort: "medium"}}
}

// GPT51Mini represents the GPT-5.1-mini reasoning model
type GPT51Mini struct{ openAIReasoningOptions }

func (m *GPT51Mini) ModelName() string      { return "gpt-5.1-mini" }
func (m *GPT51Mini) Provider() ProviderType { return ProviderOpenAI }
func (m *GPT51Mini) SystemPrompt() string   { return m.systemPrompt }
func (m *GPT51Mini) isReasoning() bool      { return true }

func (m *GPT51Mini) WithMaxCompletionTokens(n int) *GPT51Mini { m.maxCompletionTokens = n; return m }
func (m *GPT51Mini) WithReasoningEffort(e string) *GPT51Mini  { m.reasoningEffort = e; return m }
func (m *GPT51Mini) WithSystemPrompt(s string) *GPT51Mini     { m.systemPrompt = s; return m }

// NewGPT51Mini creates a new GPT-5.1-mini model with default options
func NewGPT51Mini() *GPT51Mini {
	return &GPT51Mini{openAIReasoningOptions{maxCompletionTokens: 4096, reasoningEffort: "medium"}}
}

// GPT51Nano represents the GPT-5.1-nano reasoning model
type GPT51Nano struct{ openAIReasoningOptions }

func (m *GPT51Nano) ModelName() string      { return "gpt-5.1-nano" }
func (m *GPT51Nano) Provider() ProviderType { return ProviderOpenAI }
func (m *GPT51Nano) SystemPrompt() string   { return m.systemPrompt }
func (m *GPT51Nano) isReasoning() bool      { return true }

func (m *GPT51Nano) WithMaxCompletionTokens(n int) *GPT51Nano { m.maxCompletionTokens = n; return m }
func (m *GPT51Nano) WithReasoningEffort(e string) *GPT51Nano  { m.reasoningEffort = e; return m }
func (m *GPT51Nano) WithSystemPrompt(s string) *GPT51Nano     { m.systemPrompt = s; return m }

// NewGPT51Nano creates a new GPT-5.1-nano model with default options
func NewGPT51Nano() *GPT51Nano {
	return &GPT51Nano{openAIReasoningOptions{maxCompletionTokens: 4096, reasoningEffort: "medium"}}
}

// GPT51Codex represents the GPT-5.1-codex reasoning model
type GPT51Codex struct{ openAIReasoningOptions }

func (m *GPT51Codex) ModelName() string      { return "gpt-5.1-codex" }
func (m *GPT51Codex) Provider() ProviderType { return ProviderOpenAI }
func (m *GPT51Codex) SystemPrompt() string   { return m.systemPrompt }
func (m *GPT51Codex) isReasoning() bool      { return true }

func (m *GPT51Codex) WithMaxCompletionTokens(n int) *GPT51Codex { m.maxCompletionTokens = n; return m }
func (m *GPT51Codex) WithReasoningEffort(e string) *GPT51Codex  { m.reasoningEffort = e; return m }
func (m *GPT51Codex) WithSystemPrompt(s string) *GPT51Codex     { m.systemPrompt = s; return m }

// NewGPT51Codex creates a new GPT-5.1-codex model with default options
func NewGPT51Codex() *GPT51Codex {
	return &GPT51Codex{openAIReasoningOptions{maxCompletionTokens: 8192, reasoningEffort: "medium"}}
}

// GPT51CodexMini represents the GPT-5.1-codex-mini reasoning model
type GPT51CodexMini struct{ openAIReasoningOptions }

func (m *GPT51CodexMini) ModelName() string      { return "gpt-5.1-codex-mini" }
func (m *GPT51CodexMini) Provider() ProviderType { return ProviderOpenAI }
func (m *GPT51CodexMini) SystemPrompt() string   { return m.systemPrompt }
func (m *GPT51CodexMini) isReasoning() bool      { return true }

func (m *GPT51CodexMini) WithMaxCompletionTokens(n int) *GPT51CodexMini {
	m.maxCompletionTokens = n
	return m
}
func (m *GPT51CodexMini) WithReasoningEffort(e string) *GPT51CodexMini {
	m.reasoningEffort = e
	return m
}
func (m *GPT51CodexMini) WithSystemPrompt(s string) *GPT51CodexMini { m.systemPrompt = s; return m }

// NewGPT51CodexMini creates a new GPT-5.1-codex-mini model with default options
func NewGPT51CodexMini() *GPT51CodexMini {
	return &GPT51CodexMini{openAIReasoningOptions{maxCompletionTokens: 4096, reasoningEffort: "medium"}}
}

// O3Pro represents the O3-pro reasoning model
type O3Pro struct{ openAIReasoningOptions }

func (m *O3Pro) ModelName() string      { return "o3-pro" }
func (m *O3Pro) Provider() ProviderType { return ProviderOpenAI }
func (m *O3Pro) SystemPrompt() string   { return m.systemPrompt }
func (m *O3Pro) isReasoning() bool      { return true }

func (m *O3Pro) WithMaxCompletionTokens(n int) *O3Pro { m.maxCompletionTokens = n; return m }
func (m *O3Pro) WithReasoningEffort(e string) *O3Pro  { m.reasoningEffort = e; return m }
func (m *O3Pro) WithSystemPrompt(s string) *O3Pro     { m.systemPrompt = s; return m }

// NewO3Pro creates a new O3-pro model with default options
func NewO3Pro() *O3Pro {
	return &O3Pro{openAIReasoningOptions{maxCompletionTokens: 8192, reasoningEffort: "high"}}
}

// O1Preview represents the O1-preview reasoning model
// Versions: o1-preview, o1-preview-2024-09-12
type O1Preview struct{ openAIReasoningOptions }

func (m *O1Preview) ModelName() string {
	if m.modelVersion != "" {
		return m.modelVersion
	}
	return "o1-preview"
}
func (m *O1Preview) Provider() ProviderType { return ProviderOpenAI }
func (m *O1Preview) SystemPrompt() string   { return m.systemPrompt }
func (m *O1Preview) isReasoning() bool      { return true }

func (m *O1Preview) WithVersion(v string) *O1Preview          { m.modelVersion = v; return m }
func (m *O1Preview) WithMaxCompletionTokens(n int) *O1Preview { m.maxCompletionTokens = n; return m }
func (m *O1Preview) WithReasoningEffort(e string) *O1Preview  { m.reasoningEffort = e; return m }
func (m *O1Preview) WithSystemPrompt(s string) *O1Preview     { m.systemPrompt = s; return m }

// NewO1Preview creates a new O1-preview model with default options
func NewO1Preview() *O1Preview {
	return &O1Preview{openAIReasoningOptions{maxCompletionTokens: 8192, reasoningEffort: "medium"}}
}

// ============================================================================
// OPENAI PROVIDER CLIENT
// ============================================================================

// openAIStandardModel is an interface for standard models
type openAIStandardModel interface {
	Model
	isStandard() bool
}

// openAIReasoningModel is an interface for reasoning models
type openAIReasoningModel interface {
	Model
	isReasoning() bool
}

// openAIClient implements the Provider interface for OpenAI
type openAIClient struct {
	client      openai.Client
	timeout     time.Duration
	logger      Logger
	rateLimiter *rateLimiter
}

// newOpenAIClient creates a new OpenAI client using the official SDK
func newOpenAIClient(config *OpenAIConfig, logger Logger) (*openAIClient, error) {
	if config.APIKey == "" {
		return nil, fmt.Errorf("OpenAI API key is required")
	}

	opts := []option.RequestOption{option.WithAPIKey(config.APIKey)}
	if config.BaseURL != "" {
		opts = append(opts, option.WithBaseURL(config.BaseURL))
	}

	client := openai.NewClient(opts...)

	timeout := config.Timeout
	if timeout == 0 {
		timeout = defaultTimeout()
	}

	return &openAIClient{
		client:      client,
		timeout:     timeout,
		logger:      logger,
		rateLimiter: newRateLimiter(config.RateLimiter, logger),
	}, nil
}

// Generate generates text using OpenAI's API
func (c *openAIClient) Generate(ctx context.Context, model Model, prompt string) (*GenerationResponse, error) {
	// Verify model is for OpenAI
	if model.Provider() != ProviderOpenAI {
		return nil, fmt.Errorf("model %s is not an OpenAI model", model.ModelName())
	}

	// Set timeout
	ctx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()

	// Determine if this is a reasoning model
	_, isReasoning := model.(openAIReasoningModel)

	// Build messages with optional system prompt
	var messages []openai.ChatCompletionMessageParamUnion

	if model.SystemPrompt() != "" {
		if isReasoning {
			// Reasoning models use "developer" role instead of "system"
			messages = append(messages, openai.DeveloperMessage(model.SystemPrompt()))
		} else {
			// Standard models use "system" role
			messages = append(messages, openai.SystemMessage(model.SystemPrompt()))
		}
	}
	messages = append(messages, openai.UserMessage(prompt))

	// Build request parameters
	params := openai.ChatCompletionNewParams{
		Model:    openai.ChatModel(model.ModelName()),
		Messages: messages,
	}

	// Apply options based on model type
	switch m := model.(type) {
	// Standard models
	case *GPT4o:
		if m.maxTokens > 0 {
			params.MaxTokens = openai.Int(int64(m.maxTokens))
		}
		if m.temperature > 0 {
			params.Temperature = openai.Float(m.temperature)
		}
		if m.topP > 0 {
			params.TopP = openai.Float(m.topP)
		}
	case *GPT4oMini:
		if m.maxTokens > 0 {
			params.MaxTokens = openai.Int(int64(m.maxTokens))
		}
		if m.temperature > 0 {
			params.Temperature = openai.Float(m.temperature)
		}
		if m.topP > 0 {
			params.TopP = openai.Float(m.topP)
		}
	case *GPT4Turbo:
		if m.maxTokens > 0 {
			params.MaxTokens = openai.Int(int64(m.maxTokens))
		}
		if m.temperature > 0 {
			params.Temperature = openai.Float(m.temperature)
		}
		if m.topP > 0 {
			params.TopP = openai.Float(m.topP)
		}
	case *GPT4:
		if m.maxTokens > 0 {
			params.MaxTokens = openai.Int(int64(m.maxTokens))
		}
		if m.temperature > 0 {
			params.Temperature = openai.Float(m.temperature)
		}
		if m.topP > 0 {
			params.TopP = openai.Float(m.topP)
		}
	case *GPT41:
		if m.maxTokens > 0 {
			params.MaxTokens = openai.Int(int64(m.maxTokens))
		}
		if m.temperature > 0 {
			params.Temperature = openai.Float(m.temperature)
		}
		if m.topP > 0 {
			params.TopP = openai.Float(m.topP)
		}
	case *GPT41Mini:
		if m.maxTokens > 0 {
			params.MaxTokens = openai.Int(int64(m.maxTokens))
		}
		if m.temperature > 0 {
			params.Temperature = openai.Float(m.temperature)
		}
		if m.topP > 0 {
			params.TopP = openai.Float(m.topP)
		}
	case *GPT41Nano:
		if m.maxTokens > 0 {
			params.MaxTokens = openai.Int(int64(m.maxTokens))
		}
		if m.temperature > 0 {
			params.Temperature = openai.Float(m.temperature)
		}
		if m.topP > 0 {
			params.TopP = openai.Float(m.topP)
		}
	case *GPT35Turbo:
		if m.maxTokens > 0 {
			params.MaxTokens = openai.Int(int64(m.maxTokens))
		}
		if m.temperature > 0 {
			params.Temperature = openai.Float(m.temperature)
		}
		if m.topP > 0 {
			params.TopP = openai.Float(m.topP)
		}

	// Reasoning models
	case *O1:
		if m.maxCompletionTokens > 0 {
			params.MaxCompletionTokens = openai.Int(int64(m.maxCompletionTokens))
		}
		if m.reasoningEffort != "" {
			params.ReasoningEffort = shared.ReasoningEffort(m.reasoningEffort)
		}
	case *O1Mini:
		if m.maxCompletionTokens > 0 {
			params.MaxCompletionTokens = openai.Int(int64(m.maxCompletionTokens))
		}
		if m.reasoningEffort != "" {
			params.ReasoningEffort = shared.ReasoningEffort(m.reasoningEffort)
		}
	case *O1Pro:
		if m.maxCompletionTokens > 0 {
			params.MaxCompletionTokens = openai.Int(int64(m.maxCompletionTokens))
		}
		if m.reasoningEffort != "" {
			params.ReasoningEffort = shared.ReasoningEffort(m.reasoningEffort)
		}
	case *O3:
		if m.maxCompletionTokens > 0 {
			params.MaxCompletionTokens = openai.Int(int64(m.maxCompletionTokens))
		}
		if m.reasoningEffort != "" {
			params.ReasoningEffort = shared.ReasoningEffort(m.reasoningEffort)
		}
	case *O3Mini:
		if m.maxCompletionTokens > 0 {
			params.MaxCompletionTokens = openai.Int(int64(m.maxCompletionTokens))
		}
		if m.reasoningEffort != "" {
			params.ReasoningEffort = shared.ReasoningEffort(m.reasoningEffort)
		}
	case *O4Mini:
		if m.maxCompletionTokens > 0 {
			params.MaxCompletionTokens = openai.Int(int64(m.maxCompletionTokens))
		}
		if m.reasoningEffort != "" {
			params.ReasoningEffort = shared.ReasoningEffort(m.reasoningEffort)
		}
	case *GPT5:
		if m.maxCompletionTokens > 0 {
			params.MaxCompletionTokens = openai.Int(int64(m.maxCompletionTokens))
		}
		if m.reasoningEffort != "" {
			params.ReasoningEffort = shared.ReasoningEffort(m.reasoningEffort)
		}
	case *GPT5Mini:
		if m.maxCompletionTokens > 0 {
			params.MaxCompletionTokens = openai.Int(int64(m.maxCompletionTokens))
		}
		if m.reasoningEffort != "" {
			params.ReasoningEffort = shared.ReasoningEffort(m.reasoningEffort)
		}
	case *GPT5Nano:
		if m.maxCompletionTokens > 0 {
			params.MaxCompletionTokens = openai.Int(int64(m.maxCompletionTokens))
		}
		if m.reasoningEffort != "" {
			params.ReasoningEffort = shared.ReasoningEffort(m.reasoningEffort)
		}
	case *GPT5Pro:
		if m.maxCompletionTokens > 0 {
			params.MaxCompletionTokens = openai.Int(int64(m.maxCompletionTokens))
		}
		if m.reasoningEffort != "" {
			params.ReasoningEffort = shared.ReasoningEffort(m.reasoningEffort)
		}
	case *GPT5Turbo:
		if m.maxCompletionTokens > 0 {
			params.MaxCompletionTokens = openai.Int(int64(m.maxCompletionTokens))
		}
		if m.reasoningEffort != "" {
			params.ReasoningEffort = shared.ReasoningEffort(m.reasoningEffort)
		}
	case *GPT51:
		if m.maxCompletionTokens > 0 {
			params.MaxCompletionTokens = openai.Int(int64(m.maxCompletionTokens))
		}
		if m.reasoningEffort != "" {
			params.ReasoningEffort = shared.ReasoningEffort(m.reasoningEffort)
		}
	case *GPT51Mini:
		if m.maxCompletionTokens > 0 {
			params.MaxCompletionTokens = openai.Int(int64(m.maxCompletionTokens))
		}
		if m.reasoningEffort != "" {
			params.ReasoningEffort = shared.ReasoningEffort(m.reasoningEffort)
		}
	case *GPT51Nano:
		if m.maxCompletionTokens > 0 {
			params.MaxCompletionTokens = openai.Int(int64(m.maxCompletionTokens))
		}
		if m.reasoningEffort != "" {
			params.ReasoningEffort = shared.ReasoningEffort(m.reasoningEffort)
		}
	case *GPT51Codex:
		if m.maxCompletionTokens > 0 {
			params.MaxCompletionTokens = openai.Int(int64(m.maxCompletionTokens))
		}
		if m.reasoningEffort != "" {
			params.ReasoningEffort = shared.ReasoningEffort(m.reasoningEffort)
		}
	case *GPT51CodexMini:
		if m.maxCompletionTokens > 0 {
			params.MaxCompletionTokens = openai.Int(int64(m.maxCompletionTokens))
		}
		if m.reasoningEffort != "" {
			params.ReasoningEffort = shared.ReasoningEffort(m.reasoningEffort)
		}
	case *O3Pro:
		if m.maxCompletionTokens > 0 {
			params.MaxCompletionTokens = openai.Int(int64(m.maxCompletionTokens))
		}
		if m.reasoningEffort != "" {
			params.ReasoningEffort = shared.ReasoningEffort(m.reasoningEffort)
		}
	case *O1Preview:
		if m.maxCompletionTokens > 0 {
			params.MaxCompletionTokens = openai.Int(int64(m.maxCompletionTokens))
		}
		if m.reasoningEffort != "" {
			params.ReasoningEffort = shared.ReasoningEffort(m.reasoningEffort)
		}
	}

	c.logger.Debug().
		Str("model", model.ModelName()).
		Bool("is_reasoning_model", isReasoning).
		Msg("Making OpenAI API request")

	// Make request with rate limit handling
	var resp *openai.ChatCompletion
	err := c.rateLimiter.Execute(ctx, func() error {
		var reqErr error
		resp, reqErr = c.client.Chat.Completions.New(ctx, params)
		return reqErr
	})
	if err != nil {
		c.logger.Error().
			Err(err).
			Str("model", model.ModelName()).
			Bool("is_reasoning_model", isReasoning).
			Str("prompt_preview", truncateString(prompt, 100)).
			Msg("OpenAI generation failed")
		return nil, fmt.Errorf("OpenAI generation failed: %w", err)
	}

	if len(resp.Choices) == 0 {
		return nil, fmt.Errorf("no response choices returned from OpenAI")
	}

	choice := resp.Choices[0]

	// Build response
	response := &GenerationResponse{
		Text:         choice.Message.Content,
		Model:        resp.Model,
		FinishReason: string(choice.FinishReason),
		Usage: TokenUsage{
			PromptTokens:     int(resp.Usage.PromptTokens),
			CompletionTokens: int(resp.Usage.CompletionTokens),
			TotalTokens:      int(resp.Usage.TotalTokens),
		},
		Metadata: map[string]string{
			"provider":           "openai",
			"model":              resp.Model,
			"is_reasoning_model": fmt.Sprintf("%t", isReasoning),
		},
	}

	// Add reasoning tokens to metadata if available
	if resp.Usage.CompletionTokensDetails.ReasoningTokens > 0 {
		response.Metadata["reasoning_tokens"] = fmt.Sprintf("%d", resp.Usage.CompletionTokensDetails.ReasoningTokens)
	}

	c.logger.Debug().
		Str("model", resp.Model).
		Bool("is_reasoning_model", isReasoning).
		Int64("prompt_tokens", resp.Usage.PromptTokens).
		Int64("completion_tokens", resp.Usage.CompletionTokens).
		Int64("total_tokens", resp.Usage.TotalTokens).
		Msg("OpenAI generation completed")

	return response, nil
}

// Health checks the health of the OpenAI client
func (c *openAIClient) Health(ctx context.Context) error {
	ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	params := openai.ChatCompletionNewParams{
		Model: openai.ChatModel("gpt-4o-mini"),
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage("Hello"),
		},
		MaxTokens: openai.Int(5),
	}

	_, err := c.client.Chat.Completions.New(ctx, params)
	if err != nil {
		return fmt.Errorf("OpenAI health check failed: %w", err)
	}

	return nil
}

// Close closes the OpenAI client (no-op for OpenAI)
func (c *openAIClient) Close() error {
	return nil
}
