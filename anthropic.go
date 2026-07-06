package lingo

import (
	"context"
	"fmt"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
)

func init() {
	RegisterProvider(ProviderAnthropic, func(config ProviderConfig, logger Logger) (Provider, error) {
		cfg, ok := config.(*AnthropicConfig)
		if !ok {
			return nil, fmt.Errorf("invalid config type for Anthropic provider")
		}
		return newAnthropicClient(cfg, logger)
	})
}

// ============================================================================
// ANTHROPIC PROVIDER CONFIG
// ============================================================================

// AnthropicConfig contains configuration for the Anthropic provider
type AnthropicConfig struct {
	// APIKey is the Anthropic API key (required)
	APIKey string
	// Timeout is the request timeout (default: 60s)
	Timeout time.Duration
	// RateLimiter is the optional rate limit configuration
	RateLimiter *RateLimitConfig
}

// Implement ProviderConfig interface
func (c *AnthropicConfig) providerType() ProviderType        { return ProviderAnthropic }
func (c *AnthropicConfig) apiKey() string                    { return c.APIKey }
func (c *AnthropicConfig) timeout() time.Duration            { return c.Timeout }
func (c *AnthropicConfig) rateLimitConfig() *RateLimitConfig { return c.RateLimiter }

// ============================================================================
// SHARED OPTIONS (embedded in model structs)
// ============================================================================

// anthropicOptions contains options for standard Anthropic models
type anthropicOptions struct {
	modelVersion string // Optional: override model name with specific version (e.g., "latest")
	maxTokens    int
	temperature  float64
	topP         float64
	topK         int
	systemPrompt string
}

// anthropicThinkingOptions contains options for models that support extended thinking
type anthropicThinkingOptions struct {
	anthropicOptions
	thinkingBudget int // Must be >= 1024 and less than maxTokens
}

// ============================================================================
// STANDARD MODELS (Claude 3.5 series and earlier)
// ============================================================================

// Claude35Sonnet represents the Claude 3.5 Sonnet model.
// Deprecated: retired by Anthropic (Oct 28, 2025); the API returns 404. Migrate to ClaudeSonnet46.
// Versions: claude-3-5-sonnet-20241022, claude-3-5-sonnet-latest
type Claude35Sonnet struct{ anthropicOptions }

func (m *Claude35Sonnet) ModelName() string {
	if m.modelVersion != "" {
		return m.modelVersion
	}
	return "claude-3-5-sonnet-20241022"
}
func (m *Claude35Sonnet) Provider() ProviderType { return ProviderAnthropic }
func (m *Claude35Sonnet) SystemPrompt() string   { return m.systemPrompt }
func (m *Claude35Sonnet) supportsThinking() bool { return false }

func (m *Claude35Sonnet) WithVersion(v string) *Claude35Sonnet      { m.modelVersion = v; return m }
func (m *Claude35Sonnet) WithMaxTokens(n int) *Claude35Sonnet       { m.maxTokens = n; return m }
func (m *Claude35Sonnet) WithTemperature(t float64) *Claude35Sonnet { m.temperature = t; return m }
func (m *Claude35Sonnet) WithTopP(p float64) *Claude35Sonnet        { m.topP = p; return m }
func (m *Claude35Sonnet) WithTopK(k int) *Claude35Sonnet            { m.topK = k; return m }
func (m *Claude35Sonnet) WithSystemPrompt(s string) *Claude35Sonnet { m.systemPrompt = s; return m }

// NewClaude35Sonnet creates a new Claude 3.5 Sonnet model with default options
func NewClaude35Sonnet() *Claude35Sonnet {
	return &Claude35Sonnet{anthropicOptions{maxTokens: 4096, temperature: 1.0}}
}

// Claude35Haiku represents the Claude 3.5 Haiku model.
// Deprecated: retired by Anthropic (Feb 19, 2026); the API returns 404. Migrate to ClaudeHaiku45.
// Versions: claude-3-5-haiku-20241022, claude-3-5-haiku-latest
type Claude35Haiku struct{ anthropicOptions }

func (m *Claude35Haiku) ModelName() string {
	if m.modelVersion != "" {
		return m.modelVersion
	}
	return "claude-3-5-haiku-20241022"
}
func (m *Claude35Haiku) Provider() ProviderType { return ProviderAnthropic }
func (m *Claude35Haiku) SystemPrompt() string   { return m.systemPrompt }
func (m *Claude35Haiku) supportsThinking() bool { return false }

func (m *Claude35Haiku) WithVersion(v string) *Claude35Haiku      { m.modelVersion = v; return m }
func (m *Claude35Haiku) WithMaxTokens(n int) *Claude35Haiku       { m.maxTokens = n; return m }
func (m *Claude35Haiku) WithTemperature(t float64) *Claude35Haiku { m.temperature = t; return m }
func (m *Claude35Haiku) WithTopP(p float64) *Claude35Haiku        { m.topP = p; return m }
func (m *Claude35Haiku) WithTopK(k int) *Claude35Haiku            { m.topK = k; return m }
func (m *Claude35Haiku) WithSystemPrompt(s string) *Claude35Haiku { m.systemPrompt = s; return m }

// NewClaude35Haiku creates a new Claude 3.5 Haiku model with default options
func NewClaude35Haiku() *Claude35Haiku {
	return &Claude35Haiku{anthropicOptions{maxTokens: 4096, temperature: 1.0}}
}

// Claude3Opus represents the Claude 3 Opus model.
// Deprecated: retired by Anthropic (Jan 5, 2026); the API returns 404. Migrate to ClaudeOpus48.
// Versions: claude-3-opus-20240229, claude-3-opus-latest
type Claude3Opus struct{ anthropicOptions }

func (m *Claude3Opus) ModelName() string {
	if m.modelVersion != "" {
		return m.modelVersion
	}
	return "claude-3-opus-20240229"
}
func (m *Claude3Opus) Provider() ProviderType { return ProviderAnthropic }
func (m *Claude3Opus) SystemPrompt() string   { return m.systemPrompt }
func (m *Claude3Opus) supportsThinking() bool { return false }

func (m *Claude3Opus) WithVersion(v string) *Claude3Opus      { m.modelVersion = v; return m }
func (m *Claude3Opus) WithMaxTokens(n int) *Claude3Opus       { m.maxTokens = n; return m }
func (m *Claude3Opus) WithTemperature(t float64) *Claude3Opus { m.temperature = t; return m }
func (m *Claude3Opus) WithTopP(p float64) *Claude3Opus        { m.topP = p; return m }
func (m *Claude3Opus) WithTopK(k int) *Claude3Opus            { m.topK = k; return m }
func (m *Claude3Opus) WithSystemPrompt(s string) *Claude3Opus { m.systemPrompt = s; return m }

// NewClaude3Opus creates a new Claude 3 Opus model with default options
func NewClaude3Opus() *Claude3Opus {
	return &Claude3Opus{anthropicOptions{maxTokens: 4096, temperature: 1.0}}
}

// Claude3Haiku represents the Claude 3 Haiku model.
// Deprecated: retired by Anthropic (Apr 19, 2026); the API returns 404. Migrate to ClaudeHaiku45.
type Claude3Haiku struct{ anthropicOptions }

func (m *Claude3Haiku) ModelName() string      { return "claude-3-haiku-20240307" }
func (m *Claude3Haiku) Provider() ProviderType { return ProviderAnthropic }
func (m *Claude3Haiku) SystemPrompt() string   { return m.systemPrompt }
func (m *Claude3Haiku) supportsThinking() bool { return false }

func (m *Claude3Haiku) WithMaxTokens(n int) *Claude3Haiku       { m.maxTokens = n; return m }
func (m *Claude3Haiku) WithTemperature(t float64) *Claude3Haiku { m.temperature = t; return m }
func (m *Claude3Haiku) WithTopP(p float64) *Claude3Haiku        { m.topP = p; return m }
func (m *Claude3Haiku) WithTopK(k int) *Claude3Haiku            { m.topK = k; return m }
func (m *Claude3Haiku) WithSystemPrompt(s string) *Claude3Haiku { m.systemPrompt = s; return m }

// NewClaude3Haiku creates a new Claude 3 Haiku model with default options
func NewClaude3Haiku() *Claude3Haiku {
	return &Claude3Haiku{anthropicOptions{maxTokens: 4096, temperature: 1.0}}
}

// Claude3Sonnet represents the Claude 3 Sonnet model.
// Deprecated: retired by Anthropic (Jul 21, 2025); the API returns 404. Migrate to ClaudeSonnet46.
type Claude3Sonnet struct{ anthropicOptions }

func (m *Claude3Sonnet) ModelName() string      { return "claude-3-sonnet-20240229" }
func (m *Claude3Sonnet) Provider() ProviderType { return ProviderAnthropic }
func (m *Claude3Sonnet) SystemPrompt() string   { return m.systemPrompt }
func (m *Claude3Sonnet) supportsThinking() bool { return false }

func (m *Claude3Sonnet) WithMaxTokens(n int) *Claude3Sonnet       { m.maxTokens = n; return m }
func (m *Claude3Sonnet) WithTemperature(t float64) *Claude3Sonnet { m.temperature = t; return m }
func (m *Claude3Sonnet) WithTopP(p float64) *Claude3Sonnet        { m.topP = p; return m }
func (m *Claude3Sonnet) WithTopK(k int) *Claude3Sonnet            { m.topK = k; return m }
func (m *Claude3Sonnet) WithSystemPrompt(s string) *Claude3Sonnet { m.systemPrompt = s; return m }

// NewClaude3Sonnet creates a new Claude 3 Sonnet model with default options
func NewClaude3Sonnet() *Claude3Sonnet {
	return &Claude3Sonnet{anthropicOptions{maxTokens: 4096, temperature: 1.0}}
}

// ============================================================================
// EXTENDED THINKING MODELS (Claude 3.7+, Claude 4+)
// ============================================================================

// Claude37Sonnet represents the Claude 3.7 Sonnet model (supports extended thinking).
// Deprecated: retired by Anthropic (Feb 19, 2026); the API returns 404. Migrate to ClaudeSonnet46.
// Versions: claude-3-7-sonnet-20250219, claude-3-7-sonnet-latest
type Claude37Sonnet struct{ anthropicThinkingOptions }

func (m *Claude37Sonnet) ModelName() string {
	if m.modelVersion != "" {
		return m.modelVersion
	}
	return "claude-3-7-sonnet-20250219"
}
func (m *Claude37Sonnet) Provider() ProviderType { return ProviderAnthropic }
func (m *Claude37Sonnet) SystemPrompt() string   { return m.systemPrompt }
func (m *Claude37Sonnet) supportsThinking() bool { return true }

func (m *Claude37Sonnet) WithVersion(v string) *Claude37Sonnet      { m.modelVersion = v; return m }
func (m *Claude37Sonnet) WithMaxTokens(n int) *Claude37Sonnet       { m.maxTokens = n; return m }
func (m *Claude37Sonnet) WithTemperature(t float64) *Claude37Sonnet { m.temperature = t; return m }
func (m *Claude37Sonnet) WithTopP(p float64) *Claude37Sonnet        { m.topP = p; return m }
func (m *Claude37Sonnet) WithTopK(k int) *Claude37Sonnet            { m.topK = k; return m }
func (m *Claude37Sonnet) WithSystemPrompt(s string) *Claude37Sonnet { m.systemPrompt = s; return m }
func (m *Claude37Sonnet) WithThinkingBudget(n int) *Claude37Sonnet  { m.thinkingBudget = n; return m }

// NewClaude37Sonnet creates a new Claude 3.7 Sonnet model with default options
func NewClaude37Sonnet() *Claude37Sonnet {
	return &Claude37Sonnet{anthropicThinkingOptions{
		anthropicOptions: anthropicOptions{maxTokens: 8192, temperature: 1.0},
	}}
}

// ClaudeSonnet4 represents the Claude Sonnet 4 model (supports extended thinking)
type ClaudeSonnet4 struct{ anthropicThinkingOptions }

func (m *ClaudeSonnet4) ModelName() string      { return "claude-sonnet-4-20250514" }
func (m *ClaudeSonnet4) Provider() ProviderType { return ProviderAnthropic }
func (m *ClaudeSonnet4) SystemPrompt() string   { return m.systemPrompt }
func (m *ClaudeSonnet4) supportsThinking() bool { return true }

func (m *ClaudeSonnet4) WithMaxTokens(n int) *ClaudeSonnet4       { m.maxTokens = n; return m }
func (m *ClaudeSonnet4) WithTemperature(t float64) *ClaudeSonnet4 { m.temperature = t; return m }
func (m *ClaudeSonnet4) WithTopP(p float64) *ClaudeSonnet4        { m.topP = p; return m }
func (m *ClaudeSonnet4) WithTopK(k int) *ClaudeSonnet4            { m.topK = k; return m }
func (m *ClaudeSonnet4) WithSystemPrompt(s string) *ClaudeSonnet4 { m.systemPrompt = s; return m }
func (m *ClaudeSonnet4) WithThinkingBudget(n int) *ClaudeSonnet4  { m.thinkingBudget = n; return m }

// NewClaudeSonnet4 creates a new Claude Sonnet 4 model with default options
func NewClaudeSonnet4() *ClaudeSonnet4 {
	return &ClaudeSonnet4{anthropicThinkingOptions{
		anthropicOptions: anthropicOptions{maxTokens: 8192, temperature: 1.0},
	}}
}

// ClaudeOpus4 represents the Claude Opus 4 model (supports extended thinking)
type ClaudeOpus4 struct{ anthropicThinkingOptions }

func (m *ClaudeOpus4) ModelName() string      { return "claude-opus-4-20250514" }
func (m *ClaudeOpus4) Provider() ProviderType { return ProviderAnthropic }
func (m *ClaudeOpus4) SystemPrompt() string   { return m.systemPrompt }
func (m *ClaudeOpus4) supportsThinking() bool { return true }

func (m *ClaudeOpus4) WithMaxTokens(n int) *ClaudeOpus4       { m.maxTokens = n; return m }
func (m *ClaudeOpus4) WithTemperature(t float64) *ClaudeOpus4 { m.temperature = t; return m }
func (m *ClaudeOpus4) WithTopP(p float64) *ClaudeOpus4        { m.topP = p; return m }
func (m *ClaudeOpus4) WithTopK(k int) *ClaudeOpus4            { m.topK = k; return m }
func (m *ClaudeOpus4) WithSystemPrompt(s string) *ClaudeOpus4 { m.systemPrompt = s; return m }
func (m *ClaudeOpus4) WithThinkingBudget(n int) *ClaudeOpus4  { m.thinkingBudget = n; return m }

// NewClaudeOpus4 creates a new Claude Opus 4 model with default options
func NewClaudeOpus4() *ClaudeOpus4 {
	return &ClaudeOpus4{anthropicThinkingOptions{
		anthropicOptions: anthropicOptions{maxTokens: 8192, temperature: 1.0},
	}}
}

// ClaudeSonnet45 represents the Claude Sonnet 4.5 model (supports extended thinking)
type ClaudeSonnet45 struct{ anthropicThinkingOptions }

func (m *ClaudeSonnet45) ModelName() string      { return "claude-sonnet-4-5-20250929" }
func (m *ClaudeSonnet45) Provider() ProviderType { return ProviderAnthropic }
func (m *ClaudeSonnet45) SystemPrompt() string   { return m.systemPrompt }
func (m *ClaudeSonnet45) supportsThinking() bool { return true }

func (m *ClaudeSonnet45) WithMaxTokens(n int) *ClaudeSonnet45       { m.maxTokens = n; return m }
func (m *ClaudeSonnet45) WithTemperature(t float64) *ClaudeSonnet45 { m.temperature = t; return m }
func (m *ClaudeSonnet45) WithTopP(p float64) *ClaudeSonnet45        { m.topP = p; return m }
func (m *ClaudeSonnet45) WithTopK(k int) *ClaudeSonnet45            { m.topK = k; return m }
func (m *ClaudeSonnet45) WithSystemPrompt(s string) *ClaudeSonnet45 { m.systemPrompt = s; return m }
func (m *ClaudeSonnet45) WithThinkingBudget(n int) *ClaudeSonnet45  { m.thinkingBudget = n; return m }

// NewClaudeSonnet45 creates a new Claude Sonnet 4.5 model with default options
func NewClaudeSonnet45() *ClaudeSonnet45 {
	return &ClaudeSonnet45{anthropicThinkingOptions{
		anthropicOptions: anthropicOptions{maxTokens: 8192, temperature: 1.0},
	}}
}

// ClaudeOpus45 represents the Claude Opus 4.5 model (supports extended thinking)
type ClaudeOpus45 struct{ anthropicThinkingOptions }

func (m *ClaudeOpus45) ModelName() string      { return "claude-opus-4-5-20251101" }
func (m *ClaudeOpus45) Provider() ProviderType { return ProviderAnthropic }
func (m *ClaudeOpus45) SystemPrompt() string   { return m.systemPrompt }
func (m *ClaudeOpus45) supportsThinking() bool { return true }

func (m *ClaudeOpus45) WithMaxTokens(n int) *ClaudeOpus45       { m.maxTokens = n; return m }
func (m *ClaudeOpus45) WithTemperature(t float64) *ClaudeOpus45 { m.temperature = t; return m }
func (m *ClaudeOpus45) WithTopP(p float64) *ClaudeOpus45        { m.topP = p; return m }
func (m *ClaudeOpus45) WithTopK(k int) *ClaudeOpus45            { m.topK = k; return m }
func (m *ClaudeOpus45) WithSystemPrompt(s string) *ClaudeOpus45 { m.systemPrompt = s; return m }
func (m *ClaudeOpus45) WithThinkingBudget(n int) *ClaudeOpus45  { m.thinkingBudget = n; return m }

// NewClaudeOpus45 creates a new Claude Opus 4.5 model with default options
func NewClaudeOpus45() *ClaudeOpus45 {
	return &ClaudeOpus45{anthropicThinkingOptions{
		anthropicOptions: anthropicOptions{maxTokens: 8192, temperature: 1.0},
	}}
}

// ClaudeHaiku45 represents the Claude Haiku 4.5 model (supports extended thinking)
type ClaudeHaiku45 struct{ anthropicThinkingOptions }

func (m *ClaudeHaiku45) ModelName() string      { return "claude-haiku-4-5-20251001" }
func (m *ClaudeHaiku45) Provider() ProviderType { return ProviderAnthropic }
func (m *ClaudeHaiku45) SystemPrompt() string   { return m.systemPrompt }
func (m *ClaudeHaiku45) supportsThinking() bool { return true }

func (m *ClaudeHaiku45) WithMaxTokens(n int) *ClaudeHaiku45       { m.maxTokens = n; return m }
func (m *ClaudeHaiku45) WithTemperature(t float64) *ClaudeHaiku45 { m.temperature = t; return m }
func (m *ClaudeHaiku45) WithTopP(p float64) *ClaudeHaiku45        { m.topP = p; return m }
func (m *ClaudeHaiku45) WithTopK(k int) *ClaudeHaiku45            { m.topK = k; return m }
func (m *ClaudeHaiku45) WithSystemPrompt(s string) *ClaudeHaiku45 { m.systemPrompt = s; return m }
func (m *ClaudeHaiku45) WithThinkingBudget(n int) *ClaudeHaiku45  { m.thinkingBudget = n; return m }

// NewClaudeHaiku45 creates a new Claude Haiku 4.5 model with default options
func NewClaudeHaiku45() *ClaudeHaiku45 {
	return &ClaudeHaiku45{anthropicThinkingOptions{
		anthropicOptions: anthropicOptions{maxTokens: 8192, temperature: 1.0},
	}}
}

// ClaudeOpus41 represents the Claude Opus 4.1 model (supports extended thinking)
// Versions: claude-opus-4-1-20250805, claude-opus-4-1
type ClaudeOpus41 struct{ anthropicThinkingOptions }

func (m *ClaudeOpus41) ModelName() string {
	if m.modelVersion != "" {
		return m.modelVersion
	}
	return "claude-opus-4-1-20250805"
}
func (m *ClaudeOpus41) Provider() ProviderType { return ProviderAnthropic }
func (m *ClaudeOpus41) SystemPrompt() string   { return m.systemPrompt }
func (m *ClaudeOpus41) supportsThinking() bool { return true }

func (m *ClaudeOpus41) WithVersion(v string) *ClaudeOpus41      { m.modelVersion = v; return m }
func (m *ClaudeOpus41) WithMaxTokens(n int) *ClaudeOpus41       { m.maxTokens = n; return m }
func (m *ClaudeOpus41) WithTemperature(t float64) *ClaudeOpus41 { m.temperature = t; return m }
func (m *ClaudeOpus41) WithTopP(p float64) *ClaudeOpus41        { m.topP = p; return m }
func (m *ClaudeOpus41) WithTopK(k int) *ClaudeOpus41            { m.topK = k; return m }
func (m *ClaudeOpus41) WithSystemPrompt(s string) *ClaudeOpus41 { m.systemPrompt = s; return m }
func (m *ClaudeOpus41) WithThinkingBudget(n int) *ClaudeOpus41  { m.thinkingBudget = n; return m }

// NewClaudeOpus41 creates a new Claude Opus 4.1 model with default options
func NewClaudeOpus41() *ClaudeOpus41 {
	return &ClaudeOpus41{anthropicThinkingOptions{
		anthropicOptions: anthropicOptions{maxTokens: 8192, temperature: 1.0},
	}}
}

// ClaudeOpus46 represents the Claude Opus 4.6 model (supports extended thinking)
// This is the current recommended model for complex tasks.
type ClaudeOpus46 struct{ anthropicThinkingOptions }

func (m *ClaudeOpus46) ModelName() string      { return "claude-opus-4-6" }
func (m *ClaudeOpus46) Provider() ProviderType { return ProviderAnthropic }
func (m *ClaudeOpus46) SystemPrompt() string   { return m.systemPrompt }
func (m *ClaudeOpus46) supportsThinking() bool { return true }

func (m *ClaudeOpus46) WithMaxTokens(n int) *ClaudeOpus46       { m.maxTokens = n; return m }
func (m *ClaudeOpus46) WithTemperature(t float64) *ClaudeOpus46 { m.temperature = t; return m }
func (m *ClaudeOpus46) WithTopP(p float64) *ClaudeOpus46        { m.topP = p; return m }
func (m *ClaudeOpus46) WithTopK(k int) *ClaudeOpus46            { m.topK = k; return m }
func (m *ClaudeOpus46) WithSystemPrompt(s string) *ClaudeOpus46 { m.systemPrompt = s; return m }
func (m *ClaudeOpus46) WithThinkingBudget(n int) *ClaudeOpus46  { m.thinkingBudget = n; return m }

// NewClaudeOpus46 creates a new Claude Opus 4.6 model with default options
func NewClaudeOpus46() *ClaudeOpus46 {
	return &ClaudeOpus46{anthropicThinkingOptions{
		anthropicOptions: anthropicOptions{maxTokens: 8192, temperature: 1.0},
	}}
}

// ClaudeSonnet46 represents the Claude Sonnet 4.6 model (supports extended thinking)
// This is the current recommended model for speed/intelligence balance.
type ClaudeSonnet46 struct{ anthropicThinkingOptions }

func (m *ClaudeSonnet46) ModelName() string      { return "claude-sonnet-4-6" }
func (m *ClaudeSonnet46) Provider() ProviderType { return ProviderAnthropic }
func (m *ClaudeSonnet46) SystemPrompt() string   { return m.systemPrompt }
func (m *ClaudeSonnet46) supportsThinking() bool { return true }

func (m *ClaudeSonnet46) WithMaxTokens(n int) *ClaudeSonnet46       { m.maxTokens = n; return m }
func (m *ClaudeSonnet46) WithTemperature(t float64) *ClaudeSonnet46 { m.temperature = t; return m }
func (m *ClaudeSonnet46) WithTopP(p float64) *ClaudeSonnet46        { m.topP = p; return m }
func (m *ClaudeSonnet46) WithTopK(k int) *ClaudeSonnet46            { m.topK = k; return m }
func (m *ClaudeSonnet46) WithSystemPrompt(s string) *ClaudeSonnet46 { m.systemPrompt = s; return m }
func (m *ClaudeSonnet46) WithThinkingBudget(n int) *ClaudeSonnet46  { m.thinkingBudget = n; return m }

// NewClaudeSonnet46 creates a new Claude Sonnet 4.6 model with default options
func NewClaudeSonnet46() *ClaudeSonnet46 {
	return &ClaudeSonnet46{anthropicThinkingOptions{
		anthropicOptions: anthropicOptions{maxTokens: 8192, temperature: 1.0},
	}}
}

// ClaudeOpus47 represents the Claude Opus 4.7 model (supports adaptive thinking)
// Opus 4.7 is adaptive-thinking only: sampling parameters (temperature/topP/topK)
// and fixed thinking budgets are rejected by the API, so they are left unset by default.
type ClaudeOpus47 struct{ anthropicThinkingOptions }

func (m *ClaudeOpus47) ModelName() string      { return "claude-opus-4-7" }
func (m *ClaudeOpus47) Provider() ProviderType { return ProviderAnthropic }
func (m *ClaudeOpus47) SystemPrompt() string   { return m.systemPrompt }
func (m *ClaudeOpus47) supportsThinking() bool { return true }

func (m *ClaudeOpus47) WithMaxTokens(n int) *ClaudeOpus47       { m.maxTokens = n; return m }
func (m *ClaudeOpus47) WithTemperature(t float64) *ClaudeOpus47 { m.temperature = t; return m }
func (m *ClaudeOpus47) WithTopP(p float64) *ClaudeOpus47        { m.topP = p; return m }
func (m *ClaudeOpus47) WithTopK(k int) *ClaudeOpus47            { m.topK = k; return m }
func (m *ClaudeOpus47) WithSystemPrompt(s string) *ClaudeOpus47 { m.systemPrompt = s; return m }
func (m *ClaudeOpus47) WithThinkingBudget(n int) *ClaudeOpus47  { m.thinkingBudget = n; return m }

// NewClaudeOpus47 creates a new Claude Opus 4.7 model with default options
func NewClaudeOpus47() *ClaudeOpus47 {
	return &ClaudeOpus47{anthropicThinkingOptions{
		anthropicOptions: anthropicOptions{maxTokens: 8192},
	}}
}

// ClaudeOpus48 represents the Claude Opus 4.8 model (supports adaptive thinking)
// This is the current recommended Opus-tier model for complex and long-horizon tasks.
// Opus 4.8 is adaptive-thinking only: sampling parameters (temperature/topP/topK)
// and fixed thinking budgets are rejected by the API, so they are left unset by default.
type ClaudeOpus48 struct{ anthropicThinkingOptions }

func (m *ClaudeOpus48) ModelName() string      { return "claude-opus-4-8" }
func (m *ClaudeOpus48) Provider() ProviderType { return ProviderAnthropic }
func (m *ClaudeOpus48) SystemPrompt() string   { return m.systemPrompt }
func (m *ClaudeOpus48) supportsThinking() bool { return true }

func (m *ClaudeOpus48) WithMaxTokens(n int) *ClaudeOpus48       { m.maxTokens = n; return m }
func (m *ClaudeOpus48) WithTemperature(t float64) *ClaudeOpus48 { m.temperature = t; return m }
func (m *ClaudeOpus48) WithTopP(p float64) *ClaudeOpus48        { m.topP = p; return m }
func (m *ClaudeOpus48) WithTopK(k int) *ClaudeOpus48            { m.topK = k; return m }
func (m *ClaudeOpus48) WithSystemPrompt(s string) *ClaudeOpus48 { m.systemPrompt = s; return m }
func (m *ClaudeOpus48) WithThinkingBudget(n int) *ClaudeOpus48  { m.thinkingBudget = n; return m }

// NewClaudeOpus48 creates a new Claude Opus 4.8 model with default options
func NewClaudeOpus48() *ClaudeOpus48 {
	return &ClaudeOpus48{anthropicThinkingOptions{
		anthropicOptions: anthropicOptions{maxTokens: 8192},
	}}
}

// ClaudeFable5 represents the Claude Fable 5 model (supports adaptive thinking)
// Fable 5 is Anthropic's most capable widely released model. Thinking is always on,
// sampling parameters are rejected, and it uses a different tokenizer (~30% more
// tokens for the same content), so they are left unset by default.
type ClaudeFable5 struct{ anthropicThinkingOptions }

func (m *ClaudeFable5) ModelName() string      { return "claude-fable-5" }
func (m *ClaudeFable5) Provider() ProviderType { return ProviderAnthropic }
func (m *ClaudeFable5) SystemPrompt() string   { return m.systemPrompt }
func (m *ClaudeFable5) supportsThinking() bool { return true }

func (m *ClaudeFable5) WithMaxTokens(n int) *ClaudeFable5       { m.maxTokens = n; return m }
func (m *ClaudeFable5) WithTemperature(t float64) *ClaudeFable5 { m.temperature = t; return m }
func (m *ClaudeFable5) WithTopP(p float64) *ClaudeFable5        { m.topP = p; return m }
func (m *ClaudeFable5) WithTopK(k int) *ClaudeFable5            { m.topK = k; return m }
func (m *ClaudeFable5) WithSystemPrompt(s string) *ClaudeFable5 { m.systemPrompt = s; return m }
func (m *ClaudeFable5) WithThinkingBudget(n int) *ClaudeFable5  { m.thinkingBudget = n; return m }

// NewClaudeFable5 creates a new Claude Fable 5 model with default options
func NewClaudeFable5() *ClaudeFable5 {
	return &ClaudeFable5{anthropicThinkingOptions{
		anthropicOptions: anthropicOptions{maxTokens: 8192},
	}}
}

// ============================================================================
// ANTHROPIC PROVIDER CLIENT
// ============================================================================

// anthropicThinkingModel is an interface for models that support extended thinking
type anthropicThinkingModel interface {
	Model
	supportsThinking() bool
}

// anthropicClient implements the Provider interface for Anthropic
type anthropicClient struct {
	client      anthropic.Client
	timeout     time.Duration
	logger      Logger
	rateLimiter *rateLimiter
}

// newAnthropicClient creates a new Anthropic client using the official SDK
func newAnthropicClient(config *AnthropicConfig, logger Logger) (*anthropicClient, error) {
	if config.APIKey == "" {
		return nil, fmt.Errorf("anthropic API key is required")
	}

	client := anthropic.NewClient(option.WithAPIKey(config.APIKey))

	timeout := config.Timeout
	if timeout == 0 {
		timeout = defaultTimeout()
	}

	return &anthropicClient{
		client:      client,
		timeout:     timeout,
		logger:      logger,
		rateLimiter: newRateLimiter(config.RateLimiter, logger),
	}, nil
}

// Generate generates text using Anthropic's API
func (c *anthropicClient) Generate(ctx context.Context, model Model, prompt string) (*GenerationResponse, error) {
	// Verify model is for Anthropic
	if model.Provider() != ProviderAnthropic {
		return nil, fmt.Errorf("model %s is not an Anthropic model", model.ModelName())
	}

	// Set timeout
	ctx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()

	// Build request parameters
	params := anthropic.MessageNewParams{
		Model:     anthropic.Model(model.ModelName()),
		MaxTokens: int64(4096), // Default
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock(prompt)),
		},
	}

	// Add system prompt if provided
	if model.SystemPrompt() != "" {
		params.System = []anthropic.TextBlockParam{
			{Text: model.SystemPrompt()},
		}
	}

	// Apply options based on model type
	var hasThinking bool
	switch m := model.(type) {
	// Standard models
	case *Claude35Sonnet:
		if m.maxTokens > 0 {
			params.MaxTokens = int64(m.maxTokens)
		}
		if m.temperature > 0 {
			params.Temperature = anthropic.Float(m.temperature)
		}
		if m.topP > 0 {
			params.TopP = anthropic.Float(m.topP)
		}
		if m.topK > 0 {
			params.TopK = anthropic.Int(int64(m.topK))
		}
	case *Claude35Haiku:
		if m.maxTokens > 0 {
			params.MaxTokens = int64(m.maxTokens)
		}
		if m.temperature > 0 {
			params.Temperature = anthropic.Float(m.temperature)
		}
		if m.topP > 0 {
			params.TopP = anthropic.Float(m.topP)
		}
		if m.topK > 0 {
			params.TopK = anthropic.Int(int64(m.topK))
		}
	case *Claude3Opus:
		if m.maxTokens > 0 {
			params.MaxTokens = int64(m.maxTokens)
		}
		if m.temperature > 0 {
			params.Temperature = anthropic.Float(m.temperature)
		}
		if m.topP > 0 {
			params.TopP = anthropic.Float(m.topP)
		}
		if m.topK > 0 {
			params.TopK = anthropic.Int(int64(m.topK))
		}
	case *Claude3Haiku:
		if m.maxTokens > 0 {
			params.MaxTokens = int64(m.maxTokens)
		}
		if m.temperature > 0 {
			params.Temperature = anthropic.Float(m.temperature)
		}
		if m.topP > 0 {
			params.TopP = anthropic.Float(m.topP)
		}
		if m.topK > 0 {
			params.TopK = anthropic.Int(int64(m.topK))
		}
	case *Claude3Sonnet:
		if m.maxTokens > 0 {
			params.MaxTokens = int64(m.maxTokens)
		}
		if m.temperature > 0 {
			params.Temperature = anthropic.Float(m.temperature)
		}
		if m.topP > 0 {
			params.TopP = anthropic.Float(m.topP)
		}
		if m.topK > 0 {
			params.TopK = anthropic.Int(int64(m.topK))
		}

	// Extended thinking models
	case *Claude37Sonnet:
		if m.maxTokens > 0 {
			params.MaxTokens = int64(m.maxTokens)
		}
		if m.temperature > 0 {
			params.Temperature = anthropic.Float(m.temperature)
		}
		if m.topP > 0 {
			params.TopP = anthropic.Float(m.topP)
		}
		if m.topK > 0 {
			params.TopK = anthropic.Int(int64(m.topK))
		}
		if m.thinkingBudget > 0 {
			hasThinking = true
			params.Thinking = anthropic.ThinkingConfigParamOfEnabled(int64(m.thinkingBudget))
		}
	case *ClaudeSonnet4:
		if m.maxTokens > 0 {
			params.MaxTokens = int64(m.maxTokens)
		}
		if m.temperature > 0 {
			params.Temperature = anthropic.Float(m.temperature)
		}
		if m.topP > 0 {
			params.TopP = anthropic.Float(m.topP)
		}
		if m.topK > 0 {
			params.TopK = anthropic.Int(int64(m.topK))
		}
		if m.thinkingBudget > 0 {
			hasThinking = true
			params.Thinking = anthropic.ThinkingConfigParamOfEnabled(int64(m.thinkingBudget))
		}
	case *ClaudeOpus4:
		if m.maxTokens > 0 {
			params.MaxTokens = int64(m.maxTokens)
		}
		if m.temperature > 0 {
			params.Temperature = anthropic.Float(m.temperature)
		}
		if m.topP > 0 {
			params.TopP = anthropic.Float(m.topP)
		}
		if m.topK > 0 {
			params.TopK = anthropic.Int(int64(m.topK))
		}
		if m.thinkingBudget > 0 {
			hasThinking = true
			params.Thinking = anthropic.ThinkingConfigParamOfEnabled(int64(m.thinkingBudget))
		}
	case *ClaudeSonnet45:
		if m.maxTokens > 0 {
			params.MaxTokens = int64(m.maxTokens)
		}
		if m.temperature > 0 {
			params.Temperature = anthropic.Float(m.temperature)
		}
		if m.topP > 0 {
			params.TopP = anthropic.Float(m.topP)
		}
		if m.topK > 0 {
			params.TopK = anthropic.Int(int64(m.topK))
		}
		if m.thinkingBudget > 0 {
			hasThinking = true
			params.Thinking = anthropic.ThinkingConfigParamOfEnabled(int64(m.thinkingBudget))
		}
	case *ClaudeOpus45:
		if m.maxTokens > 0 {
			params.MaxTokens = int64(m.maxTokens)
		}
		if m.temperature > 0 {
			params.Temperature = anthropic.Float(m.temperature)
		}
		if m.topP > 0 {
			params.TopP = anthropic.Float(m.topP)
		}
		if m.topK > 0 {
			params.TopK = anthropic.Int(int64(m.topK))
		}
		if m.thinkingBudget > 0 {
			hasThinking = true
			params.Thinking = anthropic.ThinkingConfigParamOfEnabled(int64(m.thinkingBudget))
		}
	case *ClaudeHaiku45:
		if m.maxTokens > 0 {
			params.MaxTokens = int64(m.maxTokens)
		}
		if m.temperature > 0 {
			params.Temperature = anthropic.Float(m.temperature)
		}
		if m.topP > 0 {
			params.TopP = anthropic.Float(m.topP)
		}
		if m.topK > 0 {
			params.TopK = anthropic.Int(int64(m.topK))
		}
		if m.thinkingBudget > 0 {
			hasThinking = true
			params.Thinking = anthropic.ThinkingConfigParamOfEnabled(int64(m.thinkingBudget))
		}
	case *ClaudeOpus41:
		if m.maxTokens > 0 {
			params.MaxTokens = int64(m.maxTokens)
		}
		if m.temperature > 0 {
			params.Temperature = anthropic.Float(m.temperature)
		}
		if m.topP > 0 {
			params.TopP = anthropic.Float(m.topP)
		}
		if m.topK > 0 {
			params.TopK = anthropic.Int(int64(m.topK))
		}
		if m.thinkingBudget > 0 {
			hasThinking = true
			params.Thinking = anthropic.ThinkingConfigParamOfEnabled(int64(m.thinkingBudget))
		}
	case *ClaudeOpus46:
		if m.maxTokens > 0 {
			params.MaxTokens = int64(m.maxTokens)
		}
		if m.temperature > 0 {
			params.Temperature = anthropic.Float(m.temperature)
		}
		if m.topP > 0 {
			params.TopP = anthropic.Float(m.topP)
		}
		if m.topK > 0 {
			params.TopK = anthropic.Int(int64(m.topK))
		}
		if m.thinkingBudget > 0 {
			hasThinking = true
			params.Thinking = anthropic.ThinkingConfigParamOfEnabled(int64(m.thinkingBudget))
		}
	case *ClaudeSonnet46:
		if m.maxTokens > 0 {
			params.MaxTokens = int64(m.maxTokens)
		}
		if m.temperature > 0 {
			params.Temperature = anthropic.Float(m.temperature)
		}
		if m.topP > 0 {
			params.TopP = anthropic.Float(m.topP)
		}
		if m.topK > 0 {
			params.TopK = anthropic.Int(int64(m.topK))
		}
		if m.thinkingBudget > 0 {
			hasThinking = true
			params.Thinking = anthropic.ThinkingConfigParamOfEnabled(int64(m.thinkingBudget))
		}
	case *ClaudeOpus47:
		if m.maxTokens > 0 {
			params.MaxTokens = int64(m.maxTokens)
		}
		if m.temperature > 0 {
			params.Temperature = anthropic.Float(m.temperature)
		}
		if m.topP > 0 {
			params.TopP = anthropic.Float(m.topP)
		}
		if m.topK > 0 {
			params.TopK = anthropic.Int(int64(m.topK))
		}
		if m.thinkingBudget > 0 {
			hasThinking = true
			params.Thinking = anthropic.ThinkingConfigParamOfEnabled(int64(m.thinkingBudget))
		}
	case *ClaudeOpus48:
		if m.maxTokens > 0 {
			params.MaxTokens = int64(m.maxTokens)
		}
		if m.temperature > 0 {
			params.Temperature = anthropic.Float(m.temperature)
		}
		if m.topP > 0 {
			params.TopP = anthropic.Float(m.topP)
		}
		if m.topK > 0 {
			params.TopK = anthropic.Int(int64(m.topK))
		}
		if m.thinkingBudget > 0 {
			hasThinking = true
			params.Thinking = anthropic.ThinkingConfigParamOfEnabled(int64(m.thinkingBudget))
		}
	case *ClaudeFable5:
		if m.maxTokens > 0 {
			params.MaxTokens = int64(m.maxTokens)
		}
		if m.temperature > 0 {
			params.Temperature = anthropic.Float(m.temperature)
		}
		if m.topP > 0 {
			params.TopP = anthropic.Float(m.topP)
		}
		if m.topK > 0 {
			params.TopK = anthropic.Int(int64(m.topK))
		}
		if m.thinkingBudget > 0 {
			hasThinking = true
			params.Thinking = anthropic.ThinkingConfigParamOfEnabled(int64(m.thinkingBudget))
		}
	}

	c.logger.Debug().
		Str("model", model.ModelName()).
		Bool("has_thinking", hasThinking).
		Msg("Making Anthropic API request")

	// Make request with rate limit handling
	var resp *anthropic.Message
	err := c.rateLimiter.Execute(ctx, func() error {
		var reqErr error
		resp, reqErr = c.client.Messages.New(ctx, params)
		return reqErr
	})
	if err != nil {
		c.logger.Error().
			Err(err).
			Str("model", model.ModelName()).
			Str("prompt_preview", truncateString(prompt, 100)).
			Msg("Anthropic generation failed")
		return nil, fmt.Errorf("anthropic generation failed: %w", err)
	}

	if len(resp.Content) == 0 {
		return nil, fmt.Errorf("no response content returned from Anthropic")
	}

	// Extract text content and thinking content
	var text string
	var thinkingText string
	for _, block := range resp.Content {
		switch block.Type {
		case "text":
			text = block.Text
		case "thinking":
			thinkingText = block.Thinking
		}
	}

	if text == "" {
		return nil, fmt.Errorf("no text content found in Anthropic response")
	}

	// Build response
	result := &GenerationResponse{
		Text:         text,
		Model:        string(resp.Model),
		FinishReason: string(resp.StopReason),
		Usage: TokenUsage{
			PromptTokens:     int(resp.Usage.InputTokens),
			CompletionTokens: int(resp.Usage.OutputTokens),
			TotalTokens:      int(resp.Usage.InputTokens + resp.Usage.OutputTokens),
		},
		Metadata: map[string]string{
			"provider": "anthropic",
			"model":    string(resp.Model),
		},
	}

	// Add thinking content to metadata if present
	if thinkingText != "" {
		result.Metadata["thinking"] = thinkingText
	}

	c.logger.Debug().
		Str("model", string(resp.Model)).
		Int64("input_tokens", resp.Usage.InputTokens).
		Int64("output_tokens", resp.Usage.OutputTokens).
		Int64("total_tokens", resp.Usage.InputTokens+resp.Usage.OutputTokens).
		Bool("has_thinking", thinkingText != "").
		Msg("Anthropic generation completed")

	return result, nil
}

// Health checks the health of the Anthropic client
func (c *anthropicClient) Health(ctx context.Context) error {
	ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	params := anthropic.MessageNewParams{
		Model:     anthropic.Model("claude-haiku-4-5"),
		MaxTokens: int64(5),
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock("Hello")),
		},
	}

	_, err := c.client.Messages.New(ctx, params)
	if err != nil {
		return fmt.Errorf("anthropic health check failed: %w", err)
	}

	return nil
}

// Close closes the Anthropic client (no-op for Anthropic)
func (c *anthropicClient) Close() error {
	return nil
}
