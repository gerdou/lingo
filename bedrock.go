package lingo

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
)

func init() {
	RegisterProvider(ProviderBedrock, func(cfg ProviderConfig, logger Logger) (Provider, error) {
		bedrockCfg, ok := cfg.(*BedrockConfig)
		if !ok {
			return nil, fmt.Errorf("invalid config type for Bedrock provider")
		}
		return newBedrockClient(bedrockCfg, logger)
	})
}

// ============================================================================
// BEDROCK PROVIDER CONFIG
// ============================================================================

// BedrockConfig contains configuration for the AWS Bedrock provider
type BedrockConfig struct {
	// Region is the AWS region (required, e.g., "us-east-1")
	Region string
	// Profile is the AWS profile name from ~/.aws/credentials or ~/.aws/config (optional)
	Profile string
	// AccessKeyID is the AWS access key ID (optional if using IAM roles, environment, or profile)
	AccessKeyID string
	// SecretAccessKey is the AWS secret access key (optional if using IAM roles, environment, or profile)
	SecretAccessKey string
	// SessionToken is the AWS session token for temporary credentials (optional)
	SessionToken string
	// Timeout is the request timeout (default: 60s)
	Timeout time.Duration
	// RateLimiter is the optional rate limit configuration
	RateLimiter *RateLimitConfig
}

// Implement ProviderConfig interface
func (c *BedrockConfig) providerType() ProviderType        { return ProviderBedrock }
func (c *BedrockConfig) apiKey() string                    { return c.AccessKeyID } // Not used directly
func (c *BedrockConfig) timeout() time.Duration            { return c.Timeout }
func (c *BedrockConfig) rateLimitConfig() *RateLimitConfig { return c.RateLimiter }

// ============================================================================
// SHARED OPTIONS (embedded in model structs)
// ============================================================================

// bedrockClaudeOptions contains options for Claude models on Bedrock
type bedrockClaudeOptions struct {
	maxTokens        int
	temperature      float64
	topP             float64
	topK             int
	systemPrompt     string
	anthropicVersion string
}

// bedrockTitanOptions contains options for Amazon Titan models on Bedrock
type bedrockTitanOptions struct {
	maxTokens    int
	temperature  float64
	topP         float64
	systemPrompt string
}

// bedrockNovaOptions contains options for Amazon Nova models on Bedrock
type bedrockNovaOptions struct {
	maxTokens    int
	temperature  float64
	topP         float64
	topK         int
	systemPrompt string
}

// bedrockLlamaOptions contains options for Llama models on Bedrock
type bedrockLlamaOptions struct {
	maxTokens    int
	temperature  float64
	topP         float64
	systemPrompt string
}

// bedrockMistralOptions contains options for Mistral models on Bedrock
type bedrockMistralOptions struct {
	maxTokens    int
	temperature  float64
	topP         float64
	topK         int
	systemPrompt string
}

// ============================================================================
// BEDROCK CLAUDE MODELS
// ============================================================================

// BedrockClaude35Sonnet represents Claude 3.5 Sonnet on Bedrock
type BedrockClaude35Sonnet struct{ bedrockClaudeOptions }

func (m *BedrockClaude35Sonnet) ModelName() string {
	return "anthropic.claude-3-5-sonnet-20241022-v2:0"
}
func (m *BedrockClaude35Sonnet) Provider() ProviderType { return ProviderBedrock }
func (m *BedrockClaude35Sonnet) SystemPrompt() string   { return m.systemPrompt }

func (m *BedrockClaude35Sonnet) WithMaxTokens(n int) *BedrockClaude35Sonnet {
	m.maxTokens = n
	return m
}
func (m *BedrockClaude35Sonnet) WithTemperature(t float64) *BedrockClaude35Sonnet {
	m.temperature = t
	return m
}
func (m *BedrockClaude35Sonnet) WithTopP(p float64) *BedrockClaude35Sonnet { m.topP = p; return m }
func (m *BedrockClaude35Sonnet) WithTopK(k int) *BedrockClaude35Sonnet     { m.topK = k; return m }
func (m *BedrockClaude35Sonnet) WithSystemPrompt(s string) *BedrockClaude35Sonnet {
	m.systemPrompt = s
	return m
}

// NewBedrockClaude35Sonnet creates a new Claude 3.5 Sonnet model for Bedrock
func NewBedrockClaude35Sonnet() *BedrockClaude35Sonnet {
	return &BedrockClaude35Sonnet{bedrockClaudeOptions{
		maxTokens:        4096,
		temperature:      1.0,
		anthropicVersion: "bedrock-2023-05-31",
	}}
}

// BedrockClaude35Haiku represents Claude 3.5 Haiku on Bedrock
type BedrockClaude35Haiku struct{ bedrockClaudeOptions }

func (m *BedrockClaude35Haiku) ModelName() string      { return "anthropic.claude-3-5-haiku-20241022-v1:0" }
func (m *BedrockClaude35Haiku) Provider() ProviderType { return ProviderBedrock }
func (m *BedrockClaude35Haiku) SystemPrompt() string   { return m.systemPrompt }

func (m *BedrockClaude35Haiku) WithMaxTokens(n int) *BedrockClaude35Haiku { m.maxTokens = n; return m }
func (m *BedrockClaude35Haiku) WithTemperature(t float64) *BedrockClaude35Haiku {
	m.temperature = t
	return m
}
func (m *BedrockClaude35Haiku) WithTopP(p float64) *BedrockClaude35Haiku { m.topP = p; return m }
func (m *BedrockClaude35Haiku) WithTopK(k int) *BedrockClaude35Haiku     { m.topK = k; return m }
func (m *BedrockClaude35Haiku) WithSystemPrompt(s string) *BedrockClaude35Haiku {
	m.systemPrompt = s
	return m
}

// NewBedrockClaude35Haiku creates a new Claude 3.5 Haiku model for Bedrock
func NewBedrockClaude35Haiku() *BedrockClaude35Haiku {
	return &BedrockClaude35Haiku{bedrockClaudeOptions{
		maxTokens:        4096,
		temperature:      1.0,
		anthropicVersion: "bedrock-2023-05-31",
	}}
}

// BedrockClaude37Sonnet represents Claude 3.7 Sonnet on Bedrock
type BedrockClaude37Sonnet struct{ bedrockClaudeOptions }

func (m *BedrockClaude37Sonnet) ModelName() string {
	return "anthropic.claude-3-7-sonnet-20250219-v1:0"
}
func (m *BedrockClaude37Sonnet) Provider() ProviderType { return ProviderBedrock }
func (m *BedrockClaude37Sonnet) SystemPrompt() string   { return m.systemPrompt }

func (m *BedrockClaude37Sonnet) WithMaxTokens(n int) *BedrockClaude37Sonnet {
	m.maxTokens = n
	return m
}
func (m *BedrockClaude37Sonnet) WithTemperature(t float64) *BedrockClaude37Sonnet {
	m.temperature = t
	return m
}
func (m *BedrockClaude37Sonnet) WithTopP(p float64) *BedrockClaude37Sonnet { m.topP = p; return m }
func (m *BedrockClaude37Sonnet) WithTopK(k int) *BedrockClaude37Sonnet     { m.topK = k; return m }
func (m *BedrockClaude37Sonnet) WithSystemPrompt(s string) *BedrockClaude37Sonnet {
	m.systemPrompt = s
	return m
}

// NewBedrockClaude37Sonnet creates a new Claude 3.7 Sonnet model for Bedrock
func NewBedrockClaude37Sonnet() *BedrockClaude37Sonnet {
	return &BedrockClaude37Sonnet{bedrockClaudeOptions{
		maxTokens:        8192,
		temperature:      1.0,
		anthropicVersion: "bedrock-2023-05-31",
	}}
}

// BedrockClaudeSonnet4 represents Claude Sonnet 4 on Bedrock
type BedrockClaudeSonnet4 struct{ bedrockClaudeOptions }

func (m *BedrockClaudeSonnet4) ModelName() string      { return "anthropic.claude-sonnet-4-20250514-v1:0" }
func (m *BedrockClaudeSonnet4) Provider() ProviderType { return ProviderBedrock }
func (m *BedrockClaudeSonnet4) SystemPrompt() string   { return m.systemPrompt }

func (m *BedrockClaudeSonnet4) WithMaxTokens(n int) *BedrockClaudeSonnet4 {
	m.maxTokens = n
	return m
}
func (m *BedrockClaudeSonnet4) WithTemperature(t float64) *BedrockClaudeSonnet4 {
	m.temperature = t
	return m
}
func (m *BedrockClaudeSonnet4) WithTopP(p float64) *BedrockClaudeSonnet4 { m.topP = p; return m }
func (m *BedrockClaudeSonnet4) WithTopK(k int) *BedrockClaudeSonnet4     { m.topK = k; return m }
func (m *BedrockClaudeSonnet4) WithSystemPrompt(s string) *BedrockClaudeSonnet4 {
	m.systemPrompt = s
	return m
}

// NewBedrockClaudeSonnet4 creates a new Claude Sonnet 4 model for Bedrock
func NewBedrockClaudeSonnet4() *BedrockClaudeSonnet4 {
	return &BedrockClaudeSonnet4{bedrockClaudeOptions{
		maxTokens:        8192,
		temperature:      1.0,
		anthropicVersion: "bedrock-2023-05-31",
	}}
}

// BedrockClaudeOpus4 represents Claude Opus 4 on Bedrock
type BedrockClaudeOpus4 struct{ bedrockClaudeOptions }

func (m *BedrockClaudeOpus4) ModelName() string      { return "anthropic.claude-opus-4-20250514-v1:0" }
func (m *BedrockClaudeOpus4) Provider() ProviderType { return ProviderBedrock }
func (m *BedrockClaudeOpus4) SystemPrompt() string   { return m.systemPrompt }

func (m *BedrockClaudeOpus4) WithMaxTokens(n int) *BedrockClaudeOpus4 { m.maxTokens = n; return m }
func (m *BedrockClaudeOpus4) WithTemperature(t float64) *BedrockClaudeOpus4 {
	m.temperature = t
	return m
}
func (m *BedrockClaudeOpus4) WithTopP(p float64) *BedrockClaudeOpus4 { m.topP = p; return m }
func (m *BedrockClaudeOpus4) WithTopK(k int) *BedrockClaudeOpus4     { m.topK = k; return m }
func (m *BedrockClaudeOpus4) WithSystemPrompt(s string) *BedrockClaudeOpus4 {
	m.systemPrompt = s
	return m
}

// NewBedrockClaudeOpus4 creates a new Claude Opus 4 model for Bedrock
func NewBedrockClaudeOpus4() *BedrockClaudeOpus4 {
	return &BedrockClaudeOpus4{bedrockClaudeOptions{
		maxTokens:        8192,
		temperature:      1.0,
		anthropicVersion: "bedrock-2023-05-31",
	}}
}

// BedrockClaudeSonnet45 represents Claude Sonnet 4.5 on Bedrock
type BedrockClaudeSonnet45 struct{ bedrockClaudeOptions }

func (m *BedrockClaudeSonnet45) ModelName() string {
	return "anthropic.claude-sonnet-4-5-20250929-v1:0"
}
func (m *BedrockClaudeSonnet45) Provider() ProviderType { return ProviderBedrock }
func (m *BedrockClaudeSonnet45) SystemPrompt() string   { return m.systemPrompt }

func (m *BedrockClaudeSonnet45) WithMaxTokens(n int) *BedrockClaudeSonnet45 {
	m.maxTokens = n
	return m
}
func (m *BedrockClaudeSonnet45) WithTemperature(t float64) *BedrockClaudeSonnet45 {
	m.temperature = t
	return m
}
func (m *BedrockClaudeSonnet45) WithTopP(p float64) *BedrockClaudeSonnet45 { m.topP = p; return m }
func (m *BedrockClaudeSonnet45) WithTopK(k int) *BedrockClaudeSonnet45     { m.topK = k; return m }
func (m *BedrockClaudeSonnet45) WithSystemPrompt(s string) *BedrockClaudeSonnet45 {
	m.systemPrompt = s
	return m
}

// NewBedrockClaudeSonnet45 creates a new Claude Sonnet 4.5 model for Bedrock
func NewBedrockClaudeSonnet45() *BedrockClaudeSonnet45 {
	return &BedrockClaudeSonnet45{bedrockClaudeOptions{
		maxTokens:        8192,
		temperature:      1.0,
		anthropicVersion: "bedrock-2023-05-31",
	}}
}

// BedrockClaudeOpus45 represents Claude Opus 4.5 on Bedrock
type BedrockClaudeOpus45 struct{ bedrockClaudeOptions }

func (m *BedrockClaudeOpus45) ModelName() string {
	return "anthropic.claude-opus-4-5-20251101-v1:0"
}
func (m *BedrockClaudeOpus45) Provider() ProviderType { return ProviderBedrock }
func (m *BedrockClaudeOpus45) SystemPrompt() string   { return m.systemPrompt }

func (m *BedrockClaudeOpus45) WithMaxTokens(n int) *BedrockClaudeOpus45 { m.maxTokens = n; return m }
func (m *BedrockClaudeOpus45) WithTemperature(t float64) *BedrockClaudeOpus45 {
	m.temperature = t
	return m
}
func (m *BedrockClaudeOpus45) WithTopP(p float64) *BedrockClaudeOpus45 { m.topP = p; return m }
func (m *BedrockClaudeOpus45) WithTopK(k int) *BedrockClaudeOpus45     { m.topK = k; return m }
func (m *BedrockClaudeOpus45) WithSystemPrompt(s string) *BedrockClaudeOpus45 {
	m.systemPrompt = s
	return m
}

// NewBedrockClaudeOpus45 creates a new Claude Opus 4.5 model for Bedrock
func NewBedrockClaudeOpus45() *BedrockClaudeOpus45 {
	return &BedrockClaudeOpus45{bedrockClaudeOptions{
		maxTokens:        8192,
		temperature:      1.0,
		anthropicVersion: "bedrock-2023-05-31",
	}}
}

// BedrockClaudeHaiku45 represents Claude Haiku 4.5 on Bedrock
type BedrockClaudeHaiku45 struct{ bedrockClaudeOptions }

func (m *BedrockClaudeHaiku45) ModelName() string {
	return "anthropic.claude-haiku-4-5-20251001-v1:0"
}
func (m *BedrockClaudeHaiku45) Provider() ProviderType { return ProviderBedrock }
func (m *BedrockClaudeHaiku45) SystemPrompt() string   { return m.systemPrompt }

func (m *BedrockClaudeHaiku45) WithMaxTokens(n int) *BedrockClaudeHaiku45 {
	m.maxTokens = n
	return m
}
func (m *BedrockClaudeHaiku45) WithTemperature(t float64) *BedrockClaudeHaiku45 {
	m.temperature = t
	return m
}
func (m *BedrockClaudeHaiku45) WithTopP(p float64) *BedrockClaudeHaiku45 { m.topP = p; return m }
func (m *BedrockClaudeHaiku45) WithTopK(k int) *BedrockClaudeHaiku45     { m.topK = k; return m }
func (m *BedrockClaudeHaiku45) WithSystemPrompt(s string) *BedrockClaudeHaiku45 {
	m.systemPrompt = s
	return m
}

// NewBedrockClaudeHaiku45 creates a new Claude Haiku 4.5 model for Bedrock
func NewBedrockClaudeHaiku45() *BedrockClaudeHaiku45 {
	return &BedrockClaudeHaiku45{bedrockClaudeOptions{
		maxTokens:        8192,
		temperature:      1.0,
		anthropicVersion: "bedrock-2023-05-31",
	}}
}

// BedrockClaudeOpus46 represents Claude Opus 4.6 on Bedrock (current recommended)
type BedrockClaudeOpus46 struct{ bedrockClaudeOptions }

func (m *BedrockClaudeOpus46) ModelName() string      { return "anthropic.claude-opus-4-6-v1" }
func (m *BedrockClaudeOpus46) Provider() ProviderType { return ProviderBedrock }
func (m *BedrockClaudeOpus46) SystemPrompt() string   { return m.systemPrompt }

func (m *BedrockClaudeOpus46) WithMaxTokens(n int) *BedrockClaudeOpus46 { m.maxTokens = n; return m }
func (m *BedrockClaudeOpus46) WithTemperature(t float64) *BedrockClaudeOpus46 {
	m.temperature = t
	return m
}
func (m *BedrockClaudeOpus46) WithTopP(p float64) *BedrockClaudeOpus46 { m.topP = p; return m }
func (m *BedrockClaudeOpus46) WithTopK(k int) *BedrockClaudeOpus46     { m.topK = k; return m }
func (m *BedrockClaudeOpus46) WithSystemPrompt(s string) *BedrockClaudeOpus46 {
	m.systemPrompt = s
	return m
}

// NewBedrockClaudeOpus46 creates a new Claude Opus 4.6 model for Bedrock
func NewBedrockClaudeOpus46() *BedrockClaudeOpus46 {
	return &BedrockClaudeOpus46{bedrockClaudeOptions{
		maxTokens:        8192,
		temperature:      1.0,
		anthropicVersion: "bedrock-2023-05-31",
	}}
}

// BedrockClaudeSonnet46 represents Claude Sonnet 4.6 on Bedrock (current recommended)
type BedrockClaudeSonnet46 struct{ bedrockClaudeOptions }

func (m *BedrockClaudeSonnet46) ModelName() string      { return "anthropic.claude-sonnet-4-6" }
func (m *BedrockClaudeSonnet46) Provider() ProviderType { return ProviderBedrock }
func (m *BedrockClaudeSonnet46) SystemPrompt() string   { return m.systemPrompt }

func (m *BedrockClaudeSonnet46) WithMaxTokens(n int) *BedrockClaudeSonnet46 {
	m.maxTokens = n
	return m
}
func (m *BedrockClaudeSonnet46) WithTemperature(t float64) *BedrockClaudeSonnet46 {
	m.temperature = t
	return m
}
func (m *BedrockClaudeSonnet46) WithTopP(p float64) *BedrockClaudeSonnet46 { m.topP = p; return m }
func (m *BedrockClaudeSonnet46) WithTopK(k int) *BedrockClaudeSonnet46     { m.topK = k; return m }
func (m *BedrockClaudeSonnet46) WithSystemPrompt(s string) *BedrockClaudeSonnet46 {
	m.systemPrompt = s
	return m
}

// NewBedrockClaudeSonnet46 creates a new Claude Sonnet 4.6 model for Bedrock
func NewBedrockClaudeSonnet46() *BedrockClaudeSonnet46 {
	return &BedrockClaudeSonnet46{bedrockClaudeOptions{
		maxTokens:        8192,
		temperature:      1.0,
		anthropicVersion: "bedrock-2023-05-31",
	}}
}

// BedrockClaudeOpus47 represents Claude Opus 4.7 on Bedrock.
// Opus 4.7 rejects sampling parameters (temperature/topP/topK) with a 400 error,
// so this type does not expose setters for them.
type BedrockClaudeOpus47 struct{ bedrockClaudeOptions }

func (m *BedrockClaudeOpus47) ModelName() string      { return "anthropic.claude-opus-4-7" }
func (m *BedrockClaudeOpus47) Provider() ProviderType { return ProviderBedrock }
func (m *BedrockClaudeOpus47) SystemPrompt() string   { return m.systemPrompt }

func (m *BedrockClaudeOpus47) WithMaxTokens(n int) *BedrockClaudeOpus47 { m.maxTokens = n; return m }
func (m *BedrockClaudeOpus47) WithSystemPrompt(s string) *BedrockClaudeOpus47 {
	m.systemPrompt = s
	return m
}

// NewBedrockClaudeOpus47 creates a new Claude Opus 4.7 model for Bedrock
func NewBedrockClaudeOpus47() *BedrockClaudeOpus47 {
	return &BedrockClaudeOpus47{bedrockClaudeOptions{
		maxTokens:        8192,
		anthropicVersion: "bedrock-2023-05-31",
	}}
}

// BedrockClaudeOpus48 represents Claude Opus 4.8 on Bedrock (current recommended).
// Opus 4.8 rejects sampling parameters (temperature/topP/topK) with a 400 error,
// so this type does not expose setters for them.
type BedrockClaudeOpus48 struct{ bedrockClaudeOptions }

func (m *BedrockClaudeOpus48) ModelName() string      { return "anthropic.claude-opus-4-8" }
func (m *BedrockClaudeOpus48) Provider() ProviderType { return ProviderBedrock }
func (m *BedrockClaudeOpus48) SystemPrompt() string   { return m.systemPrompt }

func (m *BedrockClaudeOpus48) WithMaxTokens(n int) *BedrockClaudeOpus48 { m.maxTokens = n; return m }
func (m *BedrockClaudeOpus48) WithSystemPrompt(s string) *BedrockClaudeOpus48 {
	m.systemPrompt = s
	return m
}

// NewBedrockClaudeOpus48 creates a new Claude Opus 4.8 model for Bedrock
func NewBedrockClaudeOpus48() *BedrockClaudeOpus48 {
	return &BedrockClaudeOpus48{bedrockClaudeOptions{
		maxTokens:        8192,
		anthropicVersion: "bedrock-2023-05-31",
	}}
}

// BedrockClaudeFable5 represents Claude Fable 5 on Bedrock (most capable).
// Fable 5 rejects sampling parameters (temperature/topP/topK) with a 400 error
// and thinking is always on, so this type does not expose setters for them.
type BedrockClaudeFable5 struct{ bedrockClaudeOptions }

func (m *BedrockClaudeFable5) ModelName() string      { return "anthropic.claude-fable-5" }
func (m *BedrockClaudeFable5) Provider() ProviderType { return ProviderBedrock }
func (m *BedrockClaudeFable5) SystemPrompt() string   { return m.systemPrompt }

func (m *BedrockClaudeFable5) WithMaxTokens(n int) *BedrockClaudeFable5 { m.maxTokens = n; return m }
func (m *BedrockClaudeFable5) WithSystemPrompt(s string) *BedrockClaudeFable5 {
	m.systemPrompt = s
	return m
}

// NewBedrockClaudeFable5 creates a new Claude Fable 5 model for Bedrock
func NewBedrockClaudeFable5() *BedrockClaudeFable5 {
	return &BedrockClaudeFable5{bedrockClaudeOptions{
		maxTokens:        8192,
		anthropicVersion: "bedrock-2023-05-31",
	}}
}

// BedrockClaude3Sonnet represents Claude 3 Sonnet on Bedrock
type BedrockClaude3Sonnet struct{ bedrockClaudeOptions }

func (m *BedrockClaude3Sonnet) ModelName() string      { return "anthropic.claude-3-sonnet-20240229-v1:0" }
func (m *BedrockClaude3Sonnet) Provider() ProviderType { return ProviderBedrock }
func (m *BedrockClaude3Sonnet) SystemPrompt() string   { return m.systemPrompt }

func (m *BedrockClaude3Sonnet) WithMaxTokens(n int) *BedrockClaude3Sonnet { m.maxTokens = n; return m }
func (m *BedrockClaude3Sonnet) WithTemperature(t float64) *BedrockClaude3Sonnet {
	m.temperature = t
	return m
}
func (m *BedrockClaude3Sonnet) WithTopP(p float64) *BedrockClaude3Sonnet { m.topP = p; return m }
func (m *BedrockClaude3Sonnet) WithTopK(k int) *BedrockClaude3Sonnet     { m.topK = k; return m }
func (m *BedrockClaude3Sonnet) WithSystemPrompt(s string) *BedrockClaude3Sonnet {
	m.systemPrompt = s
	return m
}

// NewBedrockClaude3Sonnet creates a new Claude 3 Sonnet model for Bedrock
func NewBedrockClaude3Sonnet() *BedrockClaude3Sonnet {
	return &BedrockClaude3Sonnet{bedrockClaudeOptions{
		maxTokens:        4096,
		temperature:      1.0,
		anthropicVersion: "bedrock-2023-05-31",
	}}
}

// BedrockClaude3Haiku represents Claude 3 Haiku on Bedrock
type BedrockClaude3Haiku struct{ bedrockClaudeOptions }

func (m *BedrockClaude3Haiku) ModelName() string      { return "anthropic.claude-3-haiku-20240307-v1:0" }
func (m *BedrockClaude3Haiku) Provider() ProviderType { return ProviderBedrock }
func (m *BedrockClaude3Haiku) SystemPrompt() string   { return m.systemPrompt }

func (m *BedrockClaude3Haiku) WithMaxTokens(n int) *BedrockClaude3Haiku { m.maxTokens = n; return m }
func (m *BedrockClaude3Haiku) WithTemperature(t float64) *BedrockClaude3Haiku {
	m.temperature = t
	return m
}
func (m *BedrockClaude3Haiku) WithTopP(p float64) *BedrockClaude3Haiku { m.topP = p; return m }
func (m *BedrockClaude3Haiku) WithTopK(k int) *BedrockClaude3Haiku     { m.topK = k; return m }
func (m *BedrockClaude3Haiku) WithSystemPrompt(s string) *BedrockClaude3Haiku {
	m.systemPrompt = s
	return m
}

// NewBedrockClaude3Haiku creates a new Claude 3 Haiku model for Bedrock
func NewBedrockClaude3Haiku() *BedrockClaude3Haiku {
	return &BedrockClaude3Haiku{bedrockClaudeOptions{
		maxTokens:        4096,
		temperature:      1.0,
		anthropicVersion: "bedrock-2023-05-31",
	}}
}

// BedrockClaude3Opus represents Claude 3 Opus on Bedrock
type BedrockClaude3Opus struct{ bedrockClaudeOptions }

func (m *BedrockClaude3Opus) ModelName() string      { return "anthropic.claude-3-opus-20240229-v1:0" }
func (m *BedrockClaude3Opus) Provider() ProviderType { return ProviderBedrock }
func (m *BedrockClaude3Opus) SystemPrompt() string   { return m.systemPrompt }

func (m *BedrockClaude3Opus) WithMaxTokens(n int) *BedrockClaude3Opus { m.maxTokens = n; return m }
func (m *BedrockClaude3Opus) WithTemperature(t float64) *BedrockClaude3Opus {
	m.temperature = t
	return m
}
func (m *BedrockClaude3Opus) WithTopP(p float64) *BedrockClaude3Opus { m.topP = p; return m }
func (m *BedrockClaude3Opus) WithTopK(k int) *BedrockClaude3Opus     { m.topK = k; return m }
func (m *BedrockClaude3Opus) WithSystemPrompt(s string) *BedrockClaude3Opus {
	m.systemPrompt = s
	return m
}

// NewBedrockClaude3Opus creates a new Claude 3 Opus model for Bedrock
func NewBedrockClaude3Opus() *BedrockClaude3Opus {
	return &BedrockClaude3Opus{bedrockClaudeOptions{
		maxTokens:        4096,
		temperature:      1.0,
		anthropicVersion: "bedrock-2023-05-31",
	}}
}

// ============================================================================
// BEDROCK TITAN MODELS
// ============================================================================

// BedrockTitanTextExpress represents Amazon Titan Text Express
type BedrockTitanTextExpress struct{ bedrockTitanOptions }

func (m *BedrockTitanTextExpress) ModelName() string      { return "amazon.titan-text-express-v1" }
func (m *BedrockTitanTextExpress) Provider() ProviderType { return ProviderBedrock }
func (m *BedrockTitanTextExpress) SystemPrompt() string   { return m.systemPrompt }

func (m *BedrockTitanTextExpress) WithMaxTokens(n int) *BedrockTitanTextExpress {
	m.maxTokens = n
	return m
}
func (m *BedrockTitanTextExpress) WithTemperature(t float64) *BedrockTitanTextExpress {
	m.temperature = t
	return m
}
func (m *BedrockTitanTextExpress) WithTopP(p float64) *BedrockTitanTextExpress { m.topP = p; return m }
func (m *BedrockTitanTextExpress) WithSystemPrompt(s string) *BedrockTitanTextExpress {
	m.systemPrompt = s
	return m
}

// NewBedrockTitanTextExpress creates a new Titan Text Express model for Bedrock
func NewBedrockTitanTextExpress() *BedrockTitanTextExpress {
	return &BedrockTitanTextExpress{bedrockTitanOptions{maxTokens: 4096, temperature: 0.7}}
}

// BedrockTitanTextLite represents Amazon Titan Text Lite
type BedrockTitanTextLite struct{ bedrockTitanOptions }

func (m *BedrockTitanTextLite) ModelName() string      { return "amazon.titan-text-lite-v1" }
func (m *BedrockTitanTextLite) Provider() ProviderType { return ProviderBedrock }
func (m *BedrockTitanTextLite) SystemPrompt() string   { return m.systemPrompt }

func (m *BedrockTitanTextLite) WithMaxTokens(n int) *BedrockTitanTextLite { m.maxTokens = n; return m }
func (m *BedrockTitanTextLite) WithTemperature(t float64) *BedrockTitanTextLite {
	m.temperature = t
	return m
}
func (m *BedrockTitanTextLite) WithTopP(p float64) *BedrockTitanTextLite { m.topP = p; return m }
func (m *BedrockTitanTextLite) WithSystemPrompt(s string) *BedrockTitanTextLite {
	m.systemPrompt = s
	return m
}

// NewBedrockTitanTextLite creates a new Titan Text Lite model for Bedrock
func NewBedrockTitanTextLite() *BedrockTitanTextLite {
	return &BedrockTitanTextLite{bedrockTitanOptions{maxTokens: 4096, temperature: 0.7}}
}

// BedrockTitanTextPremier represents Amazon Titan Text Premier
type BedrockTitanTextPremier struct{ bedrockTitanOptions }

func (m *BedrockTitanTextPremier) ModelName() string      { return "amazon.titan-text-premier-v1:0" }
func (m *BedrockTitanTextPremier) Provider() ProviderType { return ProviderBedrock }
func (m *BedrockTitanTextPremier) SystemPrompt() string   { return m.systemPrompt }

func (m *BedrockTitanTextPremier) WithMaxTokens(n int) *BedrockTitanTextPremier {
	m.maxTokens = n
	return m
}
func (m *BedrockTitanTextPremier) WithTemperature(t float64) *BedrockTitanTextPremier {
	m.temperature = t
	return m
}
func (m *BedrockTitanTextPremier) WithTopP(p float64) *BedrockTitanTextPremier { m.topP = p; return m }
func (m *BedrockTitanTextPremier) WithSystemPrompt(s string) *BedrockTitanTextPremier {
	m.systemPrompt = s
	return m
}

// NewBedrockTitanTextPremier creates a new Titan Text Premier model for Bedrock
func NewBedrockTitanTextPremier() *BedrockTitanTextPremier {
	return &BedrockTitanTextPremier{bedrockTitanOptions{maxTokens: 4096, temperature: 0.7}}
}

// ============================================================================
// BEDROCK AMAZON NOVA MODELS
// ============================================================================
// Nova is Amazon's current first-party model family, superseding Titan Text.
// Note: in many regions Nova models are only invokable through cross-region
// inference profiles — if the base ID fails with a validation error, use
// NewBedrockModel with the profile ID (e.g. "us.amazon.nova-pro-v1:0", "nova").

// BedrockNovaMicro represents Amazon Nova Micro on Bedrock (text-only, lowest latency)
type BedrockNovaMicro struct{ bedrockNovaOptions }

func (m *BedrockNovaMicro) ModelName() string      { return "amazon.nova-micro-v1:0" }
func (m *BedrockNovaMicro) Provider() ProviderType { return ProviderBedrock }
func (m *BedrockNovaMicro) SystemPrompt() string   { return m.systemPrompt }

func (m *BedrockNovaMicro) WithMaxTokens(n int) *BedrockNovaMicro       { m.maxTokens = n; return m }
func (m *BedrockNovaMicro) WithTemperature(t float64) *BedrockNovaMicro { m.temperature = t; return m }
func (m *BedrockNovaMicro) WithTopP(p float64) *BedrockNovaMicro        { m.topP = p; return m }
func (m *BedrockNovaMicro) WithTopK(k int) *BedrockNovaMicro            { m.topK = k; return m }
func (m *BedrockNovaMicro) WithSystemPrompt(s string) *BedrockNovaMicro {
	m.systemPrompt = s
	return m
}

// NewBedrockNovaMicro creates a new Amazon Nova Micro model for Bedrock
func NewBedrockNovaMicro() *BedrockNovaMicro {
	return &BedrockNovaMicro{bedrockNovaOptions{maxTokens: 4096, temperature: 0.7}}
}

// BedrockNovaLite represents Amazon Nova Lite on Bedrock (fast, low cost)
type BedrockNovaLite struct{ bedrockNovaOptions }

func (m *BedrockNovaLite) ModelName() string      { return "amazon.nova-lite-v1:0" }
func (m *BedrockNovaLite) Provider() ProviderType { return ProviderBedrock }
func (m *BedrockNovaLite) SystemPrompt() string   { return m.systemPrompt }

func (m *BedrockNovaLite) WithMaxTokens(n int) *BedrockNovaLite       { m.maxTokens = n; return m }
func (m *BedrockNovaLite) WithTemperature(t float64) *BedrockNovaLite { m.temperature = t; return m }
func (m *BedrockNovaLite) WithTopP(p float64) *BedrockNovaLite        { m.topP = p; return m }
func (m *BedrockNovaLite) WithTopK(k int) *BedrockNovaLite            { m.topK = k; return m }
func (m *BedrockNovaLite) WithSystemPrompt(s string) *BedrockNovaLite {
	m.systemPrompt = s
	return m
}

// NewBedrockNovaLite creates a new Amazon Nova Lite model for Bedrock
func NewBedrockNovaLite() *BedrockNovaLite {
	return &BedrockNovaLite{bedrockNovaOptions{maxTokens: 4096, temperature: 0.7}}
}

// BedrockNovaPro represents Amazon Nova Pro on Bedrock (balanced capability)
type BedrockNovaPro struct{ bedrockNovaOptions }

func (m *BedrockNovaPro) ModelName() string      { return "amazon.nova-pro-v1:0" }
func (m *BedrockNovaPro) Provider() ProviderType { return ProviderBedrock }
func (m *BedrockNovaPro) SystemPrompt() string   { return m.systemPrompt }

func (m *BedrockNovaPro) WithMaxTokens(n int) *BedrockNovaPro       { m.maxTokens = n; return m }
func (m *BedrockNovaPro) WithTemperature(t float64) *BedrockNovaPro { m.temperature = t; return m }
func (m *BedrockNovaPro) WithTopP(p float64) *BedrockNovaPro        { m.topP = p; return m }
func (m *BedrockNovaPro) WithTopK(k int) *BedrockNovaPro            { m.topK = k; return m }
func (m *BedrockNovaPro) WithSystemPrompt(s string) *BedrockNovaPro {
	m.systemPrompt = s
	return m
}

// NewBedrockNovaPro creates a new Amazon Nova Pro model for Bedrock
func NewBedrockNovaPro() *BedrockNovaPro {
	return &BedrockNovaPro{bedrockNovaOptions{maxTokens: 4096, temperature: 0.7}}
}

// BedrockNovaPremier represents Amazon Nova Premier on Bedrock (most capable Nova)
type BedrockNovaPremier struct{ bedrockNovaOptions }

func (m *BedrockNovaPremier) ModelName() string      { return "amazon.nova-premier-v1:0" }
func (m *BedrockNovaPremier) Provider() ProviderType { return ProviderBedrock }
func (m *BedrockNovaPremier) SystemPrompt() string   { return m.systemPrompt }

func (m *BedrockNovaPremier) WithMaxTokens(n int) *BedrockNovaPremier { m.maxTokens = n; return m }
func (m *BedrockNovaPremier) WithTemperature(t float64) *BedrockNovaPremier {
	m.temperature = t
	return m
}
func (m *BedrockNovaPremier) WithTopP(p float64) *BedrockNovaPremier { m.topP = p; return m }
func (m *BedrockNovaPremier) WithTopK(k int) *BedrockNovaPremier     { m.topK = k; return m }
func (m *BedrockNovaPremier) WithSystemPrompt(s string) *BedrockNovaPremier {
	m.systemPrompt = s
	return m
}

// NewBedrockNovaPremier creates a new Amazon Nova Premier model for Bedrock
func NewBedrockNovaPremier() *BedrockNovaPremier {
	return &BedrockNovaPremier{bedrockNovaOptions{maxTokens: 4096, temperature: 0.7}}
}

// ============================================================================
// BEDROCK LLAMA MODELS
// ============================================================================

// BedrockLlama31Instruct8B represents Meta Llama 3.1 8B Instruct on Bedrock
type BedrockLlama31Instruct8B struct{ bedrockLlamaOptions }

func (m *BedrockLlama31Instruct8B) ModelName() string      { return "meta.llama3-1-8b-instruct-v1:0" }
func (m *BedrockLlama31Instruct8B) Provider() ProviderType { return ProviderBedrock }
func (m *BedrockLlama31Instruct8B) SystemPrompt() string   { return m.systemPrompt }

func (m *BedrockLlama31Instruct8B) WithMaxTokens(n int) *BedrockLlama31Instruct8B {
	m.maxTokens = n
	return m
}
func (m *BedrockLlama31Instruct8B) WithTemperature(t float64) *BedrockLlama31Instruct8B {
	m.temperature = t
	return m
}
func (m *BedrockLlama31Instruct8B) WithTopP(p float64) *BedrockLlama31Instruct8B {
	m.topP = p
	return m
}
func (m *BedrockLlama31Instruct8B) WithSystemPrompt(s string) *BedrockLlama31Instruct8B {
	m.systemPrompt = s
	return m
}

// NewBedrockLlama31Instruct8B creates a new Llama 3.1 8B Instruct model for Bedrock
func NewBedrockLlama31Instruct8B() *BedrockLlama31Instruct8B {
	return &BedrockLlama31Instruct8B{bedrockLlamaOptions{maxTokens: 2048, temperature: 0.6}}
}

// BedrockLlama31Instruct70B represents Meta Llama 3.1 70B Instruct on Bedrock
type BedrockLlama31Instruct70B struct{ bedrockLlamaOptions }

func (m *BedrockLlama31Instruct70B) ModelName() string      { return "meta.llama3-1-70b-instruct-v1:0" }
func (m *BedrockLlama31Instruct70B) Provider() ProviderType { return ProviderBedrock }
func (m *BedrockLlama31Instruct70B) SystemPrompt() string   { return m.systemPrompt }

func (m *BedrockLlama31Instruct70B) WithMaxTokens(n int) *BedrockLlama31Instruct70B {
	m.maxTokens = n
	return m
}
func (m *BedrockLlama31Instruct70B) WithTemperature(t float64) *BedrockLlama31Instruct70B {
	m.temperature = t
	return m
}
func (m *BedrockLlama31Instruct70B) WithTopP(p float64) *BedrockLlama31Instruct70B {
	m.topP = p
	return m
}
func (m *BedrockLlama31Instruct70B) WithSystemPrompt(s string) *BedrockLlama31Instruct70B {
	m.systemPrompt = s
	return m
}

// NewBedrockLlama31Instruct70B creates a new Llama 3.1 70B Instruct model for Bedrock
func NewBedrockLlama31Instruct70B() *BedrockLlama31Instruct70B {
	return &BedrockLlama31Instruct70B{bedrockLlamaOptions{maxTokens: 2048, temperature: 0.6}}
}

// BedrockLlama31Instruct405B represents Meta Llama 3.1 405B Instruct on Bedrock
type BedrockLlama31Instruct405B struct{ bedrockLlamaOptions }

func (m *BedrockLlama31Instruct405B) ModelName() string      { return "meta.llama3-1-405b-instruct-v1:0" }
func (m *BedrockLlama31Instruct405B) Provider() ProviderType { return ProviderBedrock }
func (m *BedrockLlama31Instruct405B) SystemPrompt() string   { return m.systemPrompt }

func (m *BedrockLlama31Instruct405B) WithMaxTokens(n int) *BedrockLlama31Instruct405B {
	m.maxTokens = n
	return m
}
func (m *BedrockLlama31Instruct405B) WithTemperature(t float64) *BedrockLlama31Instruct405B {
	m.temperature = t
	return m
}
func (m *BedrockLlama31Instruct405B) WithTopP(p float64) *BedrockLlama31Instruct405B {
	m.topP = p
	return m
}
func (m *BedrockLlama31Instruct405B) WithSystemPrompt(s string) *BedrockLlama31Instruct405B {
	m.systemPrompt = s
	return m
}

// NewBedrockLlama31Instruct405B creates a new Llama 3.1 405B Instruct model for Bedrock
func NewBedrockLlama31Instruct405B() *BedrockLlama31Instruct405B {
	return &BedrockLlama31Instruct405B{bedrockLlamaOptions{maxTokens: 2048, temperature: 0.6}}
}

// BedrockLlama32Instruct1B represents Meta Llama 3.2 1B Instruct on Bedrock
type BedrockLlama32Instruct1B struct{ bedrockLlamaOptions }

func (m *BedrockLlama32Instruct1B) ModelName() string      { return "meta.llama3-2-1b-instruct-v1:0" }
func (m *BedrockLlama32Instruct1B) Provider() ProviderType { return ProviderBedrock }
func (m *BedrockLlama32Instruct1B) SystemPrompt() string   { return m.systemPrompt }

func (m *BedrockLlama32Instruct1B) WithMaxTokens(n int) *BedrockLlama32Instruct1B {
	m.maxTokens = n
	return m
}
func (m *BedrockLlama32Instruct1B) WithTemperature(t float64) *BedrockLlama32Instruct1B {
	m.temperature = t
	return m
}
func (m *BedrockLlama32Instruct1B) WithTopP(p float64) *BedrockLlama32Instruct1B {
	m.topP = p
	return m
}
func (m *BedrockLlama32Instruct1B) WithSystemPrompt(s string) *BedrockLlama32Instruct1B {
	m.systemPrompt = s
	return m
}

// NewBedrockLlama32Instruct1B creates a new Llama 3.2 1B Instruct model for Bedrock
func NewBedrockLlama32Instruct1B() *BedrockLlama32Instruct1B {
	return &BedrockLlama32Instruct1B{bedrockLlamaOptions{maxTokens: 2048, temperature: 0.6}}
}

// BedrockLlama32Instruct3B represents Meta Llama 3.2 3B Instruct on Bedrock
type BedrockLlama32Instruct3B struct{ bedrockLlamaOptions }

func (m *BedrockLlama32Instruct3B) ModelName() string      { return "meta.llama3-2-3b-instruct-v1:0" }
func (m *BedrockLlama32Instruct3B) Provider() ProviderType { return ProviderBedrock }
func (m *BedrockLlama32Instruct3B) SystemPrompt() string   { return m.systemPrompt }

func (m *BedrockLlama32Instruct3B) WithMaxTokens(n int) *BedrockLlama32Instruct3B {
	m.maxTokens = n
	return m
}
func (m *BedrockLlama32Instruct3B) WithTemperature(t float64) *BedrockLlama32Instruct3B {
	m.temperature = t
	return m
}
func (m *BedrockLlama32Instruct3B) WithTopP(p float64) *BedrockLlama32Instruct3B {
	m.topP = p
	return m
}
func (m *BedrockLlama32Instruct3B) WithSystemPrompt(s string) *BedrockLlama32Instruct3B {
	m.systemPrompt = s
	return m
}

// NewBedrockLlama32Instruct3B creates a new Llama 3.2 3B Instruct model for Bedrock
func NewBedrockLlama32Instruct3B() *BedrockLlama32Instruct3B {
	return &BedrockLlama32Instruct3B{bedrockLlamaOptions{maxTokens: 2048, temperature: 0.6}}
}

// BedrockLlama33Instruct70B represents Meta Llama 3.3 70B Instruct on Bedrock
type BedrockLlama33Instruct70B struct{ bedrockLlamaOptions }

func (m *BedrockLlama33Instruct70B) ModelName() string      { return "meta.llama3-3-70b-instruct-v1:0" }
func (m *BedrockLlama33Instruct70B) Provider() ProviderType { return ProviderBedrock }
func (m *BedrockLlama33Instruct70B) SystemPrompt() string   { return m.systemPrompt }

func (m *BedrockLlama33Instruct70B) WithMaxTokens(n int) *BedrockLlama33Instruct70B {
	m.maxTokens = n
	return m
}
func (m *BedrockLlama33Instruct70B) WithTemperature(t float64) *BedrockLlama33Instruct70B {
	m.temperature = t
	return m
}
func (m *BedrockLlama33Instruct70B) WithTopP(p float64) *BedrockLlama33Instruct70B {
	m.topP = p
	return m
}
func (m *BedrockLlama33Instruct70B) WithSystemPrompt(s string) *BedrockLlama33Instruct70B {
	m.systemPrompt = s
	return m
}

// NewBedrockLlama33Instruct70B creates a new Llama 3.3 70B Instruct model for Bedrock
func NewBedrockLlama33Instruct70B() *BedrockLlama33Instruct70B {
	return &BedrockLlama33Instruct70B{bedrockLlamaOptions{maxTokens: 2048, temperature: 0.6}}
}

// BedrockLlama4Scout represents Meta Llama 4 Scout 17B on Bedrock
type BedrockLlama4Scout struct{ bedrockLlamaOptions }

func (m *BedrockLlama4Scout) ModelName() string      { return "meta.llama4-scout-17b-instruct-v1:0" }
func (m *BedrockLlama4Scout) Provider() ProviderType { return ProviderBedrock }
func (m *BedrockLlama4Scout) SystemPrompt() string   { return m.systemPrompt }

func (m *BedrockLlama4Scout) WithMaxTokens(n int) *BedrockLlama4Scout { m.maxTokens = n; return m }
func (m *BedrockLlama4Scout) WithTemperature(t float64) *BedrockLlama4Scout {
	m.temperature = t
	return m
}
func (m *BedrockLlama4Scout) WithTopP(p float64) *BedrockLlama4Scout { m.topP = p; return m }
func (m *BedrockLlama4Scout) WithSystemPrompt(s string) *BedrockLlama4Scout {
	m.systemPrompt = s
	return m
}

// NewBedrockLlama4Scout creates a new Llama 4 Scout model for Bedrock
func NewBedrockLlama4Scout() *BedrockLlama4Scout {
	return &BedrockLlama4Scout{bedrockLlamaOptions{maxTokens: 2048, temperature: 0.6}}
}

// BedrockLlama4Maverick represents Meta Llama 4 Maverick 17B on Bedrock
type BedrockLlama4Maverick struct{ bedrockLlamaOptions }

func (m *BedrockLlama4Maverick) ModelName() string {
	return "meta.llama4-maverick-17b-instruct-v1:0"
}
func (m *BedrockLlama4Maverick) Provider() ProviderType { return ProviderBedrock }
func (m *BedrockLlama4Maverick) SystemPrompt() string   { return m.systemPrompt }

func (m *BedrockLlama4Maverick) WithMaxTokens(n int) *BedrockLlama4Maverick {
	m.maxTokens = n
	return m
}
func (m *BedrockLlama4Maverick) WithTemperature(t float64) *BedrockLlama4Maverick {
	m.temperature = t
	return m
}
func (m *BedrockLlama4Maverick) WithTopP(p float64) *BedrockLlama4Maverick { m.topP = p; return m }
func (m *BedrockLlama4Maverick) WithSystemPrompt(s string) *BedrockLlama4Maverick {
	m.systemPrompt = s
	return m
}

// NewBedrockLlama4Maverick creates a new Llama 4 Maverick model for Bedrock
func NewBedrockLlama4Maverick() *BedrockLlama4Maverick {
	return &BedrockLlama4Maverick{bedrockLlamaOptions{maxTokens: 2048, temperature: 0.6}}
}

// ============================================================================
// BEDROCK MISTRAL MODELS
// ============================================================================

// BedrockMistral7B represents Mistral 7B Instruct on Bedrock
type BedrockMistral7B struct{ bedrockMistralOptions }

func (m *BedrockMistral7B) ModelName() string      { return "mistral.mistral-7b-instruct-v0:2" }
func (m *BedrockMistral7B) Provider() ProviderType { return ProviderBedrock }
func (m *BedrockMistral7B) SystemPrompt() string   { return m.systemPrompt }

func (m *BedrockMistral7B) WithMaxTokens(n int) *BedrockMistral7B       { m.maxTokens = n; return m }
func (m *BedrockMistral7B) WithTemperature(t float64) *BedrockMistral7B { m.temperature = t; return m }
func (m *BedrockMistral7B) WithTopP(p float64) *BedrockMistral7B        { m.topP = p; return m }
func (m *BedrockMistral7B) WithTopK(k int) *BedrockMistral7B            { m.topK = k; return m }
func (m *BedrockMistral7B) WithSystemPrompt(s string) *BedrockMistral7B { m.systemPrompt = s; return m }

// NewBedrockMistral7B creates a new Mistral 7B Instruct model for Bedrock
func NewBedrockMistral7B() *BedrockMistral7B {
	return &BedrockMistral7B{bedrockMistralOptions{maxTokens: 4096, temperature: 0.7}}
}

// BedrockMixtral8x7B represents Mixtral 8x7B Instruct on Bedrock
type BedrockMixtral8x7B struct{ bedrockMistralOptions }

func (m *BedrockMixtral8x7B) ModelName() string      { return "mistral.mixtral-8x7b-instruct-v0:1" }
func (m *BedrockMixtral8x7B) Provider() ProviderType { return ProviderBedrock }
func (m *BedrockMixtral8x7B) SystemPrompt() string   { return m.systemPrompt }

func (m *BedrockMixtral8x7B) WithMaxTokens(n int) *BedrockMixtral8x7B { m.maxTokens = n; return m }
func (m *BedrockMixtral8x7B) WithTemperature(t float64) *BedrockMixtral8x7B {
	m.temperature = t
	return m
}
func (m *BedrockMixtral8x7B) WithTopP(p float64) *BedrockMixtral8x7B { m.topP = p; return m }
func (m *BedrockMixtral8x7B) WithTopK(k int) *BedrockMixtral8x7B     { m.topK = k; return m }
func (m *BedrockMixtral8x7B) WithSystemPrompt(s string) *BedrockMixtral8x7B {
	m.systemPrompt = s
	return m
}

// NewBedrockMixtral8x7B creates a new Mixtral 8x7B Instruct model for Bedrock
func NewBedrockMixtral8x7B() *BedrockMixtral8x7B {
	return &BedrockMixtral8x7B{bedrockMistralOptions{maxTokens: 4096, temperature: 0.7}}
}

// BedrockMistralLarge represents Mistral Large (24.02) on Bedrock.
// Consider BedrockMistralLarge2407 for the newer revision.
type BedrockMistralLarge struct{ bedrockMistralOptions }

func (m *BedrockMistralLarge) ModelName() string      { return "mistral.mistral-large-2402-v1:0" }
func (m *BedrockMistralLarge) Provider() ProviderType { return ProviderBedrock }
func (m *BedrockMistralLarge) SystemPrompt() string   { return m.systemPrompt }

func (m *BedrockMistralLarge) WithMaxTokens(n int) *BedrockMistralLarge { m.maxTokens = n; return m }
func (m *BedrockMistralLarge) WithTemperature(t float64) *BedrockMistralLarge {
	m.temperature = t
	return m
}
func (m *BedrockMistralLarge) WithTopP(p float64) *BedrockMistralLarge { m.topP = p; return m }
func (m *BedrockMistralLarge) WithTopK(k int) *BedrockMistralLarge     { m.topK = k; return m }
func (m *BedrockMistralLarge) WithSystemPrompt(s string) *BedrockMistralLarge {
	m.systemPrompt = s
	return m
}

// NewBedrockMistralLarge creates a new Mistral Large model for Bedrock
func NewBedrockMistralLarge() *BedrockMistralLarge {
	return &BedrockMistralLarge{bedrockMistralOptions{maxTokens: 8192, temperature: 0.7}}
}

// BedrockMistralLarge2407 represents Mistral Large 2 (24.07) on Bedrock
type BedrockMistralLarge2407 struct{ bedrockMistralOptions }

func (m *BedrockMistralLarge2407) ModelName() string      { return "mistral.mistral-large-2407-v1:0" }
func (m *BedrockMistralLarge2407) Provider() ProviderType { return ProviderBedrock }
func (m *BedrockMistralLarge2407) SystemPrompt() string   { return m.systemPrompt }

func (m *BedrockMistralLarge2407) WithMaxTokens(n int) *BedrockMistralLarge2407 {
	m.maxTokens = n
	return m
}
func (m *BedrockMistralLarge2407) WithTemperature(t float64) *BedrockMistralLarge2407 {
	m.temperature = t
	return m
}
func (m *BedrockMistralLarge2407) WithTopP(p float64) *BedrockMistralLarge2407 { m.topP = p; return m }
func (m *BedrockMistralLarge2407) WithTopK(k int) *BedrockMistralLarge2407     { m.topK = k; return m }
func (m *BedrockMistralLarge2407) WithSystemPrompt(s string) *BedrockMistralLarge2407 {
	m.systemPrompt = s
	return m
}

// NewBedrockMistralLarge2407 creates a new Mistral Large 2 (24.07) model for Bedrock
func NewBedrockMistralLarge2407() *BedrockMistralLarge2407 {
	return &BedrockMistralLarge2407{bedrockMistralOptions{maxTokens: 8192, temperature: 0.7}}
}

// ============================================================================
// GENERIC BEDROCK MODEL
// ============================================================================

// BedrockModel represents a generic Bedrock model
// Use this for any model available in your Bedrock environment, including
// cross-region inference profile IDs (e.g. "us.anthropic.claude-opus-4-8"),
// which many newer models require outside their home regions.
type BedrockModel struct {
	modelID      string
	maxTokens    int
	temperature  float64
	topP         float64
	topK         int
	systemPrompt string
	modelFamily  string // "claude", "nova", "titan", "llama", "mistral"
}

func (m *BedrockModel) ModelName() string      { return m.modelID }
func (m *BedrockModel) Provider() ProviderType { return ProviderBedrock }
func (m *BedrockModel) SystemPrompt() string   { return m.systemPrompt }

func (m *BedrockModel) WithMaxTokens(n int) *BedrockModel       { m.maxTokens = n; return m }
func (m *BedrockModel) WithTemperature(t float64) *BedrockModel { m.temperature = t; return m }
func (m *BedrockModel) WithTopP(p float64) *BedrockModel        { m.topP = p; return m }
func (m *BedrockModel) WithTopK(k int) *BedrockModel            { m.topK = k; return m }
func (m *BedrockModel) WithSystemPrompt(s string) *BedrockModel { m.systemPrompt = s; return m }
func (m *BedrockModel) WithModelFamily(f string) *BedrockModel  { m.modelFamily = f; return m }

// NewBedrockModel creates a new generic Bedrock model with the specified model ID
// modelFamily should be one of: "claude", "nova", "titan", "llama", "mistral"
func NewBedrockModel(modelID, modelFamily string) *BedrockModel {
	return &BedrockModel{
		modelID:     modelID,
		modelFamily: modelFamily,
		maxTokens:   4096,
		temperature: 0.7,
	}
}

// ============================================================================
// BEDROCK PROVIDER CLIENT
// ============================================================================

// bedrockClient implements the Provider interface for AWS Bedrock
type bedrockClient struct {
	client      *bedrockruntime.Client
	timeout     time.Duration
	logger      Logger
	rateLimiter *rateLimiter
}

// newBedrockClient creates a new Bedrock client
func newBedrockClient(bedrockCfg *BedrockConfig, logger Logger) (*bedrockClient, error) {
	if bedrockCfg.Region == "" {
		return nil, fmt.Errorf("AWS region is required for Bedrock")
	}

	ctx := context.Background()

	// Build AWS config options
	var configOpts []func(*config.LoadOptions) error
	configOpts = append(configOpts, config.WithRegion(bedrockCfg.Region))

	if bedrockCfg.AccessKeyID != "" && bedrockCfg.SecretAccessKey != "" {
		// Use explicit credentials
		configOpts = append(configOpts, config.WithCredentialsProvider(
			credentials.NewStaticCredentialsProvider(
				bedrockCfg.AccessKeyID,
				bedrockCfg.SecretAccessKey,
				bedrockCfg.SessionToken,
			),
		))
	} else if bedrockCfg.Profile != "" {
		// Use named profile from ~/.aws/credentials or ~/.aws/config
		configOpts = append(configOpts, config.WithSharedConfigProfile(bedrockCfg.Profile))
	}
	// Otherwise, use default credential chain (IAM roles, environment variables, etc.)

	awsCfg, err := config.LoadDefaultConfig(ctx, configOpts...)
	if err != nil {
		return nil, fmt.Errorf("failed to load AWS config: %w", err)
	}

	client := bedrockruntime.NewFromConfig(awsCfg)

	timeout := bedrockCfg.Timeout
	if timeout == 0 {
		timeout = defaultTimeout()
	}

	return &bedrockClient{
		client:      client,
		timeout:     timeout,
		logger:      logger,
		rateLimiter: newRateLimiter(bedrockCfg.RateLimiter, logger),
	}, nil
}

// Bedrock request/response types for different model families

// Claude Messages API format
type bedrockClaudeRequest struct {
	AnthropicVersion string                 `json:"anthropic_version"`
	MaxTokens        int                    `json:"max_tokens"`
	Messages         []bedrockClaudeMessage `json:"messages"`
	System           string                 `json:"system,omitempty"`
	Temperature      float64                `json:"temperature,omitempty"`
	TopP             float64                `json:"top_p,omitempty"`
	TopK             int                    `json:"top_k,omitempty"`
}

type bedrockClaudeMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type bedrockClaudeResponse struct {
	Content    []bedrockClaudeContent `json:"content"`
	StopReason string                 `json:"stop_reason"`
	Usage      bedrockClaudeUsage     `json:"usage"`
}

type bedrockClaudeContent struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type bedrockClaudeUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// Titan format
type bedrockTitanRequest struct {
	InputText            string             `json:"inputText"`
	TextGenerationConfig bedrockTitanConfig `json:"textGenerationConfig"`
}

type bedrockTitanConfig struct {
	MaxTokenCount int     `json:"maxTokenCount"`
	Temperature   float64 `json:"temperature"`
	TopP          float64 `json:"topP"`
}

type bedrockTitanResponse struct {
	Results []bedrockTitanResult `json:"results"`
}

type bedrockTitanResult struct {
	OutputText       string `json:"outputText"`
	CompletionReason string `json:"completionReason"`
	TokenCount       int    `json:"tokenCount"`
}

// Nova format (messages-v1 schema)
type bedrockNovaRequest struct {
	SchemaVersion   string                `json:"schemaVersion"`
	Messages        []bedrockNovaMessage  `json:"messages"`
	System          []bedrockNovaText     `json:"system,omitempty"`
	InferenceConfig *bedrockNovaInference `json:"inferenceConfig,omitempty"`
}

type bedrockNovaText struct {
	Text string `json:"text"`
}

type bedrockNovaMessage struct {
	Role    string            `json:"role"`
	Content []bedrockNovaText `json:"content"`
}

type bedrockNovaInference struct {
	MaxTokens   int     `json:"maxTokens,omitempty"`
	Temperature float64 `json:"temperature,omitempty"`
	TopP        float64 `json:"topP,omitempty"`
	TopK        int     `json:"topK,omitempty"`
}

type bedrockNovaResponse struct {
	Output struct {
		Message struct {
			Role    string            `json:"role"`
			Content []bedrockNovaText `json:"content"`
		} `json:"message"`
	} `json:"output"`
	StopReason string `json:"stopReason"`
	Usage      struct {
		InputTokens  int `json:"inputTokens"`
		OutputTokens int `json:"outputTokens"`
		TotalTokens  int `json:"totalTokens"`
	} `json:"usage"`
}

// Llama format
type bedrockLlamaRequest struct {
	Prompt      string  `json:"prompt"`
	MaxGenLen   int     `json:"max_gen_len"`
	Temperature float64 `json:"temperature"`
	TopP        float64 `json:"top_p"`
}

type bedrockLlamaResponse struct {
	Generation           string `json:"generation"`
	StopReason           string `json:"stop_reason"`
	PromptTokenCount     int    `json:"prompt_token_count"`
	GenerationTokenCount int    `json:"generation_token_count"`
}

// Mistral format
type bedrockMistralRequest struct {
	Prompt      string  `json:"prompt"`
	MaxTokens   int     `json:"max_tokens"`
	Temperature float64 `json:"temperature,omitempty"`
	TopP        float64 `json:"top_p,omitempty"`
	TopK        int     `json:"top_k,omitempty"`
}

type bedrockMistralResponse struct {
	Outputs []bedrockMistralOutput `json:"outputs"`
}

type bedrockMistralOutput struct {
	Text       string `json:"text"`
	StopReason string `json:"stop_reason"`
}

// getModelFamily determines the model family from the model ID.
// Cross-region inference profile IDs prefix the base model ID with a routing
// scope (e.g. "us.anthropic.claude-...", "global.anthropic.claude-..."); many
// newer Bedrock models are only invokable through such profiles, so the prefix
// is stripped before matching the provider.
func getModelFamily(modelID string) string {
	id := modelID
	for _, scope := range []string{"us.", "eu.", "apac.", "jp.", "au.", "ca.", "sa.", "global."} {
		if strings.HasPrefix(id, scope) {
			id = strings.TrimPrefix(id, scope)
			break
		}
	}
	switch {
	case strings.HasPrefix(id, "anthropic."):
		return "claude"
	case strings.HasPrefix(id, "amazon.nova"):
		return "nova"
	case strings.HasPrefix(id, "amazon."):
		return "titan"
	case strings.HasPrefix(id, "meta."):
		return "llama"
	case strings.HasPrefix(id, "mistral."):
		return "mistral"
	default:
		return "unknown"
	}
}

// Generate generates text using AWS Bedrock
func (c *bedrockClient) Generate(ctx context.Context, model Model, prompt string) (*GenerationResponse, error) {
	// Verify model is for Bedrock
	if model.Provider() != ProviderBedrock {
		return nil, fmt.Errorf("model %s is not a Bedrock model", model.ModelName())
	}

	// Set timeout
	ctx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()

	modelID := model.ModelName()

	// Determine model family
	var modelFamily string
	if bm, ok := model.(*BedrockModel); ok {
		modelFamily = bm.modelFamily
	} else {
		modelFamily = getModelFamily(modelID)
	}

	c.logger.Debug().
		Str("model", modelID).
		Str("family", modelFamily).
		Msg("Making Bedrock API request")

	var body []byte
	var err error

	// Build request based on model family
	switch modelFamily {
	case "claude":
		body, err = c.buildClaudeRequest(model, prompt)
	case "nova":
		body, err = c.buildNovaRequest(model, prompt)
	case "titan":
		body, err = c.buildTitanRequest(model, prompt)
	case "llama":
		body, err = c.buildLlamaRequest(model, prompt)
	case "mistral":
		body, err = c.buildMistralRequest(model, prompt)
	default:
		return nil, fmt.Errorf("unsupported model family: %s", modelFamily)
	}
	if err != nil {
		return nil, err
	}

	// Make request with rate limit handling
	var output *bedrockruntime.InvokeModelOutput
	err = c.rateLimiter.Execute(ctx, func() error {
		var reqErr error
		output, reqErr = c.client.InvokeModel(ctx, &bedrockruntime.InvokeModelInput{
			ModelId:     aws.String(modelID),
			Body:        body,
			ContentType: aws.String("application/json"),
		})
		return reqErr
	})
	if err != nil {
		c.logger.Error().
			Err(err).
			Str("model", modelID).
			Str("prompt_preview", truncateString(prompt, 100)).
			Msg("Bedrock generation failed")
		return nil, fmt.Errorf("bedrock generation failed: %w", err)
	}

	// Parse response based on model family
	var response *GenerationResponse
	switch modelFamily {
	case "claude":
		response, err = c.parseClaudeResponse(output.Body, modelID)
	case "nova":
		response, err = c.parseNovaResponse(output.Body, modelID)
	case "titan":
		response, err = c.parseTitanResponse(output.Body, modelID)
	case "llama":
		response, err = c.parseLlamaResponse(output.Body, modelID)
	case "mistral":
		response, err = c.parseMistralResponse(output.Body, modelID)
	}
	if err != nil {
		return nil, err
	}

	c.logger.Debug().
		Str("model", modelID).
		Int("prompt_tokens", response.Usage.PromptTokens).
		Int("completion_tokens", response.Usage.CompletionTokens).
		Int("total_tokens", response.Usage.TotalTokens).
		Msg("Bedrock generation completed")

	return response, nil
}

func (c *bedrockClient) buildClaudeRequest(model Model, prompt string) ([]byte, error) {
	req := bedrockClaudeRequest{
		AnthropicVersion: "bedrock-2023-05-31",
		MaxTokens:        4096,
		Messages: []bedrockClaudeMessage{
			{Role: "user", Content: prompt},
		},
	}

	// Apply model-specific options
	switch m := model.(type) {
	case *BedrockClaude35Sonnet:
		if m.maxTokens > 0 {
			req.MaxTokens = m.maxTokens
		}
		if m.temperature > 0 {
			req.Temperature = m.temperature
		}
		if m.topP > 0 {
			req.TopP = m.topP
		}
		if m.topK > 0 {
			req.TopK = m.topK
		}
		if m.systemPrompt != "" {
			req.System = m.systemPrompt
		}
	case *BedrockClaude35Haiku:
		if m.maxTokens > 0 {
			req.MaxTokens = m.maxTokens
		}
		if m.temperature > 0 {
			req.Temperature = m.temperature
		}
		if m.topP > 0 {
			req.TopP = m.topP
		}
		if m.topK > 0 {
			req.TopK = m.topK
		}
		if m.systemPrompt != "" {
			req.System = m.systemPrompt
		}
	case *BedrockClaude37Sonnet:
		if m.maxTokens > 0 {
			req.MaxTokens = m.maxTokens
		}
		if m.temperature > 0 {
			req.Temperature = m.temperature
		}
		if m.topP > 0 {
			req.TopP = m.topP
		}
		if m.topK > 0 {
			req.TopK = m.topK
		}
		if m.systemPrompt != "" {
			req.System = m.systemPrompt
		}
	case *BedrockClaudeSonnet4:
		if m.maxTokens > 0 {
			req.MaxTokens = m.maxTokens
		}
		if m.temperature > 0 {
			req.Temperature = m.temperature
		}
		if m.topP > 0 {
			req.TopP = m.topP
		}
		if m.topK > 0 {
			req.TopK = m.topK
		}
		if m.systemPrompt != "" {
			req.System = m.systemPrompt
		}
	case *BedrockClaudeOpus4:
		if m.maxTokens > 0 {
			req.MaxTokens = m.maxTokens
		}
		if m.temperature > 0 {
			req.Temperature = m.temperature
		}
		if m.topP > 0 {
			req.TopP = m.topP
		}
		if m.topK > 0 {
			req.TopK = m.topK
		}
		if m.systemPrompt != "" {
			req.System = m.systemPrompt
		}
	case *BedrockClaudeSonnet45:
		if m.maxTokens > 0 {
			req.MaxTokens = m.maxTokens
		}
		if m.temperature > 0 {
			req.Temperature = m.temperature
		}
		if m.topP > 0 {
			req.TopP = m.topP
		}
		if m.topK > 0 {
			req.TopK = m.topK
		}
		if m.systemPrompt != "" {
			req.System = m.systemPrompt
		}
	case *BedrockClaudeOpus45:
		if m.maxTokens > 0 {
			req.MaxTokens = m.maxTokens
		}
		if m.temperature > 0 {
			req.Temperature = m.temperature
		}
		if m.topP > 0 {
			req.TopP = m.topP
		}
		if m.topK > 0 {
			req.TopK = m.topK
		}
		if m.systemPrompt != "" {
			req.System = m.systemPrompt
		}
	case *BedrockClaudeHaiku45:
		if m.maxTokens > 0 {
			req.MaxTokens = m.maxTokens
		}
		if m.temperature > 0 {
			req.Temperature = m.temperature
		}
		if m.topP > 0 {
			req.TopP = m.topP
		}
		if m.topK > 0 {
			req.TopK = m.topK
		}
		if m.systemPrompt != "" {
			req.System = m.systemPrompt
		}
	case *BedrockClaudeOpus46:
		if m.maxTokens > 0 {
			req.MaxTokens = m.maxTokens
		}
		if m.temperature > 0 {
			req.Temperature = m.temperature
		}
		if m.topP > 0 {
			req.TopP = m.topP
		}
		if m.topK > 0 {
			req.TopK = m.topK
		}
		if m.systemPrompt != "" {
			req.System = m.systemPrompt
		}
	case *BedrockClaudeSonnet46:
		if m.maxTokens > 0 {
			req.MaxTokens = m.maxTokens
		}
		if m.temperature > 0 {
			req.Temperature = m.temperature
		}
		if m.topP > 0 {
			req.TopP = m.topP
		}
		if m.topK > 0 {
			req.TopK = m.topK
		}
		if m.systemPrompt != "" {
			req.System = m.systemPrompt
		}
	// Opus 4.7/4.8 and Fable 5 reject sampling parameters (temperature/topP/topK)
	// with a 400 error; only max_tokens and the system prompt are sent.
	case *BedrockClaudeOpus47:
		if m.maxTokens > 0 {
			req.MaxTokens = m.maxTokens
		}
		if m.systemPrompt != "" {
			req.System = m.systemPrompt
		}
	case *BedrockClaudeOpus48:
		if m.maxTokens > 0 {
			req.MaxTokens = m.maxTokens
		}
		if m.systemPrompt != "" {
			req.System = m.systemPrompt
		}
	case *BedrockClaudeFable5:
		if m.maxTokens > 0 {
			req.MaxTokens = m.maxTokens
		}
		if m.systemPrompt != "" {
			req.System = m.systemPrompt
		}
	case *BedrockClaude3Sonnet:
		if m.maxTokens > 0 {
			req.MaxTokens = m.maxTokens
		}
		if m.temperature > 0 {
			req.Temperature = m.temperature
		}
		if m.topP > 0 {
			req.TopP = m.topP
		}
		if m.topK > 0 {
			req.TopK = m.topK
		}
		if m.systemPrompt != "" {
			req.System = m.systemPrompt
		}
	case *BedrockClaude3Haiku:
		if m.maxTokens > 0 {
			req.MaxTokens = m.maxTokens
		}
		if m.temperature > 0 {
			req.Temperature = m.temperature
		}
		if m.topP > 0 {
			req.TopP = m.topP
		}
		if m.topK > 0 {
			req.TopK = m.topK
		}
		if m.systemPrompt != "" {
			req.System = m.systemPrompt
		}
	case *BedrockClaude3Opus:
		if m.maxTokens > 0 {
			req.MaxTokens = m.maxTokens
		}
		if m.temperature > 0 {
			req.Temperature = m.temperature
		}
		if m.topP > 0 {
			req.TopP = m.topP
		}
		if m.topK > 0 {
			req.TopK = m.topK
		}
		if m.systemPrompt != "" {
			req.System = m.systemPrompt
		}
	case *BedrockModel:
		if m.maxTokens > 0 {
			req.MaxTokens = m.maxTokens
		}
		if m.temperature > 0 {
			req.Temperature = m.temperature
		}
		if m.topP > 0 {
			req.TopP = m.topP
		}
		if m.topK > 0 {
			req.TopK = m.topK
		}
		if m.systemPrompt != "" {
			req.System = m.systemPrompt
		}
	}

	return json.Marshal(req)
}

func (c *bedrockClient) buildNovaRequest(model Model, prompt string) ([]byte, error) {
	req := bedrockNovaRequest{
		SchemaVersion: "messages-v1",
		Messages: []bedrockNovaMessage{
			{Role: "user", Content: []bedrockNovaText{{Text: prompt}}},
		},
		InferenceConfig: &bedrockNovaInference{MaxTokens: 4096},
	}

	if model.SystemPrompt() != "" {
		req.System = []bedrockNovaText{{Text: model.SystemPrompt()}}
	}

	applyOpts := func(maxTokens int, temperature, topP float64, topK int) {
		if maxTokens > 0 {
			req.InferenceConfig.MaxTokens = maxTokens
		}
		if temperature > 0 {
			req.InferenceConfig.Temperature = temperature
		}
		if topP > 0 {
			req.InferenceConfig.TopP = topP
		}
		if topK > 0 {
			req.InferenceConfig.TopK = topK
		}
	}

	switch m := model.(type) {
	case *BedrockNovaMicro:
		applyOpts(m.maxTokens, m.temperature, m.topP, m.topK)
	case *BedrockNovaLite:
		applyOpts(m.maxTokens, m.temperature, m.topP, m.topK)
	case *BedrockNovaPro:
		applyOpts(m.maxTokens, m.temperature, m.topP, m.topK)
	case *BedrockNovaPremier:
		applyOpts(m.maxTokens, m.temperature, m.topP, m.topK)
	case *BedrockModel:
		applyOpts(m.maxTokens, m.temperature, m.topP, m.topK)
	}

	return json.Marshal(req)
}

func (c *bedrockClient) buildTitanRequest(model Model, prompt string) ([]byte, error) {
	req := bedrockTitanRequest{
		InputText: prompt,
		TextGenerationConfig: bedrockTitanConfig{
			MaxTokenCount: 4096,
			Temperature:   0.7,
			TopP:          0.9,
		},
	}

	// Prepend system prompt if set
	if model.SystemPrompt() != "" {
		req.InputText = model.SystemPrompt() + "\n\n" + prompt
	}

	// Apply model-specific options
	switch m := model.(type) {
	case *BedrockTitanTextExpress:
		if m.maxTokens > 0 {
			req.TextGenerationConfig.MaxTokenCount = m.maxTokens
		}
		if m.temperature > 0 {
			req.TextGenerationConfig.Temperature = m.temperature
		}
		if m.topP > 0 {
			req.TextGenerationConfig.TopP = m.topP
		}
	case *BedrockTitanTextLite:
		if m.maxTokens > 0 {
			req.TextGenerationConfig.MaxTokenCount = m.maxTokens
		}
		if m.temperature > 0 {
			req.TextGenerationConfig.Temperature = m.temperature
		}
		if m.topP > 0 {
			req.TextGenerationConfig.TopP = m.topP
		}
	case *BedrockTitanTextPremier:
		if m.maxTokens > 0 {
			req.TextGenerationConfig.MaxTokenCount = m.maxTokens
		}
		if m.temperature > 0 {
			req.TextGenerationConfig.Temperature = m.temperature
		}
		if m.topP > 0 {
			req.TextGenerationConfig.TopP = m.topP
		}
	case *BedrockModel:
		if m.maxTokens > 0 {
			req.TextGenerationConfig.MaxTokenCount = m.maxTokens
		}
		if m.temperature > 0 {
			req.TextGenerationConfig.Temperature = m.temperature
		}
		if m.topP > 0 {
			req.TextGenerationConfig.TopP = m.topP
		}
	}

	return json.Marshal(req)
}

func (c *bedrockClient) buildLlamaRequest(model Model, prompt string) ([]byte, error) {
	// Build Llama prompt format
	var fullPrompt string
	if model.SystemPrompt() != "" {
		fullPrompt = fmt.Sprintf("<s>[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]", model.SystemPrompt(), prompt)
	} else {
		fullPrompt = fmt.Sprintf("<s>[INST] %s [/INST]", prompt)
	}

	req := bedrockLlamaRequest{
		Prompt:      fullPrompt,
		MaxGenLen:   2048,
		Temperature: 0.6,
		TopP:        0.9,
	}

	// Apply model-specific options
	switch m := model.(type) {
	case *BedrockLlama31Instruct8B:
		if m.maxTokens > 0 {
			req.MaxGenLen = m.maxTokens
		}
		if m.temperature > 0 {
			req.Temperature = m.temperature
		}
		if m.topP > 0 {
			req.TopP = m.topP
		}
	case *BedrockLlama31Instruct70B:
		if m.maxTokens > 0 {
			req.MaxGenLen = m.maxTokens
		}
		if m.temperature > 0 {
			req.Temperature = m.temperature
		}
		if m.topP > 0 {
			req.TopP = m.topP
		}
	case *BedrockLlama31Instruct405B:
		if m.maxTokens > 0 {
			req.MaxGenLen = m.maxTokens
		}
		if m.temperature > 0 {
			req.Temperature = m.temperature
		}
		if m.topP > 0 {
			req.TopP = m.topP
		}
	case *BedrockLlama32Instruct1B:
		if m.maxTokens > 0 {
			req.MaxGenLen = m.maxTokens
		}
		if m.temperature > 0 {
			req.Temperature = m.temperature
		}
		if m.topP > 0 {
			req.TopP = m.topP
		}
	case *BedrockLlama32Instruct3B:
		if m.maxTokens > 0 {
			req.MaxGenLen = m.maxTokens
		}
		if m.temperature > 0 {
			req.Temperature = m.temperature
		}
		if m.topP > 0 {
			req.TopP = m.topP
		}
	case *BedrockLlama33Instruct70B:
		if m.maxTokens > 0 {
			req.MaxGenLen = m.maxTokens
		}
		if m.temperature > 0 {
			req.Temperature = m.temperature
		}
		if m.topP > 0 {
			req.TopP = m.topP
		}
	case *BedrockLlama4Scout:
		if m.maxTokens > 0 {
			req.MaxGenLen = m.maxTokens
		}
		if m.temperature > 0 {
			req.Temperature = m.temperature
		}
		if m.topP > 0 {
			req.TopP = m.topP
		}
	case *BedrockLlama4Maverick:
		if m.maxTokens > 0 {
			req.MaxGenLen = m.maxTokens
		}
		if m.temperature > 0 {
			req.Temperature = m.temperature
		}
		if m.topP > 0 {
			req.TopP = m.topP
		}
	case *BedrockModel:
		if m.maxTokens > 0 {
			req.MaxGenLen = m.maxTokens
		}
		if m.temperature > 0 {
			req.Temperature = m.temperature
		}
		if m.topP > 0 {
			req.TopP = m.topP
		}
	}

	return json.Marshal(req)
}

func (c *bedrockClient) buildMistralRequest(model Model, prompt string) ([]byte, error) {
	// Build Mistral prompt format
	var fullPrompt string
	if model.SystemPrompt() != "" {
		fullPrompt = fmt.Sprintf("<s>[INST] %s\n\n%s [/INST]", model.SystemPrompt(), prompt)
	} else {
		fullPrompt = fmt.Sprintf("<s>[INST] %s [/INST]", prompt)
	}

	req := bedrockMistralRequest{
		Prompt:      fullPrompt,
		MaxTokens:   4096,
		Temperature: 0.7,
		TopP:        0.9,
	}

	// Apply model-specific options
	switch m := model.(type) {
	case *BedrockMistral7B:
		if m.maxTokens > 0 {
			req.MaxTokens = m.maxTokens
		}
		if m.temperature > 0 {
			req.Temperature = m.temperature
		}
		if m.topP > 0 {
			req.TopP = m.topP
		}
		if m.topK > 0 {
			req.TopK = m.topK
		}
	case *BedrockMixtral8x7B:
		if m.maxTokens > 0 {
			req.MaxTokens = m.maxTokens
		}
		if m.temperature > 0 {
			req.Temperature = m.temperature
		}
		if m.topP > 0 {
			req.TopP = m.topP
		}
		if m.topK > 0 {
			req.TopK = m.topK
		}
	case *BedrockMistralLarge:
		if m.maxTokens > 0 {
			req.MaxTokens = m.maxTokens
		}
		if m.temperature > 0 {
			req.Temperature = m.temperature
		}
		if m.topP > 0 {
			req.TopP = m.topP
		}
		if m.topK > 0 {
			req.TopK = m.topK
		}
	case *BedrockMistralLarge2407:
		if m.maxTokens > 0 {
			req.MaxTokens = m.maxTokens
		}
		if m.temperature > 0 {
			req.Temperature = m.temperature
		}
		if m.topP > 0 {
			req.TopP = m.topP
		}
		if m.topK > 0 {
			req.TopK = m.topK
		}
	case *BedrockModel:
		if m.maxTokens > 0 {
			req.MaxTokens = m.maxTokens
		}
		if m.temperature > 0 {
			req.Temperature = m.temperature
		}
		if m.topP > 0 {
			req.TopP = m.topP
		}
		if m.topK > 0 {
			req.TopK = m.topK
		}
	}

	return json.Marshal(req)
}

func (c *bedrockClient) parseClaudeResponse(body []byte, modelID string) (*GenerationResponse, error) {
	var resp bedrockClaudeResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("failed to parse Claude response: %w", err)
	}

	// Safety classifiers (notably on Claude Fable 5) can decline a request with
	// stop_reason "refusal" and empty or partial content.
	if resp.StopReason == "refusal" {
		return nil, fmt.Errorf("claude declined the request (stop_reason: refusal) for model %s; retry on a different model such as anthropic.claude-opus-4-8", modelID)
	}

	if len(resp.Content) == 0 {
		return nil, fmt.Errorf("no content in Claude response")
	}

	var text string
	for _, content := range resp.Content {
		if content.Type == "text" {
			text += content.Text
		}
	}

	return &GenerationResponse{
		Text:         text,
		Model:        modelID,
		FinishReason: resp.StopReason,
		Usage: TokenUsage{
			PromptTokens:     resp.Usage.InputTokens,
			CompletionTokens: resp.Usage.OutputTokens,
			TotalTokens:      resp.Usage.InputTokens + resp.Usage.OutputTokens,
		},
		Metadata: map[string]string{
			"provider": "bedrock",
			"model":    modelID,
			"family":   "claude",
		},
	}, nil
}

func (c *bedrockClient) parseNovaResponse(body []byte, modelID string) (*GenerationResponse, error) {
	var resp bedrockNovaResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("failed to parse Nova response: %w", err)
	}

	if len(resp.Output.Message.Content) == 0 {
		return nil, fmt.Errorf("no content in Nova response")
	}

	var text string
	for _, content := range resp.Output.Message.Content {
		text += content.Text
	}

	totalTokens := resp.Usage.TotalTokens
	if totalTokens == 0 {
		totalTokens = resp.Usage.InputTokens + resp.Usage.OutputTokens
	}

	return &GenerationResponse{
		Text:         text,
		Model:        modelID,
		FinishReason: resp.StopReason,
		Usage: TokenUsage{
			PromptTokens:     resp.Usage.InputTokens,
			CompletionTokens: resp.Usage.OutputTokens,
			TotalTokens:      totalTokens,
		},
		Metadata: map[string]string{
			"provider": "bedrock",
			"model":    modelID,
			"family":   "nova",
		},
	}, nil
}

func (c *bedrockClient) parseTitanResponse(body []byte, modelID string) (*GenerationResponse, error) {
	var resp bedrockTitanResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("failed to parse Titan response: %w", err)
	}

	if len(resp.Results) == 0 {
		return nil, fmt.Errorf("no results in Titan response")
	}

	result := resp.Results[0]
	return &GenerationResponse{
		Text:         result.OutputText,
		Model:        modelID,
		FinishReason: result.CompletionReason,
		Usage: TokenUsage{
			CompletionTokens: result.TokenCount,
			TotalTokens:      result.TokenCount,
		},
		Metadata: map[string]string{
			"provider": "bedrock",
			"model":    modelID,
			"family":   "titan",
		},
	}, nil
}

func (c *bedrockClient) parseLlamaResponse(body []byte, modelID string) (*GenerationResponse, error) {
	var resp bedrockLlamaResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("failed to parse Llama response: %w", err)
	}

	return &GenerationResponse{
		Text:         resp.Generation,
		Model:        modelID,
		FinishReason: resp.StopReason,
		Usage: TokenUsage{
			PromptTokens:     resp.PromptTokenCount,
			CompletionTokens: resp.GenerationTokenCount,
			TotalTokens:      resp.PromptTokenCount + resp.GenerationTokenCount,
		},
		Metadata: map[string]string{
			"provider": "bedrock",
			"model":    modelID,
			"family":   "llama",
		},
	}, nil
}

func (c *bedrockClient) parseMistralResponse(body []byte, modelID string) (*GenerationResponse, error) {
	var resp bedrockMistralResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("failed to parse Mistral response: %w", err)
	}

	if len(resp.Outputs) == 0 {
		return nil, fmt.Errorf("no outputs in Mistral response")
	}

	output := resp.Outputs[0]
	return &GenerationResponse{
		Text:         output.Text,
		Model:        modelID,
		FinishReason: output.StopReason,
		Usage:        TokenUsage{}, // Mistral doesn't return token counts
		Metadata: map[string]string{
			"provider": "bedrock",
			"model":    modelID,
			"family":   "mistral",
		},
	}, nil
}

// Health checks the health of the Bedrock client
func (c *bedrockClient) Health(ctx context.Context) error {
	ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	// Use a simple Titan model for health check (most widely available)
	req := bedrockTitanRequest{
		InputText: "Hello",
		TextGenerationConfig: bedrockTitanConfig{
			MaxTokenCount: 5,
			Temperature:   0.5,
			TopP:          0.9,
		},
	}

	body, err := json.Marshal(req)
	if err != nil {
		return fmt.Errorf("bedrock health check failed: %w", err)
	}

	_, err = c.client.InvokeModel(ctx, &bedrockruntime.InvokeModelInput{
		ModelId:     aws.String("amazon.titan-text-lite-v1"),
		Body:        body,
		ContentType: aws.String("application/json"),
	})
	if err != nil {
		return fmt.Errorf("bedrock health check failed: %w", err)
	}

	return nil
}

// Close closes the Bedrock client (no-op for AWS SDK)
func (c *bedrockClient) Close() error {
	return nil
}
