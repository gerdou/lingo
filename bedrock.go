package lingo

import (
	"context"
	"encoding/json"
	"fmt"
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

// BedrockMistralLarge represents Mistral Large on Bedrock
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

// ============================================================================
// GENERIC BEDROCK MODEL
// ============================================================================

// BedrockModel represents a generic Bedrock model
// Use this for any model available in your Bedrock environment
type BedrockModel struct {
	modelID      string
	maxTokens    int
	temperature  float64
	topP         float64
	topK         int
	systemPrompt string
	modelFamily  string // "claude", "titan", "llama", "mistral"
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
// modelFamily should be one of: "claude", "titan", "llama", "mistral"
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

// getModelFamily determines the model family from the model ID
func getModelFamily(modelID string) string {
	switch {
	case len(modelID) >= 9 && modelID[:9] == "anthropic":
		return "claude"
	case len(modelID) >= 6 && modelID[:6] == "amazon":
		return "titan"
	case len(modelID) >= 4 && modelID[:4] == "meta":
		return "llama"
	case len(modelID) >= 7 && modelID[:7] == "mistral":
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
