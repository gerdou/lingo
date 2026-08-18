package lingo

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/document"
	brtypes "github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"github.com/aws/smithy-go/middleware"
	smithyhttp "github.com/aws/smithy-go/transport/http"
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
	// HealthCheckModel is the model id Health generates five tokens against.
	// When empty -- the default -- Health makes no inference call at all and
	// asks Bedrock a question that names no model, which is what keeps the
	// check from failing over a model the account happens not to have. Set it
	// only when "is this specific model servable" is the question you want
	// answered.
	HealthCheckModel string
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

// bedrockCacheOptions carries the prompt caching configuration. Bedrock has one
// options struct per model family instead of a single shared one, so the
// configuration lives here and every family embeds it; this one accessor is
// what makes all Bedrock models satisfy CacheableModel. Only the Claude family
// has a breakpoint to place -- elsewhere the setting is inert, which is the
// documented behaviour for asking a provider for caching it cannot do.
type bedrockCacheOptions struct {
	cache CacheOptions
}

// CacheOptions returns the model's prompt caching configuration.
func (o *bedrockCacheOptions) CacheOptions() *CacheOptions { return &o.cache }

// bedrockThinkingOptions carries the thinking configuration. Like
// bedrockCacheOptions it is a one-field struct the families embed, but only two
// of them do: Claude, whose InvokeModel body is Anthropic's own and therefore has
// a thinking field to fill in, and Nova, which reasons upstream but is wired to
// nothing here (see the comment on buildConverseInput). Titan, Llama and Mistral
// deliberately do not embed it, so those model types structurally cannot satisfy
// ThinkingModel and no thinking knob can be handed to a model whose API has none.
type bedrockThinkingOptions struct {
	thinking ThinkingOptions
}

// ThinkingOptions returns the model's thinking configuration.
func (o *bedrockThinkingOptions) ThinkingOptions() *ThinkingOptions { return &o.thinking }

// bedrockClaudeOptions contains options for Claude models on Bedrock
type bedrockClaudeOptions struct {
	bedrockCacheOptions
	bedrockThinkingOptions
	maxTokens        int
	temperature      float64
	topP             float64
	topK             int
	systemPrompt     string
	anthropicVersion string
}

// bedrockTitanOptions contains options for Amazon Titan models on Bedrock
type bedrockTitanOptions struct {
	bedrockCacheOptions
	maxTokens    int
	temperature  float64
	topP         float64
	systemPrompt string
}

// thinkingDimensions answers for a family whose request body has no thinking
// field of any kind. Without it ModelThinkingDimensions would fall back to the
// provider-wide answer and promise Titan a knob it has never had -- the models
// cannot even carry the configuration, so there is nothing to be vague about.
func (o *bedrockTitanOptions) thinkingDimensions() ThinkingDimension { return 0 }

// bedrockNovaOptions contains options for Amazon Nova models on Bedrock
type bedrockNovaOptions struct {
	bedrockCacheOptions
	bedrockThinkingOptions
	maxTokens    int
	temperature  float64
	topP         float64
	topK         int
	systemPrompt string
}

// thinkingDimensions reports that lingo asks Nova for nothing. Nova's reasoning
// config would have to ride in the Converse AdditionalModelRequestFields
// document, whose key name for it is not modelled by bedrockruntime and could
// not be verified from any pinned source, so the configuration is stored and
// never sent. Answering 0 is what keeps ModelThinkingDimensions honest about it.
func (o *bedrockNovaOptions) thinkingDimensions() ThinkingDimension { return 0 }

// bedrockLlamaOptions contains options for Llama models on Bedrock
type bedrockLlamaOptions struct {
	bedrockCacheOptions
	maxTokens    int
	temperature  float64
	topP         float64
	systemPrompt string
}

// thinkingDimensions answers for a family whose request body has no thinking
// field of any kind. See bedrockTitanOptions.thinkingDimensions.
func (o *bedrockLlamaOptions) thinkingDimensions() ThinkingDimension { return 0 }

// bedrockMistralOptions contains options for Mistral models on Bedrock
type bedrockMistralOptions struct {
	bedrockCacheOptions
	maxTokens    int
	temperature  float64
	topP         float64
	topK         int
	systemPrompt string
}

// thinkingDimensions answers for a family whose request body has no thinking
// field of any kind. See bedrockTitanOptions.thinkingDimensions.
func (o *bedrockMistralOptions) thinkingDimensions() ThinkingDimension { return 0 }

// ============================================================================
// CLAUDE THINKING ON BEDROCK
// ============================================================================
//
// The Claude models on Bedrock are the models Anthropic serves directly, so
// which thinking knobs they honour is the same question anthropic.go already
// answers -- by API generation, resolved from the model id. The only difference
// is the id itself, which Bedrock decorates with a cross-region scope and a
// vendor prefix, and the dialect lingo can write into the InvokeModel body.
//
// That dialect is narrower than the first-party one. InvokeModel forwards the
// body to the vendor verbatim, which is why the fixed thinking config lands here
// in Anthropic's own spelling, exactly as the cache breakpoint does. But the two
// newer shapes -- the adaptive thinking config and output_config.effort -- could
// not be verified as accepted on this path from any pinned source, and a body
// field Bedrock does not know is a 400, not a silent drop. So lingo writes the
// one shape Bedrock has taken since Claude 3.7 and no other, and the eras that
// accept nothing else report no depth knob at all rather than sending a guess.

// bedrockScopes are the cross-region inference prefixes a Bedrock model id can
// carry. They say where the request may be routed, not which model it is.
var bedrockScopes = []string{"us.", "eu.", "apac.", "jp.", "au.", "ca.", "sa.", "global."}

// bedrockStripScope removes a cross-region inference prefix from a model id.
func bedrockStripScope(modelID string) string {
	for _, scope := range bedrockScopes {
		if strings.HasPrefix(modelID, scope) {
			return strings.TrimPrefix(modelID, scope)
		}
	}
	return modelID
}

// bedrockResolveModelID reduces whatever the caller put in the model id down to
// the plain vendor id every classifier in this file is written against.
//
// Bedrock's ModelId parameter takes four spellings, and until this existed only
// the first two were understood:
//
//	meta.llama4-scout-17b-instruct-v1:0     a base model id
//	us.meta.llama4-scout-17b-instruct-v1:0  a cross-region inference profile id
//	arn:aws:bedrock:us-east-1:1234:inference-profile/us.meta.llama4-scout-17b-instruct-v1:0
//	arn:aws:bedrock:us-east-1:1234:provisioned-model/abc123
//
// An ARN carries its subject in the segment after the last "/", so taking that
// segment and then stripping the scope yields exactly the string a caller who
// passed the id directly would have given. No Bedrock model id contains a "/",
// so the cut is a no-op on the first two forms and needs no test on the "arn:"
// prefix to stay safe.
//
// A provisioned-throughput ARN names an opaque id instead, and reduces to
// "abc123": no vendor, no generation, nothing to classify. That is not an error
// here. It matches nothing, so every classifier lands on the default it has
// always used for an id it does not recognise -- "unknown" for the family, the
// 3.x template for Llama, no thinking dialect for Claude.
func bedrockResolveModelID(modelID string) string {
	if i := strings.LastIndex(modelID, "/"); i >= 0 {
		modelID = modelID[i+1:]
	}
	return bedrockStripScope(modelID)
}

// bedrockClaudeThinkingEra resolves a Bedrock model id to the Claude generation
// whose thinking dialect it speaks, by reducing the id to the upstream Anthropic
// one the era table in anthropic.go is written against:
//
//	us.anthropic.claude-opus-4-6-v1 -> claude-opus-4-6-v1 -> 4.6
//
// An inference-profile ARN reduces to the same thing, which is what makes the
// thinking config reachable on a Claude invoked through one: an unresolved ARN
// starts with neither "claude-" nor a known prefix, so it used to land on the
// era with no thinking field and drop the caller's thinking config on the floor.
//
// A genuinely non-Anthropic id still reduces to something the era table does not
// know, and still lands on the era with no thinking field at all.
func bedrockClaudeThinkingEra(modelID string) anthropicThinkingEra {
	return anthropicThinkingEraFor(strings.TrimPrefix(bedrockResolveModelID(modelID), "anthropic."))
}

// bedrockEraDimensions reports which thinking knobs lingo can actually ask a
// Bedrock Claude generation for, which is a subset of what the model honours
// first-party:
//
//	3.5 and earlier   nothing: the API has no thinking field
//	3.7 - 4.5         toggle | budget: the fixed thinking config, as always
//	4.6               toggle | budget: the fixed config is deprecated upstream
//	                  but still accepted, and it is the only one lingo can write
//	4.7, 4.8, 5.x     toggle only: a fixed budget is rejected, so thinking can be
//	                  switched off but its depth cannot be set from here
//	Fable 5           nothing: thinking is server-side and any thinking config
//	                  is a 400
//
// Every era that reasons reports its trace, whether or not lingo asked for it.
func bedrockEraDimensions(e anthropicThinkingEra) ThinkingDimension {
	switch e {
	case anthropicThinkingEraBudget, anthropicThinkingEraAdaptiveBudget:
		return ThinkingCanToggle | ThinkingCanSetBudget | ThinkingCanReportTrace
	case anthropicThinkingEraAdaptive, anthropicThinkingEraDefaultOn:
		return ThinkingCanToggle | ThinkingCanReportTrace
	case anthropicThinkingEraAlwaysOn:
		return ThinkingCanReportTrace
	default:
		return 0
	}
}

// bedrockThinkingDimensions answers ModelThinkingDimensions for one Bedrock
// model, resolved from the model id so that a zero-value literal and the generic
// BedrockModel both get the right answer without a constructor having stored
// anything.
func bedrockThinkingDimensions(modelID string) ThinkingDimension {
	return bedrockEraDimensions(bedrockClaudeThinkingEra(modelID))
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
func (m *BedrockClaude35Sonnet) thinkingDimensions() ThinkingDimension {
	return bedrockThinkingDimensions(m.ModelName())
}

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
func (m *BedrockClaude35Haiku) thinkingDimensions() ThinkingDimension {
	return bedrockThinkingDimensions(m.ModelName())
}

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
func (m *BedrockClaude37Sonnet) thinkingDimensions() ThinkingDimension {
	return bedrockThinkingDimensions(m.ModelName())
}

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
func (m *BedrockClaudeSonnet4) thinkingDimensions() ThinkingDimension {
	return bedrockThinkingDimensions(m.ModelName())
}

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
func (m *BedrockClaudeOpus4) thinkingDimensions() ThinkingDimension {
	return bedrockThinkingDimensions(m.ModelName())
}

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
func (m *BedrockClaudeSonnet45) thinkingDimensions() ThinkingDimension {
	return bedrockThinkingDimensions(m.ModelName())
}

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
func (m *BedrockClaudeOpus45) thinkingDimensions() ThinkingDimension {
	return bedrockThinkingDimensions(m.ModelName())
}

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
func (m *BedrockClaudeHaiku45) thinkingDimensions() ThinkingDimension {
	return bedrockThinkingDimensions(m.ModelName())
}

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
func (m *BedrockClaudeOpus46) thinkingDimensions() ThinkingDimension {
	return bedrockThinkingDimensions(m.ModelName())
}

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
func (m *BedrockClaudeSonnet46) thinkingDimensions() ThinkingDimension {
	return bedrockThinkingDimensions(m.ModelName())
}

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
func (m *BedrockClaudeOpus47) thinkingDimensions() ThinkingDimension {
	return bedrockThinkingDimensions(m.ModelName())
}

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
func (m *BedrockClaudeOpus48) thinkingDimensions() ThinkingDimension {
	return bedrockThinkingDimensions(m.ModelName())
}

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
func (m *BedrockClaudeFable5) thinkingDimensions() ThinkingDimension {
	return bedrockThinkingDimensions(m.ModelName())
}

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

// BedrockClaudeOpus5 represents Claude Opus 5 on Bedrock (current recommended).
// Opus 5 rejects sampling parameters (temperature/topP/topK) with a 400 error,
// so this type does not expose setters for them. Thinking is adaptive and on
// by default.
type BedrockClaudeOpus5 struct{ bedrockClaudeOptions }

func (m *BedrockClaudeOpus5) ModelName() string      { return "anthropic.claude-opus-5" }
func (m *BedrockClaudeOpus5) Provider() ProviderType { return ProviderBedrock }
func (m *BedrockClaudeOpus5) SystemPrompt() string   { return m.systemPrompt }
func (m *BedrockClaudeOpus5) thinkingDimensions() ThinkingDimension {
	return bedrockThinkingDimensions(m.ModelName())
}

func (m *BedrockClaudeOpus5) WithMaxTokens(n int) *BedrockClaudeOpus5 { m.maxTokens = n; return m }
func (m *BedrockClaudeOpus5) WithSystemPrompt(s string) *BedrockClaudeOpus5 {
	m.systemPrompt = s
	return m
}

// NewBedrockClaudeOpus5 creates a new Claude Opus 5 model for Bedrock
func NewBedrockClaudeOpus5() *BedrockClaudeOpus5 {
	return &BedrockClaudeOpus5{bedrockClaudeOptions{
		maxTokens:        8192,
		anthropicVersion: "bedrock-2023-05-31",
	}}
}

// BedrockClaudeSonnet5 represents Claude Sonnet 5 on Bedrock.
// Sonnet 5 rejects non-default sampling parameters (temperature/topP/topK) with
// a 400 error, so this type does not expose setters for them. Thinking is
// adaptive and on by default.
// Note: on Bedrock, a forced tool choice additionally requires thinking to be
// disabled — a constraint that does not apply on the first-party Claude API.
type BedrockClaudeSonnet5 struct{ bedrockClaudeOptions }

func (m *BedrockClaudeSonnet5) ModelName() string      { return "anthropic.claude-sonnet-5" }
func (m *BedrockClaudeSonnet5) Provider() ProviderType { return ProviderBedrock }
func (m *BedrockClaudeSonnet5) SystemPrompt() string   { return m.systemPrompt }
func (m *BedrockClaudeSonnet5) thinkingDimensions() ThinkingDimension {
	return bedrockThinkingDimensions(m.ModelName())
}

func (m *BedrockClaudeSonnet5) WithMaxTokens(n int) *BedrockClaudeSonnet5 { m.maxTokens = n; return m }
func (m *BedrockClaudeSonnet5) WithSystemPrompt(s string) *BedrockClaudeSonnet5 {
	m.systemPrompt = s
	return m
}

// NewBedrockClaudeSonnet5 creates a new Claude Sonnet 5 model for Bedrock
func NewBedrockClaudeSonnet5() *BedrockClaudeSonnet5 {
	return &BedrockClaudeSonnet5{bedrockClaudeOptions{
		maxTokens:        8192,
		anthropicVersion: "bedrock-2023-05-31",
	}}
}

// BedrockClaude3Sonnet represents Claude 3 Sonnet on Bedrock
type BedrockClaude3Sonnet struct{ bedrockClaudeOptions }

func (m *BedrockClaude3Sonnet) ModelName() string      { return "anthropic.claude-3-sonnet-20240229-v1:0" }
func (m *BedrockClaude3Sonnet) Provider() ProviderType { return ProviderBedrock }
func (m *BedrockClaude3Sonnet) SystemPrompt() string   { return m.systemPrompt }
func (m *BedrockClaude3Sonnet) thinkingDimensions() ThinkingDimension {
	return bedrockThinkingDimensions(m.ModelName())
}

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
func (m *BedrockClaude3Haiku) thinkingDimensions() ThinkingDimension {
	return bedrockThinkingDimensions(m.ModelName())
}

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
func (m *BedrockClaude3Opus) thinkingDimensions() ThinkingDimension {
	return bedrockThinkingDimensions(m.ModelName())
}

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
	bedrockCacheOptions
	// The escape hatch carries thinking configuration for the same reason it
	// carries caching configuration: it is the only way to reach a Claude that
	// shipped after this build. What it sends is decided by the model id it was
	// handed, so a Titan or Llama id stores the configuration and sends nothing.
	bedrockThinkingOptions
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
func (m *BedrockModel) thinkingDimensions() ThinkingDimension {
	return bedrockThinkingDimensions(m.ModelName())
}

func (m *BedrockModel) WithMaxTokens(n int) *BedrockModel       { m.maxTokens = n; return m }
func (m *BedrockModel) WithTemperature(t float64) *BedrockModel { m.temperature = t; return m }
func (m *BedrockModel) WithTopP(p float64) *BedrockModel        { m.topP = p; return m }
func (m *BedrockModel) WithTopK(k int) *BedrockModel            { m.topK = k; return m }
func (m *BedrockModel) WithSystemPrompt(s string) *BedrockModel { m.systemPrompt = s; return m }

// WithModelFamily declares which request format the model speaks: "claude",
// "nova", "titan", "llama" or "mistral". Setting it back to "" restores the
// default, which is to classify the model id.
func (m *BedrockModel) WithModelFamily(f string) *BedrockModel { m.modelFamily = f; return m }

// NewBedrockModel creates a new generic Bedrock model with the specified model ID.
//
// modelFamily should be one of: "claude", "nova", "titan", "llama", "mistral".
// Leave it empty to have the family classified from the model id, which works
// for a base id, a cross-region inference profile id and an inference-profile
// ARN alike. Name it explicitly when the id cannot be classified -- a
// provisioned-throughput ARN carries an opaque id and nothing else -- or when
// the id would classify to the wrong thing.
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
	client *bedrockruntime.Client
	// healthModel is generated against by Health. Empty -- the default -- means
	// Health probes without invoking anything; see Health.
	healthModel string
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

	client := newBedrockRuntimeClient(awsCfg)

	timeout := bedrockCfg.Timeout
	if timeout == 0 {
		timeout = defaultTimeout()
	}

	return &bedrockClient{
		client:      client,
		healthModel: bedrockCfg.HealthCheckModel,
		timeout:     timeout,
		logger:      logger,
		rateLimiter: newRateLimiter(bedrockCfg.RateLimiter, logger),
	}, nil
}

// newBedrockRuntimeClient builds the SDK client every Bedrock call goes
// through. It exists so the token-header middleware and the retryer split have
// exactly one registration site, shared by the production constructor and by
// the wire tests, instead of a stack the tests do not actually exercise.
func newBedrockRuntimeClient(awsCfg aws.Config, optFns ...func(*bedrockruntime.Options)) *bedrockruntime.Client {
	return bedrockruntime.NewFromConfig(awsCfg,
		append(optFns,
			bedrockruntime.WithAPIOptions(captureBedrockTokenHeaders),
			// Resolved defaults are in place by the time an optFn runs
			// (bedrockruntime@v1.57.3 api_client.go:204,224), so this wraps a
			// real retryer rather than a nil one.
			func(o *bedrockruntime.Options) { o.Retryer = lingoOwnedRetryer{Retryer: o.Retryer} },
		)...)
}

// ============================================================================
// INVOKEMODEL TOKEN-COUNT HEADERS
// ============================================================================
//
// Bedrock reports the tokens it billed on every InvokeModel response as
// X-Amzn-Bedrock-Input-Token-Count and X-Amzn-Bedrock-Output-Token-Count.
// Neither header is in the Smithy model -- the InvokeModel API reference
// documents only contentType, performanceConfigLatency and serviceTier as
// response headers, and bedrockruntime.InvokeModelOutput has fields for exactly
// those three -- so the SDK parses them into nothing and the only way to see
// them is to read the raw response before it is discarded.
//
// That is what this middleware is for. It is modelled on the SDK's own
// RequestIDRetriever (aws-sdk-go-v2 aws/middleware/request_id_retriever.go),
// which pulls X-Amzn-Requestid off the same raw response and stashes it in
// middleware.Metadata; ResultMetadata on the operation output is that same
// metadata, so a value set here is readable by the caller with no plumbing.
//
// Because the headers are unmodelled they are also unpromised: a response that
// omits them sets no metadata, and every reader must treat absence as normal.
// Nothing about the request changes, so no model sees different bytes.

// bedrockTokenHeaderKey is the middleware.Metadata key the captured counts live
// under. An empty struct type, like the SDK's own metadata keys, so it can
// collide with nothing else in the map.
type bedrockTokenHeaderKey struct{}

// bedrockHeaderUsage is what Bedrock said it billed. present is false when the
// response carried no usable token headers, which is the normal state for any
// response this middleware did not recognise -- it is never the same as a
// genuine zero.
type bedrockHeaderUsage struct {
	inputTokens  int
	outputTokens int
	present      bool
}

// captureBedrockTokenHeaders registers on the Deserialize step, ahead of the
// operation deserializer, so it runs after the whole inner chain has returned
// and can read out.RawResponse. Add is used rather than Insert because Add
// cannot fail: an Insert that could not find its anchor would return an error
// from every operation on the client, and a token counter is not worth that.
func captureBedrockTokenHeaders(stack *middleware.Stack) error {
	return stack.Deserialize.Add(middleware.DeserializeMiddlewareFunc(
		"lingoBedrockTokenHeaders",
		func(ctx context.Context, in middleware.DeserializeInput, next middleware.DeserializeHandler) (
			middleware.DeserializeOutput, middleware.Metadata, error,
		) {
			out, metadata, err := next.HandleDeserialize(ctx, in)
			if err != nil {
				return out, metadata, err
			}
			resp, ok := out.RawResponse.(*smithyhttp.Response)
			if !ok {
				return out, metadata, err
			}
			var u bedrockHeaderUsage
			if n, ok := bedrockAtoi(resp.Header.Get("X-Amzn-Bedrock-Input-Token-Count")); ok {
				u.inputTokens, u.present = n, true
			}
			if n, ok := bedrockAtoi(resp.Header.Get("X-Amzn-Bedrock-Output-Token-Count")); ok {
				u.outputTokens, u.present = n, true
			}
			if u.present {
				metadata.Set(bedrockTokenHeaderKey{}, u)
			}
			return out, metadata, err
		},
	), middleware.Before)
}

// bedrockAtoi reads one unmodelled header value. A missing, malformed or
// negative count reads as "nothing was reported" rather than as a zero, because
// for cost tracking those are not the same claim.
func bedrockAtoi(s string) (int, bool) {
	if s == "" {
		return 0, false
	}
	n, err := strconv.Atoi(strings.TrimSpace(s))
	if err != nil || n < 0 {
		return 0, false
	}
	return n, true
}

// bedrockCompleteUsage fills the gaps in a family's body-reported usage from the
// counts Bedrock put in the response headers, and names the sources the result
// came from -- "" when the headers contributed nothing, so a family that
// reported for itself is never relabelled.
//
// The body wins wherever it reported, for three reasons: it is the model's own
// accounting, it is modelled and documented where the headers are neither, and
// on at least one family it does not even mean the same thing -- Claude's
// input_tokens excludes the cache counters reported beside it, so a header input
// count is a different number rather than a better one. The headers are a
// fallback here and never an override.
//
// Only a zero counts as a gap. That is exact for the two families this is used
// on -- Titan reports inputTextTokenCount and results[].tokenCount, Mistral
// reports neither, and no real generation costs zero prompt tokens or produces
// zero completion tokens -- and it is why the rule is applied per family instead
// of centrally in Generate, where it would eventually meet a family whose zero
// is a real answer.
//
// TotalTokens is recomputed from the two parts whenever either moves, so a
// filled-in count can never leave the three numbers disagreeing.
func bedrockCompleteUsage(u TokenUsage, hdr bedrockHeaderUsage) (TokenUsage, string) {
	if !hdr.present {
		return u, ""
	}
	fromBody := u.PromptTokens > 0 || u.CompletionTokens > 0
	var filled bool
	if u.PromptTokens == 0 && hdr.inputTokens > 0 {
		u.PromptTokens, filled = hdr.inputTokens, true
	}
	if u.CompletionTokens == 0 && hdr.outputTokens > 0 {
		u.CompletionTokens, filled = hdr.outputTokens, true
	}
	if !filled {
		return u, ""
	}
	u.TotalTokens = u.PromptTokens + u.CompletionTokens
	if fromBody {
		return u, "body+response_headers"
	}
	return u, "response_headers"
}

// bedrockHeaderUsageFrom reads back what captureBedrockTokenHeaders stashed.
// The zero value means the response reported nothing, so callers can use the
// result unconditionally.
func bedrockHeaderUsageFrom(md middleware.Metadata) bedrockHeaderUsage {
	u, _ := md.Get(bedrockTokenHeaderKey{}).(bedrockHeaderUsage)
	return u
}

// Bedrock request/response types for different model families

// Claude Messages API format.
//
// System and Content are typed as any because Anthropic accepts either a plain
// string or an array of content blocks, and only the block form can carry a
// cache breakpoint. They hold a string unless caching is on, so an untouched
// model marshals byte for byte the body it always did.
type bedrockClaudeRequest struct {
	AnthropicVersion string                 `json:"anthropic_version"`
	MaxTokens        int                    `json:"max_tokens"`
	Messages         []bedrockClaudeMessage `json:"messages"`
	System           any                    `json:"system,omitempty"`
	Temperature      float64                `json:"temperature,omitempty"`
	TopP             float64                `json:"top_p,omitempty"`
	TopK             int                    `json:"top_k,omitempty"`
	// Thinking is a pointer with omitempty for the same reason System is typed
	// any: a model whose ThinkingOptions were never touched marshals byte for
	// byte the body it always did.
	Thinking *bedrockClaudeThinking `json:"thinking,omitempty"`
}

// bedrockClaudeThinking is Anthropic's thinking config in its own spelling,
// which is what the InvokeModel body speaks. Only the two shapes Bedrock has
// carried since Claude 3.7 are written here: {"type":"enabled","budget_tokens":N}
// and {"type":"disabled"}. BudgetTokens has omitempty so the disabled form is
// exactly those two words -- the API requires a budget with "enabled" and
// rejects one with "disabled".
type bedrockClaudeThinking struct {
	Type         string `json:"type"`
	BudgetTokens int    `json:"budget_tokens,omitempty"`
}

type bedrockClaudeMessage struct {
	Role    string `json:"role"`
	Content any    `json:"content"`
}

// bedrockClaudeText is one text content block. InvokeModel forwards the body to
// the vendor verbatim, so a Bedrock cache breakpoint is written in Anthropic's
// own dialect rather than as a Converse cachePoint.
type bedrockClaudeText struct {
	Type         string               `json:"type"`
	Text         string               `json:"text"`
	CacheControl *bedrockCacheControl `json:"cache_control,omitempty"`
}

type bedrockCacheControl struct {
	Type string `json:"type"`
	TTL  string `json:"ttl,omitempty"`
}

// bedrockCacheBlock wraps text in a content block ending a cached prefix.
// CacheTTLDefault leaves ttl unset, which the API reads as 5 minutes.
func bedrockCacheBlock(text string, ttl CacheTTL) []bedrockClaudeText {
	cc := &bedrockCacheControl{Type: "ephemeral"}
	switch ttl {
	case CacheTTL5m, CacheTTL1h:
		cc.TTL = string(ttl)
	}
	return []bedrockClaudeText{{Type: "text", Text: text, CacheControl: cc}}
}

type bedrockClaudeResponse struct {
	Content    []bedrockClaudeContent `json:"content"`
	StopReason string                 `json:"stop_reason"`
	Usage      bedrockClaudeUsage     `json:"usage"`
}

// bedrockClaudeContent is one block of the answer. It is response-only -- the
// request builds bedrockClaudeText -- so the thinking fields cost no request
// bytes. Without them a {"type":"thinking"} block unmarshals into an empty
// struct and the trace is lost before anything gets a chance to read it.
type bedrockClaudeContent struct {
	Type string `json:"type"`
	Text string `json:"text"`
	// Thinking is the reasoning text of a "thinking" block, and Signature the
	// token that authenticates it for replay on a later turn.
	Thinking  string `json:"thinking"`
	Signature string `json:"signature"`
	// Data is the encrypted payload of a "redacted_thinking" block: the model
	// reasoned, but the trace came back opaque.
	Data string `json:"data"`
}

type bedrockClaudeUsage struct {
	InputTokens              int                  `json:"input_tokens"`
	OutputTokens             int                  `json:"output_tokens"`
	CacheCreationInputTokens int                  `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     int                  `json:"cache_read_input_tokens"`
	CacheCreation            bedrockCacheCreation `json:"cache_creation"`
	// OutputTokensDetails is Anthropic's thinking-token breakdown. Bedrock is not
	// known to return it -- neither the Claude body nor the Converse TokenUsage
	// has carried one at any version this library has been built against -- so
	// this is a read-if-present: the counter reports 0 rather than lingo
	// pretending to a number it never received.
	OutputTokensDetails bedrockClaudeOutputTokensDetails `json:"output_tokens_details"`
}

// bedrockClaudeOutputTokensDetails mirrors Anthropic's usage.output_tokens_details.
// thinking_tokens is documented as always <= output_tokens, so it is a subset of
// the completion total rather than an addition to it.
type bedrockClaudeOutputTokensDetails struct {
	ThinkingTokens int `json:"thinking_tokens"`
}

// bedrockCacheCreation is the per-TTL split of the cache write, reported only
// by model versions new enough to bill the two lifetimes differently.
type bedrockCacheCreation struct {
	Ephemeral5mInputTokens int `json:"ephemeral_5m_input_tokens"`
	Ephemeral1hInputTokens int `json:"ephemeral_1h_input_tokens"`
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
	// InputTextTokenCount is Titan's own prompt count, reported beside the
	// results rather than inside them. It was never modelled, which is why
	// PromptTokens came back 0 on every Titan call; a response that omits it
	// leaves the gap for the response headers to fill.
	InputTextTokenCount int                  `json:"inputTextTokenCount"`
	Results             []bedrockTitanResult `json:"results"`
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
// is stripped before matching the provider. An inference-profile ARN carries the
// same id in its last segment and resolves to the same family; a
// provisioned-throughput ARN carries an opaque id and stays "unknown", which
// Generate reports as an unsupported family rather than guessing.
func getModelFamily(modelID string) string {
	id := bedrockResolveModelID(modelID)
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

// bedrockFamilyFor decides which family builds the request for one model.
//
// Every model type but the escape hatch answers from its id, because its id is
// the only thing it has. BedrockModel can also carry a declared family, and a
// declaration wins: the caller may know something the id does not say, and a
// provisioned-throughput ARN says nothing at all, so overriding it would take
// away the only way to use one.
//
// An empty family is not a declaration. NewBedrockModel("us.anthropic.claude-opus-5", "")
// used to be a dead end -- the empty string reached the family switch verbatim
// and came back "unsupported model family: " for an id that classifies
// perfectly -- so an unset family now means what it reads as, "work it out from
// the id". That matters more since ARNs classify too: an inference-profile ARN
// names its model in its last segment and needs no second opinion from the
// caller.
//
// An id that classifies to nothing still lands on "unknown", which Generate
// still refuses loudly. Silence is the one thing this must not buy.
func bedrockFamilyFor(model Model, modelID string) string {
	if bm, ok := model.(*BedrockModel); ok && bm.modelFamily != "" {
		return bm.modelFamily
	}
	return getModelFamily(modelID)
}

// bedrockUsesConverse reports whether a family is served by the Converse API
// instead of InvokeModel. Only Nova is: it is the one family besides Claude
// that supports prompt caching, and unlike Claude it reports no cache counters
// on the InvokeModel response, so Converse is the only place its cache
// accounting exists. Claude keeps its own dialect, which already carries a
// richer per-TTL write split; Llama and Mistral support no caching at all and
// would have their prompt templating silently rewritten by Converse; Titan
// Text is past end-of-life on Bedrock.
//
// Nova's InvokeModel body does accept a cachePoint block, so the marker alone
// is not the reason for the move -- the response is. AWS documents
// cacheReadInputTokens, cacheWriteInputTokens and cacheDetails only on the
// Converse response, so placing a cachePoint on InvokeModel would cache
// without ever being able to report that it did.
//
// Flipping this to false is the complete rollback: buildNovaRequest and
// parseNovaResponse are deliberately kept, so the InvokeModel path for Nova
// still works and nothing else has to be restored.
func bedrockUsesConverse(family string) bool { return family == "nova" }

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

	modelFamily := bedrockFamilyFor(model, modelID)

	if bedrockUsesConverse(modelFamily) {
		return c.generateConverse(ctx, model, prompt, modelID, modelFamily)
	}

	var body []byte
	var cacheBreakpoint bool
	var thinking thinkingPlan
	var err error

	// Build request based on model family
	switch modelFamily {
	case "claude":
		body, cacheBreakpoint, thinking, err = c.buildClaudeRequest(model, prompt)
	case "nova":
		body, err = c.buildNovaRequest(model, prompt)
	case "titan":
		body, err = c.buildTitanRequest(model, prompt)
	case "llama":
		body, err = c.buildLlamaRequest(model, prompt)
	case "mistral":
		body, err = c.buildMistralRequest(model, prompt)
	default:
		// Reached by a declared family lingo has no builder for, and by an id
		// that classifies to nothing -- a provisioned-throughput ARN, say.
		// Naming the id as well as the family is what tells the second case
		// apart from the first, and the fix for it is WithModelFamily.
		return nil, fmt.Errorf("unsupported model family: %s (model %s)", modelFamily, modelID)
	}
	if err != nil {
		return nil, err
	}

	// Logged after the body is built, so cache_breakpoint can report the marker
	// the request actually carries rather than the one that was asked for, and
	// thinking_translation whatever had to be adapted to get there.
	c.logger.Debug().
		Str("model", modelID).
		Str("family", modelFamily).
		Bool("cache_breakpoint", cacheBreakpoint).
		Str("thinking_translation", thinking.translation()).
		Msg("Making Bedrock API request")

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

	// What Bedrock said it billed. Only the two families whose bodies leave a
	// gap consume it: Mistral, whose response carries no counts at all, and
	// Titan, whose response carries a completion count and -- in the shape lingo
	// modelled -- no prompt one. Claude, Llama and Nova each report a complete
	// set for themselves and are left reading their own body, so no number lingo
	// already reports can move.
	headerUsage := bedrockHeaderUsageFrom(output.ResultMetadata)

	// Parse response based on model family
	var response *GenerationResponse
	switch modelFamily {
	case "claude":
		response, err = c.parseClaudeResponse(output.Body, modelID)
	case "nova":
		response, err = c.parseNovaResponse(output.Body, modelID)
	case "titan":
		response, err = c.parseTitanResponse(output.Body, modelID, headerUsage)
	case "llama":
		response, err = c.parseLlamaResponse(output.Body, modelID)
	case "mistral":
		response, err = c.parseMistralResponse(output.Body, modelID, headerUsage)
	}
	if err != nil {
		return nil, err
	}

	// Whatever lingo had to translate or drop to fit the caller's request onto
	// this model's dialect, so a silent adaptation is never invisible. Only the
	// Claude path builds a plan; everywhere else this is empty.
	if s := thinking.translation(); s != "" {
		response.Metadata["thinking_translation"] = s
	}

	c.logger.Debug().
		Str("model", modelID).
		Int("prompt_tokens", response.Usage.PromptTokens).
		Int("completion_tokens", response.Usage.CompletionTokens).
		Int("total_tokens", response.Usage.TotalTokens).
		Int("cache_read_tokens", response.Usage.CacheReadTokens).
		Int("cache_write_tokens", response.Usage.CacheWriteTokens).
		Bool("has_thinking", response.Thinking != "").
		Msg("Bedrock generation completed")

	return response, nil
}

// buildClaudeRequest also reports whether a cache breakpoint was actually
// placed, which is not the same as the caller having asked for one: enabling
// caching on a model with no system prompt marks nothing. It returns the
// thinking plan for the same reason: what lingo had to translate or drop to fit
// the request onto this model's dialect has to reach the response metadata, and
// the decision cannot be made before max_tokens is known.
func (c *bedrockClient) buildClaudeRequest(model Model, prompt string) ([]byte, bool, thinkingPlan, error) {
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
	// Opus 4.7/4.8, Fable 5 and the Claude 5 series reject sampling parameters
	// (temperature/topP/topK) with a 400 error; only max_tokens and the system
	// prompt are sent.
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
	case *BedrockClaudeOpus5:
		if m.maxTokens > 0 {
			req.MaxTokens = m.maxTokens
		}
		if m.systemPrompt != "" {
			req.System = m.systemPrompt
		}
	case *BedrockClaudeSonnet5:
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

	// Thinking is opt-in and, like caching, is applied once from a plan built
	// outside the switch -- but unlike caching it cannot be applied uniformly,
	// because the wire shape below belongs to some Claude generations and not
	// others. A model whose ThinkingOptions were never touched produces a zero
	// plan and leaves req exactly as built above.
	era := bedrockClaudeThinkingEra(model.ModelName())
	dims := bedrockEraDimensions(era)

	// budget_tokens must be >= 1024 and strictly below max_tokens. A request whose
	// max_tokens leaves no room for a legal budget has no budget knob at all, so
	// the depth is translated or dropped rather than sent to be rejected.
	br := budgetRange{min: anthropicMinThinkingBudget, max: req.MaxTokens - 1}
	if br.max < br.min {
		br = budgetRange{}
		dims &^= ThinkingCanSetBudget
	}

	// No effort ladder is passed: output_config is not written on this path, so a
	// caller's effort is translated into a token budget for the generations that
	// take one and dropped, with a note, for the ones that do not.
	plan := planThinking(modelThinkingOptions(model), dims, br)

	switch {
	case plan.disable:
		req.Thinking = &bedrockClaudeThinking{Type: "disabled"}
	case plan.budget > 0:
		req.Thinking = &bedrockClaudeThinking{Type: "enabled", BudgetTokens: plan.budget}
	case plan.enable || plan.dynamic:
		// Thinking was asked for without a depth lingo can name. On the
		// generations that speak in fixed budgets it has to become one, since the
		// adaptive config is not written here.
		switch {
		case dims.Has(ThinkingCanSetBudget):
			// The window above already guarantees a positive budget here. The
			// guard is what keeps that guarantee cheap to hold: "enabled" with no
			// budget_tokens is a body the API rejects, so a zero means no thinking
			// config at all rather than a malformed one.
			if n := ThinkingBudgetForEffort(ThinkingEffortHigh, br.min, br.max); n > 0 {
				plan.note("thinking enabled as a fixed budget of %d tokens: lingo sends no adaptive thinking config on Bedrock", n)
				req.Thinking = &bedrockClaudeThinking{Type: "enabled", BudgetTokens: n}
			} else {
				plan.note("thinking enabled but dropped: max_tokens leaves no room for a legal budget")
			}
		case era == anthropicThinkingEraBudget || era == anthropicThinkingEraAdaptiveBudget:
			// The generation does take a budget; this request has no room for a
			// legal one, which is why the dimension was withdrawn above.
			plan.note("thinking enabled but dropped: max_tokens leaves no room for a legal budget")
		case era == anthropicThinkingEraDefaultOn:
			// Claude 5 reasons unless told not to, so asking for thinking is a
			// no-op that changes no bytes rather than something to translate.
		default:
			plan.note("thinking enabled but dropped: this generation takes only an adaptive thinking config, which lingo does not send on Bedrock")
		}
	}

	// Prompt caching is opt-in and works the same for every Claude type, so it
	// sits outside the switch. The breakpoint only fits a content block, so the
	// strings assigned above are widened to block arrays -- and only then, which
	// leaves an untouched model's body unchanged.
	co := modelCacheOptions(model)
	var breakpoint bool
	if co.SystemPromptCached() {
		if system, ok := req.System.(string); ok && system != "" {
			req.System = bedrockCacheBlock(system, co.TTL())
			breakpoint = true
		}
	}
	if co.PromptCached() && len(req.Messages) > 0 {
		last := &req.Messages[len(req.Messages)-1]
		if text, ok := last.Content.(string); ok && text != "" {
			last.Content = bedrockCacheBlock(text, co.TTL())
			breakpoint = true
		}
	}

	body, err := json.Marshal(req)
	return body, breakpoint, plan, err
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

// buildConverseInput assembles a Converse request. It is pure so that the wire
// shape can be asserted in a test without a network stub, the same property
// the build*Request functions have. Like buildClaudeRequest it also reports
// whether a cache point was actually appended, not merely asked for.
func (c *bedrockClient) buildConverseInput(model Model, prompt, modelID string) (*bedrockruntime.ConverseInput, bool) {
	in := &bedrockruntime.ConverseInput{
		ModelId:         aws.String(modelID),
		InferenceConfig: &brtypes.InferenceConfiguration{MaxTokens: aws.Int32(4096)},
		Messages: []brtypes.Message{{
			Role:    brtypes.ConversationRoleUser,
			Content: []brtypes.ContentBlock{&brtypes.ContentBlockMemberText{Value: prompt}},
		}},
	}

	if s := model.SystemPrompt(); s != "" {
		in.System = []brtypes.SystemContentBlock{&brtypes.SystemContentBlockMemberText{Value: s}}
	}

	// Mirrors buildNovaRequest's applyOpts closure. InferenceConfiguration
	// carries no TopK, so it rides in AdditionalModelRequestFields under Nova's
	// own inferenceConfig path.
	var topK int
	applyOpts := func(maxTokens int, temperature, topP float64, k int) {
		if maxTokens > 0 {
			in.InferenceConfig.MaxTokens = aws.Int32(int32(maxTokens))
		}
		if temperature > 0 {
			in.InferenceConfig.Temperature = aws.Float32(float32(temperature))
		}
		if topP > 0 {
			in.InferenceConfig.TopP = aws.Float32(float32(topP))
		}
		topK = k
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
	if topK > 0 {
		in.AdditionalModelRequestFields = document.NewLazyDocument(
			map[string]any{"inferenceConfig": map[string]any{"topK": topK}})
	}

	// Nova's thinking configuration is deliberately not written here. Converse
	// models no reasoning request field -- InferenceConfiguration is max tokens,
	// stop sequences, temperature and top-p, and nothing in bedrockruntime models
	// a reasoning config -- so it could only ride in the schemaless
	// AdditionalModelRequestFields document, and the key Nova expects there could
	// not be verified from any pinned source. A guessed key is a 400, not a silent
	// drop, so a Nova model stores whatever thinking configuration it was given
	// and sends none of it; bedrockNovaOptions.thinkingDimensions reports 0 so
	// that ModelThinkingDimensions says so before the request is ever made.
	//
	// Wiring it up is a one-line change here plus a dimension in that method, but
	// note the hazard: the assignment above replaces the whole document, so a
	// second key has to be merged into one accumulated map or topK regresses.

	// Caching is opt-in, so a cache point is appended only when asked for.
	// Nova accepts checkpoints in system and messages, not in tools.
	co := modelCacheOptions(model)
	var breakpoint bool
	if co.SystemPromptCached() && len(in.System) > 0 {
		in.System = append(in.System, &brtypes.SystemContentBlockMemberCachePoint{
			Value: bedrockCachePoint(co.TTL()),
		})
		breakpoint = true
	}
	if co.PromptCached() {
		last := &in.Messages[len(in.Messages)-1]
		last.Content = append(last.Content, &brtypes.ContentBlockMemberCachePoint{
			Value: bedrockCachePoint(co.TTL()),
		})
		breakpoint = true
	}

	return in, breakpoint
}

// bedrockCachePoint builds a Converse cache checkpoint. The ttl is deliberately
// dropped: Nova, the only family routed through Converse, documents a 5 minute
// lifetime only, and the zero Ttl means exactly that -- so a 1h request is
// clamped away rather than rejected, which is the documented behaviour for
// asking a provider for a lifetime it does not model. The parameter stays so
// that a family which does model extended TTLs has one place to change.
func bedrockCachePoint(ttl CacheTTL) brtypes.CachePointBlock {
	_ = ttl
	return brtypes.CachePointBlock{Type: brtypes.CachePointTypeDefault}
}

// generateConverse runs a request through the Converse API. It reproduces the
// request log, error wrapping and completion log of the InvokeModel path in
// Generate, so an operator's log output and error strings do not move.
func (c *bedrockClient) generateConverse(ctx context.Context, model Model, prompt, modelID, family string) (*GenerationResponse, error) {
	in, cacheBreakpoint := c.buildConverseInput(model, prompt, modelID)

	c.logger.Debug().
		Str("model", modelID).
		Str("family", family).
		Bool("cache_breakpoint", cacheBreakpoint).
		Msg("Making Bedrock API request")

	var output *bedrockruntime.ConverseOutput
	err := c.rateLimiter.Execute(ctx, func() error {
		var reqErr error
		output, reqErr = c.client.Converse(ctx, in)
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

	response, err := c.parseConverseOutput(output, modelID, family)
	if err != nil {
		return nil, err
	}

	c.logger.Debug().
		Str("model", modelID).
		Int("prompt_tokens", response.Usage.PromptTokens).
		Int("completion_tokens", response.Usage.CompletionTokens).
		Int("total_tokens", response.Usage.TotalTokens).
		Int("cache_read_tokens", response.Usage.CacheReadTokens).
		Int("cache_write_tokens", response.Usage.CacheWriteTokens).
		Msg("Bedrock generation completed")

	return response, nil
}

// buildTitanRequest builds the Titan Text InvokeModel body.
//
// Titan has no system prompt, and lingo cannot give it one. AWS documents the
// whole request body as
//
//	{"inputText": string,
//	 "textGenerationConfig": {"temperature", "topP", "maxTokenCount", "stopSequences"}}
//
// (Amazon Bedrock User Guide, "Amazon Titan Text models", Request) -- there is
// no system field, no instructions field and no role structure anywhere in it,
// and AWS's own guidance for expressing a conversation is to write the roles as
// plain text inside inputText, as in its documented `"inputText": "User:
// {{prompt}}\nBot:"` form. Titan Text is not served by Converse either, so
// there is no second dialect here with a system block to borrow.
//
// So a system prompt set on a Titan model is concatenated ahead of the user
// prompt with a blank line between them, and callers should know what that
// buys: the blank line is the entire boundary. There are no control tokens on
// this API to forge -- which is why the Llama and Mistral escaping in this file
// has no Titan equivalent -- but there is also nothing that tells the model the
// first paragraph outranks the second, so prompt text that contradicts the
// system text is arguing on equal terms. A caller who needs an instruction the
// user's text cannot restate needs a family whose API models one: Claude, Nova,
// Llama and Mistral all do.
func (c *bedrockClient) buildTitanRequest(model Model, prompt string) ([]byte, error) {
	req := bedrockTitanRequest{
		InputText: prompt,
		TextGenerationConfig: bedrockTitanConfig{
			MaxTokenCount: 4096,
			Temperature:   0.7,
			TopP:          0.9,
		},
	}

	// Prepend system prompt if set. A blank line is the only boundary this API
	// offers; see the doc comment.
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

// bedrockLlamaTokens is the set of role/turn markers one Llama generation uses.
// The generations do not share a template: Llama 3 (and 3.1, 3.2, 3.3) encloses
// a role in <|start_header_id|>/<|end_header_id|> and closes a turn with
// <|eot_id|>, while Llama 4 renamed all three to
// <|header_start|>/<|header_end|>/<|eot|> and does not carry the 3.x spellings
// in its vocabulary at all, so a 3.x prompt reaches Llama 4 as ordinary text.
type bedrockLlamaTokens struct {
	headerOpen  string
	headerClose string
	endOfTurn   string
	// endOfMessage is the other way a turn can end -- the tool-calling variant
	// of endOfTurn. The template never writes it, but it is a reserved id all
	// the same, so caller-supplied text must not be able to smuggle one in.
	endOfMessage string
}

// The two text-boundary tokens are spelled the same in both generations'
// vocabularies, so they sit outside the per-generation set.
const (
	bedrockLlamaBeginOfText = "<|begin_of_text|>"
	bedrockLlamaEndOfText   = "<|end_of_text|>"
)

var (
	bedrockLlama3Tokens = bedrockLlamaTokens{
		headerOpen:   "<|start_header_id|>",
		headerClose:  "<|end_header_id|>",
		endOfTurn:    "<|eot_id|>",
		endOfMessage: "<|eom_id|>",
	}
	bedrockLlama4Tokens = bedrockLlamaTokens{
		headerOpen:   "<|header_start|>",
		headerClose:  "<|header_end|>",
		endOfTurn:    "<|eot|>",
		endOfMessage: "<|eom|>",
	}
)

// specials is everything this generation reserves that could restructure a
// prompt: the tokens the template writes, the two text boundaries and the
// end-of-message variant of the turn boundary. Deliberately nothing else --
// the multimodal and tool ids are reserved too, but they cannot forge a turn,
// and every token listed here is one that stops being ordinary text in a
// caller's document.
//
// The two lists share only the text boundaries; every turn marker in one is
// ordinary text to the other, which is the point: "<|eot_id|>" is a reserved id
// to Llama 3 and plain text to Llama 4.
func (t bedrockLlamaTokens) specials() []string {
	return []string{
		bedrockLlamaBeginOfText, bedrockLlamaEndOfText,
		t.headerOpen, t.headerClose, t.endOfTurn, t.endOfMessage,
	}
}

// render lays out one system message (when there is one), one user message and
// the opening header of the assistant turn the model is asked to complete. The
// two newlines after every header and the two that end the prompt are part of
// the template, not decoration: Meta's own encoder appends "\n\n" after each
// header and ends a generation prompt with the assistant header, so dropping
// the trailing pair leaves the model completing a header rather than a reply.
//
// Both caller-supplied strings are neutralized here rather than at the call
// site, so no future caller of render can forget to do it. See
// bedrockNeutralizeControlTokens for what that means for the text.
func (t bedrockLlamaTokens) render(system, prompt string) string {
	specials := t.specials()
	system = bedrockNeutralizeControlTokens(system, specials)
	prompt = bedrockNeutralizeControlTokens(prompt, specials)

	var b strings.Builder
	b.WriteString(bedrockLlamaBeginOfText)
	if system != "" {
		b.WriteString(t.headerOpen + "system" + t.headerClose + "\n\n" + system + t.endOfTurn)
	}
	b.WriteString(t.headerOpen + "user" + t.headerClose + "\n\n" + prompt + t.endOfTurn)
	b.WriteString(t.headerOpen + "assistant" + t.headerClose + "\n\n")
	return b.String()
}

// bedrockLlama2Specials and bedrockMistralSpecials are the turn boundaries of
// the two templates that still speak in bracket markers.
//
// Whether a tokenizer reserves an id for "[INST]" varies by version -- Mistral
// v3 does, v0.1 and v0.2 do not -- but that is not what makes the marker
// dangerous. These templates delimit turns with literal text, so an injected
// "[/INST]" ends the instruction and opens the answer whether or not the
// tokenizer gave it an id of its own. "<s>" is a reserved id in every version.
var (
	bedrockLlama2Specials  = []string{"<s>", "</s>", "[INST]", "[/INST]", "<<SYS>>", "<</SYS>>"}
	bedrockMistralSpecials = []string{"<s>", "</s>", "[INST]", "[/INST]"}
)

// bedrockNeutralizeControlTokens rewrites every occurrence of the given control
// tokens in caller-supplied text, so that text can only ever be read as content.
//
// This is the price of emitting a generation's real control tokens. Under the
// old Llama 2 template the interpolation was inert on a Llama 3 or 4 model,
// because "<s>" and "[INST]" are ordinary text to those tokenizers. Now that
// lingo writes the tokens the model actually reserves, a prompt or system
// prompt containing "<|eot_id|><|start_header_id|>system<|end_header_id|>"
// would close its own turn and open a forged system turn. Meta's own encoder
// never has this problem because it tokenizes message content with
// special-token parsing disabled; lingo hands Bedrock one raw string and has no
// way to ask for that, so the neutralization has to happen in the string.
//
// A token is neutralized by escaping it, not by removing it -- see
// bedrockEscapeControlToken for the shape. Escaping rather than stripping, for
// two reasons:
//
//   - Deleting a token can create one. "<|eot<|eot_id|>_id|>" is a string a
//     caller may legitimately send, and deleting the inner token splices the
//     surviving halves into a real "<|eot_id|>". Escaping only ever inserts, and
//     an insertion cannot bring previously separated characters together, so a
//     single pass is provably enough.
//   - The caller wrote that text as content. A document that quotes a Llama chat
//     template still reads as itself with a backslash in it; with the tokens
//     deleted it reads as a corrupted template.
//
// Only whole tokens match, so ordinary prose that merely contains angle
// brackets or pipes -- "a < b | c > d", "<div>", "|col|col|", "<<EOF" -- comes
// back byte for byte. And only the tokens of the generation being rendered are
// passed in, so nothing is escaped that the model would have read as text
// anyway.
func bedrockNeutralizeControlTokens(s string, tokens []string) string {
	for _, tok := range tokens {
		// An empty token would match everywhere and mean nothing; skipping it
		// keeps a half-filled token set from turning into a rewrite of the
		// whole string.
		if tok != "" && strings.Contains(s, tok) {
			s = strings.ReplaceAll(s, tok, bedrockEscapeControlToken(tok))
		}
	}
	return s
}

// bedrockEscapeControlToken breaks one control token by inserting a backslash
// after its opening character: "<|eot_id|>" becomes "<\|eot_id|>", "[INST]"
// becomes "[\INST]", "<s>" becomes "<\s>".
//
// The backslash has to go inside the token. Prefixing one would leave the token
// intact as a substring, and a tokenizer scanning for reserved ids would match
// it one byte further along and be forged exactly as before.
func bedrockEscapeControlToken(tok string) string {
	if len(tok) < 2 {
		return tok
	}
	return tok[:1] + `\` + tok[1:]
}

// bedrockLlamaGeneration reads the major Llama generation out of a Bedrock model
// id, reusing getModelFamily's convention of reducing the id to its vendor form
// first:
//
//	us.meta.llama3-3-70b-instruct-v1:0    -> meta.llama3-3-... -> 3
//	meta.llama4-scout-17b-instruct-v1:0                        -> 4
//	arn:...:inference-profile/us.meta.llama4-scout-17b-instruct-v1:0 -> 4
//
// The ARN form matters as much as the other two: a Llama 4 reached through an
// inference-profile ARN that classified as "not a Llama at all" would be sent
// the 3.x template, which is the exact off-template bug this classifier exists
// to prevent.
//
// An id that does not name a generation reports 0, which the caller reads as
// "the 3.x template", the shape every Llama on Bedrock except Llama 4 speaks.
func bedrockLlamaGeneration(modelID string) int {
	rest, ok := strings.CutPrefix(strings.TrimPrefix(bedrockResolveModelID(modelID), "meta."), "llama")
	if !ok {
		return 0
	}
	gen := 0
	for _, r := range rest {
		if r < '0' || r > '9' {
			break
		}
		gen = gen*10 + int(r-'0')
	}
	return gen
}

// bedrockLlamaPrompt renders a single-turn conversation in the template the
// given Llama model id speaks.
func bedrockLlamaPrompt(modelID, system, prompt string) string {
	switch gen := bedrockLlamaGeneration(modelID); {
	case gen == 2:
		// Llama 2. No Llama 2 model ships in lingo and Bedrock has retired them,
		// but the escape hatch can still be handed a meta.llama2- id, and this is
		// the template that id does speak. Its markers are neutralized in the
		// caller's text for the same reason the 3.x and 4 ones are: they are
		// this template's turn boundaries, so an injected "[/INST]" ends the
		// instruction here just as an injected "<|eot_id|>" ends the turn there.
		system = bedrockNeutralizeControlTokens(system, bedrockLlama2Specials)
		prompt = bedrockNeutralizeControlTokens(prompt, bedrockLlama2Specials)
		if system != "" {
			return fmt.Sprintf("<s>[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]", system, prompt)
		}
		return fmt.Sprintf("<s>[INST] %s [/INST]", prompt)
	case gen >= 4:
		return bedrockLlama4Tokens.render(system, prompt)
	default:
		return bedrockLlama3Tokens.render(system, prompt)
	}
}

func (c *bedrockClient) buildLlamaRequest(model Model, prompt string) ([]byte, error) {
	req := bedrockLlamaRequest{
		Prompt:      bedrockLlamaPrompt(model.ModelName(), model.SystemPrompt(), prompt),
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
	// Build Mistral prompt format. Mistral's markers are as load-bearing in its
	// template as Llama's are in theirs -- an injected "[/INST]" ends the
	// instruction and opens the answer -- so the caller's text is neutralized
	// the same way. This exposure predates the Llama template fix rather than
	// coming with it, but it is the same hole and it is treated the same.
	system := bedrockNeutralizeControlTokens(model.SystemPrompt(), bedrockMistralSpecials)
	prompt = bedrockNeutralizeControlTokens(prompt, bedrockMistralSpecials)

	var fullPrompt string
	if system != "" {
		fullPrompt = fmt.Sprintf("<s>[INST] %s\n\n%s [/INST]", system, prompt)
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

	// Extract the answer and the reasoning trace. Both accumulate: a response is
	// a list of blocks, and once thinking is on it routinely arrives as several
	// of each.
	var text, thinkingText, thinkingSignature, redactedThinking string
	for _, content := range resp.Content {
		switch content.Type {
		case "text":
			text += content.Text
		case "thinking":
			thinkingText += content.Thinking
			// The signature authenticates the block for replay on a later turn.
			// A multi-block response has one per block and Metadata holds one
			// string, so this is the last of them; faithful replay needs a typed
			// content API, which lingo's single-turn Generate does not have.
			if content.Signature != "" {
				thinkingSignature = content.Signature
			}
		case "redacted_thinking":
			// The model reasoned but the trace came back encrypted. It is opaque
			// to the caller and useful only for replay, so it stays out of
			// Thinking and rides in metadata.
			redactedThinking += content.Data
		}
	}

	result := &GenerationResponse{
		Text:         text,
		Thinking:     thinkingText,
		Model:        modelID,
		FinishReason: resp.StopReason,
		// The Claude body speaks Anthropic's dialect, which reports cache tokens
		// alongside input_tokens rather than inside it, so withCache folds them
		// back into the prompt total. Thinking tokens are the other way round --
		// Anthropic documents thinking_tokens as always <= output_tokens -- and
		// stay 0 here anyway, because Bedrock is not known to report them.
		Usage: TokenUsage{
			PromptTokens:     resp.Usage.InputTokens,
			CompletionTokens: resp.Usage.OutputTokens,
			TotalTokens:      resp.Usage.InputTokens + resp.Usage.OutputTokens,
		}.withCache(resp.Usage.CacheReadInputTokens, resp.Usage.CacheCreationInputTokens, false).
			withThinking(resp.Usage.OutputTokensDetails.ThinkingTokens, true),
		Metadata: map[string]string{
			"provider": "bedrock",
			"model":    modelID,
			"family":   "claude",
		},
	}

	// The trace now has a typed home in GenerationResponse.Thinking; the metadata
	// key the Anthropic provider has always mirrored it into is written here too,
	// so a reader that already handles first-party Claude handles Bedrock Claude
	// without a second code path.
	//
	// Deprecated: read GenerationResponse.Thinking instead of Metadata["thinking"].
	if thinkingText != "" {
		result.Metadata["thinking"] = thinkingText
	}
	if thinkingSignature != "" {
		result.Metadata["thinking_signature"] = thinkingSignature
	}
	if redactedThinking != "" {
		result.Metadata["thinking_redacted"] = redactedThinking
	}

	// Bedrock bills the two cache lifetimes differently and reports the split;
	// it does not fit two counters, so it rides along as metadata.
	if n := resp.Usage.CacheCreation.Ephemeral5mInputTokens; n > 0 {
		result.Metadata["cache_write_tokens_5m"] = fmt.Sprintf("%d", n)
	}
	if n := resp.Usage.CacheCreation.Ephemeral1hInputTokens; n > 0 {
		result.Metadata["cache_write_tokens_1h"] = fmt.Sprintf("%d", n)
	}

	return result, nil
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

// parseConverseOutput reads a Converse response. It is pure, so the accounting
// can be asserted in a test from a hand-built ConverseOutput.
func (c *bedrockClient) parseConverseOutput(output *bedrockruntime.ConverseOutput, modelID, family string) (*GenerationResponse, error) {
	if output == nil {
		return nil, fmt.Errorf("empty Converse response")
	}

	msg, ok := output.Output.(*brtypes.ConverseOutputMemberMessage)
	if !ok {
		return nil, fmt.Errorf("unexpected Converse output type %T", output.Output)
	}

	// Reasoning is read unconditionally, whether or not anyone asked for it: lingo
	// asks no Converse model to reason (see bedrockNovaOptions.thinkingDimensions),
	// but a model that reasons on its own terms returns the trace in its own
	// content block, and a type switch that only knows about text drops it.
	var text, thinkingText, thinkingSignature, redactedThinking string
	for _, block := range msg.Value.Content {
		switch b := block.(type) {
		case *brtypes.ContentBlockMemberText:
			text += b.Value
		case *brtypes.ContentBlockMemberReasoningContent:
			switch r := b.Value.(type) {
			case *brtypes.ReasoningContentBlockMemberReasoningText:
				thinkingText += aws.ToString(r.Value.Text)
				if s := aws.ToString(r.Value.Signature); s != "" {
					thinkingSignature = s
				}
			case *brtypes.ReasoningContentBlockMemberRedactedContent:
				// Encrypted by the provider and opaque to the caller. The SDK has
				// already base64-decoded the blob, so re-encoding is what puts it
				// back in the form the Claude InvokeModel path records it in.
				redactedThinking += base64.StdEncoding.EncodeToString(r.Value)
			}
		}
	}
	if text == "" {
		return nil, fmt.Errorf("no content in Converse response")
	}

	// Converse reports cache tokens alongside inputTokens rather than inside it
	// ("total input tokens = inputTokens + cacheReadInputTokens +
	// cacheWriteInputTokens"), so withCache folds them into the prompt total.
	// The response also carries its own totalTokens, but that field already
	// counts the cache tokens, so summing input and output here is what keeps
	// withCache from adding them a second time.
	usage := TokenUsage{}
	var read, write int
	if u := output.Usage; u != nil {
		usage.PromptTokens = int(aws.ToInt32(u.InputTokens))
		usage.CompletionTokens = int(aws.ToInt32(u.OutputTokens))
		usage.TotalTokens = usage.PromptTokens + usage.CompletionTokens
		read = int(aws.ToInt32(u.CacheReadInputTokens))
		write = int(aws.ToInt32(u.CacheWriteInputTokens))
	}

	result := &GenerationResponse{
		Text:         text,
		Thinking:     thinkingText,
		Model:        modelID,
		FinishReason: string(output.StopReason),
		// Converse reports no thinking token count at any version this library
		// has been built against, so ThinkingTokens stays 0 even when a trace
		// came back.
		Usage: usage.withCache(read, write, false),
		Metadata: map[string]string{
			"provider": "bedrock",
			"model":    modelID,
			"family":   family,
		},
	}

	// Deprecated: read GenerationResponse.Thinking instead of Metadata["thinking"].
	if thinkingText != "" {
		result.Metadata["thinking"] = thinkingText
	}
	if thinkingSignature != "" {
		result.Metadata["thinking_signature"] = thinkingSignature
	}
	if redactedThinking != "" {
		result.Metadata["thinking_redacted"] = redactedThinking
	}

	// The same per-TTL write split the Claude path records, here delivered as a
	// typed list rather than two named fields.
	if output.Usage != nil {
		for _, d := range output.Usage.CacheDetails {
			if n := aws.ToInt32(d.InputTokens); n > 0 {
				result.Metadata["cache_write_tokens_"+string(d.Ttl)] = fmt.Sprintf("%d", n)
			}
		}
	}

	return result, nil
}

// parseTitanResponse takes the header counts because Titan's body reports half
// the story in practice: results[].tokenCount is the completion count and
// inputTextTokenCount the prompt one, and lingo modelled only the first, so
// every Titan call reported PromptTokens 0 and a TotalTokens that was really
// the completion count. The body is still preferred wherever it reports; hdr
// fills what is missing, and when hdr is the zero value the numbers are the
// ones Titan itself gave, never a fabricated one.
//
// The answer is results[0], and that is a stated assumption rather than an
// accident. AWS documents the field as "results -- An array of one item"
// (Amazon Bedrock User Guide, "Amazon Titan Text models", InvokeModel
// Response), and lingo could not ask for a second completion even if it wanted
// one: bedrockTitanRequest and bedrockTitanConfig model inputText,
// maxTokenCount, temperature and topP, and no numResults field exists in either
// for a caller or this package to set. The alternative -- returning several
// completions -- would need a GenerationResponse that holds more than one Text,
// which no provider in this library has and none of them can currently produce.
//
// What the assumption must not do is hide a surprise, so it is not trusted
// blindly. Bedrock bills every result it generated, so CompletionTokens sums
// the token count of all of them rather than of the one whose text is returned;
// and a response that carried more than one records how many in
// Metadata["results"], which turns a silent drop into a visible one. In the
// documented single-result case both are exactly what they always were.
func (c *bedrockClient) parseTitanResponse(body []byte, modelID string, hdr bedrockHeaderUsage) (*GenerationResponse, error) {
	var resp bedrockTitanResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("failed to parse Titan response: %w", err)
	}

	if len(resp.Results) == 0 {
		return nil, fmt.Errorf("no results in Titan response")
	}

	result := resp.Results[0]
	completionTokens := 0
	for _, r := range resp.Results {
		completionTokens += r.TokenCount
	}
	usage, source := bedrockCompleteUsage(TokenUsage{
		PromptTokens:     resp.InputTextTokenCount,
		CompletionTokens: completionTokens,
		TotalTokens:      resp.InputTextTokenCount + completionTokens,
	}, hdr)

	metadata := map[string]string{
		"provider": "bedrock",
		"model":    modelID,
		"family":   "titan",
	}
	if source != "" {
		metadata["usage_source"] = source
	}
	if n := len(resp.Results); n > 1 {
		metadata["results"] = fmt.Sprintf("%d", n)
	}

	return &GenerationResponse{
		Text:         result.OutputText,
		Model:        modelID,
		FinishReason: result.CompletionReason,
		Usage:        usage,
		Metadata:     metadata,
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

// parseMistralResponse takes the header counts because Mistral's InvokeModel
// response body has nowhere to put them: AWS documents the body as
// {"outputs":[{"text","stop_reason"}]} and nothing else, which is why this
// function used to report a zeroed TokenUsage and why cost tracking silently
// under-reported every Mistral call. hdr is the zero value when the response
// carried no token headers, in which case the old zeroed usage is what comes
// back -- the same numbers as before, never a fabricated one.
//
// The answer is outputs[0] on the same terms Titan's results[0] is: no
// numResults or n field exists in bedrockMistralRequest for anything to set, so
// one output is what a lingo request can produce, and GenerationResponse holds
// one Text either way. Nothing billed is dropped by the choice -- the outputs
// carry no token counts at all, and the header counts cover the whole call
// however many outputs it produced -- so only alternative text could go
// missing, and a response that carried more than one output says so in
// Metadata["outputs"] rather than saying nothing.
func (c *bedrockClient) parseMistralResponse(body []byte, modelID string, hdr bedrockHeaderUsage) (*GenerationResponse, error) {
	var resp bedrockMistralResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("failed to parse Mistral response: %w", err)
	}

	if len(resp.Outputs) == 0 {
		return nil, fmt.Errorf("no outputs in Mistral response")
	}

	// Every number here is a gap, so bedrockCompleteUsage fills all of them and
	// labels the result "response_headers" rather than the mixed label Titan
	// gets. The label is recorded because these numbers did not come from the
	// model the way every other family's do, and an unmodelled header can stop
	// arriving without any version of anything changing.
	usage, source := bedrockCompleteUsage(TokenUsage{}, hdr)
	metadata := map[string]string{
		"provider": "bedrock",
		"model":    modelID,
		"family":   "mistral",
	}
	if source != "" {
		metadata["usage_source"] = source
	}
	if n := len(resp.Outputs); n > 1 {
		metadata["outputs"] = fmt.Sprintf("%d", n)
	}

	output := resp.Outputs[0]
	return &GenerationResponse{
		Text:         output.Text,
		Model:        modelID,
		FinishReason: output.StopReason,
		Usage:        usage,
		Metadata:     metadata,
	}, nil
}

// Health checks the health of the Bedrock client.
//
// By default it invokes nothing. Health is asked whether this client can reach
// Bedrock with usable credentials, and every question of the form "can it
// generate against model X" answers something narrower: a customer without
// access to X gets a failure that says nothing about the provider. Bedrock has
// no models endpoint on the runtime API, but it does have one operation that
// names no model at all -- ListAsyncInvokes, "GET /async-invoke", "The request
// does not have a request body" (Amazon Bedrock API Reference,
// ListAsyncInvokes) -- so that is what the default probe calls: it proves
// credentials, region and endpoint, generates no tokens and costs nothing.
// This is the choice cohere.go (CheckApiKey), ollama.go (/api/tags) and
// oaicompat.go (Models.List) already make wherever the provider offers a call
// cheaper than a generation.
//
// It used to invoke amazon.titan-text-lite-v1, chosen as "most widely
// available". That stopped being true: the Titan Text models are past
// end-of-life on Bedrock and no longer in the model catalogue, so on an account
// without Titan access the health check failed for a reason unrelated to
// health -- exactly the failure mode this design avoids.
//
// A caller who does want the narrower question answered sets
// BedrockConfig.HealthCheckModel, and that model is generated against for five
// tokens through the ordinary Generate path, so the probe exercises the same
// family routing and the same request builders a real call would. It is the
// caller's to name because only they know which models their account has --
// the same reason AnthropicConfig and OpenAICompatibleConfig take one.
func (c *bedrockClient) Health(ctx context.Context) error {
	ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	if c.healthModel != "" {
		// An empty family classifies from the id, so a health model may be
		// named as a base id, a cross-region profile id or an ARN.
		if _, err := c.Generate(ctx, NewBedrockModel(c.healthModel, "").WithMaxTokens(5), "Hello"); err != nil {
			return fmt.Errorf("bedrock health check failed: %w", err)
		}
		return nil
	}

	if _, err := c.client.ListAsyncInvokes(ctx, &bedrockruntime.ListAsyncInvokesInput{
		MaxResults: aws.Int32(1),
	}); err != nil {
		return fmt.Errorf("bedrock health check failed: %w", err)
	}

	return nil
}

// Close closes the Bedrock client (no-op for AWS SDK)
func (c *bedrockClient) Close() error {
	return nil
}
