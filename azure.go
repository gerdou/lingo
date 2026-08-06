package lingo

import (
	"fmt"
	"time"

	"github.com/Azure/azure-sdk-for-go/sdk/azcore"
	"github.com/openai/openai-go/v3/option"

	azureopenai "github.com/openai/openai-go/v3/azure"
)

func init() {
	RegisterProvider(ProviderAzure, func(config ProviderConfig, logger Logger) (Provider, error) {
		cfg, ok := config.(*AzureOpenAIConfig)
		if !ok {
			return nil, fmt.Errorf("invalid config type for Azure OpenAI provider")
		}
		return newAzureClient(cfg, logger)
	})
}

// AzureAPIVersionDefault is the api-version lingo sends when the config does
// not set one. Azure requires an explicit api-version on the deployment
// routes; override it to opt into newer preview features.
const AzureAPIVersionDefault = "2024-10-21"

// ============================================================================
// AZURE OPENAI PROVIDER CONFIG
// ============================================================================

// AzureOpenAIConfig contains configuration for Azure OpenAI.
//
// Azure is not reachable by pointing OpenAIConfig.BaseURL at an Azure
// resource: it authenticates with an api-key header rather than a bearer
// token, requires an api-version query parameter, and routes by deployment
// name instead of model name. This provider handles all three.
//
// Models are addressed by deployment name, which you choose when deploying:
// NewAzureOpenAIModel("my-gpt4o-deployment").
type AzureOpenAIConfig struct {
	// Endpoint is the Azure OpenAI resource endpoint (required),
	// e.g. "https://my-resource.openai.azure.com"
	Endpoint string
	// APIKey is the Azure OpenAI key. Provide this or TokenCredential.
	APIKey string
	// TokenCredential authenticates with Microsoft Entra ID instead of a key,
	// e.g. an azidentity.DefaultAzureCredential for managed identity.
	// Provide this or APIKey.
	TokenCredential azcore.TokenCredential
	// TokenCredentialScopes overrides the default token scopes. Rarely needed;
	// set it for sovereign clouds.
	TokenCredentialScopes []string
	// APIVersion is the api-version query parameter (default: AzureAPIVersionDefault)
	APIVersion string
	// Timeout is the request timeout (default: 60s)
	Timeout time.Duration
	// RateLimiter is the optional rate limit configuration
	RateLimiter *RateLimitConfig
}

// Implement ProviderConfig interface
func (c *AzureOpenAIConfig) providerType() ProviderType        { return ProviderAzure }
func (c *AzureOpenAIConfig) apiKey() string                    { return c.APIKey }
func (c *AzureOpenAIConfig) timeout() time.Duration            { return c.Timeout }
func (c *AzureOpenAIConfig) rateLimitConfig() *RateLimitConfig { return c.RateLimiter }

// ============================================================================
// MODELS
// ============================================================================

// AzureOpenAIModel is a standard chat deployment (GPT-4o, GPT-4.1 and the
// like), addressed by deployment name rather than model name.
type AzureOpenAIModel struct {
	oaiOptions
	deployment string
}

func (m *AzureOpenAIModel) ModelName() string      { return m.deployment }
func (m *AzureOpenAIModel) Provider() ProviderType { return ProviderAzure }

func (m *AzureOpenAIModel) WithMaxTokens(n int) *AzureOpenAIModel       { m.maxTokens = n; return m }
func (m *AzureOpenAIModel) WithTemperature(t float64) *AzureOpenAIModel { m.temperature = t; return m }
func (m *AzureOpenAIModel) WithTopP(p float64) *AzureOpenAIModel        { m.topP = p; return m }
func (m *AzureOpenAIModel) WithSystemPrompt(s string) *AzureOpenAIModel {
	m.systemPrompt = s
	return m
}
func (m *AzureOpenAIModel) WithExtraField(k string, v any) *AzureOpenAIModel {
	m.setExtra(k, v)
	return m
}

// NewAzureOpenAIModel creates a standard chat model for the named deployment
func NewAzureOpenAIModel(deployment string) *AzureOpenAIModel {
	return &AzureOpenAIModel{deployment: deployment}
}

// AzureOpenAIReasoningModel is a reasoning deployment (o-series, GPT-5.x),
// which takes max_completion_tokens and reasoning_effort instead of
// max_tokens and the sampling parameters.
type AzureOpenAIReasoningModel struct {
	oaiOptions
	deployment string
}

func (m *AzureOpenAIReasoningModel) ModelName() string      { return m.deployment }
func (m *AzureOpenAIReasoningModel) Provider() ProviderType { return ProviderAzure }

func (m *AzureOpenAIReasoningModel) WithMaxCompletionTokens(n int) *AzureOpenAIReasoningModel {
	m.maxCompletionTokens = n
	return m
}
func (m *AzureOpenAIReasoningModel) WithReasoningEffort(e string) *AzureOpenAIReasoningModel {
	m.reasoningEffort = e
	return m
}
func (m *AzureOpenAIReasoningModel) WithSystemPrompt(s string) *AzureOpenAIReasoningModel {
	m.systemPrompt = s
	return m
}
func (m *AzureOpenAIReasoningModel) WithExtraField(k string, v any) *AzureOpenAIReasoningModel {
	m.setExtra(k, v)
	return m
}

// NewAzureOpenAIReasoningModel creates a reasoning model for the named deployment
func NewAzureOpenAIReasoningModel(deployment string) *AzureOpenAIReasoningModel {
	return &AzureOpenAIReasoningModel{oaiOptions{maxCompletionTokens: 8192, reasoning: true}, deployment}
}

// Compile-time check that every Azure model routes through the shared client
var (
	_ oaiCompatibleModel = (*AzureOpenAIModel)(nil)
	_ oaiCompatibleModel = (*AzureOpenAIReasoningModel)(nil)
)

// ============================================================================
// CLIENT
// ============================================================================

// newAzureClient creates a new Azure OpenAI client
func newAzureClient(config *AzureOpenAIConfig, logger Logger) (Provider, error) {
	if config.Endpoint == "" {
		return nil, fmt.Errorf("Azure OpenAI endpoint is required")
	}
	if config.APIKey == "" && config.TokenCredential == nil {
		return nil, fmt.Errorf("Azure OpenAI requires either APIKey or TokenCredential")
	}

	apiVersion := config.APIVersion
	if apiVersion == "" {
		apiVersion = AzureAPIVersionDefault
	}

	// WithEndpoint adds the api-version query parameter and rewrites paths to
	// /openai/deployments/{deployment}/... using the request's model field
	opts := []option.RequestOption{azureopenai.WithEndpoint(config.Endpoint, apiVersion)}

	if config.TokenCredential != nil {
		var credOpts []azureopenai.TokenCredentialOption
		if len(config.TokenCredentialScopes) > 0 {
			credOpts = append(credOpts, azureopenai.WithTokenCredentialScopes(config.TokenCredentialScopes))
		}
		opts = append(opts, azureopenai.WithTokenCredential(config.TokenCredential, credOpts...))
	} else {
		opts = append(opts, azureopenai.WithAPIKey(config.APIKey))
	}

	return newOAICompatClient(
		ProviderAzure,
		"Azure OpenAI",
		"", // Health lists models; Azure maps this to /openai/models
		config.Timeout,
		config.RateLimiter,
		logger,
		opts...,
	), nil
}
