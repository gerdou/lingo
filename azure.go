package lingo

import (
	"fmt"
	"strings"
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
// routes; it is the newest dated GA version, and the dated line is frozen
// there. Newer features live on the v1 surface instead.
const AzureAPIVersionDefault = "2024-10-21"

// AzureAPIVersionV1 selects Azure's next-generation v1 API surface instead of a
// dated api-version. Set it on AzureOpenAIConfig.APIVersion. The v1 surface is
// GA, needs no api-version at all, and is the only Azure route that models
// prompt_cache_key or reports cached_tokens.
const AzureAPIVersionV1 = "v1"

// AzureAPIVersionV1Preview is the v1 surface with preview features enabled. It
// routes identically to AzureAPIVersionV1 but pins api-version=preview.
const AzureAPIVersionV1Preview = "preview"

// azureUsesV1 reports whether an api-version selects the v1 surface. These two
// values are the only members of Azure's api-version enum for v1; everything
// else is a dated api-version served from the legacy deployment routes.
func azureUsesV1(apiVersion string) bool {
	return apiVersion == AzureAPIVersionV1 || apiVersion == AzureAPIVersionV1Preview
}

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
	// APIVersion is the api-version query parameter (default:
	// AzureAPIVersionDefault). Set AzureAPIVersionV1 to use Azure's v1 API
	// surface instead, which needs no api-version and is the only route that
	// accepts a prompt cache key. Entra ID on v1 may need a different token
	// audience; TokenCredentialScopes is the escape hatch.
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

// A standard chat deployment has no reasoning field: gpt-4o and its siblings
// reject reasoning_effort, so this type stores thinking configuration like
// every oaiOptions model and sends none of it.
func (m *AzureOpenAIModel) thinkingDimensions() ThinkingDimension { return 0 }
func (m *AzureOpenAIModel) thinkingEfforts() []ThinkingEffort     { return nil }

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

// Azure rides OpenAI's chat completions params, so the knob is the same one:
// an effort, no token budget, no toggle, and no trace to hide -- reasoning
// summaries are a Responses-API feature and Azure's acceptable use policy
// specifically forbids scraping raw chain of thought by any other route.
//
// Deployments are addressed by name, so lingo cannot tell which model is behind
// one and cannot gate the ladder per model the way it does for first-party
// OpenAI. The ladder below is the widest one Azure's current reasoning
// deployments accept; the per-model legality table (gpt-5-pro takes only high,
// minimal is original-GPT-5 only, o1-mini takes none of it) is the caller's,
// and WithReasoningEffort forwards their choice unexamined.
func (m *AzureOpenAIReasoningModel) thinkingDimensions() ThinkingDimension {
	return ThinkingCanSetEffort | ThinkingCanReportTokens
}

func (m *AzureOpenAIReasoningModel) thinkingEfforts() []ThinkingEffort {
	return []ThinkingEffort{
		ThinkingEffortNone, ThinkingEffortLow, ThinkingEffortMedium,
		ThinkingEffortHigh, ThinkingEffortXHigh,
	}
}

func (m *AzureOpenAIReasoningModel) WithMaxCompletionTokens(n int) *AzureOpenAIReasoningModel {
	m.maxCompletionTokens = n
	return m
}
func (m *AzureOpenAIReasoningModel) WithReasoningEffort(e string) *AzureOpenAIReasoningModel {
	m.setReasoningEffort(e)
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

	// The v1 surface keeps the plain OpenAI paths under /openai/v1 and routes by
	// the body's model field, so azureopenai.WithEndpoint -- which rewrites paths
	// to /openai/deployments/{deployment}/... and pins a dated api-version -- has
	// to be bypassed for it. Deployment names still travel in model, so
	// ModelName() is unchanged either way.
	var opts []option.RequestOption
	v1 := azureUsesV1(apiVersion)
	if v1 {
		opts = append(opts, option.WithBaseURL(strings.TrimSuffix(config.Endpoint, "/")+"/openai/v1/"))
		if apiVersion == AzureAPIVersionV1Preview {
			opts = append(opts, option.WithQueryAdd("api-version", AzureAPIVersionV1Preview))
		}
	} else {
		opts = append(opts, azureopenai.WithEndpoint(config.Endpoint, apiVersion))
	}

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
		"", // Health lists models: /openai/models dated, /openai/v1/models on v1
		config.Timeout,
		config.RateLimiter,
		logger,
		// prompt_cache_key exists only on Azure's v1 surface -- no dated
		// api-version, GA or preview, models it, and Azure rejects body fields
		// its api-version does not know. Dated routes therefore never send it.
		oaiCacheCaps{promptCacheKey: v1},
		// reasoning_effort is not gated on the api-version the way
		// prompt_cache_key is. lingo has always sent it on every route once a
		// caller asked for it, and Microsoft publishes no minimum api-version
		// for the field, so gating it now would silently stop sending a
		// parameter that works for callers today. A deployment whose
		// api-version does not know the field rejects it, which is the caller's
		// signal to move to AzureAPIVersionV1.
		oaiThinkingCaps{flatEffort: true},
		opts...,
	), nil
}
