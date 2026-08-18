package lingo

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/shared"
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
	// BaseURL is an optional custom base URL for proxies and gateways that
	// keep OpenAI's authentication and routing.
	//
	// This is not the way to reach Azure OpenAI: Azure authenticates with an
	// api-key header rather than a bearer token, requires an api-version
	// query parameter, and routes by deployment name. Use AzureOpenAIConfig.
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
	cache        CacheOptions
}

// CacheOptions returns the model's prompt caching configuration. Every standard
// OpenAI model embeds openAIStandardOptions, so this one declaration makes them
// all satisfy CacheableModel.
func (o *openAIStandardOptions) CacheOptions() *CacheOptions { return &o.cache }

// thinkingDimensions answers for the standard models, whose API has no
// reasoning field: gpt-4o and its siblings reject reasoning_effort outright.
// They deliberately do NOT get a ThinkingOptions accessor -- the sibling option
// structs are the gate, exactly as they are for the reasoning-only parameters.
func (o *openAIStandardOptions) thinkingDimensions() ThinkingDimension { return 0 }

// openAIReasoningOptions contains options for reasoning models (o1, o3, o4, GPT-5)
type openAIReasoningOptions struct {
	modelVersion        string // Optional: override model name with specific version
	maxCompletionTokens int
	thinking            ThinkingOptions
	systemPrompt        string
	cache               CacheOptions
}

// CacheOptions returns the model's prompt caching configuration. The reasoning
// options are a sibling of openAIStandardOptions rather than an extension of
// it, so the accessor has to be declared on both.
func (o *openAIReasoningOptions) CacheOptions() *CacheOptions { return &o.cache }

// ThinkingOptions returns the model's thinking configuration. Every OpenAI
// reasoning model embeds openAIReasoningOptions, so this one declaration makes
// them all satisfy ThinkingModel -- and, deliberately, only them.
//
// It is the single storage behind WithReasoningEffort, so the portable surface
// and the per-model setter can never disagree about what the request carries.
func (o *openAIReasoningOptions) ThinkingOptions() *ThinkingOptions { return &o.thinking }

// setReasoningEffort backs the per-model WithReasoningEffort setters.
//
// It writes the level and pins it, which is what keeps a caller's own string on
// the wire byte for byte: pinned means never clamped to the model's ladder,
// never translated, never dropped for being a value this library has not heard
// of. The setter takes a bare string and has always forwarded whatever it was
// given, including "none" on a model with no off switch, and that is preserved.
//
// It leaves the mode alone, which is where it parts company with the portable
// ThinkingOptions.WithEffort. reasoning_effort is the only thinking field this
// dialect has, so a mode of its own would have nowhere to go, and letting
// WithReasoningEffort("none") read as ThinkingModeOff would send the portable
// surface's off switch instead of the caller's literal value.
func (o *openAIReasoningOptions) setReasoningEffort(e string) {
	o.thinking.effort = ThinkingEffort(e)
	o.thinking.pin(ThinkingCanSetEffort)
}

// openAISeededEffort is the pinned reasoning_effort several OpenAI constructors
// have always shipped -- "medium" for most, "high" for the pro tier, "low" for
// GPT-5.6 Luna. Those defaults predate the portable thinking surface, so they
// are seeded exactly as a per-model setter would have set them: pinned, and
// therefore on the wire unchanged. Dropping them would silently alter every
// request written before this feature existed.
func openAISeededEffort(e ThinkingEffort) ThinkingOptions {
	return ThinkingOptions{effort: e, pinned: ThinkingCanSetEffort}
}

// ============================================================================
// REASONING EFFORT LADDERS
// ============================================================================
//
// reasoning_effort is per MODEL, not per provider: the vocabulary has grown
// with each family and OpenAI rejects a rung a given model does not know. The
// table below is what the portable surface is clamped to. It never constrains
// WithReasoningEffort, whose values are pinned and forwarded as given.
//
// Two rungs are missing everywhere on purpose. "max" exists only on the
// Responses API, which lingo does not speak, so it can never legally be sent
// here and clamps down to xhigh. "minimal" is the original GPT-5 family only.

var (
	// openAIEffortsMinimal is the original GPT-5 ladder: minimal, no none.
	openAIEffortsMinimal = []ThinkingEffort{
		ThinkingEffortMinimal, ThinkingEffortLow, ThinkingEffortMedium, ThinkingEffortHigh,
	}
	// openAIEffortsNone is gpt-5.1 and 5.2: none arrived, minimal left.
	openAIEffortsNone = []ThinkingEffort{
		ThinkingEffortNone, ThinkingEffortLow, ThinkingEffortMedium, ThinkingEffortHigh,
	}
	// openAIEffortsXHigh is gpt-5.4 and later, plus gpt-5.1-codex-max.
	openAIEffortsXHigh = []ThinkingEffort{
		ThinkingEffortNone, ThinkingEffortLow, ThinkingEffortMedium,
		ThinkingEffortHigh, ThinkingEffortXHigh,
	}
)

// openAIEffortLadders maps model id prefixes to the reasoning_effort values
// that id accepts, first match wins, so longer prefixes are listed before the
// shorter ones they extend.
var openAIEffortLadders = []struct {
	prefix  string
	efforts []ThinkingEffort
}{
	// o1-mini takes no reasoning_effort at all. Its constructor has always
	// seeded one anyway, which is pinned and so still sent; the empty ladder
	// only stops the portable surface from adding to a known-bad request.
	{"o1-mini", nil},
	{"o1", []ThinkingEffort{ThinkingEffortLow, ThinkingEffortMedium, ThinkingEffortHigh}},
	{"o3", []ThinkingEffort{ThinkingEffortLow, ThinkingEffortMedium, ThinkingEffortHigh}},
	{"o4", []ThinkingEffort{ThinkingEffortLow, ThinkingEffortMedium, ThinkingEffortHigh}},
	// gpt-5-pro accepts high and nothing else.
	{"gpt-5-pro", []ThinkingEffort{ThinkingEffortHigh}},
	// The original GPT-5 family is the only one with "minimal" and the only
	// pre-5.1 family without "none".
	{"gpt-5-codex", []ThinkingEffort{ThinkingEffortLow, ThinkingEffortMedium, ThinkingEffortHigh}},
	{"gpt-5-", openAIEffortsMinimal},
	{"gpt-5.1-codex-max", openAIEffortsXHigh},
	{"gpt-5.1", openAIEffortsNone},
	{"gpt-5.2", openAIEffortsNone},
	{"gpt-5.4", openAIEffortsXHigh},
	{"gpt-5.5", openAIEffortsXHigh},
	{"gpt-5.6", openAIEffortsXHigh},
	// Bare "gpt-5" is the original family; it sorts after the dotted ids above
	// so gpt-5.1 and friends never match it.
	{"gpt-5", openAIEffortsMinimal},
}

// openAIEffortLadder returns the reasoning_effort rungs a model id accepts.
//
// An id this library has not seen gets the current family's ladder rather than
// nothing, so the generic OpenAIReasoningModel is not the one place the
// portable surface silently does nothing. The rungs OpenAI has withdrawn --
// "minimal" from 5.1 onward -- are absent from it, so the optimistic answer
// cannot ask a newer model for something a newer model is likely to reject.
func openAIEffortLadder(modelID string) []ThinkingEffort {
	for _, l := range openAIEffortLadders {
		if strings.HasPrefix(modelID, l.prefix) {
			return l.efforts
		}
	}
	return openAIEffortsXHigh
}

// openAIThinkingDimensions answers ModelThinkingDimensions for one reasoning
// model, resolved from the model id so a zero-value literal and the generic
// OpenAIReasoningModel both get the right answer without a constructor having
// stored anything.
//
// Only two dimensions are ever real here. Chat completions has no thinking
// token budget and no boolean toggle -- "off" is the effort rung "none", which
// only gpt-5.1 and later accept -- and it returns no reasoning trace at all, so
// there is nothing to hide and nothing to report but the token count.
func openAIThinkingDimensions(modelID string) ThinkingDimension {
	if len(openAIEffortLadder(modelID)) == 0 {
		return ThinkingCanReportTokens
	}
	return ThinkingCanSetEffort | ThinkingCanReportTokens
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
//
// Deprecated: scheduled for shutdown by OpenAI on Oct 23, 2026. Migrate to GPT56Terra.
type GPT4Turbo struct{ openAIStandardOptions }

func (m *GPT4Turbo) ModelName() string {
	if m.modelVersion != "" {
		return m.modelVersion
	}
	return "gpt-4-turbo"
}
func (m *GPT4Turbo) Provider() ProviderType { return ProviderOpenAI }
func (m *GPT4Turbo) SystemPrompt() string   { return m.systemPrompt }

func (m *GPT4Turbo) WithVersion(v string) *GPT4Turbo      { m.modelVersion = v; return m }
func (m *GPT4Turbo) WithMaxTokens(n int) *GPT4Turbo       { m.maxTokens = n; return m }
func (m *GPT4Turbo) WithTemperature(t float64) *GPT4Turbo { m.temperature = t; return m }
func (m *GPT4Turbo) WithTopP(p float64) *GPT4Turbo        { m.topP = p; return m }
func (m *GPT4Turbo) WithSystemPrompt(s string) *GPT4Turbo { m.systemPrompt = s; return m }

// NewGPT4Turbo creates a new GPT-4-turbo model with default options
//
// Deprecated: scheduled for shutdown by OpenAI on Oct 23, 2026. Migrate to GPT56Terra.
func NewGPT4Turbo() *GPT4Turbo {
	return &GPT4Turbo{openAIStandardOptions{maxTokens: 4096, temperature: 1.0}}
}

// GPT4 represents the GPT-4 model
// Versions: gpt-4, gpt-4-0613
//
// Deprecated: scheduled for shutdown by OpenAI on Oct 23, 2026. Migrate to GPT56Terra.
type GPT4 struct{ openAIStandardOptions }

func (m *GPT4) ModelName() string {
	if m.modelVersion != "" {
		return m.modelVersion
	}
	return "gpt-4"
}
func (m *GPT4) Provider() ProviderType { return ProviderOpenAI }
func (m *GPT4) SystemPrompt() string   { return m.systemPrompt }

func (m *GPT4) WithVersion(v string) *GPT4      { m.modelVersion = v; return m }
func (m *GPT4) WithMaxTokens(n int) *GPT4       { m.maxTokens = n; return m }
func (m *GPT4) WithTemperature(t float64) *GPT4 { m.temperature = t; return m }
func (m *GPT4) WithTopP(p float64) *GPT4        { m.topP = p; return m }
func (m *GPT4) WithSystemPrompt(s string) *GPT4 { m.systemPrompt = s; return m }

// NewGPT4 creates a new GPT-4 model with default options
//
// Deprecated: scheduled for shutdown by OpenAI on Oct 23, 2026. Migrate to GPT56Terra.
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
//
// Deprecated: scheduled for shutdown by OpenAI on Oct 23, 2026. Migrate to GPT56Luna.
type GPT35Turbo struct{ openAIStandardOptions }

func (m *GPT35Turbo) ModelName() string {
	if m.modelVersion != "" {
		return m.modelVersion
	}
	return "gpt-3.5-turbo"
}
func (m *GPT35Turbo) Provider() ProviderType { return ProviderOpenAI }
func (m *GPT35Turbo) SystemPrompt() string   { return m.systemPrompt }

func (m *GPT35Turbo) WithVersion(v string) *GPT35Turbo      { m.modelVersion = v; return m }
func (m *GPT35Turbo) WithMaxTokens(n int) *GPT35Turbo       { m.maxTokens = n; return m }
func (m *GPT35Turbo) WithTemperature(t float64) *GPT35Turbo { m.temperature = t; return m }
func (m *GPT35Turbo) WithTopP(p float64) *GPT35Turbo        { m.topP = p; return m }
func (m *GPT35Turbo) WithSystemPrompt(s string) *GPT35Turbo { m.systemPrompt = s; return m }

// NewGPT35Turbo creates a new GPT-3.5-turbo model with default options
//
// Deprecated: scheduled for shutdown by OpenAI on Oct 23, 2026. Migrate to GPT56Luna.
func NewGPT35Turbo() *GPT35Turbo {
	return &GPT35Turbo{openAIStandardOptions{maxTokens: 4096, temperature: 1.0}}
}

// ============================================================================
// REASONING MODELS (O1, O3, O4, GPT-5 series)
// ============================================================================

// O1 represents the O1 reasoning model
// Versions: o1, o1-2024-12-17
//
// Deprecated: scheduled for shutdown by OpenAI on Oct 23, 2026. Migrate to GPT56Sol.
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
func (m *O1) thinkingDimensions() ThinkingDimension {
	return openAIThinkingDimensions(m.ModelName())
}

func (m *O1) WithVersion(v string) *O1          { m.modelVersion = v; return m }
func (m *O1) WithMaxCompletionTokens(n int) *O1 { m.maxCompletionTokens = n; return m }
func (m *O1) WithReasoningEffort(e string) *O1  { m.setReasoningEffort(e); return m }
func (m *O1) WithSystemPrompt(s string) *O1     { m.systemPrompt = s; return m }

// NewO1 creates a new O1 model with default options
//
// Deprecated: scheduled for shutdown by OpenAI on Oct 23, 2026. Migrate to GPT56Sol.
func NewO1() *O1 {
	return &O1{openAIReasoningOptions{maxCompletionTokens: 4096, thinking: openAISeededEffort(ThinkingEffortMedium)}}
}

// O1Mini represents the O1-mini reasoning model.
// Versions: o1-mini, o1-mini-2024-09-12
//
// Deprecated: removed from the OpenAI API (deprecation announced Apr 2025); requests return 404. Migrate to O4Mini or GPT5Mini.
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
func (m *O1Mini) thinkingDimensions() ThinkingDimension {
	return openAIThinkingDimensions(m.ModelName())
}

func (m *O1Mini) WithVersion(v string) *O1Mini          { m.modelVersion = v; return m }
func (m *O1Mini) WithMaxCompletionTokens(n int) *O1Mini { m.maxCompletionTokens = n; return m }
func (m *O1Mini) WithReasoningEffort(e string) *O1Mini  { m.setReasoningEffort(e); return m }
func (m *O1Mini) WithSystemPrompt(s string) *O1Mini     { m.systemPrompt = s; return m }

// NewO1Mini creates a new O1-mini model with default options
//
// Deprecated: removed from the OpenAI API (deprecation announced Apr 2025); requests return 404. Migrate to O4Mini or GPT5Mini.
func NewO1Mini() *O1Mini {
	return &O1Mini{openAIReasoningOptions{maxCompletionTokens: 4096, thinking: openAISeededEffort(ThinkingEffortMedium)}}
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
func (m *O1Pro) thinkingDimensions() ThinkingDimension {
	return openAIThinkingDimensions(m.ModelName())
}

func (m *O1Pro) WithVersion(v string) *O1Pro          { m.modelVersion = v; return m }
func (m *O1Pro) WithMaxCompletionTokens(n int) *O1Pro { m.maxCompletionTokens = n; return m }
func (m *O1Pro) WithReasoningEffort(e string) *O1Pro  { m.setReasoningEffort(e); return m }
func (m *O1Pro) WithSystemPrompt(s string) *O1Pro     { m.systemPrompt = s; return m }

// NewO1Pro creates a new O1-pro model with default options
func NewO1Pro() *O1Pro {
	return &O1Pro{openAIReasoningOptions{maxCompletionTokens: 8192, thinking: openAISeededEffort(ThinkingEffortHigh)}}
}

// O3 represents the O3 reasoning model
// Versions: o3, o3-2025-04-16
//
// Deprecated: scheduled for shutdown by OpenAI on Dec 11, 2026. Migrate to GPT56Sol.
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
func (m *O3) thinkingDimensions() ThinkingDimension {
	return openAIThinkingDimensions(m.ModelName())
}

func (m *O3) WithVersion(v string) *O3          { m.modelVersion = v; return m }
func (m *O3) WithMaxCompletionTokens(n int) *O3 { m.maxCompletionTokens = n; return m }
func (m *O3) WithReasoningEffort(e string) *O3  { m.setReasoningEffort(e); return m }
func (m *O3) WithSystemPrompt(s string) *O3     { m.systemPrompt = s; return m }

// NewO3 creates a new O3 model with default options
//
// Deprecated: scheduled for shutdown by OpenAI on Dec 11, 2026. Migrate to GPT56Sol.
func NewO3() *O3 {
	return &O3{openAIReasoningOptions{maxCompletionTokens: 8192, thinking: openAISeededEffort(ThinkingEffortMedium)}}
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
func (m *O3Mini) thinkingDimensions() ThinkingDimension {
	return openAIThinkingDimensions(m.ModelName())
}

func (m *O3Mini) WithVersion(v string) *O3Mini          { m.modelVersion = v; return m }
func (m *O3Mini) WithMaxCompletionTokens(n int) *O3Mini { m.maxCompletionTokens = n; return m }
func (m *O3Mini) WithReasoningEffort(e string) *O3Mini  { m.setReasoningEffort(e); return m }
func (m *O3Mini) WithSystemPrompt(s string) *O3Mini     { m.systemPrompt = s; return m }

// NewO3Mini creates a new O3-mini model with default options
func NewO3Mini() *O3Mini {
	return &O3Mini{openAIReasoningOptions{maxCompletionTokens: 4096, thinking: openAISeededEffort(ThinkingEffortMedium)}}
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
func (m *O4Mini) thinkingDimensions() ThinkingDimension {
	return openAIThinkingDimensions(m.ModelName())
}

func (m *O4Mini) WithVersion(v string) *O4Mini          { m.modelVersion = v; return m }
func (m *O4Mini) WithMaxCompletionTokens(n int) *O4Mini { m.maxCompletionTokens = n; return m }
func (m *O4Mini) WithReasoningEffort(e string) *O4Mini  { m.setReasoningEffort(e); return m }
func (m *O4Mini) WithSystemPrompt(s string) *O4Mini     { m.systemPrompt = s; return m }

// NewO4Mini creates a new O4-mini model with default options
func NewO4Mini() *O4Mini {
	return &O4Mini{openAIReasoningOptions{maxCompletionTokens: 4096, thinking: openAISeededEffort(ThinkingEffortMedium)}}
}

// GPT5 represents the GPT-5 reasoning model
//
// Deprecated: scheduled for shutdown by OpenAI on Dec 11, 2026. Migrate to GPT56Sol.
type GPT5 struct{ openAIReasoningOptions }

func (m *GPT5) ModelName() string      { return "gpt-5" }
func (m *GPT5) Provider() ProviderType { return ProviderOpenAI }
func (m *GPT5) SystemPrompt() string   { return m.systemPrompt }
func (m *GPT5) isReasoning() bool      { return true }
func (m *GPT5) thinkingDimensions() ThinkingDimension {
	return openAIThinkingDimensions(m.ModelName())
}

func (m *GPT5) WithMaxCompletionTokens(n int) *GPT5 { m.maxCompletionTokens = n; return m }
func (m *GPT5) WithReasoningEffort(e string) *GPT5  { m.setReasoningEffort(e); return m }
func (m *GPT5) WithSystemPrompt(s string) *GPT5     { m.systemPrompt = s; return m }

// NewGPT5 creates a new GPT-5 model with default options
//
// Deprecated: scheduled for shutdown by OpenAI on Dec 11, 2026. Migrate to GPT56Sol.
func NewGPT5() *GPT5 {
	return &GPT5{openAIReasoningOptions{maxCompletionTokens: 8192, thinking: openAISeededEffort(ThinkingEffortMedium)}}
}

// GPT5Mini represents the GPT-5-mini reasoning model
//
// Deprecated: scheduled for shutdown by OpenAI on Dec 11, 2026. Migrate to GPT56Terra.
type GPT5Mini struct{ openAIReasoningOptions }

func (m *GPT5Mini) ModelName() string      { return "gpt-5-mini" }
func (m *GPT5Mini) Provider() ProviderType { return ProviderOpenAI }
func (m *GPT5Mini) SystemPrompt() string   { return m.systemPrompt }
func (m *GPT5Mini) isReasoning() bool      { return true }
func (m *GPT5Mini) thinkingDimensions() ThinkingDimension {
	return openAIThinkingDimensions(m.ModelName())
}

func (m *GPT5Mini) WithMaxCompletionTokens(n int) *GPT5Mini { m.maxCompletionTokens = n; return m }
func (m *GPT5Mini) WithReasoningEffort(e string) *GPT5Mini  { m.setReasoningEffort(e); return m }
func (m *GPT5Mini) WithSystemPrompt(s string) *GPT5Mini     { m.systemPrompt = s; return m }

// NewGPT5Mini creates a new GPT-5-mini model with default options
//
// Deprecated: scheduled for shutdown by OpenAI on Dec 11, 2026. Migrate to GPT56Terra.
func NewGPT5Mini() *GPT5Mini {
	return &GPT5Mini{openAIReasoningOptions{maxCompletionTokens: 4096, thinking: openAISeededEffort(ThinkingEffortMedium)}}
}

// GPT5Nano represents the GPT-5-nano reasoning model
//
// Deprecated: scheduled for shutdown by OpenAI on Dec 11, 2026. Migrate to GPT56Luna.
type GPT5Nano struct{ openAIReasoningOptions }

func (m *GPT5Nano) ModelName() string      { return "gpt-5-nano" }
func (m *GPT5Nano) Provider() ProviderType { return ProviderOpenAI }
func (m *GPT5Nano) SystemPrompt() string   { return m.systemPrompt }
func (m *GPT5Nano) isReasoning() bool      { return true }
func (m *GPT5Nano) thinkingDimensions() ThinkingDimension {
	return openAIThinkingDimensions(m.ModelName())
}

func (m *GPT5Nano) WithMaxCompletionTokens(n int) *GPT5Nano { m.maxCompletionTokens = n; return m }
func (m *GPT5Nano) WithReasoningEffort(e string) *GPT5Nano  { m.setReasoningEffort(e); return m }
func (m *GPT5Nano) WithSystemPrompt(s string) *GPT5Nano     { m.systemPrompt = s; return m }

// NewGPT5Nano creates a new GPT-5-nano model with default options
//
// Deprecated: scheduled for shutdown by OpenAI on Dec 11, 2026. Migrate to GPT56Luna.
func NewGPT5Nano() *GPT5Nano {
	return &GPT5Nano{openAIReasoningOptions{maxCompletionTokens: 4096, thinking: openAISeededEffort(ThinkingEffortMedium)}}
}

// GPT5Pro represents the GPT-5-pro reasoning model
type GPT5Pro struct{ openAIReasoningOptions }

func (m *GPT5Pro) ModelName() string      { return "gpt-5-pro" }
func (m *GPT5Pro) Provider() ProviderType { return ProviderOpenAI }
func (m *GPT5Pro) SystemPrompt() string   { return m.systemPrompt }
func (m *GPT5Pro) isReasoning() bool      { return true }
func (m *GPT5Pro) thinkingDimensions() ThinkingDimension {
	return openAIThinkingDimensions(m.ModelName())
}

func (m *GPT5Pro) WithMaxCompletionTokens(n int) *GPT5Pro { m.maxCompletionTokens = n; return m }
func (m *GPT5Pro) WithReasoningEffort(e string) *GPT5Pro  { m.setReasoningEffort(e); return m }
func (m *GPT5Pro) WithSystemPrompt(s string) *GPT5Pro     { m.systemPrompt = s; return m }

// NewGPT5Pro creates a new GPT-5-pro model with default options
func NewGPT5Pro() *GPT5Pro {
	return &GPT5Pro{openAIReasoningOptions{maxCompletionTokens: 8192, thinking: openAISeededEffort(ThinkingEffortHigh)}}
}

// GPT51 represents the GPT-5.1 reasoning model
type GPT51 struct{ openAIReasoningOptions }

func (m *GPT51) ModelName() string      { return "gpt-5.1" }
func (m *GPT51) Provider() ProviderType { return ProviderOpenAI }
func (m *GPT51) SystemPrompt() string   { return m.systemPrompt }
func (m *GPT51) isReasoning() bool      { return true }
func (m *GPT51) thinkingDimensions() ThinkingDimension {
	return openAIThinkingDimensions(m.ModelName())
}

func (m *GPT51) WithMaxCompletionTokens(n int) *GPT51 { m.maxCompletionTokens = n; return m }
func (m *GPT51) WithReasoningEffort(e string) *GPT51  { m.setReasoningEffort(e); return m }
func (m *GPT51) WithSystemPrompt(s string) *GPT51     { m.systemPrompt = s; return m }

// NewGPT51 creates a new GPT-5.1 model with default options
func NewGPT51() *GPT51 {
	return &GPT51{openAIReasoningOptions{maxCompletionTokens: 8192, thinking: openAISeededEffort(ThinkingEffortMedium)}}
}

// GPT51Mini represents the GPT-5.1-mini reasoning model
type GPT51Mini struct{ openAIReasoningOptions }

func (m *GPT51Mini) ModelName() string      { return "gpt-5.1-mini" }
func (m *GPT51Mini) Provider() ProviderType { return ProviderOpenAI }
func (m *GPT51Mini) SystemPrompt() string   { return m.systemPrompt }
func (m *GPT51Mini) isReasoning() bool      { return true }
func (m *GPT51Mini) thinkingDimensions() ThinkingDimension {
	return openAIThinkingDimensions(m.ModelName())
}

func (m *GPT51Mini) WithMaxCompletionTokens(n int) *GPT51Mini { m.maxCompletionTokens = n; return m }
func (m *GPT51Mini) WithReasoningEffort(e string) *GPT51Mini  { m.setReasoningEffort(e); return m }
func (m *GPT51Mini) WithSystemPrompt(s string) *GPT51Mini     { m.systemPrompt = s; return m }

// NewGPT51Mini creates a new GPT-5.1-mini model with default options
func NewGPT51Mini() *GPT51Mini {
	return &GPT51Mini{openAIReasoningOptions{maxCompletionTokens: 4096, thinking: openAISeededEffort(ThinkingEffortMedium)}}
}

// GPT51Nano represents the GPT-5.1-nano reasoning model
type GPT51Nano struct{ openAIReasoningOptions }

func (m *GPT51Nano) ModelName() string      { return "gpt-5.1-nano" }
func (m *GPT51Nano) Provider() ProviderType { return ProviderOpenAI }
func (m *GPT51Nano) SystemPrompt() string   { return m.systemPrompt }
func (m *GPT51Nano) isReasoning() bool      { return true }
func (m *GPT51Nano) thinkingDimensions() ThinkingDimension {
	return openAIThinkingDimensions(m.ModelName())
}

func (m *GPT51Nano) WithMaxCompletionTokens(n int) *GPT51Nano { m.maxCompletionTokens = n; return m }
func (m *GPT51Nano) WithReasoningEffort(e string) *GPT51Nano  { m.setReasoningEffort(e); return m }
func (m *GPT51Nano) WithSystemPrompt(s string) *GPT51Nano     { m.systemPrompt = s; return m }

// NewGPT51Nano creates a new GPT-5.1-nano model with default options
func NewGPT51Nano() *GPT51Nano {
	return &GPT51Nano{openAIReasoningOptions{maxCompletionTokens: 4096, thinking: openAISeededEffort(ThinkingEffortMedium)}}
}

// GPT51Codex represents the GPT-5.1-codex reasoning model
//
// Deprecated: retired by OpenAI (Jul 23, 2026); the API returns 404. Migrate to GPT56Sol.
type GPT51Codex struct{ openAIReasoningOptions }

func (m *GPT51Codex) ModelName() string      { return "gpt-5.1-codex" }
func (m *GPT51Codex) Provider() ProviderType { return ProviderOpenAI }
func (m *GPT51Codex) SystemPrompt() string   { return m.systemPrompt }
func (m *GPT51Codex) isReasoning() bool      { return true }
func (m *GPT51Codex) thinkingDimensions() ThinkingDimension {
	return openAIThinkingDimensions(m.ModelName())
}

func (m *GPT51Codex) WithMaxCompletionTokens(n int) *GPT51Codex { m.maxCompletionTokens = n; return m }
func (m *GPT51Codex) WithReasoningEffort(e string) *GPT51Codex  { m.setReasoningEffort(e); return m }
func (m *GPT51Codex) WithSystemPrompt(s string) *GPT51Codex     { m.systemPrompt = s; return m }

// NewGPT51Codex creates a new GPT-5.1-codex model with default options
//
// Deprecated: retired by OpenAI (Jul 23, 2026); the API returns 404. Migrate to GPT56Sol.
func NewGPT51Codex() *GPT51Codex {
	return &GPT51Codex{openAIReasoningOptions{maxCompletionTokens: 8192, thinking: openAISeededEffort(ThinkingEffortMedium)}}
}

// GPT51CodexMini represents the GPT-5.1-codex-mini reasoning model
//
// Deprecated: retired by OpenAI (Jul 23, 2026); the API returns 404. Migrate to GPT56Luna.
type GPT51CodexMini struct{ openAIReasoningOptions }

func (m *GPT51CodexMini) ModelName() string      { return "gpt-5.1-codex-mini" }
func (m *GPT51CodexMini) Provider() ProviderType { return ProviderOpenAI }
func (m *GPT51CodexMini) SystemPrompt() string   { return m.systemPrompt }
func (m *GPT51CodexMini) isReasoning() bool      { return true }
func (m *GPT51CodexMini) thinkingDimensions() ThinkingDimension {
	return openAIThinkingDimensions(m.ModelName())
}

func (m *GPT51CodexMini) WithMaxCompletionTokens(n int) *GPT51CodexMini {
	m.maxCompletionTokens = n
	return m
}
func (m *GPT51CodexMini) WithReasoningEffort(e string) *GPT51CodexMini {
	m.setReasoningEffort(e)
	return m
}
func (m *GPT51CodexMini) WithSystemPrompt(s string) *GPT51CodexMini { m.systemPrompt = s; return m }

// NewGPT51CodexMini creates a new GPT-5.1-codex-mini model with default options
//
// Deprecated: retired by OpenAI (Jul 23, 2026); the API returns 404. Migrate to GPT56Luna.
func NewGPT51CodexMini() *GPT51CodexMini {
	return &GPT51CodexMini{openAIReasoningOptions{maxCompletionTokens: 4096, thinking: openAISeededEffort(ThinkingEffortMedium)}}
}

// GPT54Nano represents the GPT-5.4-nano reasoning model (cheapest GPT-5.4-class model)
type GPT54Nano struct{ openAIReasoningOptions }

func (m *GPT54Nano) ModelName() string      { return "gpt-5.4-nano" }
func (m *GPT54Nano) Provider() ProviderType { return ProviderOpenAI }
func (m *GPT54Nano) SystemPrompt() string   { return m.systemPrompt }
func (m *GPT54Nano) isReasoning() bool      { return true }
func (m *GPT54Nano) thinkingDimensions() ThinkingDimension {
	return openAIThinkingDimensions(m.ModelName())
}

func (m *GPT54Nano) WithMaxCompletionTokens(n int) *GPT54Nano { m.maxCompletionTokens = n; return m }
func (m *GPT54Nano) WithReasoningEffort(e string) *GPT54Nano  { m.setReasoningEffort(e); return m }
func (m *GPT54Nano) WithSystemPrompt(s string) *GPT54Nano     { m.systemPrompt = s; return m }

// NewGPT54Nano creates a new GPT-5.4-nano model with default options
func NewGPT54Nano() *GPT54Nano {
	return &GPT54Nano{openAIReasoningOptions{maxCompletionTokens: 4096, thinking: openAISeededEffort(ThinkingEffortMedium)}}
}

// GPT54Mini represents the GPT-5.4-mini reasoning model (strong mini model for coding and subagents)
type GPT54Mini struct{ openAIReasoningOptions }

func (m *GPT54Mini) ModelName() string      { return "gpt-5.4-mini" }
func (m *GPT54Mini) Provider() ProviderType { return ProviderOpenAI }
func (m *GPT54Mini) SystemPrompt() string   { return m.systemPrompt }
func (m *GPT54Mini) isReasoning() bool      { return true }
func (m *GPT54Mini) thinkingDimensions() ThinkingDimension {
	return openAIThinkingDimensions(m.ModelName())
}

func (m *GPT54Mini) WithMaxCompletionTokens(n int) *GPT54Mini { m.maxCompletionTokens = n; return m }
func (m *GPT54Mini) WithReasoningEffort(e string) *GPT54Mini  { m.setReasoningEffort(e); return m }
func (m *GPT54Mini) WithSystemPrompt(s string) *GPT54Mini     { m.systemPrompt = s; return m }

// NewGPT54Mini creates a new GPT-5.4-mini model with default options
func NewGPT54Mini() *GPT54Mini {
	return &GPT54Mini{openAIReasoningOptions{maxCompletionTokens: 4096, thinking: openAISeededEffort(ThinkingEffortMedium)}}
}

// GPT54 represents the GPT-5.4 reasoning model (affordable model for coding and professional work)
type GPT54 struct{ openAIReasoningOptions }

func (m *GPT54) ModelName() string      { return "gpt-5.4" }
func (m *GPT54) Provider() ProviderType { return ProviderOpenAI }
func (m *GPT54) SystemPrompt() string   { return m.systemPrompt }
func (m *GPT54) isReasoning() bool      { return true }
func (m *GPT54) thinkingDimensions() ThinkingDimension {
	return openAIThinkingDimensions(m.ModelName())
}

func (m *GPT54) WithMaxCompletionTokens(n int) *GPT54 { m.maxCompletionTokens = n; return m }
func (m *GPT54) WithReasoningEffort(e string) *GPT54  { m.setReasoningEffort(e); return m }
func (m *GPT54) WithSystemPrompt(s string) *GPT54     { m.systemPrompt = s; return m }

// NewGPT54 creates a new GPT-5.4 model with default options
func NewGPT54() *GPT54 {
	return &GPT54{openAIReasoningOptions{maxCompletionTokens: 8192, thinking: openAISeededEffort(ThinkingEffortMedium)}}
}

// GPT54Pro represents the GPT-5.4-pro reasoning model (higher-precision GPT-5.4)
type GPT54Pro struct{ openAIReasoningOptions }

func (m *GPT54Pro) ModelName() string      { return "gpt-5.4-pro" }
func (m *GPT54Pro) Provider() ProviderType { return ProviderOpenAI }
func (m *GPT54Pro) SystemPrompt() string   { return m.systemPrompt }
func (m *GPT54Pro) isReasoning() bool      { return true }
func (m *GPT54Pro) thinkingDimensions() ThinkingDimension {
	return openAIThinkingDimensions(m.ModelName())
}

func (m *GPT54Pro) WithMaxCompletionTokens(n int) *GPT54Pro { m.maxCompletionTokens = n; return m }
func (m *GPT54Pro) WithReasoningEffort(e string) *GPT54Pro  { m.setReasoningEffort(e); return m }
func (m *GPT54Pro) WithSystemPrompt(s string) *GPT54Pro     { m.systemPrompt = s; return m }

// NewGPT54Pro creates a new GPT-5.4-pro model with default options
func NewGPT54Pro() *GPT54Pro {
	return &GPT54Pro{openAIReasoningOptions{maxCompletionTokens: 8192, thinking: openAISeededEffort(ThinkingEffortHigh)}}
}

// GPT55 represents the GPT-5.5 reasoning model
// GPT-5.5 is OpenAI's frontier model for the most complex coding and professional work.
type GPT55 struct{ openAIReasoningOptions }

func (m *GPT55) ModelName() string      { return "gpt-5.5" }
func (m *GPT55) Provider() ProviderType { return ProviderOpenAI }
func (m *GPT55) SystemPrompt() string   { return m.systemPrompt }
func (m *GPT55) isReasoning() bool      { return true }
func (m *GPT55) thinkingDimensions() ThinkingDimension {
	return openAIThinkingDimensions(m.ModelName())
}

func (m *GPT55) WithMaxCompletionTokens(n int) *GPT55 { m.maxCompletionTokens = n; return m }
func (m *GPT55) WithReasoningEffort(e string) *GPT55  { m.setReasoningEffort(e); return m }
func (m *GPT55) WithSystemPrompt(s string) *GPT55     { m.systemPrompt = s; return m }

// NewGPT55 creates a new GPT-5.5 model with default options
func NewGPT55() *GPT55 {
	return &GPT55{openAIReasoningOptions{maxCompletionTokens: 8192, thinking: openAISeededEffort(ThinkingEffortMedium)}}
}

// GPT55Pro represents the GPT-5.5-pro reasoning model (smarter, more precise GPT-5.5)
type GPT55Pro struct{ openAIReasoningOptions }

func (m *GPT55Pro) ModelName() string      { return "gpt-5.5-pro" }
func (m *GPT55Pro) Provider() ProviderType { return ProviderOpenAI }
func (m *GPT55Pro) SystemPrompt() string   { return m.systemPrompt }
func (m *GPT55Pro) isReasoning() bool      { return true }
func (m *GPT55Pro) thinkingDimensions() ThinkingDimension {
	return openAIThinkingDimensions(m.ModelName())
}

func (m *GPT55Pro) WithMaxCompletionTokens(n int) *GPT55Pro { m.maxCompletionTokens = n; return m }
func (m *GPT55Pro) WithReasoningEffort(e string) *GPT55Pro  { m.setReasoningEffort(e); return m }
func (m *GPT55Pro) WithSystemPrompt(s string) *GPT55Pro     { m.systemPrompt = s; return m }

// NewGPT55Pro creates a new GPT-5.5-pro model with default options
func NewGPT55Pro() *GPT55Pro {
	return &GPT55Pro{openAIReasoningOptions{maxCompletionTokens: 8192, thinking: openAISeededEffort(ThinkingEffortHigh)}}
}

// GPT56Sol represents the GPT-5.6 Sol reasoning model.
// Sol is OpenAI's current frontier model for the most complex professional
// work and is what the bare "gpt-5.6" alias resolves to.
// 1.05M token context window, 128K max output tokens.
type GPT56Sol struct{ openAIReasoningOptions }

func (m *GPT56Sol) ModelName() string      { return "gpt-5.6-sol" }
func (m *GPT56Sol) Provider() ProviderType { return ProviderOpenAI }
func (m *GPT56Sol) SystemPrompt() string   { return m.systemPrompt }
func (m *GPT56Sol) isReasoning() bool      { return true }
func (m *GPT56Sol) thinkingDimensions() ThinkingDimension {
	return openAIThinkingDimensions(m.ModelName())
}
func (m *GPT56Sol) supportsExplicitCache() bool { return true }

func (m *GPT56Sol) WithMaxCompletionTokens(n int) *GPT56Sol { m.maxCompletionTokens = n; return m }
func (m *GPT56Sol) WithReasoningEffort(e string) *GPT56Sol  { m.setReasoningEffort(e); return m }
func (m *GPT56Sol) WithSystemPrompt(s string) *GPT56Sol     { m.systemPrompt = s; return m }

// NewGPT56Sol creates a new GPT-5.6 Sol model with default options
func NewGPT56Sol() *GPT56Sol {
	return &GPT56Sol{openAIReasoningOptions{maxCompletionTokens: 8192, thinking: openAISeededEffort(ThinkingEffortMedium)}}
}

// GPT56Terra represents the GPT-5.6 Terra reasoning model.
// Terra balances intelligence and cost and is the general-purpose choice in
// the GPT-5.6 family. 1.05M token context window, 128K max output tokens.
type GPT56Terra struct{ openAIReasoningOptions }

func (m *GPT56Terra) ModelName() string      { return "gpt-5.6-terra" }
func (m *GPT56Terra) Provider() ProviderType { return ProviderOpenAI }
func (m *GPT56Terra) SystemPrompt() string   { return m.systemPrompt }
func (m *GPT56Terra) isReasoning() bool      { return true }
func (m *GPT56Terra) thinkingDimensions() ThinkingDimension {
	return openAIThinkingDimensions(m.ModelName())
}
func (m *GPT56Terra) supportsExplicitCache() bool { return true }

func (m *GPT56Terra) WithMaxCompletionTokens(n int) *GPT56Terra { m.maxCompletionTokens = n; return m }
func (m *GPT56Terra) WithReasoningEffort(e string) *GPT56Terra  { m.setReasoningEffort(e); return m }
func (m *GPT56Terra) WithSystemPrompt(s string) *GPT56Terra     { m.systemPrompt = s; return m }

// NewGPT56Terra creates a new GPT-5.6 Terra model with default options
func NewGPT56Terra() *GPT56Terra {
	return &GPT56Terra{openAIReasoningOptions{maxCompletionTokens: 8192, thinking: openAISeededEffort(ThinkingEffortMedium)}}
}

// GPT56Luna represents the GPT-5.6 Luna reasoning model.
// Luna is optimized for cost-sensitive, high-volume workloads.
// 1.05M token context window, 128K max output tokens.
type GPT56Luna struct{ openAIReasoningOptions }

func (m *GPT56Luna) ModelName() string      { return "gpt-5.6-luna" }
func (m *GPT56Luna) Provider() ProviderType { return ProviderOpenAI }
func (m *GPT56Luna) SystemPrompt() string   { return m.systemPrompt }
func (m *GPT56Luna) isReasoning() bool      { return true }
func (m *GPT56Luna) thinkingDimensions() ThinkingDimension {
	return openAIThinkingDimensions(m.ModelName())
}
func (m *GPT56Luna) supportsExplicitCache() bool { return true }

func (m *GPT56Luna) WithMaxCompletionTokens(n int) *GPT56Luna { m.maxCompletionTokens = n; return m }
func (m *GPT56Luna) WithReasoningEffort(e string) *GPT56Luna  { m.setReasoningEffort(e); return m }
func (m *GPT56Luna) WithSystemPrompt(s string) *GPT56Luna     { m.systemPrompt = s; return m }

// NewGPT56Luna creates a new GPT-5.6 Luna model with default options
func NewGPT56Luna() *GPT56Luna {
	return &GPT56Luna{openAIReasoningOptions{maxCompletionTokens: 8192, thinking: openAISeededEffort(ThinkingEffortLow)}}
}

// O3Pro represents the O3-pro reasoning model
type O3Pro struct{ openAIReasoningOptions }

func (m *O3Pro) ModelName() string      { return "o3-pro" }
func (m *O3Pro) Provider() ProviderType { return ProviderOpenAI }
func (m *O3Pro) SystemPrompt() string   { return m.systemPrompt }
func (m *O3Pro) isReasoning() bool      { return true }
func (m *O3Pro) thinkingDimensions() ThinkingDimension {
	return openAIThinkingDimensions(m.ModelName())
}

func (m *O3Pro) WithMaxCompletionTokens(n int) *O3Pro { m.maxCompletionTokens = n; return m }
func (m *O3Pro) WithReasoningEffort(e string) *O3Pro  { m.setReasoningEffort(e); return m }
func (m *O3Pro) WithSystemPrompt(s string) *O3Pro     { m.systemPrompt = s; return m }

// NewO3Pro creates a new O3-pro model with default options
func NewO3Pro() *O3Pro {
	return &O3Pro{openAIReasoningOptions{maxCompletionTokens: 8192, thinking: openAISeededEffort(ThinkingEffortHigh)}}
}

// O1Preview represents the O1-preview reasoning model.
// Versions: o1-preview, o1-preview-2024-09-12
//
// Deprecated: removed from the OpenAI API (deprecation announced Apr 2025); requests return 404. Migrate to O3 or GPT5.
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
func (m *O1Preview) thinkingDimensions() ThinkingDimension {
	return openAIThinkingDimensions(m.ModelName())
}

func (m *O1Preview) WithVersion(v string) *O1Preview          { m.modelVersion = v; return m }
func (m *O1Preview) WithMaxCompletionTokens(n int) *O1Preview { m.maxCompletionTokens = n; return m }
func (m *O1Preview) WithReasoningEffort(e string) *O1Preview  { m.setReasoningEffort(e); return m }
func (m *O1Preview) WithSystemPrompt(s string) *O1Preview     { m.systemPrompt = s; return m }

// NewO1Preview creates a new O1-preview model with default options
//
// Deprecated: removed from the OpenAI API (deprecation announced Apr 2025); requests return 404. Migrate to O3 or GPT5.
func NewO1Preview() *O1Preview {
	return &O1Preview{openAIReasoningOptions{maxCompletionTokens: 8192, thinking: openAISeededEffort(ThinkingEffortMedium)}}
}

// ============================================================================
// GENERIC OPENAI MODELS
// ============================================================================

// OpenAIModel represents a generic standard (non-reasoning) OpenAI model.
// Use this for any chat-completions model this library has no named type for,
// so new model releases don't require a library update.
type OpenAIModel struct {
	modelID string
	openAIStandardOptions
}

func (m *OpenAIModel) ModelName() string      { return m.modelID }
func (m *OpenAIModel) Provider() ProviderType { return ProviderOpenAI }
func (m *OpenAIModel) SystemPrompt() string   { return m.systemPrompt }

func (m *OpenAIModel) WithMaxTokens(n int) *OpenAIModel       { m.maxTokens = n; return m }
func (m *OpenAIModel) WithTemperature(t float64) *OpenAIModel { m.temperature = t; return m }
func (m *OpenAIModel) WithTopP(p float64) *OpenAIModel        { m.topP = p; return m }
func (m *OpenAIModel) WithSystemPrompt(s string) *OpenAIModel { m.systemPrompt = s; return m }

// NewOpenAIModel creates a generic standard OpenAI model with the specified model ID
func NewOpenAIModel(modelID string) *OpenAIModel {
	return &OpenAIModel{modelID: modelID, openAIStandardOptions: openAIStandardOptions{maxTokens: 4096}}
}

// OpenAIReasoningModel represents a generic reasoning OpenAI model (o-series, GPT-5+).
// Use this for any reasoning model this library has no named type for,
// so new model releases don't require a library update.
type OpenAIReasoningModel struct {
	modelID       string
	explicitCache bool
	openAIReasoningOptions
}

func (m *OpenAIReasoningModel) ModelName() string      { return m.modelID }
func (m *OpenAIReasoningModel) Provider() ProviderType { return ProviderOpenAI }
func (m *OpenAIReasoningModel) SystemPrompt() string   { return m.systemPrompt }
func (m *OpenAIReasoningModel) isReasoning() bool      { return true }
func (m *OpenAIReasoningModel) thinkingDimensions() ThinkingDimension {
	return openAIThinkingDimensions(m.ModelName())
}
func (m *OpenAIReasoningModel) supportsExplicitCache() bool { return m.explicitCache }

// WithExplicitPromptCache declares that this model ID accepts per-content-part
// prompt_cache_breakpoint markers. lingo ships the marker on the GPT-5.6 family;
// set this for a newer model ID lingo has no named type for yet. Off by default:
// a breakpoint sent to a model that predates the field is an error OpenAI raises,
// not one lingo can catch. It does not enable caching by itself -- pair it with
// Cached.
func (m *OpenAIReasoningModel) WithExplicitPromptCache(ok bool) *OpenAIReasoningModel {
	m.explicitCache = ok
	return m
}

func (m *OpenAIReasoningModel) WithMaxCompletionTokens(n int) *OpenAIReasoningModel {
	m.maxCompletionTokens = n
	return m
}
func (m *OpenAIReasoningModel) WithReasoningEffort(e string) *OpenAIReasoningModel {
	m.setReasoningEffort(e)
	return m
}
func (m *OpenAIReasoningModel) WithSystemPrompt(s string) *OpenAIReasoningModel {
	m.systemPrompt = s
	return m
}

// NewOpenAIReasoningModel creates a generic reasoning OpenAI model with the specified model ID
func NewOpenAIReasoningModel(modelID string) *OpenAIReasoningModel {
	return &OpenAIReasoningModel{modelID: modelID, openAIReasoningOptions: openAIReasoningOptions{maxCompletionTokens: 8192}}
}

// ============================================================================
// OPENAI PROVIDER CLIENT
// ============================================================================

// openAIReasoningModel is an interface for reasoning models
type openAIReasoningModel interface {
	Model
	isReasoning() bool
}

// openAIExplicitCacheModel is implemented by the OpenAI models documented to
// accept a per-content-part prompt_cache_breakpoint. OpenAI gates the field to
// gpt-5.6 and later, so the marker is declared per model type rather than
// inferred, in the same idiom as isReasoning.
type openAIExplicitCacheModel interface {
	Model
	supportsExplicitCache() bool
}

// openAISupportsExplicitCache reports whether this model may carry breakpoints.
// The generic model types answer at runtime, so the method's value is what
// counts, not merely that the type implements the interface.
func openAISupportsExplicitCache(model Model) bool {
	m, ok := model.(openAIExplicitCacheModel)
	return ok && m.supportsExplicitCache()
}

// openAICachedTextPart builds a text content part that ends a reusable prefix.
// The breakpoint inherits the request's TTL, which OpenAI fixes at 30m, so
// CacheOptions.TTL has nothing to say here and is deliberately not consulted.
func openAICachedTextPart(text string) openai.ChatCompletionContentPartTextParam {
	return openai.ChatCompletionContentPartTextParam{
		Text:                  text,
		PromptCacheBreakpoint: openai.NewChatCompletionContentPartTextPromptCacheBreakpointParam(),
	}
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

	// The retry suppression goes on first so a caller's own options still get
	// the last word.
	opts := []option.RequestOption{
		option.WithMiddleware(suppressStainlessRetry),
		option.WithAPIKey(config.APIKey),
	}
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

	// Determine if this is a reasoning model. Only the presence of the marker
	// method matters here, unlike the cache gate below, whose value is settable.
	//
	// Metadata["is_reasoning_model"] means exactly that on this provider: it is a
	// property of the model TYPE, not of the request. It reports true for
	// NoThinking(NewGPT51()) and for NewGPT51().WithReasoningEffort("none"),
	// both of which send reasoning_effort "none" -- o3 is still an o3. That is
	// deliberate and predates the portable surface, so it stays; the
	// OpenAI-compatible client's flag of the same name is the other reading
	// ("this REQUEST reasons") and answers false for those bodies. Read
	// Metadata["reasoning_effort"] or the request itself if what you need is
	// whether one particular call reasoned.
	_, isReasoning := model.(openAIReasoningModel)

	// Breakpoints are opt-in and model-gated: with either half missing the
	// messages below stay in the plain string form they have always had.
	// cacheBreakpoint records the marker that was placed, which is not the same
	// as the one that was asked for -- a model with no system prompt gets none.
	co := modelCacheOptions(model)
	breakpoints := openAISupportsExplicitCache(model)
	var cacheBreakpoint bool

	// Build messages with optional system prompt
	var messages []openai.ChatCompletionMessageParamUnion

	if s := model.SystemPrompt(); s != "" {
		switch {
		case breakpoints && co.SystemPromptCached() && isReasoning:
			messages = append(messages, openai.DeveloperMessage([]openai.ChatCompletionContentPartTextParam{openAICachedTextPart(s)}))
			cacheBreakpoint = true
		case breakpoints && co.SystemPromptCached():
			messages = append(messages, openai.SystemMessage([]openai.ChatCompletionContentPartTextParam{openAICachedTextPart(s)}))
			cacheBreakpoint = true
		case isReasoning:
			// Reasoning models use "developer" role instead of "system"
			messages = append(messages, openai.DeveloperMessage(s))
		default:
			// Standard models use "system" role
			messages = append(messages, openai.SystemMessage(s))
		}
	}
	if breakpoints && co.PromptCached() {
		part := openAICachedTextPart(prompt)
		messages = append(messages, openai.UserMessage([]openai.ChatCompletionContentPartUnionParam{{OfText: &part}}))
		cacheBreakpoint = true
	} else {
		messages = append(messages, openai.UserMessage(prompt))
	}

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
	case *O1Mini:
		if m.maxCompletionTokens > 0 {
			params.MaxCompletionTokens = openai.Int(int64(m.maxCompletionTokens))
		}
	case *O1Pro:
		if m.maxCompletionTokens > 0 {
			params.MaxCompletionTokens = openai.Int(int64(m.maxCompletionTokens))
		}
	case *O3:
		if m.maxCompletionTokens > 0 {
			params.MaxCompletionTokens = openai.Int(int64(m.maxCompletionTokens))
		}
	case *O3Mini:
		if m.maxCompletionTokens > 0 {
			params.MaxCompletionTokens = openai.Int(int64(m.maxCompletionTokens))
		}
	case *O4Mini:
		if m.maxCompletionTokens > 0 {
			params.MaxCompletionTokens = openai.Int(int64(m.maxCompletionTokens))
		}
	case *GPT5:
		if m.maxCompletionTokens > 0 {
			params.MaxCompletionTokens = openai.Int(int64(m.maxCompletionTokens))
		}
	case *GPT5Mini:
		if m.maxCompletionTokens > 0 {
			params.MaxCompletionTokens = openai.Int(int64(m.maxCompletionTokens))
		}
	case *GPT5Nano:
		if m.maxCompletionTokens > 0 {
			params.MaxCompletionTokens = openai.Int(int64(m.maxCompletionTokens))
		}
	case *GPT5Pro:
		if m.maxCompletionTokens > 0 {
			params.MaxCompletionTokens = openai.Int(int64(m.maxCompletionTokens))
		}
	case *GPT51:
		if m.maxCompletionTokens > 0 {
			params.MaxCompletionTokens = openai.Int(int64(m.maxCompletionTokens))
		}
	case *GPT51Mini:
		if m.maxCompletionTokens > 0 {
			params.MaxCompletionTokens = openai.Int(int64(m.maxCompletionTokens))
		}
	case *GPT51Nano:
		if m.maxCompletionTokens > 0 {
			params.MaxCompletionTokens = openai.Int(int64(m.maxCompletionTokens))
		}
	case *GPT51Codex:
		if m.maxCompletionTokens > 0 {
			params.MaxCompletionTokens = openai.Int(int64(m.maxCompletionTokens))
		}
	case *GPT51CodexMini:
		if m.maxCompletionTokens > 0 {
			params.MaxCompletionTokens = openai.Int(int64(m.maxCompletionTokens))
		}
	case *GPT54Nano:
		if m.maxCompletionTokens > 0 {
			params.MaxCompletionTokens = openai.Int(int64(m.maxCompletionTokens))
		}
	case *GPT54Mini:
		if m.maxCompletionTokens > 0 {
			params.MaxCompletionTokens = openai.Int(int64(m.maxCompletionTokens))
		}
	case *GPT54:
		if m.maxCompletionTokens > 0 {
			params.MaxCompletionTokens = openai.Int(int64(m.maxCompletionTokens))
		}
	case *GPT54Pro:
		if m.maxCompletionTokens > 0 {
			params.MaxCompletionTokens = openai.Int(int64(m.maxCompletionTokens))
		}
	case *GPT55:
		if m.maxCompletionTokens > 0 {
			params.MaxCompletionTokens = openai.Int(int64(m.maxCompletionTokens))
		}
	case *GPT55Pro:
		if m.maxCompletionTokens > 0 {
			params.MaxCompletionTokens = openai.Int(int64(m.maxCompletionTokens))
		}
	case *GPT56Sol:
		if m.maxCompletionTokens > 0 {
			params.MaxCompletionTokens = openai.Int(int64(m.maxCompletionTokens))
		}
	case *GPT56Terra:
		if m.maxCompletionTokens > 0 {
			params.MaxCompletionTokens = openai.Int(int64(m.maxCompletionTokens))
		}
	case *GPT56Luna:
		if m.maxCompletionTokens > 0 {
			params.MaxCompletionTokens = openai.Int(int64(m.maxCompletionTokens))
		}
	case *O3Pro:
		if m.maxCompletionTokens > 0 {
			params.MaxCompletionTokens = openai.Int(int64(m.maxCompletionTokens))
		}
	case *O1Preview:
		if m.maxCompletionTokens > 0 {
			params.MaxCompletionTokens = openai.Int(int64(m.maxCompletionTokens))
		}

	// Generic models
	case *OpenAIModel:
		if m.maxTokens > 0 {
			params.MaxTokens = openai.Int(int64(m.maxTokens))
		}
		if m.temperature > 0 {
			params.Temperature = openai.Float(m.temperature)
		}
		if m.topP > 0 {
			params.TopP = openai.Float(m.topP)
		}
	case *OpenAIReasoningModel:
		if m.maxCompletionTokens > 0 {
			params.MaxCompletionTokens = openai.Int(int64(m.maxCompletionTokens))
		}
	}

	// Thinking is applied once from a plan built outside the switch, where the
	// effort used to be one hand-copied block per reasoning model type. It has
	// exactly one place to go on this API -- reasoning_effort -- so the plan's
	// budget, toggle and trace fields have nowhere to land here and are dropped
	// with a note rather than guessed at. Chat completions has no thinking
	// token budget, no boolean toggle and returns no trace; the reasoning
	// object that carries all three is Responses-API only.
	//
	// A model whose ThinkingOptions were never touched produces a zero plan and
	// leaves params exactly as built above, and a standard model carries no
	// ThinkingOptions at all.
	to := modelThinkingOptions(model)

	// A dimension a per-model setter pinned is always on the wire, whatever the
	// ladder says. The caller reached for an OpenAI-specific knob on an
	// OpenAI-specific type -- including the generic OpenAIReasoningModel, whose
	// whole job is to send what it was told, and including the effort every
	// named reasoning constructor has always seeded -- so lingo forwards it and
	// lets the API answer, exactly as it did before the portable surface existed.
	dims := ModelThinkingDimensions(model) | oaiPinnedThinking(to)
	plan := planThinking(to, dims, budgetRange{}, openAIEffortLadder(model.ModelName())...)
	if plan.effort != "" {
		params.ReasoningEffort = shared.ReasoningEffort(plan.effort)
	}

	// OpenAI caches prompt prefixes on its own, so most models have no
	// breakpoint to place; the GPT-5.6 family additionally accepts an explicit
	// one, handled with the messages above. The partition key is accepted by
	// every chat model and is meaningful even without Enable(). Models that
	// carry no key leave params untouched.
	if !co.Disabled() && co.Key() != "" {
		params.PromptCacheKey = openai.String(co.Key())
	}

	c.logger.Debug().
		Str("model", model.ModelName()).
		Bool("is_reasoning_model", isReasoning).
		Str("reasoning_effort", string(plan.effort)).
		Str("thinking_translation", plan.translation()).
		Bool("cache_breakpoint", cacheBreakpoint).
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

	// One choice is all there can be: this package never sets n, so the API
	// default of one applies, and usage is reported once for the whole response
	// rather than per choice -- so reading choices[0] cannot drop a token
	// anyone was billed for, only an alternative wording nobody asked for.
	choice := resp.Choices[0]

	// Build response
	response := &GenerationResponse{
		Text:         choice.Message.Content,
		Model:        resp.Model,
		FinishReason: string(choice.FinishReason),
		// OpenAI counts cached tokens inside prompt_tokens, so they are already
		// part of the prompt total: promptIncludesCache is true. Reasoning
		// tokens are the same shape on the completion side -- the API reports
		// them as a breakdown of completion_tokens -- so they are already inside
		// the completion total too.
		//
		// Reporting is unconditional: a reasoning model spends thinking tokens
		// whether or not the caller configured anything.
		Usage: TokenUsage{
			PromptTokens:     int(resp.Usage.PromptTokens),
			CompletionTokens: int(resp.Usage.CompletionTokens),
			TotalTokens:      int(resp.Usage.TotalTokens),
		}.withCache(
			int(resp.Usage.PromptTokensDetails.CachedTokens),
			int(resp.Usage.PromptTokensDetails.CacheWriteTokens),
			true,
		).withThinking(int(resp.Usage.CompletionTokensDetails.ReasoningTokens), true),
		Metadata: map[string]string{
			"provider":           "openai",
			"model":              resp.Model,
			"is_reasoning_model": fmt.Sprintf("%t", isReasoning),
		},
	}

	// The count now has a typed home in TokenUsage.ThinkingTokens; the metadata
	// key it used to live under is kept for one release so existing readers keep
	// working. Note the wire spelling differs from lingo's: this dialect calls
	// them reasoning tokens, lingo calls the concept thinking.
	//
	// Deprecated: read Usage.ThinkingTokens instead of Metadata["reasoning_tokens"].
	if resp.Usage.CompletionTokensDetails.ReasoningTokens > 0 {
		response.Metadata["reasoning_tokens"] = fmt.Sprintf("%d", resp.Usage.CompletionTokensDetails.ReasoningTokens)
	}

	// Whatever lingo had to translate or drop to fit the caller's request onto
	// this model's ladder, so a silent adaptation is never invisible. OpenAI
	// returns no reasoning trace on chat completions, so GenerationResponse
	// .Thinking stays empty here however much the model thought.
	if s := plan.translation(); s != "" {
		response.Metadata["thinking_translation"] = s
	}

	c.logger.Debug().
		Str("model", resp.Model).
		Bool("is_reasoning_model", isReasoning).
		Int64("prompt_tokens", resp.Usage.PromptTokens).
		Int64("completion_tokens", resp.Usage.CompletionTokens).
		Int64("total_tokens", resp.Usage.TotalTokens).
		Int("cache_read_tokens", response.Usage.CacheReadTokens).
		Int("cache_write_tokens", response.Usage.CacheWriteTokens).
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
