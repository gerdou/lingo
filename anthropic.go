package lingo

import (
	"context"
	"fmt"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/anthropics/anthropic-sdk-go/vertex"
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

// AnthropicConfig contains configuration for the Anthropic provider.
//
// By default requests go to the Anthropic API and APIKey is required. Set
// Vertex to reach Claude through Google Cloud instead, the counterpart to the
// Bedrock provider for AWS.
type AnthropicConfig struct {
	// APIKey is the Anthropic API key. Required unless Vertex is set.
	APIKey string
	// Timeout is the request timeout (default: 60s)
	Timeout time.Duration
	// RateLimiter is the optional rate limit configuration
	RateLimiter *RateLimitConfig
	// Vertex routes requests through Claude on Google Cloud Vertex AI,
	// authenticating with Google Cloud credentials instead of an API key.
	Vertex *AnthropicVertexConfig
	// HealthCheckModel is the model id Health generates against. It defaults
	// to a small current Claude on the Anthropic API, and has no default on
	// Vertex, where model ids are project- and region-specific.
	HealthCheckModel string
}

// AnthropicVertexConfig configures Claude on Google Cloud Vertex AI.
//
// Vertex publishes Claude under its own model IDs, which carry an @-suffixed
// version (for example "claude-opus-4-5@20251101") and differ from the ids the
// Anthropic API uses. The typed constructors in this package emit Anthropic
// API ids, so address Vertex models with NewAnthropicModel and the id from the
// Vertex Model Garden.
type AnthropicVertexConfig struct {
	// ProjectID is the Google Cloud project ID (required)
	ProjectID string
	// Region is the Vertex AI region, e.g. "us-east5" or "global" (required)
	Region string
	// Scopes overrides the default Google auth scopes. Rarely needed.
	Scopes []string
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
	cache        CacheOptions
}

// CacheOptions returns the model's prompt caching configuration. Every Anthropic
// model embeds anthropicOptions, so this one declaration makes them all satisfy
// CacheableModel.
func (o *anthropicOptions) CacheOptions() *CacheOptions { return &o.cache }

// thinkingDimensions answers for the Claude 3.5 and earlier models, whose API
// generation has no thinking field of any kind. The models that do carry
// thinking configuration override it per type, resolving their generation from
// their own model id.
func (o *anthropicOptions) thinkingDimensions() ThinkingDimension { return 0 }

// anthropicThinkingOptions contains options for models that support extended
// thinking. It is a sibling of anthropicOptions rather than an extension of the
// setters, and that split is load-bearing: Claude 3.5 and earlier embed the
// plain struct, so they structurally cannot satisfy ThinkingModel and no
// thinking knob can be handed to a model whose API has no thinking field.
type anthropicThinkingOptions struct {
	anthropicOptions
	thinking ThinkingOptions
}

// ThinkingOptions returns the model's thinking configuration. Every Claude that
// can think embeds anthropicThinkingOptions, so this one declaration makes them
// all satisfy ThinkingModel -- and, deliberately, only them.
//
// It is the single storage behind WithThinkingBudget, WithAdaptiveThinking,
// WithThinkingDisabled and WithEffort, so the portable surface and the
// per-model setters can never disagree about what the request will carry.
func (o *anthropicThinkingOptions) ThinkingOptions() *ThinkingOptions { return &o.thinking }

// The three toggle-ish setters used to own a field each, and the wire builder
// read them in a fixed order: disabled beat adaptive, and adaptive beat a fixed
// budget, whatever order the caller called them in. One storage would resolve
// the same contradictions by call order instead, so the shims keep the old
// precedence explicitly -- a weaker setter arriving after a stronger one is
// ignored, exactly as it was before it had a field of its own to be ignored in.
//
// The guard is on the pinned dimension, so it applies only among these setters.
// The portable surface has no such precedence: there, the last call wins.

// disabledByASetter reports whether WithThinkingDisabled already spoke.
func (o *anthropicThinkingOptions) disabledByASetter() bool {
	return o.thinking.isPinned(ThinkingCanToggle) && o.thinking.mode == ThinkingModeOff
}

// adaptiveByASetter reports whether WithAdaptiveThinking already spoke.
func (o *anthropicThinkingOptions) adaptiveByASetter() bool {
	return o.thinking.isPinned(ThinkingCanSetBudget) && o.thinking.budget == ThinkingBudgetDynamic
}

// setThinkingBudget backs the per-model WithThinkingBudget setters.
//
// A non-positive budget clears the request rather than enabling thinking with a
// nonsense ceiling, which is what the pre-existing `if thinkingBudget > 0` wire
// guard did. A positive one is pinned: the caller named a Claude-specific knob
// on a Claude-specific type, so it goes on the wire exactly as given, including
// the values below the API's 1024 floor that lingo has never validated.
func (o *anthropicThinkingOptions) setThinkingBudget(n int) {
	if o.disabledByASetter() || o.adaptiveByASetter() {
		return
	}
	if n > 0 {
		o.thinking.WithBudget(n).pin(ThinkingCanSetBudget)
		return
	}
	o.thinking.budget = 0
	if o.thinking.mode == ThinkingModeOn && o.thinking.effort == "" {
		o.thinking.mode = ThinkingModeDefault
	}
}

// setAdaptiveThinking backs the per-model WithAdaptiveThinking setters. Adaptive
// is "budget, but you decide", so it is the dynamic budget of the portable
// surface, pinned to the same dimension.
func (o *anthropicThinkingOptions) setAdaptiveThinking() {
	if o.disabledByASetter() {
		return
	}
	o.thinking.WithDynamicBudget().pin(ThinkingCanSetBudget)
}

// setThinkingDisabled backs the per-model WithThinkingDisabled setters. It is
// the strongest of the three and overrides whatever the others left behind.
func (o *anthropicThinkingOptions) setThinkingDisabled() {
	o.thinking.Disable().pin(ThinkingCanToggle)
}

// setEffort backs the per-model WithEffort setters.
//
// It writes the effort without touching the mode, which is where it parts
// company with the portable ThinkingOptions.WithEffort. On Anthropic
// output_config.effort is a separate field from the thinking config: it caps
// overall token spend rather than switching reasoning on, and Claude 5
// documents pairing a low effort with thinking switched off. Turning the mode
// on here would put a thinking config on the wire for callers who only ever
// asked about spend.
func (o *anthropicThinkingOptions) setEffort(e AnthropicEffort) {
	o.thinking.effort = e
	o.thinking.pin(ThinkingCanSetEffort)
}

// AnthropicEffort controls thinking depth and overall token spend for a request
// (the API's output_config.effort). Higher effort means deeper reasoning at
// greater cost and latency. The API default is EffortHigh.
//
// It is an alias for the provider-neutral ThinkingEffort, whose ladder is a
// superset of Anthropic's five levels, so WithEffort accepts both spellings and
// the two surfaces share one storage. Levels Anthropic does not accept
// (ThinkingEffortNone, ThinkingEffortMinimal) are clamped by the translator
// rather than forwarded.
type AnthropicEffort = ThinkingEffort

const (
	// EffortLow suits short, scoped tasks and latency-sensitive workloads.
	EffortLow AnthropicEffort = "low"
	// EffortMedium trades some intelligence for reduced token usage.
	EffortMedium AnthropicEffort = "medium"
	// EffortHigh is the API default and the recommended minimum for
	// intelligence-sensitive work.
	EffortHigh AnthropicEffort = "high"
	// EffortXHigh sits between high and max and is the best setting for most
	// coding and agentic use cases. Claude 4.7 and later only.
	EffortXHigh AnthropicEffort = "xhigh"
	// EffortMax is the deepest setting; use when correctness matters more than cost.
	EffortMax AnthropicEffort = "max"
)

// ============================================================================
// THINKING GENERATIONS
// ============================================================================
//
// Which thinking knobs a Claude honours is a property of its API generation,
// not of the provider, and Anthropic has changed the dialect four times. The
// era is what the portable surface is projected onto; the per-model setters
// stay exactly as they were and pin what they set, so nothing below can change
// what a request written before this feature existed puts on the wire.

// anthropicThinkingEra is the thinking dialect one Claude generation speaks.
type anthropicThinkingEra int

const (
	// anthropicThinkingEraNone is Claude 3.5 and earlier: no thinking field of
	// any kind, and a 400 for sending one.
	anthropicThinkingEraNone anthropicThinkingEra = iota
	// anthropicThinkingEraBudget is Claude 3.7 through 4.5: thinking is off
	// until a fixed thinking.budget_tokens turns it on. There is no adaptive
	// config and no output_config.effort.
	anthropicThinkingEraBudget
	// anthropicThinkingEraAdaptiveBudget is Claude 4.6: adaptive thinking
	// arrived alongside output_config.effort, and the fixed budget still works
	// but is deprecated. effort tops out at max -- xhigh came with 4.7.
	anthropicThinkingEraAdaptiveBudget
	// anthropicThinkingEraAdaptive is Claude 4.7 and 4.8: adaptive thinking
	// only, opt-in, and a fixed budget is rejected.
	anthropicThinkingEraAdaptive
	// anthropicThinkingEraAlwaysOn is Claude Fable 5: thinking is on
	// server-side and any thinking config is a 400. Only effort is settable.
	anthropicThinkingEraAlwaysOn
	// anthropicThinkingEraDefaultOn is the Claude 5 series: adaptive thinking
	// is on by default, so the thinking field is sent only to change it.
	anthropicThinkingEraDefaultOn
)

// anthropicThinkingEras maps model id prefixes to eras, first match wins, so
// longer prefixes are listed before the shorter ones they extend. Prefix
// matching rather than equality is what makes the dated ids, the undated
// aliases and Vertex's @-suffixed ids all resolve the same way.
var anthropicThinkingEras = []struct {
	prefix string
	era    anthropicThinkingEra
}{
	{"claude-3-7-", anthropicThinkingEraBudget},
	{"claude-3-", anthropicThinkingEraNone},
	{"claude-opus-4-6", anthropicThinkingEraAdaptiveBudget},
	{"claude-opus-4-7", anthropicThinkingEraAdaptive},
	{"claude-opus-4-8", anthropicThinkingEraAdaptive},
	{"claude-opus-4", anthropicThinkingEraBudget},
	{"claude-sonnet-4-6", anthropicThinkingEraAdaptiveBudget},
	{"claude-sonnet-4", anthropicThinkingEraBudget},
	{"claude-haiku-4", anthropicThinkingEraBudget},
	{"claude-fable-5", anthropicThinkingEraAlwaysOn},
}

// anthropicThinkingEraFor resolves a model id to its thinking generation.
//
// An id this library has not seen resolves to the current generation's dialect,
// which is the one every Claude since 4.6 speaks and the only one that has
// never had a knob withdrawn from it. That is deliberate: the alternative is to
// answer "nothing" for a Claude released after this build, which would make the
// generic AnthropicModel the one place the portable surface silently does
// nothing. A fixed budget is the only knob Anthropic has ever taken away, and
// the current dialect does not include it, so the optimistic answer cannot ask
// a new model for something a new model is likely to reject.
func anthropicThinkingEraFor(modelID string) anthropicThinkingEra {
	for _, e := range anthropicThinkingEras {
		if len(modelID) >= len(e.prefix) && modelID[:len(e.prefix)] == e.prefix {
			return e.era
		}
	}
	if len(modelID) >= 7 && modelID[:7] == "claude-" {
		return anthropicThinkingEraDefaultOn
	}
	return anthropicThinkingEraNone
}

// adaptive reports whether the era's API accepts a thinking config of type
// "adaptive", which is how "think, and decide for yourself how much" is spelled.
func (e anthropicThinkingEra) adaptive() bool {
	switch e {
	case anthropicThinkingEraAdaptiveBudget, anthropicThinkingEraAdaptive, anthropicThinkingEraDefaultOn:
		return true
	}
	return false
}

// dimensions reports which thinking knobs the era honours.
//
// Note what is missing from each: 3.7-4.5 has no effort because
// output_config predates it, 4.7 onwards has no budget because fixed budgets
// are rejected, and Fable 5 has neither a toggle nor a trace control because
// its thinking config cannot be sent at all.
func (e anthropicThinkingEra) dimensions() ThinkingDimension {
	const report = ThinkingCanReportTokens | ThinkingCanReportTrace
	switch e {
	case anthropicThinkingEraBudget:
		return ThinkingCanToggle | ThinkingCanSetBudget | ThinkingCanHideTrace | report
	case anthropicThinkingEraAdaptiveBudget:
		return ThinkingCanToggle | ThinkingCanSetEffort | ThinkingCanSetBudget |
			ThinkingCanHideTrace | report
	case anthropicThinkingEraAdaptive, anthropicThinkingEraDefaultOn:
		return ThinkingCanToggle | ThinkingCanSetEffort | ThinkingCanHideTrace | report
	case anthropicThinkingEraAlwaysOn:
		return ThinkingCanSetEffort | report
	default:
		return 0
	}
}

// efforts is the output_config.effort ladder the era accepts. Anthropic has no
// "none" and no "minimal": the portable surface's two shallowest rungs clamp up
// to low rather than being forwarded and rejected.
func (e anthropicThinkingEra) efforts() []ThinkingEffort {
	switch e {
	case anthropicThinkingEraAdaptiveBudget:
		// xhigh arrived with Opus 4.7 and is rejected by 4.6.
		return []ThinkingEffort{EffortLow, EffortMedium, EffortHigh, EffortMax}
	case anthropicThinkingEraAdaptive, anthropicThinkingEraAlwaysOn, anthropicThinkingEraDefaultOn:
		return []ThinkingEffort{EffortLow, EffortMedium, EffortHigh, EffortXHigh, EffortMax}
	default:
		return nil
	}
}

// anthropicMinThinkingBudget is the API's floor for thinking.budget_tokens.
// A budget must also stay below max_tokens, which is why the window's ceiling
// is per request rather than per model.
const anthropicMinThinkingBudget = 1024

// anthropicThinkingDimensions answers ModelThinkingDimensions for one Claude,
// resolved from the model id so a zero-value literal and the generic
// AnthropicModel both get the right answer without a constructor having stored
// anything.
func anthropicThinkingDimensions(modelID string) ThinkingDimension {
	return anthropicThinkingEraFor(modelID).dimensions()
}

// anthropicPinnedDimensions reports which dimensions a per-model setter pinned,
// nil-safe for the models that carry no thinking configuration.
func anthropicPinnedDimensions(o *ThinkingOptions) ThinkingDimension {
	if o == nil {
		return 0
	}
	return o.pinned
}

// ============================================================================
// STANDARD MODELS (Claude 3.5 series and earlier)
// ============================================================================

// Claude35Sonnet represents the Claude 3.5 Sonnet model.
// Versions: claude-3-5-sonnet-20241022, claude-3-5-sonnet-latest
//
// Deprecated: retired by Anthropic (Oct 28, 2025); the API returns 404. Migrate to ClaudeSonnet46.
type Claude35Sonnet struct{ anthropicOptions }

func (m *Claude35Sonnet) ModelName() string {
	if m.modelVersion != "" {
		return m.modelVersion
	}
	return "claude-3-5-sonnet-20241022"
}
func (m *Claude35Sonnet) Provider() ProviderType { return ProviderAnthropic }
func (m *Claude35Sonnet) SystemPrompt() string   { return m.systemPrompt }

func (m *Claude35Sonnet) WithVersion(v string) *Claude35Sonnet      { m.modelVersion = v; return m }
func (m *Claude35Sonnet) WithMaxTokens(n int) *Claude35Sonnet       { m.maxTokens = n; return m }
func (m *Claude35Sonnet) WithTemperature(t float64) *Claude35Sonnet { m.temperature = t; return m }
func (m *Claude35Sonnet) WithTopP(p float64) *Claude35Sonnet        { m.topP = p; return m }
func (m *Claude35Sonnet) WithTopK(k int) *Claude35Sonnet            { m.topK = k; return m }
func (m *Claude35Sonnet) WithSystemPrompt(s string) *Claude35Sonnet { m.systemPrompt = s; return m }

// NewClaude35Sonnet creates a new Claude 3.5 Sonnet model with default options
//
// Deprecated: retired by Anthropic (Oct 28, 2025); the API returns 404. Migrate to ClaudeSonnet46.
func NewClaude35Sonnet() *Claude35Sonnet {
	return &Claude35Sonnet{anthropicOptions{maxTokens: 4096, temperature: 1.0}}
}

// Claude35Haiku represents the Claude 3.5 Haiku model.
// Versions: claude-3-5-haiku-20241022, claude-3-5-haiku-latest
//
// Deprecated: retired by Anthropic (Feb 19, 2026); the API returns 404. Migrate to ClaudeHaiku45.
type Claude35Haiku struct{ anthropicOptions }

func (m *Claude35Haiku) ModelName() string {
	if m.modelVersion != "" {
		return m.modelVersion
	}
	return "claude-3-5-haiku-20241022"
}
func (m *Claude35Haiku) Provider() ProviderType { return ProviderAnthropic }
func (m *Claude35Haiku) SystemPrompt() string   { return m.systemPrompt }

func (m *Claude35Haiku) WithVersion(v string) *Claude35Haiku      { m.modelVersion = v; return m }
func (m *Claude35Haiku) WithMaxTokens(n int) *Claude35Haiku       { m.maxTokens = n; return m }
func (m *Claude35Haiku) WithTemperature(t float64) *Claude35Haiku { m.temperature = t; return m }
func (m *Claude35Haiku) WithTopP(p float64) *Claude35Haiku        { m.topP = p; return m }
func (m *Claude35Haiku) WithTopK(k int) *Claude35Haiku            { m.topK = k; return m }
func (m *Claude35Haiku) WithSystemPrompt(s string) *Claude35Haiku { m.systemPrompt = s; return m }

// NewClaude35Haiku creates a new Claude 3.5 Haiku model with default options
//
// Deprecated: retired by Anthropic (Feb 19, 2026); the API returns 404. Migrate to ClaudeHaiku45.
func NewClaude35Haiku() *Claude35Haiku {
	return &Claude35Haiku{anthropicOptions{maxTokens: 4096, temperature: 1.0}}
}

// Claude3Opus represents the Claude 3 Opus model.
// Versions: claude-3-opus-20240229, claude-3-opus-latest
//
// Deprecated: retired by Anthropic (Jan 5, 2026); the API returns 404. Migrate to ClaudeOpus48.
type Claude3Opus struct{ anthropicOptions }

func (m *Claude3Opus) ModelName() string {
	if m.modelVersion != "" {
		return m.modelVersion
	}
	return "claude-3-opus-20240229"
}
func (m *Claude3Opus) Provider() ProviderType { return ProviderAnthropic }
func (m *Claude3Opus) SystemPrompt() string   { return m.systemPrompt }

func (m *Claude3Opus) WithVersion(v string) *Claude3Opus      { m.modelVersion = v; return m }
func (m *Claude3Opus) WithMaxTokens(n int) *Claude3Opus       { m.maxTokens = n; return m }
func (m *Claude3Opus) WithTemperature(t float64) *Claude3Opus { m.temperature = t; return m }
func (m *Claude3Opus) WithTopP(p float64) *Claude3Opus        { m.topP = p; return m }
func (m *Claude3Opus) WithTopK(k int) *Claude3Opus            { m.topK = k; return m }
func (m *Claude3Opus) WithSystemPrompt(s string) *Claude3Opus { m.systemPrompt = s; return m }

// NewClaude3Opus creates a new Claude 3 Opus model with default options
//
// Deprecated: retired by Anthropic (Jan 5, 2026); the API returns 404. Migrate to ClaudeOpus48.
func NewClaude3Opus() *Claude3Opus {
	return &Claude3Opus{anthropicOptions{maxTokens: 4096, temperature: 1.0}}
}

// Claude3Haiku represents the Claude 3 Haiku model.
//
// Deprecated: retired by Anthropic (Apr 19, 2026); the API returns 404. Migrate to ClaudeHaiku45.
type Claude3Haiku struct{ anthropicOptions }

func (m *Claude3Haiku) ModelName() string      { return "claude-3-haiku-20240307" }
func (m *Claude3Haiku) Provider() ProviderType { return ProviderAnthropic }
func (m *Claude3Haiku) SystemPrompt() string   { return m.systemPrompt }

func (m *Claude3Haiku) WithMaxTokens(n int) *Claude3Haiku       { m.maxTokens = n; return m }
func (m *Claude3Haiku) WithTemperature(t float64) *Claude3Haiku { m.temperature = t; return m }
func (m *Claude3Haiku) WithTopP(p float64) *Claude3Haiku        { m.topP = p; return m }
func (m *Claude3Haiku) WithTopK(k int) *Claude3Haiku            { m.topK = k; return m }
func (m *Claude3Haiku) WithSystemPrompt(s string) *Claude3Haiku { m.systemPrompt = s; return m }

// NewClaude3Haiku creates a new Claude 3 Haiku model with default options
//
// Deprecated: retired by Anthropic (Apr 19, 2026); the API returns 404. Migrate to ClaudeHaiku45.
func NewClaude3Haiku() *Claude3Haiku {
	return &Claude3Haiku{anthropicOptions{maxTokens: 4096, temperature: 1.0}}
}

// Claude3Sonnet represents the Claude 3 Sonnet model.
//
// Deprecated: retired by Anthropic (Jul 21, 2025); the API returns 404. Migrate to ClaudeSonnet46.
type Claude3Sonnet struct{ anthropicOptions }

func (m *Claude3Sonnet) ModelName() string      { return "claude-3-sonnet-20240229" }
func (m *Claude3Sonnet) Provider() ProviderType { return ProviderAnthropic }
func (m *Claude3Sonnet) SystemPrompt() string   { return m.systemPrompt }

func (m *Claude3Sonnet) WithMaxTokens(n int) *Claude3Sonnet       { m.maxTokens = n; return m }
func (m *Claude3Sonnet) WithTemperature(t float64) *Claude3Sonnet { m.temperature = t; return m }
func (m *Claude3Sonnet) WithTopP(p float64) *Claude3Sonnet        { m.topP = p; return m }
func (m *Claude3Sonnet) WithTopK(k int) *Claude3Sonnet            { m.topK = k; return m }
func (m *Claude3Sonnet) WithSystemPrompt(s string) *Claude3Sonnet { m.systemPrompt = s; return m }

// NewClaude3Sonnet creates a new Claude 3 Sonnet model with default options
//
// Deprecated: retired by Anthropic (Jul 21, 2025); the API returns 404. Migrate to ClaudeSonnet46.
func NewClaude3Sonnet() *Claude3Sonnet {
	return &Claude3Sonnet{anthropicOptions{maxTokens: 4096, temperature: 1.0}}
}

// ============================================================================
// EXTENDED THINKING MODELS (Claude 3.7+, Claude 4+)
// ============================================================================

// Claude37Sonnet represents the Claude 3.7 Sonnet model (supports extended thinking).
// Versions: claude-3-7-sonnet-20250219, claude-3-7-sonnet-latest
//
// Deprecated: retired by Anthropic (Feb 19, 2026); the API returns 404. Migrate to ClaudeSonnet46.
type Claude37Sonnet struct{ anthropicThinkingOptions }

func (m *Claude37Sonnet) ModelName() string {
	if m.modelVersion != "" {
		return m.modelVersion
	}
	return "claude-3-7-sonnet-20250219"
}
func (m *Claude37Sonnet) Provider() ProviderType { return ProviderAnthropic }
func (m *Claude37Sonnet) SystemPrompt() string   { return m.systemPrompt }
func (m *Claude37Sonnet) thinkingDimensions() ThinkingDimension {
	return anthropicThinkingDimensions(m.ModelName())
}

func (m *Claude37Sonnet) WithVersion(v string) *Claude37Sonnet      { m.modelVersion = v; return m }
func (m *Claude37Sonnet) WithMaxTokens(n int) *Claude37Sonnet       { m.maxTokens = n; return m }
func (m *Claude37Sonnet) WithTemperature(t float64) *Claude37Sonnet { m.temperature = t; return m }
func (m *Claude37Sonnet) WithTopP(p float64) *Claude37Sonnet        { m.topP = p; return m }
func (m *Claude37Sonnet) WithTopK(k int) *Claude37Sonnet            { m.topK = k; return m }
func (m *Claude37Sonnet) WithSystemPrompt(s string) *Claude37Sonnet { m.systemPrompt = s; return m }
func (m *Claude37Sonnet) WithThinkingBudget(n int) *Claude37Sonnet  { m.setThinkingBudget(n); return m }

// NewClaude37Sonnet creates a new Claude 3.7 Sonnet model with default options
//
// Deprecated: retired by Anthropic (Feb 19, 2026); the API returns 404. Migrate to ClaudeSonnet46.
func NewClaude37Sonnet() *Claude37Sonnet {
	return &Claude37Sonnet{anthropicThinkingOptions{
		anthropicOptions: anthropicOptions{maxTokens: 8192, temperature: 1.0},
	}}
}

// ClaudeSonnet4 represents the Claude Sonnet 4 model (supports extended thinking).
//
// Deprecated: retired by Anthropic (Jun 15, 2026); the API returns 404. Migrate to ClaudeSonnet46.
type ClaudeSonnet4 struct{ anthropicThinkingOptions }

func (m *ClaudeSonnet4) ModelName() string      { return "claude-sonnet-4-20250514" }
func (m *ClaudeSonnet4) Provider() ProviderType { return ProviderAnthropic }
func (m *ClaudeSonnet4) SystemPrompt() string   { return m.systemPrompt }
func (m *ClaudeSonnet4) thinkingDimensions() ThinkingDimension {
	return anthropicThinkingDimensions(m.ModelName())
}

func (m *ClaudeSonnet4) WithMaxTokens(n int) *ClaudeSonnet4       { m.maxTokens = n; return m }
func (m *ClaudeSonnet4) WithTemperature(t float64) *ClaudeSonnet4 { m.temperature = t; return m }
func (m *ClaudeSonnet4) WithTopP(p float64) *ClaudeSonnet4        { m.topP = p; return m }
func (m *ClaudeSonnet4) WithTopK(k int) *ClaudeSonnet4            { m.topK = k; return m }
func (m *ClaudeSonnet4) WithSystemPrompt(s string) *ClaudeSonnet4 { m.systemPrompt = s; return m }
func (m *ClaudeSonnet4) WithThinkingBudget(n int) *ClaudeSonnet4  { m.setThinkingBudget(n); return m }

// NewClaudeSonnet4 creates a new Claude Sonnet 4 model with default options
//
// Deprecated: retired by Anthropic (Jun 15, 2026); the API returns 404. Migrate to ClaudeSonnet46.
func NewClaudeSonnet4() *ClaudeSonnet4 {
	return &ClaudeSonnet4{anthropicThinkingOptions{
		anthropicOptions: anthropicOptions{maxTokens: 8192, temperature: 1.0},
	}}
}

// ClaudeOpus4 represents the Claude Opus 4 model (supports extended thinking).
//
// Deprecated: retired by Anthropic (Jun 15, 2026); the API returns 404. Migrate to ClaudeOpus48.
type ClaudeOpus4 struct{ anthropicThinkingOptions }

func (m *ClaudeOpus4) ModelName() string      { return "claude-opus-4-20250514" }
func (m *ClaudeOpus4) Provider() ProviderType { return ProviderAnthropic }
func (m *ClaudeOpus4) SystemPrompt() string   { return m.systemPrompt }
func (m *ClaudeOpus4) thinkingDimensions() ThinkingDimension {
	return anthropicThinkingDimensions(m.ModelName())
}

func (m *ClaudeOpus4) WithMaxTokens(n int) *ClaudeOpus4       { m.maxTokens = n; return m }
func (m *ClaudeOpus4) WithTemperature(t float64) *ClaudeOpus4 { m.temperature = t; return m }
func (m *ClaudeOpus4) WithTopP(p float64) *ClaudeOpus4        { m.topP = p; return m }
func (m *ClaudeOpus4) WithTopK(k int) *ClaudeOpus4            { m.topK = k; return m }
func (m *ClaudeOpus4) WithSystemPrompt(s string) *ClaudeOpus4 { m.systemPrompt = s; return m }
func (m *ClaudeOpus4) WithThinkingBudget(n int) *ClaudeOpus4  { m.setThinkingBudget(n); return m }

// NewClaudeOpus4 creates a new Claude Opus 4 model with default options
//
// Deprecated: retired by Anthropic (Jun 15, 2026); the API returns 404. Migrate to ClaudeOpus48.
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
func (m *ClaudeSonnet45) thinkingDimensions() ThinkingDimension {
	return anthropicThinkingDimensions(m.ModelName())
}

func (m *ClaudeSonnet45) WithMaxTokens(n int) *ClaudeSonnet45       { m.maxTokens = n; return m }
func (m *ClaudeSonnet45) WithTemperature(t float64) *ClaudeSonnet45 { m.temperature = t; return m }
func (m *ClaudeSonnet45) WithTopP(p float64) *ClaudeSonnet45        { m.topP = p; return m }
func (m *ClaudeSonnet45) WithTopK(k int) *ClaudeSonnet45            { m.topK = k; return m }
func (m *ClaudeSonnet45) WithSystemPrompt(s string) *ClaudeSonnet45 { m.systemPrompt = s; return m }
func (m *ClaudeSonnet45) WithThinkingBudget(n int) *ClaudeSonnet45  { m.setThinkingBudget(n); return m }

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
func (m *ClaudeOpus45) thinkingDimensions() ThinkingDimension {
	return anthropicThinkingDimensions(m.ModelName())
}

func (m *ClaudeOpus45) WithMaxTokens(n int) *ClaudeOpus45       { m.maxTokens = n; return m }
func (m *ClaudeOpus45) WithTemperature(t float64) *ClaudeOpus45 { m.temperature = t; return m }
func (m *ClaudeOpus45) WithTopP(p float64) *ClaudeOpus45        { m.topP = p; return m }
func (m *ClaudeOpus45) WithTopK(k int) *ClaudeOpus45            { m.topK = k; return m }
func (m *ClaudeOpus45) WithSystemPrompt(s string) *ClaudeOpus45 { m.systemPrompt = s; return m }
func (m *ClaudeOpus45) WithThinkingBudget(n int) *ClaudeOpus45  { m.setThinkingBudget(n); return m }

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
func (m *ClaudeHaiku45) thinkingDimensions() ThinkingDimension {
	return anthropicThinkingDimensions(m.ModelName())
}

func (m *ClaudeHaiku45) WithMaxTokens(n int) *ClaudeHaiku45       { m.maxTokens = n; return m }
func (m *ClaudeHaiku45) WithTemperature(t float64) *ClaudeHaiku45 { m.temperature = t; return m }
func (m *ClaudeHaiku45) WithTopP(p float64) *ClaudeHaiku45        { m.topP = p; return m }
func (m *ClaudeHaiku45) WithTopK(k int) *ClaudeHaiku45            { m.topK = k; return m }
func (m *ClaudeHaiku45) WithSystemPrompt(s string) *ClaudeHaiku45 { m.systemPrompt = s; return m }
func (m *ClaudeHaiku45) WithThinkingBudget(n int) *ClaudeHaiku45  { m.setThinkingBudget(n); return m }

// NewClaudeHaiku45 creates a new Claude Haiku 4.5 model with default options
func NewClaudeHaiku45() *ClaudeHaiku45 {
	return &ClaudeHaiku45{anthropicThinkingOptions{
		anthropicOptions: anthropicOptions{maxTokens: 8192, temperature: 1.0},
	}}
}

// ClaudeOpus41 represents the Claude Opus 4.1 model (supports extended thinking).
// Versions: claude-opus-4-1-20250805, claude-opus-4-1
//
// Deprecated: retired by Anthropic (Aug 5, 2026); the API returns 404. Migrate to ClaudeOpus5.
type ClaudeOpus41 struct{ anthropicThinkingOptions }

func (m *ClaudeOpus41) ModelName() string {
	if m.modelVersion != "" {
		return m.modelVersion
	}
	return "claude-opus-4-1-20250805"
}
func (m *ClaudeOpus41) Provider() ProviderType { return ProviderAnthropic }
func (m *ClaudeOpus41) SystemPrompt() string   { return m.systemPrompt }
func (m *ClaudeOpus41) thinkingDimensions() ThinkingDimension {
	return anthropicThinkingDimensions(m.ModelName())
}

func (m *ClaudeOpus41) WithVersion(v string) *ClaudeOpus41      { m.modelVersion = v; return m }
func (m *ClaudeOpus41) WithMaxTokens(n int) *ClaudeOpus41       { m.maxTokens = n; return m }
func (m *ClaudeOpus41) WithTemperature(t float64) *ClaudeOpus41 { m.temperature = t; return m }
func (m *ClaudeOpus41) WithTopP(p float64) *ClaudeOpus41        { m.topP = p; return m }
func (m *ClaudeOpus41) WithTopK(k int) *ClaudeOpus41            { m.topK = k; return m }
func (m *ClaudeOpus41) WithSystemPrompt(s string) *ClaudeOpus41 { m.systemPrompt = s; return m }
func (m *ClaudeOpus41) WithThinkingBudget(n int) *ClaudeOpus41  { m.setThinkingBudget(n); return m }

// NewClaudeOpus41 creates a new Claude Opus 4.1 model with default options
//
// Deprecated: retired by Anthropic (Aug 5, 2026); the API returns 404. Migrate to ClaudeOpus5.
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
func (m *ClaudeOpus46) thinkingDimensions() ThinkingDimension {
	return anthropicThinkingDimensions(m.ModelName())
}

func (m *ClaudeOpus46) WithMaxTokens(n int) *ClaudeOpus46       { m.maxTokens = n; return m }
func (m *ClaudeOpus46) WithTemperature(t float64) *ClaudeOpus46 { m.temperature = t; return m }
func (m *ClaudeOpus46) WithTopP(p float64) *ClaudeOpus46        { m.topP = p; return m }
func (m *ClaudeOpus46) WithTopK(k int) *ClaudeOpus46            { m.topK = k; return m }
func (m *ClaudeOpus46) WithSystemPrompt(s string) *ClaudeOpus46 { m.systemPrompt = s; return m }

// WithAdaptiveThinking enables adaptive thinking: the model decides when and how
// much to think. This is the recommended thinking mode for Claude 4.6+.
func (m *ClaudeOpus46) WithAdaptiveThinking() *ClaudeOpus46 { m.setAdaptiveThinking(); return m }

// WithThinkingBudget sets a fixed thinking token budget.
//
// Deprecated: fixed budgets are deprecated on Claude 4.6 models; use WithAdaptiveThinking.
func (m *ClaudeOpus46) WithThinkingBudget(n int) *ClaudeOpus46 { m.setThinkingBudget(n); return m }

// WithEffort sets output_config.effort. Opus 4.6 supports low, medium, high
// and max; EffortXHigh arrived with Opus 4.7 and is rejected here.
func (m *ClaudeOpus46) WithEffort(e AnthropicEffort) *ClaudeOpus46 { m.setEffort(e); return m }

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
func (m *ClaudeSonnet46) thinkingDimensions() ThinkingDimension {
	return anthropicThinkingDimensions(m.ModelName())
}

func (m *ClaudeSonnet46) WithMaxTokens(n int) *ClaudeSonnet46       { m.maxTokens = n; return m }
func (m *ClaudeSonnet46) WithTemperature(t float64) *ClaudeSonnet46 { m.temperature = t; return m }
func (m *ClaudeSonnet46) WithTopP(p float64) *ClaudeSonnet46        { m.topP = p; return m }
func (m *ClaudeSonnet46) WithTopK(k int) *ClaudeSonnet46            { m.topK = k; return m }
func (m *ClaudeSonnet46) WithSystemPrompt(s string) *ClaudeSonnet46 { m.systemPrompt = s; return m }

// WithAdaptiveThinking enables adaptive thinking: the model decides when and how
// much to think. This is the recommended thinking mode for Claude 4.6+.
func (m *ClaudeSonnet46) WithAdaptiveThinking() *ClaudeSonnet46 { m.setAdaptiveThinking(); return m }

// WithThinkingBudget sets a fixed thinking token budget.
//
// Deprecated: fixed budgets are deprecated on Claude 4.6 models; use WithAdaptiveThinking.
func (m *ClaudeSonnet46) WithThinkingBudget(n int) *ClaudeSonnet46 { m.setThinkingBudget(n); return m }

// WithEffort sets output_config.effort. Sonnet 4.6 supports low, medium, high
// and max; EffortXHigh arrived with Opus 4.7 and is rejected here.
func (m *ClaudeSonnet46) WithEffort(e AnthropicEffort) *ClaudeSonnet46 { m.setEffort(e); return m }

// NewClaudeSonnet46 creates a new Claude Sonnet 4.6 model with default options
func NewClaudeSonnet46() *ClaudeSonnet46 {
	return &ClaudeSonnet46{anthropicThinkingOptions{
		anthropicOptions: anthropicOptions{maxTokens: 8192, temperature: 1.0},
	}}
}

// ClaudeOpus47 represents the Claude Opus 4.7 model (supports adaptive thinking).
// Opus 4.7 is adaptive-thinking only: sampling parameters (temperature/topP/topK)
// and fixed thinking budgets are rejected by the API with a 400 error, so this
// type does not expose setters for them. Use WithAdaptiveThinking to enable thinking.
type ClaudeOpus47 struct{ anthropicThinkingOptions }

func (m *ClaudeOpus47) ModelName() string      { return "claude-opus-4-7" }
func (m *ClaudeOpus47) Provider() ProviderType { return ProviderAnthropic }
func (m *ClaudeOpus47) SystemPrompt() string   { return m.systemPrompt }
func (m *ClaudeOpus47) thinkingDimensions() ThinkingDimension {
	return anthropicThinkingDimensions(m.ModelName())
}

func (m *ClaudeOpus47) WithMaxTokens(n int) *ClaudeOpus47       { m.maxTokens = n; return m }
func (m *ClaudeOpus47) WithSystemPrompt(s string) *ClaudeOpus47 { m.systemPrompt = s; return m }

// WithAdaptiveThinking enables adaptive thinking: the model decides when and how
// much to think. Without it, Opus 4.7 runs without thinking.
func (m *ClaudeOpus47) WithAdaptiveThinking() *ClaudeOpus47 { m.setAdaptiveThinking(); return m }

// WithEffort sets output_config.effort (low through max).
func (m *ClaudeOpus47) WithEffort(e AnthropicEffort) *ClaudeOpus47 { m.setEffort(e); return m }

// NewClaudeOpus47 creates a new Claude Opus 4.7 model with default options
func NewClaudeOpus47() *ClaudeOpus47 {
	return &ClaudeOpus47{anthropicThinkingOptions{
		anthropicOptions: anthropicOptions{maxTokens: 8192},
	}}
}

// ClaudeOpus48 represents the Claude Opus 4.8 model (supports adaptive thinking).
// This is the current recommended Opus-tier model for complex and long-horizon tasks.
// Opus 4.8 is adaptive-thinking only: sampling parameters (temperature/topP/topK)
// and fixed thinking budgets are rejected by the API with a 400 error, so this
// type does not expose setters for them. Use WithAdaptiveThinking to enable thinking.
type ClaudeOpus48 struct{ anthropicThinkingOptions }

func (m *ClaudeOpus48) ModelName() string      { return "claude-opus-4-8" }
func (m *ClaudeOpus48) Provider() ProviderType { return ProviderAnthropic }
func (m *ClaudeOpus48) SystemPrompt() string   { return m.systemPrompt }
func (m *ClaudeOpus48) thinkingDimensions() ThinkingDimension {
	return anthropicThinkingDimensions(m.ModelName())
}

func (m *ClaudeOpus48) WithMaxTokens(n int) *ClaudeOpus48       { m.maxTokens = n; return m }
func (m *ClaudeOpus48) WithSystemPrompt(s string) *ClaudeOpus48 { m.systemPrompt = s; return m }

// WithAdaptiveThinking enables adaptive thinking: the model decides when and how
// much to think. Without it, Opus 4.8 runs without thinking.
func (m *ClaudeOpus48) WithAdaptiveThinking() *ClaudeOpus48 { m.setAdaptiveThinking(); return m }

// WithEffort sets output_config.effort (low through max).
func (m *ClaudeOpus48) WithEffort(e AnthropicEffort) *ClaudeOpus48 { m.setEffort(e); return m }

// NewClaudeOpus48 creates a new Claude Opus 4.8 model with default options
func NewClaudeOpus48() *ClaudeOpus48 {
	return &ClaudeOpus48{anthropicThinkingOptions{
		anthropicOptions: anthropicOptions{maxTokens: 8192},
	}}
}

// ClaudeFable5 represents the Claude Fable 5 model.
// Fable 5 is Anthropic's most capable widely released model. Thinking is always on
// (adaptive) and cannot be configured, and sampling parameters (temperature/topP/topK)
// are rejected by the API with a 400 error, so this type does not expose setters for
// them. It uses the same tokenizer as Opus 4.8, so token counts are roughly unchanged
// when migrating from Opus 4.7/4.8 (older models tokenize differently — re-baseline).
// Note: Fable 5 requires 30-day data retention and may decline requests with
// stop_reason "refusal" (surfaced by this library as a refusal error).
type ClaudeFable5 struct{ anthropicThinkingOptions }

func (m *ClaudeFable5) ModelName() string      { return "claude-fable-5" }
func (m *ClaudeFable5) Provider() ProviderType { return ProviderAnthropic }
func (m *ClaudeFable5) SystemPrompt() string   { return m.systemPrompt }
func (m *ClaudeFable5) thinkingDimensions() ThinkingDimension {
	return anthropicThinkingDimensions(m.ModelName())
}

func (m *ClaudeFable5) WithMaxTokens(n int) *ClaudeFable5       { m.maxTokens = n; return m }
func (m *ClaudeFable5) WithSystemPrompt(s string) *ClaudeFable5 { m.systemPrompt = s; return m }

// WithEffort sets output_config.effort (low through max). This is the only
// depth control on Fable 5, since thinking itself cannot be configured.
func (m *ClaudeFable5) WithEffort(e AnthropicEffort) *ClaudeFable5 { m.setEffort(e); return m }

// NewClaudeFable5 creates a new Claude Fable 5 model with default options
func NewClaudeFable5() *ClaudeFable5 {
	return &ClaudeFable5{anthropicThinkingOptions{
		anthropicOptions: anthropicOptions{maxTokens: 8192},
	}}
}

// ClaudeOpus5 represents the Claude Opus 5 model.
// This is the current recommended model for complex agentic coding and
// long-horizon work, priced the same as Opus 4.8 with a 1M context window.
// Unlike Opus 4.7/4.8, thinking is ON by default (adaptive) — call
// WithThinkingDisabled to opt out. Sampling parameters (temperature/topP/topK)
// and fixed thinking budgets are rejected by the API with a 400 error, so this
// type does not expose setters for them; use WithEffort to control depth.
// Note: Opus 5 has elevated cybersecurity safeguards and may decline a request
// with stop_reason "refusal" (surfaced by this library as a refusal error).
type ClaudeOpus5 struct{ anthropicThinkingOptions }

func (m *ClaudeOpus5) ModelName() string      { return "claude-opus-5" }
func (m *ClaudeOpus5) Provider() ProviderType { return ProviderAnthropic }
func (m *ClaudeOpus5) SystemPrompt() string   { return m.systemPrompt }
func (m *ClaudeOpus5) thinkingDimensions() ThinkingDimension {
	return anthropicThinkingDimensions(m.ModelName())
}

func (m *ClaudeOpus5) WithMaxTokens(n int) *ClaudeOpus5       { m.maxTokens = n; return m }
func (m *ClaudeOpus5) WithSystemPrompt(s string) *ClaudeOpus5 { m.systemPrompt = s; return m }

// WithEffort sets output_config.effort. Opus 5 supports the full ladder
// (low through max). Start at EffortXHigh for coding and agentic work and
// EffortHigh elsewhere, then sweep down — low and medium perform unusually
// well on this model.
func (m *ClaudeOpus5) WithEffort(e AnthropicEffort) *ClaudeOpus5 { m.setEffort(e); return m }

// WithThinkingDisabled turns off the on-by-default adaptive thinking.
// The API accepts this only at EffortHigh or below; pairing it with
// EffortXHigh or EffortMax returns a 400 error. Prefer a lower effort level
// over disabling thinking: with thinking off, Opus 5 can emit tool calls as
// plain text and leak internal XML tags into the response.
func (m *ClaudeOpus5) WithThinkingDisabled() *ClaudeOpus5 { m.setThinkingDisabled(); return m }

// NewClaudeOpus5 creates a new Claude Opus 5 model with default options
func NewClaudeOpus5() *ClaudeOpus5 {
	return &ClaudeOpus5{anthropicThinkingOptions{
		anthropicOptions: anthropicOptions{maxTokens: 8192},
	}}
}

// ClaudeSonnet5 represents the Claude Sonnet 5 model.
// This is the current recommended model for the speed/intelligence balance,
// reaching near-Opus quality on coding and agentic work with a 1M context
// window. Thinking is adaptive and ON by default — call WithThinkingDisabled
// to opt out. Non-default sampling parameters (temperature/topP/topK) and
// fixed thinking budgets are rejected by the API with a 400 error, so this
// type does not expose setters for them; use WithEffort to control depth.
// Note: Sonnet 5 uses a new tokenizer that produces roughly 30% more tokens
// than Sonnet 4.6 for the same text — re-baseline token budgets when migrating.
type ClaudeSonnet5 struct{ anthropicThinkingOptions }

func (m *ClaudeSonnet5) ModelName() string      { return "claude-sonnet-5" }
func (m *ClaudeSonnet5) Provider() ProviderType { return ProviderAnthropic }
func (m *ClaudeSonnet5) SystemPrompt() string   { return m.systemPrompt }
func (m *ClaudeSonnet5) thinkingDimensions() ThinkingDimension {
	return anthropicThinkingDimensions(m.ModelName())
}

func (m *ClaudeSonnet5) WithMaxTokens(n int) *ClaudeSonnet5       { m.maxTokens = n; return m }
func (m *ClaudeSonnet5) WithSystemPrompt(s string) *ClaudeSonnet5 { m.systemPrompt = s; return m }

// WithEffort sets output_config.effort. Sonnet 5 supports the full ladder
// (low through max) and defaults to EffortHigh; raise to EffortXHigh for the
// hardest coding and agentic tasks.
func (m *ClaudeSonnet5) WithEffort(e AnthropicEffort) *ClaudeSonnet5 { m.setEffort(e); return m }

// WithThinkingDisabled turns off the on-by-default adaptive thinking.
// Prefer adaptive thinking at EffortLow instead: with thinking off, Sonnet 5
// is markedly less likely to reach for tools.
func (m *ClaudeSonnet5) WithThinkingDisabled() *ClaudeSonnet5 { m.setThinkingDisabled(); return m }

// NewClaudeSonnet5 creates a new Claude Sonnet 5 model with default options
func NewClaudeSonnet5() *ClaudeSonnet5 {
	return &ClaudeSonnet5{anthropicThinkingOptions{
		anthropicOptions: anthropicOptions{maxTokens: 8192},
	}}
}

// ============================================================================
// GENERIC ANTHROPIC MODEL
// ============================================================================

// AnthropicModel represents a generic Anthropic model.
// Use this for any Claude model this library has no named type for,
// so new model releases don't require a library update.
// The caller is responsible for only setting options the target model accepts
// (e.g. Claude 4.7+ rejects temperature/topP/topK and fixed thinking budgets).
type AnthropicModel struct {
	modelID string
	anthropicThinkingOptions
}

func (m *AnthropicModel) ModelName() string      { return m.modelID }
func (m *AnthropicModel) Provider() ProviderType { return ProviderAnthropic }
func (m *AnthropicModel) SystemPrompt() string   { return m.systemPrompt }
func (m *AnthropicModel) thinkingDimensions() ThinkingDimension {
	return anthropicThinkingDimensions(m.ModelName())
}

func (m *AnthropicModel) WithMaxTokens(n int) *AnthropicModel       { m.maxTokens = n; return m }
func (m *AnthropicModel) WithTemperature(t float64) *AnthropicModel { m.temperature = t; return m }
func (m *AnthropicModel) WithTopP(p float64) *AnthropicModel        { m.topP = p; return m }
func (m *AnthropicModel) WithTopK(k int) *AnthropicModel            { m.topK = k; return m }
func (m *AnthropicModel) WithSystemPrompt(s string) *AnthropicModel { m.systemPrompt = s; return m }

// WithAdaptiveThinking enables adaptive thinking (Claude 4.6+).
func (m *AnthropicModel) WithAdaptiveThinking() *AnthropicModel { m.setAdaptiveThinking(); return m }

// WithThinkingBudget sets a fixed thinking token budget (legacy models only).
func (m *AnthropicModel) WithThinkingBudget(n int) *AnthropicModel { m.setThinkingBudget(n); return m }

// WithEffort sets output_config.effort (Claude 4.6+).
func (m *AnthropicModel) WithEffort(e AnthropicEffort) *AnthropicModel { m.setEffort(e); return m }

// WithThinkingDisabled explicitly disables thinking (Claude 5 series, where
// adaptive thinking is on by default).
func (m *AnthropicModel) WithThinkingDisabled() *AnthropicModel { m.setThinkingDisabled(); return m }

// NewAnthropicModel creates a generic Anthropic model with the specified model ID
func NewAnthropicModel(modelID string) *AnthropicModel {
	return &AnthropicModel{modelID: modelID, anthropicThinkingOptions: anthropicThinkingOptions{
		anthropicOptions: anthropicOptions{maxTokens: 8192},
	}}
}

// ============================================================================
// ANTHROPIC PROVIDER CLIENT
// ============================================================================

// anthropicCacheControl builds the cache breakpoint marker for a requested TTL.
// CacheTTLDefault leaves the field unset, which the API reads as 5 minutes.
func anthropicCacheControl(ttl CacheTTL) anthropic.CacheControlEphemeralParam {
	cc := anthropic.NewCacheControlEphemeralParam()
	switch ttl {
	case CacheTTL5m:
		cc.TTL = anthropic.CacheControlEphemeralTTLTTL5m
	case CacheTTL1h:
		cc.TTL = anthropic.CacheControlEphemeralTTLTTL1h
	}
	return cc
}

// anthropicClient implements the Provider interface for Anthropic
type anthropicClient struct {
	client anthropic.Client
	// healthModel is generated against by Health. Vertex publishes Claude
	// under different ids, so it has no usable default there.
	healthModel string
	timeout     time.Duration
	logger      Logger
	rateLimiter *rateLimiter
}

// anthropicVertexAuth resolves Google application default credentials for
// Claude on Vertex AI and returns them as a request option.
//
// The SDK reports every credential failure by panicking and offers no variant
// that returns: vertex.WithGoogleAuth panics on an empty region and on a
// FindDefaultCredentials error (anthropic-sdk-go@v1.63.1/vertex/vertex.go:45-53),
// and the WithCredentials it delegates to panics again if the OAuth transport
// will not build (vertex.go:91-93). lingo.New promises an error when a provider
// fails to initialize, so absent ADC or a rotated GOOGLE_APPLICATION_CREDENTIALS
// file must not take the caller's whole process down at construction. The panic
// is recovered here and handed back as the documented error.
func anthropicVertexAuth(ctx context.Context, v *AnthropicVertexConfig) (opt option.RequestOption, err error) {
	defer func() {
		if r := recover(); r != nil {
			opt = nil
			err = fmt.Errorf("anthropic on Vertex AI credentials: %v", r)
		}
	}()
	return vertex.WithGoogleAuth(ctx, v.Region, v.ProjectID, v.Scopes...), nil
}

// newAnthropicClient creates a new Anthropic client using the official SDK
func newAnthropicClient(config *AnthropicConfig, logger Logger) (*anthropicClient, error) {
	var client anthropic.Client
	healthModel := config.HealthCheckModel

	if v := config.Vertex; v != nil {
		if v.ProjectID == "" || v.Region == "" {
			return nil, fmt.Errorf("anthropic on Vertex AI requires both ProjectID and Region")
		}
		// Resolves Google application default credentials at construction
		opt, err := anthropicVertexAuth(context.Background(), v)
		if err != nil {
			return nil, err
		}
		client = anthropic.NewClient(option.WithMiddleware(suppressStainlessRetry), opt)
	} else {
		if config.APIKey == "" {
			return nil, fmt.Errorf("anthropic API key is required")
		}
		client = anthropic.NewClient(option.WithMiddleware(suppressStainlessRetry), option.WithAPIKey(config.APIKey))
		if healthModel == "" {
			healthModel = "claude-haiku-4-5"
		}
	}

	timeout := config.Timeout
	if timeout == 0 {
		timeout = defaultTimeout()
	}

	return &anthropicClient{
		client:      client,
		healthModel: healthModel,
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

	// Apply the sampling options each model type accepts. Thinking used to be
	// applied here too, one hand-copied block per type; it now comes from a
	// single plan below, so what is left is only the per-type disagreement about
	// which sampling parameters the API will take.
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
	// Opus 4.7/4.8 reject sampling parameters, so only max_tokens is set here.
	case *ClaudeOpus47:
		if m.maxTokens > 0 {
			params.MaxTokens = int64(m.maxTokens)
		}
	case *ClaudeOpus48:
		if m.maxTokens > 0 {
			params.MaxTokens = int64(m.maxTokens)
		}
	// Fable 5 and the Claude 5 series reject sampling parameters too.
	case *ClaudeFable5:
		if m.maxTokens > 0 {
			params.MaxTokens = int64(m.maxTokens)
		}
	case *ClaudeOpus5:
		if m.maxTokens > 0 {
			params.MaxTokens = int64(m.maxTokens)
		}
	case *ClaudeSonnet5:
		if m.maxTokens > 0 {
			params.MaxTokens = int64(m.maxTokens)
		}

	// Generic model: sends whatever the caller set.
	case *AnthropicModel:
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
	}

	// Thinking is opt-in and, like caching, is applied once from a plan built
	// outside the switch. Unlike caching it cannot be applied uniformly: the
	// three mutually exclusive wire shapes and output_config.effort belong to
	// different Claude generations, so the plan is projected onto the model's
	// own era rather than onto the provider.
	//
	// A model whose ThinkingOptions were never touched produces a zero plan and
	// leaves params exactly as built above.
	to := modelThinkingOptions(model)
	era := anthropicThinkingEraFor(model.ModelName())
	dims := era.dimensions()

	// budget_tokens must be >= 1024 and strictly below max_tokens. A request
	// whose max_tokens leaves no room for a legal budget has no budget knob at
	// all, so an unpinned budget is translated or dropped rather than sent to be
	// rejected.
	br := budgetRange{min: anthropicMinThinkingBudget, max: int(params.MaxTokens) - 1}
	if br.max < br.min {
		br = budgetRange{}
		dims &^= ThinkingCanSetBudget
	}

	// A dimension a per-model setter pinned is always on the wire, whatever the
	// era says. The caller reached for a Claude-specific knob on a Claude-
	// specific type -- including the generic AnthropicModel, whose whole job is
	// to send what it was told -- so lingo forwards it and lets the API answer,
	// exactly as it did before the portable surface existed.
	pinned := anthropicPinnedDimensions(to)
	dims |= pinned
	plan := planThinking(to, dims, br, era.efforts()...)

	// Thinking is on for the eras that reason unless told not to.
	hasThinking := era == anthropicThinkingEraAlwaysOn || era == anthropicThinkingEraDefaultOn

	// display is sent only when the caller asked about the trace. Anthropic's
	// own default is left alone: the SDK documents it as summarized, so naming
	// it unprompted would be a wire change with nothing to gain.
	var adaptiveDisplay anthropic.ThinkingConfigAdaptiveDisplay
	var enabledDisplay anthropic.ThinkingConfigEnabledDisplay
	switch {
	case plan.showTrace:
		adaptiveDisplay = anthropic.ThinkingConfigAdaptiveDisplaySummarized
		enabledDisplay = anthropic.ThinkingConfigEnabledDisplaySummarized
	case plan.hideTrace:
		adaptiveDisplay = anthropic.ThinkingConfigAdaptiveDisplayOmitted
		enabledDisplay = anthropic.ThinkingConfigEnabledDisplayOmitted
	}

	adaptive := func() {
		params.Thinking = anthropic.ThinkingConfigParamUnion{
			OfAdaptive: &anthropic.ThinkingConfigAdaptiveParam{Display: adaptiveDisplay},
		}
		hasThinking = true
	}
	enabled := func(budget int) {
		cfg := anthropic.ThinkingConfigParamOfEnabled(int64(budget))
		cfg.OfEnabled.Display = enabledDisplay
		params.Thinking = cfg
		hasThinking = true
	}

	switch {
	case plan.disable:
		params.Thinking = anthropic.ThinkingConfigParamUnion{OfDisabled: &anthropic.ThinkingConfigDisabledParam{}}
		hasThinking = false
	case plan.dynamic && (era.adaptive() || pinned.Has(ThinkingCanSetBudget)):
		adaptive()
	case plan.budget > 0:
		enabled(plan.budget)
	case plan.dynamic, plan.enable, plan.showTrace, plan.hideTrace:
		// Thinking was asked for without a depth. On the generations that model
		// "you decide" that is the adaptive config; on 3.7 through 4.5, which
		// only ever spoke in fixed budgets, it has to become one.
		switch {
		case era.adaptive():
			adaptive()
		case dims.Has(ThinkingCanSetBudget):
			if n := ThinkingBudgetForEffort(ThinkingEffortHigh, br.min, br.max); n > 0 {
				plan.note("thinking enabled as a fixed budget of %d tokens: this model has no adaptive setting", n)
				enabled(n)
			} else {
				plan.note("thinking enabled but dropped: max_tokens leaves no room for a legal budget")
			}
		}
	}

	// output_config.effort is its own field, not part of the thinking config, so
	// a pinned effort survives decisions the plan made about thinking -- notably
	// the plan's rule that depth is meaningless once thinking is off, which does
	// not hold here: Claude 5 documents disabling thinking at EffortHigh or below.
	effort := plan.effort
	if pinned.Has(ThinkingCanSetEffort) {
		effort = to.Effort()
	}
	if effort != "" {
		params.OutputConfig.Effort = anthropic.OutputConfigEffort(effort)
	}

	// Prompt caching is opt-in and applies to every Claude type uniformly, so it
	// sits outside the switch. A model whose CacheOptions were never touched
	// leaves params exactly as built above.
	//
	// cacheBreakpoint records what the request actually carries, not what the
	// caller asked for: enabling caching on a model with no system prompt places
	// no marker at all, and the log has to say so.
	co := modelCacheOptions(model)
	var cacheBreakpoint bool
	if co.SystemPromptCached() && len(params.System) > 0 {
		params.System[len(params.System)-1].CacheControl = anthropicCacheControl(co.TTL())
		cacheBreakpoint = true
	}
	if co.PromptCached() && len(params.Messages) > 0 {
		blocks := params.Messages[len(params.Messages)-1].Content
		if len(blocks) > 0 {
			if text := blocks[len(blocks)-1].OfText; text != nil {
				text.CacheControl = anthropicCacheControl(co.TTL())
				cacheBreakpoint = true
			}
		}
	}

	c.logger.Debug().
		Str("model", model.ModelName()).
		Bool("has_thinking", hasThinking).
		Str("thinking_translation", plan.translation()).
		Bool("cache_breakpoint", cacheBreakpoint).
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

	// Safety classifiers (notably on Claude Fable 5) can decline a request with
	// HTTP 200 and stop_reason "refusal"; content is empty (pre-output) or partial
	// (mid-stream). Surface this explicitly instead of a generic empty-content error.
	if resp.StopReason == anthropic.StopReasonRefusal {
		return nil, fmt.Errorf("anthropic declined the request (stop_reason: refusal) for model %s; retry on a different model such as claude-opus-4-8", model.ModelName())
	}

	if len(resp.Content) == 0 {
		return nil, fmt.Errorf("no response content returned from Anthropic")
	}

	// Extract the answer and the reasoning trace. Both accumulate: a response is
	// a list of blocks, and once thinking is on it routinely arrives as several
	// of each. Reading only the last one silently truncated the answer.
	var text, thinkingText, thinkingSignature, redactedThinking string
	for _, block := range resp.Content {
		switch block.Type {
		case "text":
			text += block.Text
		case "thinking":
			thinkingText += block.Thinking
			// The signature authenticates the block for replay on a later turn.
			// A multi-block response has one per block and Metadata holds one
			// string, so this is the last of them; faithful replay needs a typed
			// content API, which lingo's single-turn Generate does not have.
			if block.Signature != "" {
				thinkingSignature = block.Signature
			}
		case "redacted_thinking":
			// The model reasoned but the trace came back encrypted. It is opaque
			// to the caller and useful only for replay, so it stays out of
			// Thinking and rides in metadata.
			redactedThinking += block.Data
		}
	}

	if text == "" {
		return nil, fmt.Errorf("no text content found in Anthropic response")
	}

	// Build response
	result := &GenerationResponse{
		Text:         text,
		Thinking:     thinkingText,
		Model:        string(resp.Model),
		FinishReason: string(resp.StopReason),
		// Anthropic reports cache tokens alongside InputTokens rather than inside
		// it, so withCache folds them back into the prompt total. Thinking tokens
		// are the other way round: the SDK documents thinking_tokens as always
		// <= output_tokens, so they are already inside the completion total.
		Usage: TokenUsage{
			PromptTokens:     int(resp.Usage.InputTokens),
			CompletionTokens: int(resp.Usage.OutputTokens),
			TotalTokens:      int(resp.Usage.InputTokens + resp.Usage.OutputTokens),
		}.withCache(int(resp.Usage.CacheReadInputTokens), int(resp.Usage.CacheCreationInputTokens), false).
			withThinking(int(resp.Usage.OutputTokensDetails.ThinkingTokens), true),
		Metadata: map[string]string{
			"provider": "anthropic",
			"model":    string(resp.Model),
		},
	}

	// The trace now has a typed home in GenerationResponse.Thinking; the
	// metadata key it used to live under is kept for one release so existing
	// readers keep working.
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

	// Whatever lingo had to translate or drop to fit the caller's request onto
	// this model's dialect, so a silent adaptation is never invisible.
	if s := plan.translation(); s != "" {
		result.Metadata["thinking_translation"] = s
	}

	// Anthropic bills cache writes differently per TTL and reports the split;
	// two counters on TokenUsage cannot carry it, so it rides in metadata.
	if n := resp.Usage.CacheCreation.Ephemeral5mInputTokens; n > 0 {
		result.Metadata["cache_write_tokens_5m"] = fmt.Sprintf("%d", n)
	}
	if n := resp.Usage.CacheCreation.Ephemeral1hInputTokens; n > 0 {
		result.Metadata["cache_write_tokens_1h"] = fmt.Sprintf("%d", n)
	}

	c.logger.Debug().
		Str("model", string(resp.Model)).
		Int64("input_tokens", resp.Usage.InputTokens).
		Int64("output_tokens", resp.Usage.OutputTokens).
		Int64("total_tokens", resp.Usage.InputTokens+resp.Usage.OutputTokens).
		Int64("cache_read_tokens", resp.Usage.CacheReadInputTokens).
		Int64("cache_write_tokens", resp.Usage.CacheCreationInputTokens).
		Int64("thinking_tokens", resp.Usage.OutputTokensDetails.ThinkingTokens).
		Bool("has_thinking", thinkingText != "").
		Msg("Anthropic generation completed")

	return result, nil
}

// Health checks the health of the Anthropic client
func (c *anthropicClient) Health(ctx context.Context) error {
	ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	if c.healthModel == "" {
		return fmt.Errorf("anthropic health check needs a model: set AnthropicConfig.HealthCheckModel to a model id available in this Vertex project and region")
	}

	params := anthropic.MessageNewParams{
		Model:     anthropic.Model(c.healthModel),
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
