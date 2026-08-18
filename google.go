package lingo

import (
	"context"
	"encoding/base64"
	"fmt"
	"strings"
	"time"

	"google.golang.org/genai"
)

func init() {
	RegisterProvider(ProviderGoogle, func(config ProviderConfig, logger Logger) (Provider, error) {
		cfg, ok := config.(*GoogleConfig)
		if !ok {
			return nil, fmt.Errorf("invalid config type for Google provider")
		}
		return newGoogleClient(cfg, logger)
	})
}

// ============================================================================
// GOOGLE PROVIDER CONFIG
// ============================================================================

// GoogleConfig contains configuration for the Google AI provider.
//
// By default requests go to the Gemini Developer API and APIKey is required.
// Set UseVertexAI to reach the same models through Vertex AI instead, which
// authenticates with Google Cloud credentials and bills through GCP.
type GoogleConfig struct {
	// APIKey is the Google AI API key. Required for the Gemini Developer API;
	// on Vertex AI it is only used for express mode.
	APIKey string
	// Timeout is the request timeout (default: 60s)
	Timeout time.Duration
	// RateLimiter is the optional rate limit configuration
	RateLimiter *RateLimitConfig
	// UseVertexAI routes requests through Vertex AI rather than the Gemini
	// Developer API. Set Project and Location alongside it, or supply APIKey
	// alone for Vertex express mode.
	UseVertexAI bool
	// Project is the Google Cloud project ID (Vertex AI only)
	Project string
	// Location is the Vertex AI region, e.g. "us-central1" or "global"
	// (Vertex AI only)
	Location string
}

// Implement ProviderConfig interface
func (c *GoogleConfig) providerType() ProviderType        { return ProviderGoogle }
func (c *GoogleConfig) apiKey() string                    { return c.APIKey }
func (c *GoogleConfig) timeout() time.Duration            { return c.Timeout }
func (c *GoogleConfig) rateLimitConfig() *RateLimitConfig { return c.RateLimiter }

// ============================================================================
// SHARED OPTIONS (embedded in model structs)
// ============================================================================

// googleOptions contains options for Google Gemini models
type googleOptions struct {
	modelVersion string // Optional: override model name with specific version (e.g., "latest", "preview")
	maxTokens    int
	temperature  float64
	topP         float64
	topK         int
	systemPrompt string
	cache        CacheOptions
	thinking     ThinkingOptions
}

// CacheOptions returns the model's prompt caching configuration. Every Gemini
// model embeds googleOptions, so this one declaration makes them all satisfy
// CacheableModel.
//
// Google's explicit cache is a resource with its own lifecycle rather than a
// per-request breakpoint, so on Gemini only the resource name reaches the wire
// and the TTL and breakpoint settings are inert. Name a resource with
// WithCachedContent, or create one through LLMGateway.CacheManager and pass it
// to WithPromptCache.
func (o *googleOptions) CacheOptions() *CacheOptions { return &o.cache }

// ThinkingOptions returns the model's thinking configuration. Every Gemini model
// embeds googleOptions, so this one declaration makes them all satisfy
// ThinkingModel.
//
// Carrying the configuration is not a promise that any of it reaches the wire:
// Gemini 1.5 and 2.0 answer 400 to a thinkingConfig of any shape, so those
// models store what they are told and send nothing. Which knobs a given Gemini
// honours is answered per model by thinkingDimensions, resolved from the model
// id rather than from anything a constructor stored.
//
// lingo has never had a Google-specific thinking setter, so unlike Anthropic and
// OpenAI there is no legacy vocabulary sharing this storage: everything here
// arrives through the portable surface and is adapted to the model's dialect.
func (o *googleOptions) ThinkingOptions() *ThinkingOptions { return &o.thinking }

// ============================================================================
// THINKING GENERATIONS
// ============================================================================
//
// Which thinking knobs a Gemini honours is a property of its generation, and the
// three generations lingo knows about have mutually incompatible ones:
//
//	1.5 / 2.0   no thinkingConfig at all; sending one is a 400
//	2.5         thinkingBudget in tokens, per-model window, -1 for dynamic
//	3.x         thinkingLevel on a four-rung ladder, and no budget
//
// The last two are not merely different spellings. Setting thinkingLevel and
// thinkingBudget in one request is a hard error, so lingo must never merge them,
// which it gets for free: a generation is granted exactly one of
// ThinkingCanSetBudget and ThinkingCanSetEffort, and the translator in
// thinking.go maps whichever vocabulary the caller used onto the one the model
// speaks.

// googleThinkingEra is the thinking dialect one Gemini generation speaks.
type googleThinkingEra int

const (
	// googleThinkingEraNone is Gemini 1.5 and 2.0, including the retired
	// gemini-2.0-flash-thinking-exp: the model may reason, but there is nothing
	// to ask for and a thinkingConfig is rejected.
	googleThinkingEraNone googleThinkingEra = iota
	// googleThinkingEraBudget is Gemini 2.5: depth is a ceiling in thinking
	// tokens, with -1 asking the model to decide and 0 switching thinking off
	// where the model allows it.
	googleThinkingEraBudget
	// googleThinkingEraLevel is Gemini 3.x: depth is an ordinal level, thinking
	// cannot be switched off, and a thinkingBudget is not accepted.
	googleThinkingEraLevel
)

// googleThinkingDialect is one generation's complete answer: the vocabulary it
// speaks, the window a thinking budget has to land in, and whether thinking can
// be switched off at all.
type googleThinkingDialect struct {
	era googleThinkingEra
	// budget is the legal thinkingBudget window, consulted only on the 2.5
	// generation. The floors differ per model and are not interchangeable:
	// Flash-Lite rejects a budget below 512, and Pro below 128.
	budget budgetRange
	// canDisable reports whether a thinkingBudget of 0 is accepted. Gemini 2.5
	// Pro reasons unconditionally and rejects 0; Flash and Flash-Lite accept it.
	canDisable bool
}

// googleThinkingDialects maps model id prefixes to dialects, first match wins,
// so longer prefixes are listed before the shorter ones they extend. Prefix
// matching rather than equality is what makes the dated preview ids and the
// undated aliases resolve the same way.
//
// Gemini 1.5, 2.0 and the 2.0 thinking preview are deliberately absent: they
// fall through to the no-knobs answer below, which is what their API models.
var googleThinkingDialects = []struct {
	prefix  string
	dialect googleThinkingDialect
}{
	{"gemini-2.5-pro", googleThinkingDialect{era: googleThinkingEraBudget, budget: budgetRange{min: 128, max: 32768}}},
	{"gemini-2.5-flash-lite", googleThinkingDialect{era: googleThinkingEraBudget, budget: budgetRange{min: 512, max: 24576}, canDisable: true}},
	{"gemini-2.5-flash", googleThinkingDialect{era: googleThinkingEraBudget, budget: budgetRange{min: 1, max: 24576}, canDisable: true}},
	{"gemini-3", googleThinkingDialect{era: googleThinkingEraLevel}},
}

// googleThinkingDialectFor resolves a model id to its thinking generation.
//
// An id this library has not seen resolves to no knobs at all, which is the
// opposite of what the Anthropic side does with an unrecognised Claude, and the
// difference is deliberate. There the generations agree on a vocabulary and
// differ only in which rungs they accept, so guessing the current dialect is
// safe. Here they do not: guessing wrong sends a thinkingBudget to a model that
// takes a thinkingLevel, or a thinkingConfig to a generation that rejects every
// shape of one. A silent no-op is the documented posture; a 400 is not.
func googleThinkingDialectFor(modelID string) googleThinkingDialect {
	id := googleModelID(modelID)
	for _, d := range googleThinkingDialects {
		if strings.HasPrefix(id, d.prefix) {
			return d.dialect
		}
	}
	return googleThinkingDialect{era: googleThinkingEraNone}
}

// googleModelID strips the resource path a Vertex AI model name carries, so
// "publishers/google/models/gemini-3-pro-preview" and "gemini-3-pro-preview"
// resolve to the same dialect. lingo passes the name through to the SDK
// untouched; only the dialect lookup uses the trimmed form.
func googleModelID(name string) string {
	if i := strings.LastIndex(name, "/"); i >= 0 {
		return name[i+1:]
	}
	return name
}

// dimensions reports which thinking knobs the dialect honours.
//
// Note what each one is missing: 2.5 has no effort ladder because thinkingLevel
// did not exist yet, 3.x has no budget because thinkingBudget was withdrawn, and
// neither 2.5 Pro nor any 3.x model can be switched off.
func (d googleThinkingDialect) dimensions() ThinkingDimension {
	const report = ThinkingCanReportTokens | ThinkingCanReportTrace
	switch d.era {
	case googleThinkingEraBudget:
		dims := ThinkingCanSetBudget | ThinkingCanHideTrace | report
		if d.canDisable {
			dims |= ThinkingCanToggle
		}
		return dims
	case googleThinkingEraLevel:
		return ThinkingCanSetEffort | ThinkingCanHideTrace | report
	default:
		return 0
	}
}

// efforts is the thinkingLevel ladder the dialect accepts. Google has no "none"
// and nothing above high, so the portable surface's shallowest rung clamps up to
// minimal and its two deepest clamp down to high.
func (d googleThinkingDialect) efforts() []ThinkingEffort {
	if d.era == googleThinkingEraLevel {
		return []ThinkingEffort{
			ThinkingEffortMinimal, ThinkingEffortLow, ThinkingEffortMedium, ThinkingEffortHigh,
		}
	}
	return nil
}

// googleThinkingDimensions answers ModelThinkingDimensions for one Gemini,
// resolved from the model id so a zero-value literal and the generic GoogleModel
// both get the right answer without a constructor having stored anything.
func googleThinkingDimensions(modelID string) ThinkingDimension {
	return googleThinkingDialectFor(modelID).dimensions()
}

// googleThinkingLevel maps a planned effort onto Gemini's thinkingLevel. It
// returns "" for a level Gemini does not model, so the field is dropped rather
// than sent as THINKING_LEVEL_UNSPECIFIED, which the API treats as a value in
// its own right.
func googleThinkingLevel(e ThinkingEffort) genai.ThinkingLevel {
	switch e {
	case ThinkingEffortMinimal:
		return genai.ThinkingLevelMinimal
	case ThinkingEffortLow:
		return genai.ThinkingLevelLow
	case ThinkingEffortMedium:
		return genai.ThinkingLevelMedium
	case ThinkingEffortHigh:
		return genai.ThinkingLevelHigh
	default:
		return ""
	}
}

// googleThinkingConfig projects a model's thinking options onto the one config
// object Gemini takes, and returns the plan alongside it so the caller can
// report what had to be translated.
//
// A model whose ThinkingOptions were never touched yields a nil config, and a
// nil config leaves the request byte-for-byte what it was before this feature
// existed. So does asking only for the trace to be withheld: includeThoughts
// already defaults to false, so there is nothing to send.
func googleThinkingConfig(model Model) (*genai.ThinkingConfig, thinkingPlan) {
	dialect := googleThinkingDialectFor(model.ModelName())
	plan := planThinking(modelThinkingOptions(model), dialect.dimensions(), dialect.budget, dialect.efforts()...)

	var tc *genai.ThinkingConfig
	config := func() *genai.ThinkingConfig {
		if tc == nil {
			tc = &genai.ThinkingConfig{}
		}
		return tc
	}

	switch {
	case plan.disable:
		config().ThinkingBudget = genai.Ptr(int32(0))
	case plan.dynamic:
		config().ThinkingBudget = genai.Ptr(int32(ThinkingBudgetDynamic))
	case plan.budget > 0:
		config().ThinkingBudget = genai.Ptr(int32(plan.budget))
	case plan.enable:
		// Thinking was asked for without a depth. Only the 2.5 generation has a
		// toggle to have set this, and there "on, you decide" is the dynamic
		// budget -- which is what Flash already defaults to and what Flash-Lite,
		// whose default is off, needs to be told.
		config().ThinkingBudget = genai.Ptr(int32(ThinkingBudgetDynamic))
	}

	// Only one generation is ever granted ThinkingCanSetEffort, and it is not the
	// one granted ThinkingCanSetBudget, so a level and a budget can never both be
	// planned -- which is what keeps lingo away from the combination Gemini
	// rejects outright.
	if plan.effort != "" {
		if level := googleThinkingLevel(plan.effort); level != "" {
			config().ThinkingLevel = level
		}
	}
	if plan.showTrace {
		config().IncludeThoughts = true
	}
	return tc, plan
}

// googleSplitParts separates a candidate's parts into the answer, the reasoning
// trace and the last thought signature it carried.
//
// Splitting on Part.Thought is what stops a thought summary being returned to
// the caller as the answer once includeThoughts is on: thought parts carry their
// text in the same field the answer uses, so a loop that tests only Text
// concatenates the model's reasoning into GenerationResponse.Text. The SDK's own
// Text helper skips them for the same reason.
//
// The signature is opaque and exists to be replayed on a later turn, so it is
// base64-encoded into metadata rather than decoded. A response can carry several
// and metadata holds one string, so this is the last of them; faithful replay
// needs a typed content API, which lingo's single-turn Generate does not have.
func googleSplitParts(parts []*genai.Part) (text, thinking, signature string) {
	for _, part := range parts {
		if part == nil {
			continue
		}
		if len(part.ThoughtSignature) > 0 {
			signature = base64.StdEncoding.EncodeToString(part.ThoughtSignature)
		}
		if part.Text == "" {
			continue
		}
		if part.Thought {
			thinking += part.Text
			continue
		}
		text += part.Text
	}
	return text, thinking, signature
}

// ============================================================================
// GEMINI MODELS
// ============================================================================

// Gemini25Pro represents the Gemini 2.5 Pro model
// Versions: gemini-2.5-pro, gemini-2.5-pro-preview-05-06
type Gemini25Pro struct{ googleOptions }

func (m *Gemini25Pro) ModelName() string {
	if m.modelVersion != "" {
		return m.modelVersion
	}
	return "gemini-2.5-pro"
}
func (m *Gemini25Pro) Provider() ProviderType { return ProviderGoogle }
func (m *Gemini25Pro) SystemPrompt() string   { return m.systemPrompt }
func (m *Gemini25Pro) thinkingDimensions() ThinkingDimension {
	return googleThinkingDimensions(m.ModelName())
}

func (m *Gemini25Pro) WithVersion(v string) *Gemini25Pro      { m.modelVersion = v; return m }
func (m *Gemini25Pro) WithMaxTokens(n int) *Gemini25Pro       { m.maxTokens = n; return m }
func (m *Gemini25Pro) WithTemperature(t float64) *Gemini25Pro { m.temperature = t; return m }
func (m *Gemini25Pro) WithTopP(p float64) *Gemini25Pro        { m.topP = p; return m }
func (m *Gemini25Pro) WithTopK(k int) *Gemini25Pro            { m.topK = k; return m }
func (m *Gemini25Pro) WithSystemPrompt(s string) *Gemini25Pro { m.systemPrompt = s; return m }

// NewGemini25Pro creates a new Gemini 2.5 Pro model with default options
func NewGemini25Pro() *Gemini25Pro {
	return &Gemini25Pro{googleOptions{maxTokens: 8192, temperature: 1.0}}
}

// Gemini25Flash represents the Gemini 2.5 Flash model
// Versions: gemini-2.5-flash, gemini-2.5-flash-preview-05-20
type Gemini25Flash struct{ googleOptions }

func (m *Gemini25Flash) ModelName() string {
	if m.modelVersion != "" {
		return m.modelVersion
	}
	return "gemini-2.5-flash"
}
func (m *Gemini25Flash) Provider() ProviderType { return ProviderGoogle }
func (m *Gemini25Flash) SystemPrompt() string   { return m.systemPrompt }
func (m *Gemini25Flash) thinkingDimensions() ThinkingDimension {
	return googleThinkingDimensions(m.ModelName())
}

func (m *Gemini25Flash) WithVersion(v string) *Gemini25Flash      { m.modelVersion = v; return m }
func (m *Gemini25Flash) WithMaxTokens(n int) *Gemini25Flash       { m.maxTokens = n; return m }
func (m *Gemini25Flash) WithTemperature(t float64) *Gemini25Flash { m.temperature = t; return m }
func (m *Gemini25Flash) WithTopP(p float64) *Gemini25Flash        { m.topP = p; return m }
func (m *Gemini25Flash) WithTopK(k int) *Gemini25Flash            { m.topK = k; return m }
func (m *Gemini25Flash) WithSystemPrompt(s string) *Gemini25Flash { m.systemPrompt = s; return m }

// NewGemini25Flash creates a new Gemini 2.5 Flash model with default options
func NewGemini25Flash() *Gemini25Flash {
	return &Gemini25Flash{googleOptions{maxTokens: 8192, temperature: 1.0}}
}

// Gemini25FlashLite represents the Gemini 2.5 Flash Lite model (fast, low-cost)
type Gemini25FlashLite struct{ googleOptions }

func (m *Gemini25FlashLite) ModelName() string      { return "gemini-2.5-flash-lite" }
func (m *Gemini25FlashLite) Provider() ProviderType { return ProviderGoogle }
func (m *Gemini25FlashLite) SystemPrompt() string   { return m.systemPrompt }
func (m *Gemini25FlashLite) thinkingDimensions() ThinkingDimension {
	return googleThinkingDimensions(m.ModelName())
}

func (m *Gemini25FlashLite) WithMaxTokens(n int) *Gemini25FlashLite { m.maxTokens = n; return m }
func (m *Gemini25FlashLite) WithTemperature(t float64) *Gemini25FlashLite {
	m.temperature = t
	return m
}
func (m *Gemini25FlashLite) WithTopP(p float64) *Gemini25FlashLite { m.topP = p; return m }
func (m *Gemini25FlashLite) WithTopK(k int) *Gemini25FlashLite     { m.topK = k; return m }
func (m *Gemini25FlashLite) WithSystemPrompt(s string) *Gemini25FlashLite {
	m.systemPrompt = s
	return m
}

// NewGemini25FlashLite creates a new Gemini 2.5 Flash Lite model with default options
func NewGemini25FlashLite() *Gemini25FlashLite {
	return &Gemini25FlashLite{googleOptions{maxTokens: 8192, temperature: 1.0}}
}

// Gemini20Flash represents the Gemini 2.0 Flash model.
//
// Deprecated: shut down by Google on June 1, 2026; the API returns an error. Migrate to Gemini25Flash.
type Gemini20Flash struct{ googleOptions }

func (m *Gemini20Flash) ModelName() string      { return "gemini-2.0-flash" }
func (m *Gemini20Flash) Provider() ProviderType { return ProviderGoogle }
func (m *Gemini20Flash) SystemPrompt() string   { return m.systemPrompt }
func (m *Gemini20Flash) thinkingDimensions() ThinkingDimension {
	return googleThinkingDimensions(m.ModelName())
}

func (m *Gemini20Flash) WithMaxTokens(n int) *Gemini20Flash       { m.maxTokens = n; return m }
func (m *Gemini20Flash) WithTemperature(t float64) *Gemini20Flash { m.temperature = t; return m }
func (m *Gemini20Flash) WithTopP(p float64) *Gemini20Flash        { m.topP = p; return m }
func (m *Gemini20Flash) WithTopK(k int) *Gemini20Flash            { m.topK = k; return m }
func (m *Gemini20Flash) WithSystemPrompt(s string) *Gemini20Flash { m.systemPrompt = s; return m }

// NewGemini20Flash creates a new Gemini 2.0 Flash model with default options
//
// Deprecated: shut down by Google on June 1, 2026; the API returns an error. Migrate to Gemini25Flash.
func NewGemini20Flash() *Gemini20Flash {
	return &Gemini20Flash{googleOptions{maxTokens: 8192, temperature: 1.0}}
}

// Gemini20FlashLite represents the Gemini 2.0 Flash Lite model.
//
// Deprecated: shut down by Google on June 1, 2026; the API returns an error. Migrate to Gemini25FlashLite.
type Gemini20FlashLite struct{ googleOptions }

func (m *Gemini20FlashLite) ModelName() string      { return "gemini-2.0-flash-lite" }
func (m *Gemini20FlashLite) Provider() ProviderType { return ProviderGoogle }
func (m *Gemini20FlashLite) SystemPrompt() string   { return m.systemPrompt }
func (m *Gemini20FlashLite) thinkingDimensions() ThinkingDimension {
	return googleThinkingDimensions(m.ModelName())
}

func (m *Gemini20FlashLite) WithMaxTokens(n int) *Gemini20FlashLite { m.maxTokens = n; return m }
func (m *Gemini20FlashLite) WithTemperature(t float64) *Gemini20FlashLite {
	m.temperature = t
	return m
}
func (m *Gemini20FlashLite) WithTopP(p float64) *Gemini20FlashLite { m.topP = p; return m }
func (m *Gemini20FlashLite) WithTopK(k int) *Gemini20FlashLite     { m.topK = k; return m }
func (m *Gemini20FlashLite) WithSystemPrompt(s string) *Gemini20FlashLite {
	m.systemPrompt = s
	return m
}

// NewGemini20FlashLite creates a new Gemini 2.0 Flash Lite model with default options
//
// Deprecated: shut down by Google on June 1, 2026; the API returns an error. Migrate to Gemini25FlashLite.
func NewGemini20FlashLite() *Gemini20FlashLite {
	return &Gemini20FlashLite{googleOptions{maxTokens: 4096, temperature: 1.0}}
}

// Gemini15Pro represents the Gemini 1.5 Pro model.
// Versions: gemini-1.5-pro, gemini-1.5-pro-latest
//
// Deprecated: retired by Google; the API returns an error. Migrate to Gemini25Pro or Gemini31Pro.
type Gemini15Pro struct{ googleOptions }

func (m *Gemini15Pro) ModelName() string {
	if m.modelVersion != "" {
		return m.modelVersion
	}
	return "gemini-1.5-pro"
}
func (m *Gemini15Pro) Provider() ProviderType { return ProviderGoogle }
func (m *Gemini15Pro) SystemPrompt() string   { return m.systemPrompt }
func (m *Gemini15Pro) thinkingDimensions() ThinkingDimension {
	return googleThinkingDimensions(m.ModelName())
}

func (m *Gemini15Pro) WithVersion(v string) *Gemini15Pro      { m.modelVersion = v; return m }
func (m *Gemini15Pro) WithMaxTokens(n int) *Gemini15Pro       { m.maxTokens = n; return m }
func (m *Gemini15Pro) WithTemperature(t float64) *Gemini15Pro { m.temperature = t; return m }
func (m *Gemini15Pro) WithTopP(p float64) *Gemini15Pro        { m.topP = p; return m }
func (m *Gemini15Pro) WithTopK(k int) *Gemini15Pro            { m.topK = k; return m }
func (m *Gemini15Pro) WithSystemPrompt(s string) *Gemini15Pro { m.systemPrompt = s; return m }

// NewGemini15Pro creates a new Gemini 1.5 Pro model with default options
//
// Deprecated: retired by Google; the API returns an error. Migrate to Gemini25Pro or Gemini31Pro.
func NewGemini15Pro() *Gemini15Pro {
	return &Gemini15Pro{googleOptions{maxTokens: 8192, temperature: 1.0}}
}

// Gemini15Flash represents the Gemini 1.5 Flash model.
// Versions: gemini-1.5-flash, gemini-1.5-flash-latest
//
// Deprecated: retired by Google; the API returns an error. Migrate to Gemini35Flash.
type Gemini15Flash struct{ googleOptions }

func (m *Gemini15Flash) ModelName() string {
	if m.modelVersion != "" {
		return m.modelVersion
	}
	return "gemini-1.5-flash"
}
func (m *Gemini15Flash) Provider() ProviderType { return ProviderGoogle }
func (m *Gemini15Flash) SystemPrompt() string   { return m.systemPrompt }
func (m *Gemini15Flash) thinkingDimensions() ThinkingDimension {
	return googleThinkingDimensions(m.ModelName())
}

func (m *Gemini15Flash) WithVersion(v string) *Gemini15Flash      { m.modelVersion = v; return m }
func (m *Gemini15Flash) WithMaxTokens(n int) *Gemini15Flash       { m.maxTokens = n; return m }
func (m *Gemini15Flash) WithTemperature(t float64) *Gemini15Flash { m.temperature = t; return m }
func (m *Gemini15Flash) WithTopP(p float64) *Gemini15Flash        { m.topP = p; return m }
func (m *Gemini15Flash) WithTopK(k int) *Gemini15Flash            { m.topK = k; return m }
func (m *Gemini15Flash) WithSystemPrompt(s string) *Gemini15Flash { m.systemPrompt = s; return m }

// NewGemini15Flash creates a new Gemini 1.5 Flash model with default options
//
// Deprecated: retired by Google; the API returns an error. Migrate to Gemini35Flash.
func NewGemini15Flash() *Gemini15Flash {
	return &Gemini15Flash{googleOptions{maxTokens: 8192, temperature: 1.0}}
}

// Gemini15Flash8b represents the Gemini 1.5 Flash 8B model.
//
// Deprecated: retired by Google; the API returns an error. Migrate to Gemini31FlashLite.
type Gemini15Flash8b struct{ googleOptions }

func (m *Gemini15Flash8b) ModelName() string      { return "gemini-1.5-flash-8b" }
func (m *Gemini15Flash8b) Provider() ProviderType { return ProviderGoogle }
func (m *Gemini15Flash8b) SystemPrompt() string   { return m.systemPrompt }
func (m *Gemini15Flash8b) thinkingDimensions() ThinkingDimension {
	return googleThinkingDimensions(m.ModelName())
}

func (m *Gemini15Flash8b) WithMaxTokens(n int) *Gemini15Flash8b       { m.maxTokens = n; return m }
func (m *Gemini15Flash8b) WithTemperature(t float64) *Gemini15Flash8b { m.temperature = t; return m }
func (m *Gemini15Flash8b) WithTopP(p float64) *Gemini15Flash8b        { m.topP = p; return m }
func (m *Gemini15Flash8b) WithTopK(k int) *Gemini15Flash8b            { m.topK = k; return m }
func (m *Gemini15Flash8b) WithSystemPrompt(s string) *Gemini15Flash8b { m.systemPrompt = s; return m }

// NewGemini15Flash8b creates a new Gemini 1.5 Flash 8B model with default options
//
// Deprecated: retired by Google; the API returns an error. Migrate to Gemini31FlashLite.
func NewGemini15Flash8b() *Gemini15Flash8b {
	return &Gemini15Flash8b{googleOptions{maxTokens: 8192, temperature: 1.0}}
}

// Gemini20FlashExp represents the Gemini 2.0 Flash Experimental model.
//
// Deprecated: shut down by Google on June 1, 2026; the API returns an error. Migrate to Gemini25Flash.
type Gemini20FlashExp struct{ googleOptions }

func (m *Gemini20FlashExp) ModelName() string      { return "gemini-2.0-flash-exp" }
func (m *Gemini20FlashExp) Provider() ProviderType { return ProviderGoogle }
func (m *Gemini20FlashExp) SystemPrompt() string   { return m.systemPrompt }
func (m *Gemini20FlashExp) thinkingDimensions() ThinkingDimension {
	return googleThinkingDimensions(m.ModelName())
}

func (m *Gemini20FlashExp) WithMaxTokens(n int) *Gemini20FlashExp       { m.maxTokens = n; return m }
func (m *Gemini20FlashExp) WithTemperature(t float64) *Gemini20FlashExp { m.temperature = t; return m }
func (m *Gemini20FlashExp) WithTopP(p float64) *Gemini20FlashExp        { m.topP = p; return m }
func (m *Gemini20FlashExp) WithTopK(k int) *Gemini20FlashExp            { m.topK = k; return m }
func (m *Gemini20FlashExp) WithSystemPrompt(s string) *Gemini20FlashExp { m.systemPrompt = s; return m }

// NewGemini20FlashExp creates a new Gemini 2.0 Flash Exp model with default options
//
// Deprecated: shut down by Google on June 1, 2026; the API returns an error. Migrate to Gemini25Flash.
func NewGemini20FlashExp() *Gemini20FlashExp {
	return &Gemini20FlashExp{googleOptions{maxTokens: 8192, temperature: 1.0}}
}

// Gemini20FlashThinking represents the Gemini 2.0 Flash Thinking Experimental model.
//
// Deprecated: experimental endpoint removed by Google; the API returns an error. Migrate to Gemini25Flash.
type Gemini20FlashThinking struct{ googleOptions }

func (m *Gemini20FlashThinking) ModelName() string      { return "gemini-2.0-flash-thinking-exp" }
func (m *Gemini20FlashThinking) Provider() ProviderType { return ProviderGoogle }
func (m *Gemini20FlashThinking) SystemPrompt() string   { return m.systemPrompt }
func (m *Gemini20FlashThinking) thinkingDimensions() ThinkingDimension {
	return googleThinkingDimensions(m.ModelName())
}

func (m *Gemini20FlashThinking) WithMaxTokens(n int) *Gemini20FlashThinking {
	m.maxTokens = n
	return m
}
func (m *Gemini20FlashThinking) WithTemperature(t float64) *Gemini20FlashThinking {
	m.temperature = t
	return m
}
func (m *Gemini20FlashThinking) WithTopP(p float64) *Gemini20FlashThinking { m.topP = p; return m }
func (m *Gemini20FlashThinking) WithTopK(k int) *Gemini20FlashThinking     { m.topK = k; return m }
func (m *Gemini20FlashThinking) WithSystemPrompt(s string) *Gemini20FlashThinking {
	m.systemPrompt = s
	return m
}

// NewGemini20FlashThinking creates a new Gemini 2.0 Flash Thinking model with default options
//
// Deprecated: experimental endpoint removed by Google; the API returns an error. Migrate to Gemini25Flash.
func NewGemini20FlashThinking() *Gemini20FlashThinking {
	return &Gemini20FlashThinking{googleOptions{maxTokens: 8192, temperature: 1.0}}
}

// Gemini20ProExp represents the Gemini 2.0 Pro Experimental model.
//
// Deprecated: experimental endpoint removed by Google; the API returns an error. Migrate to Gemini25Pro or Gemini31Pro.
type Gemini20ProExp struct{ googleOptions }

func (m *Gemini20ProExp) ModelName() string      { return "gemini-2.0-pro-exp" }
func (m *Gemini20ProExp) Provider() ProviderType { return ProviderGoogle }
func (m *Gemini20ProExp) SystemPrompt() string   { return m.systemPrompt }
func (m *Gemini20ProExp) thinkingDimensions() ThinkingDimension {
	return googleThinkingDimensions(m.ModelName())
}

func (m *Gemini20ProExp) WithMaxTokens(n int) *Gemini20ProExp       { m.maxTokens = n; return m }
func (m *Gemini20ProExp) WithTemperature(t float64) *Gemini20ProExp { m.temperature = t; return m }
func (m *Gemini20ProExp) WithTopP(p float64) *Gemini20ProExp        { m.topP = p; return m }
func (m *Gemini20ProExp) WithTopK(k int) *Gemini20ProExp            { m.topK = k; return m }
func (m *Gemini20ProExp) WithSystemPrompt(s string) *Gemini20ProExp { m.systemPrompt = s; return m }

// NewGemini20ProExp creates a new Gemini 2.0 Pro Exp model with default options
//
// Deprecated: experimental endpoint removed by Google; the API returns an error. Migrate to Gemini25Pro or Gemini31Pro.
func NewGemini20ProExp() *Gemini20ProExp {
	return &Gemini20ProExp{googleOptions{maxTokens: 8192, temperature: 1.0}}
}

// Gemini3Pro represents the Gemini 3 Pro model (preview).
// Versions: gemini-3-pro-preview
//
// Deprecated: Google redirects gemini-3-pro-preview to gemini-3.1-pro-preview
// (since March 9, 2026). Use Gemini31Pro to target that model directly.
type Gemini3Pro struct{ googleOptions }

func (m *Gemini3Pro) ModelName() string {
	if m.modelVersion != "" {
		return m.modelVersion
	}
	return "gemini-3-pro-preview"
}
func (m *Gemini3Pro) Provider() ProviderType { return ProviderGoogle }
func (m *Gemini3Pro) SystemPrompt() string   { return m.systemPrompt }
func (m *Gemini3Pro) thinkingDimensions() ThinkingDimension {
	return googleThinkingDimensions(m.ModelName())
}

func (m *Gemini3Pro) WithVersion(v string) *Gemini3Pro      { m.modelVersion = v; return m }
func (m *Gemini3Pro) WithMaxTokens(n int) *Gemini3Pro       { m.maxTokens = n; return m }
func (m *Gemini3Pro) WithTemperature(t float64) *Gemini3Pro { m.temperature = t; return m }
func (m *Gemini3Pro) WithTopP(p float64) *Gemini3Pro        { m.topP = p; return m }
func (m *Gemini3Pro) WithTopK(k int) *Gemini3Pro            { m.topK = k; return m }
func (m *Gemini3Pro) WithSystemPrompt(s string) *Gemini3Pro { m.systemPrompt = s; return m }

// NewGemini3Pro creates a new Gemini 3 Pro model with default options
//
// Deprecated: Google redirects gemini-3-pro-preview to gemini-3.1-pro-preview
// (since March 9, 2026). Use Gemini31Pro to target that model directly.
func NewGemini3Pro() *Gemini3Pro {
	return &Gemini3Pro{googleOptions{maxTokens: 8192, temperature: 1.0}}
}

// Gemini3Flash represents the Gemini 3 Flash model (preview).
// Versions: gemini-3-flash-preview
type Gemini3Flash struct{ googleOptions }

func (m *Gemini3Flash) ModelName() string {
	if m.modelVersion != "" {
		return m.modelVersion
	}
	return "gemini-3-flash-preview"
}
func (m *Gemini3Flash) Provider() ProviderType { return ProviderGoogle }
func (m *Gemini3Flash) SystemPrompt() string   { return m.systemPrompt }
func (m *Gemini3Flash) thinkingDimensions() ThinkingDimension {
	return googleThinkingDimensions(m.ModelName())
}

func (m *Gemini3Flash) WithVersion(v string) *Gemini3Flash      { m.modelVersion = v; return m }
func (m *Gemini3Flash) WithMaxTokens(n int) *Gemini3Flash       { m.maxTokens = n; return m }
func (m *Gemini3Flash) WithTemperature(t float64) *Gemini3Flash { m.temperature = t; return m }
func (m *Gemini3Flash) WithTopP(p float64) *Gemini3Flash        { m.topP = p; return m }
func (m *Gemini3Flash) WithTopK(k int) *Gemini3Flash            { m.topK = k; return m }
func (m *Gemini3Flash) WithSystemPrompt(s string) *Gemini3Flash { m.systemPrompt = s; return m }

// NewGemini3Flash creates a new Gemini 3 Flash model with default options
func NewGemini3Flash() *Gemini3Flash {
	return &Gemini3Flash{googleOptions{maxTokens: 8192, temperature: 1.0}}
}

// Gemini31Pro represents the Gemini 3.1 Pro model (preview).
// Top-tier reasoning and complex problem-solving. Versions: gemini-3.1-pro-preview
type Gemini31Pro struct{ googleOptions }

func (m *Gemini31Pro) ModelName() string {
	if m.modelVersion != "" {
		return m.modelVersion
	}
	return "gemini-3.1-pro-preview"
}
func (m *Gemini31Pro) Provider() ProviderType { return ProviderGoogle }
func (m *Gemini31Pro) SystemPrompt() string   { return m.systemPrompt }
func (m *Gemini31Pro) thinkingDimensions() ThinkingDimension {
	return googleThinkingDimensions(m.ModelName())
}

func (m *Gemini31Pro) WithVersion(v string) *Gemini31Pro      { m.modelVersion = v; return m }
func (m *Gemini31Pro) WithMaxTokens(n int) *Gemini31Pro       { m.maxTokens = n; return m }
func (m *Gemini31Pro) WithTemperature(t float64) *Gemini31Pro { m.temperature = t; return m }
func (m *Gemini31Pro) WithTopP(p float64) *Gemini31Pro        { m.topP = p; return m }
func (m *Gemini31Pro) WithTopK(k int) *Gemini31Pro            { m.topK = k; return m }
func (m *Gemini31Pro) WithSystemPrompt(s string) *Gemini31Pro { m.systemPrompt = s; return m }

// NewGemini31Pro creates a new Gemini 3.1 Pro model with default options
func NewGemini31Pro() *Gemini31Pro {
	return &Gemini31Pro{googleOptions{maxTokens: 8192, temperature: 1.0}}
}

// Gemini35Flash represents the Gemini 3.5 Flash model (stable/GA).
// The most intelligent Flash model for sustained frontier performance on agentic and coding tasks.
type Gemini35Flash struct{ googleOptions }

func (m *Gemini35Flash) ModelName() string      { return "gemini-3.5-flash" }
func (m *Gemini35Flash) Provider() ProviderType { return ProviderGoogle }
func (m *Gemini35Flash) SystemPrompt() string   { return m.systemPrompt }
func (m *Gemini35Flash) thinkingDimensions() ThinkingDimension {
	return googleThinkingDimensions(m.ModelName())
}

func (m *Gemini35Flash) WithMaxTokens(n int) *Gemini35Flash       { m.maxTokens = n; return m }
func (m *Gemini35Flash) WithTemperature(t float64) *Gemini35Flash { m.temperature = t; return m }
func (m *Gemini35Flash) WithTopP(p float64) *Gemini35Flash        { m.topP = p; return m }
func (m *Gemini35Flash) WithTopK(k int) *Gemini35Flash            { m.topK = k; return m }
func (m *Gemini35Flash) WithSystemPrompt(s string) *Gemini35Flash { m.systemPrompt = s; return m }

// NewGemini35Flash creates a new Gemini 3.5 Flash model with default options
func NewGemini35Flash() *Gemini35Flash {
	return &Gemini35Flash{googleOptions{maxTokens: 8192, temperature: 1.0}}
}

// Gemini31FlashLite represents the Gemini 3.1 Flash-Lite model (stable).
// The most cost-efficient Gemini model, optimized for low-latency, high-volume traffic.
type Gemini31FlashLite struct{ googleOptions }

func (m *Gemini31FlashLite) ModelName() string      { return "gemini-3.1-flash-lite" }
func (m *Gemini31FlashLite) Provider() ProviderType { return ProviderGoogle }
func (m *Gemini31FlashLite) SystemPrompt() string   { return m.systemPrompt }
func (m *Gemini31FlashLite) thinkingDimensions() ThinkingDimension {
	return googleThinkingDimensions(m.ModelName())
}

func (m *Gemini31FlashLite) WithMaxTokens(n int) *Gemini31FlashLite { m.maxTokens = n; return m }
func (m *Gemini31FlashLite) WithTemperature(t float64) *Gemini31FlashLite {
	m.temperature = t
	return m
}
func (m *Gemini31FlashLite) WithTopP(p float64) *Gemini31FlashLite { m.topP = p; return m }
func (m *Gemini31FlashLite) WithTopK(k int) *Gemini31FlashLite     { m.topK = k; return m }
func (m *Gemini31FlashLite) WithSystemPrompt(s string) *Gemini31FlashLite {
	m.systemPrompt = s
	return m
}

// NewGemini31FlashLite creates a new Gemini 3.1 Flash-Lite model with default options
func NewGemini31FlashLite() *Gemini31FlashLite {
	return &Gemini31FlashLite{googleOptions{maxTokens: 8192, temperature: 1.0}}
}

// Gemini36Flash represents the Gemini 3.6 Flash model (stable/GA).
// The current Flash flagship: improved token efficiency and code/agentic
// planning at a lower price point than Gemini 3.5 Flash.
type Gemini36Flash struct{ googleOptions }

func (m *Gemini36Flash) ModelName() string      { return "gemini-3.6-flash" }
func (m *Gemini36Flash) Provider() ProviderType { return ProviderGoogle }
func (m *Gemini36Flash) SystemPrompt() string   { return m.systemPrompt }
func (m *Gemini36Flash) thinkingDimensions() ThinkingDimension {
	return googleThinkingDimensions(m.ModelName())
}

func (m *Gemini36Flash) WithMaxTokens(n int) *Gemini36Flash       { m.maxTokens = n; return m }
func (m *Gemini36Flash) WithTemperature(t float64) *Gemini36Flash { m.temperature = t; return m }
func (m *Gemini36Flash) WithTopP(p float64) *Gemini36Flash        { m.topP = p; return m }
func (m *Gemini36Flash) WithTopK(k int) *Gemini36Flash            { m.topK = k; return m }
func (m *Gemini36Flash) WithSystemPrompt(s string) *Gemini36Flash { m.systemPrompt = s; return m }

// NewGemini36Flash creates a new Gemini 3.6 Flash model with default options
func NewGemini36Flash() *Gemini36Flash {
	return &Gemini36Flash{googleOptions{maxTokens: 8192, temperature: 1.0}}
}

// Gemini35FlashLite represents the Gemini 3.5 Flash-Lite model (stable/GA).
// The fastest, most cost-effective model in the 3.5 series, for
// high-throughput execution.
type Gemini35FlashLite struct{ googleOptions }

func (m *Gemini35FlashLite) ModelName() string      { return "gemini-3.5-flash-lite" }
func (m *Gemini35FlashLite) Provider() ProviderType { return ProviderGoogle }
func (m *Gemini35FlashLite) SystemPrompt() string   { return m.systemPrompt }
func (m *Gemini35FlashLite) thinkingDimensions() ThinkingDimension {
	return googleThinkingDimensions(m.ModelName())
}

func (m *Gemini35FlashLite) WithMaxTokens(n int) *Gemini35FlashLite { m.maxTokens = n; return m }
func (m *Gemini35FlashLite) WithTemperature(t float64) *Gemini35FlashLite {
	m.temperature = t
	return m
}
func (m *Gemini35FlashLite) WithTopP(p float64) *Gemini35FlashLite { m.topP = p; return m }
func (m *Gemini35FlashLite) WithTopK(k int) *Gemini35FlashLite     { m.topK = k; return m }
func (m *Gemini35FlashLite) WithSystemPrompt(s string) *Gemini35FlashLite {
	m.systemPrompt = s
	return m
}

// NewGemini35FlashLite creates a new Gemini 3.5 Flash-Lite model with default options
func NewGemini35FlashLite() *Gemini35FlashLite {
	return &Gemini35FlashLite{googleOptions{maxTokens: 8192, temperature: 1.0}}
}

// GoogleModel represents a generic Google Gemini model.
// Use this for any Gemini model this library has no named type for
// (e.g. new previews), so new model releases don't require a library update.
type GoogleModel struct {
	modelID string
	googleOptions
}

func (m *GoogleModel) ModelName() string      { return m.modelID }
func (m *GoogleModel) Provider() ProviderType { return ProviderGoogle }
func (m *GoogleModel) SystemPrompt() string   { return m.systemPrompt }
func (m *GoogleModel) thinkingDimensions() ThinkingDimension {
	return googleThinkingDimensions(m.ModelName())
}

func (m *GoogleModel) WithMaxTokens(n int) *GoogleModel       { m.maxTokens = n; return m }
func (m *GoogleModel) WithTemperature(t float64) *GoogleModel { m.temperature = t; return m }
func (m *GoogleModel) WithTopP(p float64) *GoogleModel        { m.topP = p; return m }
func (m *GoogleModel) WithTopK(k int) *GoogleModel            { m.topK = k; return m }
func (m *GoogleModel) WithSystemPrompt(s string) *GoogleModel { m.systemPrompt = s; return m }

// NewGoogleModel creates a generic Google Gemini model with the specified model ID
func NewGoogleModel(modelID string) *GoogleModel {
	return &GoogleModel{modelID: modelID, googleOptions: googleOptions{maxTokens: 8192, temperature: 1.0}}
}

// ============================================================================
// GOOGLE PROVIDER CLIENT
// ============================================================================

// googleClient implements the Provider interface for Google AI (Gemini)
// Uses the new Google GenAI SDK (google.golang.org/genai)
type googleClient struct {
	client      *genai.Client
	timeout     time.Duration
	logger      Logger
	rateLimiter *rateLimiter
}

// newGoogleClient creates a new Google AI client using the Google GenAI SDK
func newGoogleClient(config *GoogleConfig, logger Logger) (*googleClient, error) {
	clientConfig := &genai.ClientConfig{APIKey: config.APIKey}

	if config.UseVertexAI {
		// Vertex AI authenticates with application default credentials when
		// Project and Location are set, or with an API key in express mode
		if config.Project == "" && config.APIKey == "" {
			return nil, fmt.Errorf("vertex AI requires Project (with Location, or GOOGLE_CLOUD_LOCATION), or an APIKey for express mode")
		}
		clientConfig.Backend = genai.BackendVertexAI
		clientConfig.Project = config.Project
		clientConfig.Location = config.Location
	} else {
		if config.APIKey == "" {
			return nil, fmt.Errorf("google API key is required")
		}
		clientConfig.Backend = genai.BackendGeminiAPI
	}

	ctx := context.Background()
	client, err := genai.NewClient(ctx, clientConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create Google AI client: %w", err)
	}

	timeout := config.Timeout
	if timeout == 0 {
		timeout = defaultTimeout()
	}

	return &googleClient{
		client:      client,
		timeout:     timeout,
		logger:      logger,
		rateLimiter: newRateLimiter(config.RateLimiter, logger),
	}, nil
}

// getGoogleOptions extracts googleOptions from any model type
func getGoogleOptions(model Model) *googleOptions {
	switch m := model.(type) {
	case *Gemini25Pro:
		return &m.googleOptions
	case *Gemini25Flash:
		return &m.googleOptions
	case *Gemini25FlashLite:
		return &m.googleOptions
	case *Gemini20Flash:
		return &m.googleOptions
	case *Gemini20FlashLite:
		return &m.googleOptions
	case *Gemini15Pro:
		return &m.googleOptions
	case *Gemini15Flash:
		return &m.googleOptions
	case *Gemini15Flash8b:
		return &m.googleOptions
	case *Gemini20FlashExp:
		return &m.googleOptions
	case *Gemini20FlashThinking:
		return &m.googleOptions
	case *Gemini20ProExp:
		return &m.googleOptions
	case *Gemini3Pro:
		return &m.googleOptions
	case *Gemini3Flash:
		return &m.googleOptions
	case *Gemini31Pro:
		return &m.googleOptions
	case *Gemini35Flash:
		return &m.googleOptions
	case *Gemini31FlashLite:
		return &m.googleOptions
	case *Gemini36Flash:
		return &m.googleOptions
	case *Gemini35FlashLite:
		return &m.googleOptions
	case *GoogleModel:
		return &m.googleOptions
	default:
		return nil
	}
}

// Generate generates text using Google's Gemini API
func (c *googleClient) Generate(ctx context.Context, model Model, prompt string) (*GenerationResponse, error) {
	// Verify model is for Google
	if model.Provider() != ProviderGoogle {
		return nil, fmt.Errorf("model %s is not a Google model", model.ModelName())
	}

	// Set timeout
	ctx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()

	// Get model options
	opts := getGoogleOptions(model)
	if opts == nil {
		return nil, fmt.Errorf("unsupported Google model type: %T", model)
	}

	// Build generation config
	config := &genai.GenerateContentConfig{}

	if opts.temperature > 0 {
		temp := float32(opts.temperature)
		config.Temperature = &temp
	}
	if opts.maxTokens > 0 {
		config.MaxOutputTokens = int32(opts.maxTokens)
	}
	if opts.topP > 0 {
		topP := float32(opts.topP)
		config.TopP = &topP
	}
	if opts.topK > 0 {
		topK := float32(opts.topK)
		config.TopK = &topK
	}
	// Thinking is opt-in and, like caching, is applied once from a plan built
	// against the model's own generation rather than against the provider: which
	// of thinkingBudget and thinkingLevel a Gemini takes -- and whether it takes
	// either -- changed twice between 1.5 and 3.x.
	//
	// A model whose ThinkingOptions were never touched yields a nil config and
	// leaves the request exactly as built above.
	thinkingConfig, plan := googleThinkingConfig(model)
	config.ThinkingConfig = thinkingConfig

	// Explicit caching: point the request at a cache resource, either named
	// directly or created through the CacheManager below.
	var cachedContent string
	if co := modelCacheOptions(model); !co.Disabled() {
		cachedContent = co.CachedContent()
	}

	if opts.systemPrompt != "" && cachedContent == "" {
		config.SystemInstruction = &genai.Content{
			Parts: []*genai.Part{{Text: opts.systemPrompt}},
		}
	}
	if cachedContent != "" {
		// Gemini rejects a request that sets both, because the cache resource
		// already carries the system instruction it was created with. Sending
		// the cache resource wins; the model's own system prompt is dropped.
		config.CachedContent = cachedContent
		if opts.systemPrompt != "" {
			c.logger.Debug().
				Str("model", model.ModelName()).
				Str("cached_content", cachedContent).
				Msg("Ignoring system prompt: it must be baked into the cached content resource")
		}
	}

	// Build content
	contents := []*genai.Content{
		{
			Role:  "user",
			Parts: []*genai.Part{{Text: prompt}},
		},
	}

	c.logger.Debug().
		Str("model", model.ModelName()).
		// A disable is carried by a thinkingConfig too -- Gemini spells off as
		// thinkingBudget 0 -- so the presence of the config alone would report a
		// request that switched thinking off as one that asked for it. The field
		// means "this request asks the model to think", as it does on Anthropic.
		Bool("has_thinking", thinkingConfig != nil && !plan.disable).
		Str("thinking_translation", plan.translation()).
		Msg("Making Google AI API request")

	// Make the request with rate limit handling
	var resp *genai.GenerateContentResponse
	err := c.rateLimiter.Execute(ctx, func() error {
		var reqErr error
		resp, reqErr = c.client.Models.GenerateContent(ctx, model.ModelName(), contents, config)
		return reqErr
	})
	if err != nil {
		c.logger.Error().
			Err(err).
			Str("model", model.ModelName()).
			Str("prompt_preview", truncateString(prompt, 100)).
			Msg("Google AI generation failed")
		return nil, fmt.Errorf("google AI generation failed: %w", err)
	}

	if len(resp.Candidates) == 0 {
		return nil, fmt.Errorf("no candidates returned from Google AI")
	}

	candidate := resp.Candidates[0]
	if candidate.Content == nil || len(candidate.Content.Parts) == 0 {
		return nil, fmt.Errorf("no content in Google AI response")
	}

	// Extract the answer and the reasoning trace. A thought part carries its
	// text in the same field the answer uses and is distinguished only by its
	// Thought flag, so the split is what keeps the model's reasoning out of
	// GenerationResponse.Text once includeThoughts is on.
	text, thinkingText, thoughtSignature := googleSplitParts(candidate.Content.Parts)

	if text == "" {
		return nil, fmt.Errorf("no text content found in Google AI response")
	}

	// Extract token usage
	var promptTokens, completionTokens, totalTokens, cachedTokens, thoughtsTokens int
	if resp.UsageMetadata != nil {
		promptTokens = int(resp.UsageMetadata.PromptTokenCount)
		completionTokens = int(resp.UsageMetadata.CandidatesTokenCount)
		totalTokens = int(resp.UsageMetadata.TotalTokenCount)
		cachedTokens = int(resp.UsageMetadata.CachedContentTokenCount)
		thoughtsTokens = int(resp.UsageMetadata.ThoughtsTokenCount)
	}

	// Determine finish reason
	finishReason := "stop"
	if candidate.FinishReason != "" {
		finishReason = string(candidate.FinishReason)
	}

	// Build response
	response := &GenerationResponse{
		Text:         text,
		Thinking:     thinkingText,
		Model:        model.ModelName(),
		FinishReason: finishReason,
		// Gemini counts cached tokens inside PromptTokenCount, so they are
		// already part of the prompt total: promptIncludesCache is true. There
		// is no cache-write counter, so the write side stays zero.
		//
		// Thinking is the one place Gemini counts the other way round, and the
		// only provider in the library that does: thoughtsTokenCount is inside
		// totalTokenCount but outside candidatesTokenCount, which the SDK states
		// in the doc on TotalTokenCount itself ("the sum of prompt_token_count,
		// candidates_token_count, tool_use_prompt_token_count, and
		// thoughts_token_count"). Passing false folds the thoughts into the
		// completion total so CompletionTokens means the same thing here as
		// everywhere else, and leaves the reported total alone because it
		// already covers them.
		Usage: TokenUsage{
			PromptTokens:     promptTokens,
			CompletionTokens: completionTokens,
			TotalTokens:      totalTokens,
		}.withCache(cachedTokens, 0, true).
			withThinking(thoughtsTokens, false),
		Metadata: map[string]string{
			"provider": "google",
			"model":    model.ModelName(),
		},
	}

	// The signature authenticates a thought for replay on a later turn. It is
	// opaque bytes, so it rides in metadata base64-encoded rather than in a
	// typed field.
	if thoughtSignature != "" {
		response.Metadata["thinking_signature"] = thoughtSignature
	}

	// Whatever lingo had to translate or drop to fit the caller's request onto
	// this model's dialect, so a silent adaptation is never invisible.
	if s := plan.translation(); s != "" {
		response.Metadata["thinking_translation"] = s
	}

	c.logger.Debug().
		Str("model", model.ModelName()).
		Int("prompt_tokens", promptTokens).
		Int("completion_tokens", response.Usage.CompletionTokens).
		Int("total_tokens", response.Usage.TotalTokens).
		Int("cache_read_tokens", response.Usage.CacheReadTokens).
		Int("thinking_tokens", response.Usage.ThinkingTokens).
		Msg("Google AI generation completed")

	return response, nil
}

// Health checks the health of the Google AI client
func (c *googleClient) Health(ctx context.Context) error {
	ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	config := &genai.GenerateContentConfig{
		MaxOutputTokens: 5,
	}

	contents := []*genai.Content{
		{
			Role:  "user",
			Parts: []*genai.Part{{Text: "Hello"}},
		},
	}

	_, err := c.client.Models.GenerateContent(ctx, "gemini-2.5-flash-lite", contents, config)
	if err != nil {
		return fmt.Errorf("google AI health check failed: %w", err)
	}

	return nil
}

// Close closes the Google AI client
func (c *googleClient) Close() error {
	// The new SDK client doesn't require explicit closing
	return nil
}

// ============================================================================
// GOOGLE CACHE RESOURCE LIFECYCLE
// ============================================================================

// googleClient is the only provider whose prompt cache is a resource with a
// lifecycle, so it is the only PromptCacheManager.
var _ PromptCacheManager = (*googleClient)(nil)

// promptCacheFromGenai converts a genai CachedContent into lingo's shape. The
// resource names it carries differ between backends -- "cachedContents/abc123"
// on the Gemini Developer API, "projects/p/locations/l/cachedContents/abc123"
// on Vertex AI -- and are passed through verbatim: the SDK's own resource-name
// transformer accepts either form on either backend, and trimming the Vertex
// prefix would throw away the project and location it encodes.
func promptCacheFromGenai(cc *genai.CachedContent) *PromptCache {
	if cc == nil {
		return nil
	}

	cache := &PromptCache{
		Name:        cc.Name,
		DisplayName: cc.DisplayName,
		Model:       cc.Model,
		CreatedAt:   cc.CreateTime,
		ExpiresAt:   cc.ExpireTime,
	}
	if cc.UsageMetadata != nil {
		cache.Tokens = int(cc.UsageMetadata.TotalTokenCount)
	}
	return cache
}

// CreateCache stores content in a Gemini CachedContent resource. The system
// instruction belongs in the spec rather than on the model, because Gemini
// rejects a generate request that carries both a cache resource and a system
// instruction.
func (c *googleClient) CreateCache(ctx context.Context, spec PromptCacheSpec) (*PromptCache, error) {
	if spec.Model == nil {
		return nil, fmt.Errorf("google cache: Model is required")
	}
	if spec.Model.Provider() != ProviderGoogle {
		return nil, fmt.Errorf("google cache: model %s is not a Google model", spec.Model.ModelName())
	}
	if spec.Content == "" && spec.SystemInstruction == "" {
		return nil, fmt.Errorf("google cache: Content or SystemInstruction is required")
	}

	ctx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()

	config := &genai.CreateCachedContentConfig{
		TTL:         spec.TTL,
		DisplayName: spec.DisplayName,
	}
	if spec.Content != "" {
		config.Contents = []*genai.Content{
			{
				Role:  "user",
				Parts: []*genai.Part{{Text: spec.Content}},
			},
		}
	}
	if spec.SystemInstruction != "" {
		config.SystemInstruction = &genai.Content{
			Parts: []*genai.Part{{Text: spec.SystemInstruction}},
		}
	}

	var cc *genai.CachedContent
	err := c.rateLimiter.Execute(ctx, func() error {
		var reqErr error
		cc, reqErr = c.client.Caches.Create(ctx, spec.Model.ModelName(), config)
		return reqErr
	})
	if err != nil {
		c.logger.Error().
			Err(err).
			Str("model", spec.Model.ModelName()).
			Msg("Google cache create failed")
		return nil, fmt.Errorf("google cache create failed: %w", err)
	}

	cache := promptCacheFromGenai(cc)

	c.logger.Debug().
		Str("model", spec.Model.ModelName()).
		Str("cached_content", cache.Name).
		Int("tokens", cache.Tokens).
		Msg("Google cache created")

	return cache, nil
}

// GetCache reads a CachedContent resource back. Only its metadata comes back:
// the API never returns the content or system instruction it was created with.
func (c *googleClient) GetCache(ctx context.Context, name string) (*PromptCache, error) {
	if name == "" {
		return nil, fmt.Errorf("google cache: name is required")
	}

	ctx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()

	var cc *genai.CachedContent
	err := c.rateLimiter.Execute(ctx, func() error {
		var reqErr error
		cc, reqErr = c.client.Caches.Get(ctx, name, nil)
		return reqErr
	})
	if err != nil {
		return nil, fmt.Errorf("google cache get failed: %w", err)
	}

	return promptCacheFromGenai(cc), nil
}

// ListCaches walks every page of CachedContent resources.
func (c *googleClient) ListCaches(ctx context.Context) ([]*PromptCache, error) {
	ctx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()

	// The walk retries as a unit, so the accumulator is reset per attempt:
	// pagination cannot resume mid-iterator, and listing is a read, so
	// re-walking is safe. Without this the sibling methods would retry a 429
	// and this one alone would surface it.
	var caches []*PromptCache
	err := c.rateLimiter.Execute(ctx, func() error {
		caches = nil
		for cc, reqErr := range c.client.Caches.All(ctx) {
			if reqErr != nil {
				return reqErr
			}
			caches = append(caches, promptCacheFromGenai(cc))
		}
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("google cache list failed: %w", err)
	}

	return caches, nil
}

// RefreshCache extends a resource's lifetime to ttl measured from now. Gemini's
// update accepts nothing else: content, model and system instruction are fixed
// at creation.
func (c *googleClient) RefreshCache(ctx context.Context, name string, ttl time.Duration) (*PromptCache, error) {
	if name == "" {
		return nil, fmt.Errorf("google cache: name is required")
	}
	if ttl <= 0 {
		return nil, fmt.Errorf("google cache: ttl must be positive")
	}

	ctx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()

	var cc *genai.CachedContent
	err := c.rateLimiter.Execute(ctx, func() error {
		var reqErr error
		cc, reqErr = c.client.Caches.Update(ctx, name, &genai.UpdateCachedContentConfig{TTL: ttl})
		return reqErr
	})
	if err != nil {
		return nil, fmt.Errorf("google cache refresh failed: %w", err)
	}

	return promptCacheFromGenai(cc), nil
}

// DeleteCache drops a CachedContent resource. The delete response carries only
// the raw HTTP exchange, so there is nothing to hand back.
func (c *googleClient) DeleteCache(ctx context.Context, name string) error {
	if name == "" {
		return fmt.Errorf("google cache: name is required")
	}

	ctx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()

	err := c.rateLimiter.Execute(ctx, func() error {
		_, reqErr := c.client.Caches.Delete(ctx, name, nil)
		return reqErr
	})
	if err != nil {
		return fmt.Errorf("google cache delete failed: %w", err)
	}

	return nil
}
