package lingo

import (
	"context"
	"time"
)

// ============================================================================
// PROMPT CACHING
// ============================================================================
//
// Prompt caching lets a provider reuse the work it already did on a stable
// prefix of a request (typically the system prompt) instead of re-processing
// it on every call. Providers implement it in three different ways:
//
//   - explicit  the caller marks a cache breakpoint in the request
//               (Anthropic, Bedrock, OpenRouter). Google is explicit in a
//               different shape: there is no breakpoint at all, the cache is a
//               resource you create yourself and name in the request
//   - automatic the provider caches on its own and only reports what it did
//               (OpenAI, Azure, DeepSeek, xAI, Cohere, most OpenAI-compatible
//               endpoints)
//   - none      no caching, or none that is observable (Perplexity, Ollama)
//
// Use CachingSupport to ask which category a provider falls into. It answers
// per provider; OpenAI's GPT-5.6 models are the one case where individual
// models go further than their provider's level (see CacheSupportUsageOnly).
//
// lingo exposes one surface over all three. Caching is opt-in: a model whose
// CacheOptions were never touched produces byte-for-byte the same request it
// produced before this feature existed. Providers that cannot honour a request
// side option ignore it silently -- asking for caching is never an error.
//
// Read-side reporting is unconditional: every provider that reports cache
// token counts fills TokenUsage.CacheReadTokens / CacheWriteTokens, whether or
// not the caller opted in. TokenUsage is the portable contract; the cache_*
// keys some providers add to GenerationResponse.Metadata are provider-specific
// detail on top of it, not a cross-provider promise.
//
// Nothing here streams: lingo's Generate is a single round trip. Whoever adds a
// streaming API should know cache counters arrive only on the terminal usage
// event, so they have to be read off the accumulated usage, not the first one.

// CacheMode is the tri-state opt-in for prompt caching. The zero value,
// CacheModeDefault, leaves the provider's own default behaviour untouched and
// is what every model carries until a caller changes it.
type CacheMode int

const (
	// CacheModeDefault leaves caching to the provider. lingo sends no cache
	// fields at all, so providers that cache implicitly keep doing so and
	// providers that need an explicit breakpoint do not cache.
	CacheModeDefault CacheMode = iota
	// CacheModeOn asks the provider to cache the configured parts of the
	// request. Providers without a request-side knob ignore it.
	CacheModeOn
	// CacheModeOff suppresses every cache field lingo would otherwise send.
	// It cannot switch off caching a provider performs automatically, because
	// none of the supported providers offer an opt-out that is safe to send to
	// every model; on those providers it behaves like CacheModeDefault.
	CacheModeOff
)

// CacheTTL is the requested lifetime of a cache entry. Providers clamp this to
// the values their API accepts and ignore it when they do not model a TTL.
type CacheTTL string

const (
	// CacheTTLDefault lets the provider pick (Anthropic and Bedrock: 5m).
	CacheTTLDefault CacheTTL = ""
	// CacheTTL5m is honoured by Anthropic and Bedrock.
	CacheTTL5m CacheTTL = "5m"
	// CacheTTL1h is honoured by Anthropic and Bedrock, and costs more to write.
	CacheTTL1h CacheTTL = "1h"
)

// CacheSupport describes how far a provider can go with prompt caching.
type CacheSupport int

const (
	// CacheSupportNone means neither caching nor cache accounting is available.
	CacheSupportNone CacheSupport = iota
	// CacheSupportUsageOnly means the provider decides for itself what to cache
	// and lingo reports the token counts. There is no breakpoint to place,
	// though some of these providers accept a routing key (see WithCacheKey).
	// The level is per provider, so it reports the answer for the bulk of a
	// provider's catalogue: OpenAI is usage-only, but its GPT-5.6 models are a
	// per-model exception that does take a breakpoint when caching is enabled.
	CacheSupportUsageOnly
	// CacheSupportExplicit means the provider takes a request-side cache
	// instruction. On Anthropic, Bedrock and OpenRouter that is a breakpoint,
	// which lingo places for you when caching is enabled. Google is the odd one
	// out: the instruction is a cache resource you created yourself (see
	// WithCachedContent), lingo places no marker, and Gemini caches implicitly
	// besides -- so reads can be non-zero on a model nobody opted in.
	CacheSupportExplicit
)

// String returns the support level as a short lowercase label.
func (s CacheSupport) String() string {
	switch s {
	case CacheSupportUsageOnly:
		return "usage-only"
	case CacheSupportExplicit:
		return "explicit"
	default:
		return "none"
	}
}

// CachingSupport reports how much of the caching surface a provider honours.
// It is advisory: calling a cache setter on a provider below CacheSupportExplicit
// is a no-op, not an error.
func CachingSupport(provider ProviderType) CacheSupport {
	switch provider {
	case ProviderAnthropic, ProviderGoogle, ProviderBedrock, ProviderOpenRouter:
		return CacheSupportExplicit
	case ProviderOpenAI, ProviderDeepSeek, ProviderXAI, ProviderCohere, ProviderAzure, ProviderOpenAICompatible:
		return CacheSupportUsageOnly
	default:
		// ProviderPerplexity, ProviderOllama and anything unknown.
		return CacheSupportNone
	}
}

// CacheOptions is the provider-neutral prompt caching configuration carried by
// a model. Obtain it from a model with CacheOptions(), or configure it in a
// fluent chain with Cached.
//
//	m := lingo.Cached(lingo.NewClaudeSonnet5().WithMaxTokens(8192),
//		lingo.WithCacheTTL(lingo.CacheTTL1h))
//
//	// equivalent, statement form
//	m := lingo.NewClaudeSonnet5().WithMaxTokens(8192)
//	m.CacheOptions().Enable().WithTTL(lingo.CacheTTL1h)
type CacheOptions struct {
	mode          CacheMode
	ttl           CacheTTL
	key           string // OpenAI prompt_cache_key: routes similar requests to the same cache
	cachedContent string // Google: resource name of a pre-created CachedContent
	system        bool   // place a breakpoint at the end of the system prompt
	prompt        bool   // place a breakpoint at the end of the user prompt
}

// Enable turns caching on. Unless CacheSystemPrompt or CachePrompt was already
// set, it marks the system prompt as the cached prefix, which is the only
// stable prefix a single-turn Generate call has.
//
// It follows that enabling caching on a model with no system prompt and no
// WithPrompt(true) places no breakpoint at all: there is nothing to mark, so the
// request goes out unchanged. That is the usual reason an opted-in model reports
// no cache activity; the providers that place a breakpoint log whether one
// actually landed under the key cache_breakpoint.
func (o *CacheOptions) Enable() *CacheOptions {
	o.mode = CacheModeOn
	if !o.system && !o.prompt {
		o.system = true
	}
	return o
}

// Disable asks the provider not to cache this request.
func (o *CacheOptions) Disable() *CacheOptions { o.mode = CacheModeOff; return o }

// WithTTL requests a cache lifetime. Providers clamp unsupported values.
func (o *CacheOptions) WithTTL(ttl CacheTTL) *CacheOptions { o.ttl = ttl; return o }

// WithKey sets a cache partition key (OpenAI prompt_cache_key). Requests that
// share a key are routed to the same cache; it does not enable caching by
// itself and is ignored by providers that do not model it.
func (o *CacheOptions) WithKey(key string) *CacheOptions { o.key = key; return o }

// WithCachedContent points the request at a pre-created provider-side cache
// resource. Google only: the value is a CachedContent resource name such as
// "cachedContents/1234". Ignored elsewhere.
func (o *CacheOptions) WithCachedContent(name string) *CacheOptions {
	o.cachedContent = name
	return o
}

// WithSystemPrompt controls whether the system prompt ends a cached prefix.
func (o *CacheOptions) WithSystemPrompt(cache bool) *CacheOptions { o.system = cache; return o }

// WithPrompt controls whether the user prompt ends a cached prefix. Useful
// when the same long document is sent with different instructions appended
// downstream; pointless for prompts that change on every call.
func (o *CacheOptions) WithPrompt(cache bool) *CacheOptions { o.prompt = cache; return o }

// Mode reports the configured mode.
func (o *CacheOptions) Mode() CacheMode {
	if o == nil {
		return CacheModeDefault
	}
	return o.mode
}

// Enabled reports whether caching was explicitly turned on.
func (o *CacheOptions) Enabled() bool { return o != nil && o.mode == CacheModeOn }

// Disabled reports whether caching was explicitly turned off.
func (o *CacheOptions) Disabled() bool { return o != nil && o.mode == CacheModeOff }

// TTL reports the requested lifetime, CacheTTLDefault when unset.
func (o *CacheOptions) TTL() CacheTTL {
	if o == nil {
		return CacheTTLDefault
	}
	return o.ttl
}

// Key reports the cache partition key, "" when unset.
func (o *CacheOptions) Key() string {
	if o == nil {
		return ""
	}
	return o.key
}

// CachedContent reports the provider-side cache resource name, "" when unset.
func (o *CacheOptions) CachedContent() string {
	if o == nil {
		return ""
	}
	return o.cachedContent
}

// SystemPromptCached reports whether a breakpoint belongs after the system prompt.
func (o *CacheOptions) SystemPromptCached() bool { return o.Enabled() && o.system }

// PromptCached reports whether a breakpoint belongs after the user prompt.
func (o *CacheOptions) PromptCached() bool { return o.Enabled() && o.prompt }

// CacheOption configures a CacheOptions in a fluent chain. See Cached.
type CacheOption func(*CacheOptions)

// WithCacheTTL requests a cache lifetime.
func WithCacheTTL(ttl CacheTTL) CacheOption {
	return func(o *CacheOptions) { o.WithTTL(ttl) }
}

// WithCacheKey sets a cache partition key (OpenAI prompt_cache_key).
func WithCacheKey(key string) CacheOption {
	return func(o *CacheOptions) { o.WithKey(key) }
}

// WithCachedContent points a Google request at a pre-created cache resource by
// name. Use WithPromptCache to pass a resource a PromptCacheManager returned.
func WithCachedContent(name string) CacheOption {
	return func(o *CacheOptions) { o.WithCachedContent(name) }
}

// WithCacheSystemPrompt controls caching of the system prompt (default true).
func WithCacheSystemPrompt(cache bool) CacheOption {
	return func(o *CacheOptions) { o.WithSystemPrompt(cache) }
}

// WithCachePrompt controls caching of the user prompt (default false).
func WithCachePrompt(cache bool) CacheOption {
	return func(o *CacheOptions) { o.WithPrompt(cache) }
}

// ============================================================================
// PROVIDER-SIDE CACHE RESOURCES
// ============================================================================

// PromptCache describes a provider-side cache resource: content the provider
// has already processed and stored under a name, which later requests point at
// instead of resending. Only Google models caching this way; everywhere else a
// cache is a property of one request and does not outlive it.
type PromptCache struct {
	// Name is the resource name a request refers to. On the Gemini Developer
	// API it looks like "cachedContents/abc123"; on Vertex AI it is fully
	// qualified, "projects/p/locations/l/cachedContents/abc123". Either form
	// round-trips through WithCachedContent unchanged.
	Name string `json:"name"`
	// DisplayName is the label supplied at creation, "" when none was given.
	DisplayName string `json:"display_name,omitempty"`
	// Model is the model the cache is bound to, in the provider's own
	// qualified form ("models/gemini-3.1-pro-preview"), not lingo's ModelName().
	// A request with a different model is rejected by the provider.
	Model string `json:"model"`
	// CreatedAt is when the provider created the resource.
	CreatedAt time.Time `json:"created_at,omitzero"`
	// ExpiresAt is when the provider will drop the resource.
	ExpiresAt time.Time `json:"expires_at,omitzero"`
	// Tokens is the number of tokens the cached content occupies, 0 when the
	// provider does not report it.
	Tokens int `json:"tokens,omitempty"`
}

// Expired reports whether the resource's expiry has passed. A zero ExpiresAt
// is never expired.
func (c *PromptCache) Expired() bool {
	return c != nil && !c.ExpiresAt.IsZero() && time.Now().After(c.ExpiresAt)
}

// TimeToLive returns the time left before the resource expires, 0 once it has
// expired or when the provider reported no expiry.
func (c *PromptCache) TimeToLive() time.Duration {
	if c == nil || c.ExpiresAt.IsZero() {
		return 0
	}
	if d := time.Until(c.ExpiresAt); d > 0 {
		return d
	}
	return 0
}

// PromptCacheSpec describes a cache resource to create.
type PromptCacheSpec struct {
	// Model binds the cache to one model. Required: a resource created for one
	// model cannot be used by another.
	Model Model
	// Content is the text to cache. It must clear the provider's minimum
	// (Gemini enforces a per-model floor in the low thousands of tokens) or
	// the provider rejects the create with a 400.
	Content string
	// SystemInstruction is baked into the resource. Gemini rejects a generate
	// request that carries both a cache resource and a system instruction, so
	// the system prompt has to live here rather than on the model.
	SystemInstruction string
	// TTL is the requested lifetime, measured from creation. Zero leaves the
	// provider's default.
	TTL time.Duration
	// DisplayName is an optional human label the provider stores with it.
	DisplayName string
}

// PromptCacheManager is the optional capability interface a provider
// implements when its prompt cache is a resource with a lifecycle the caller
// owns, rather than a per-request flag. Reach it with LLMGateway.CacheManager.
//
// Only the Google provider implements it, on both the Gemini Developer API and
// Vertex AI. Every other provider caches per request: there is no resource to
// create, refresh or delete, so there is nothing for them to implement. Treat
// it as Google's lifecycle surface, not as a portability promise.
//
// These are direct resource calls, not request-side options, so unlike the
// rest of the caching surface they report failures as errors. What stays a
// silent no-op is the discovery: CacheManager returns false, never an error.
type PromptCacheManager interface {
	// CreateCache stores content provider-side and returns the resource,
	// whose Name you hand to WithPromptCache or WithCachedContent.
	CreateCache(ctx context.Context, spec PromptCacheSpec) (*PromptCache, error)
	// GetCache reads a resource back, so you can check its expiry before use.
	GetCache(ctx context.Context, name string) (*PromptCache, error)
	// ListCaches returns every cache resource the credentials can see, walking
	// pagination internally. Every page is fetched under the provider's single
	// request timeout, so a tenant with thousands of resources can time out
	// mid-walk and lose the whole result.
	ListCaches(ctx context.Context) ([]*PromptCache, error)
	// RefreshCache extends a resource's lifetime to ttl measured from now. It
	// cannot change the content, model or system instruction the resource was
	// created with.
	RefreshCache(ctx context.Context, name string, ttl time.Duration) (*PromptCache, error)
	// DeleteCache drops a resource.
	DeleteCache(ctx context.Context, name string) error
}

// WithPromptCache points a Google request at a cache resource returned by a
// PromptCacheManager. It is WithCachedContent(cache.Name) with the nil check
// folded in, so a create-then-generate flow needs no name plumbing.
func WithPromptCache(cache *PromptCache) CacheOption {
	return func(o *CacheOptions) {
		if cache != nil {
			o.WithCachedContent(cache.Name)
		}
	}
}

// CacheableModel is the optional capability interface a model implements when
// it can carry caching configuration. Providers type-assert it; models that do
// not implement it are generated exactly as before. Model itself is unchanged,
// so external implementations keep compiling.
//
// Carrying CacheOptions is not a promise that any of it reaches the wire: the
// DeepSeek and xAI models carry one and send nothing, because their provider
// caches on its own. The reverse holds too -- Cohere is CacheSupportUsageOnly
// yet its models are deliberately not CacheableModel, because its chat request
// has no cache field to set. Use CachingSupport to ask what a provider does;
// this interface only says where configuration can be stored.
type CacheableModel interface {
	Model
	// CacheOptions returns the model's caching configuration, never nil.
	CacheOptions() *CacheOptions
}

// Cached turns prompt caching on for a model and returns the same model, so it
// slots into the existing builder chain and keeps the concrete type:
//
//	m := lingo.Cached(lingo.NewClaudeSonnet5(), lingo.WithCacheTTL(lingo.CacheTTL1h)).
//		WithMaxTokens(8192)
//
// Models whose provider cannot place a breakpoint are returned untouched, so
// the call is safe to make generically across providers.
func Cached[M Model](model M, opts ...CacheOption) M {
	if c, ok := any(model).(CacheableModel); ok {
		co := c.CacheOptions()
		co.Enable()
		for _, opt := range opts {
			opt(co)
		}
	}
	return model
}

// NotCached asks the provider to skip caching for this model. Only providers
// with an explicit opt-out honour it; elsewhere it is a no-op.
func NotCached[M Model](model M) M {
	if c, ok := any(model).(CacheableModel); ok {
		c.CacheOptions().Disable()
	}
	return model
}

// modelCacheOptions returns the model's cache configuration, or nil when the
// model does not carry one. Providers call this instead of type-asserting.
func modelCacheOptions(model Model) *CacheOptions {
	if c, ok := model.(CacheableModel); ok {
		return c.CacheOptions()
	}
	return nil
}
