package lingo

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	cohere "github.com/cohere-ai/cohere-go/v2"
)

// ============================================================================
// COMPILE-TIME CAPABILITY ASSERTIONS
// ============================================================================
//
// One representative model per cacheable provider, including the thinking and
// reasoning variants, whose option structs are siblings of the standard ones
// rather than extensions and so need their own accessor. A provider that loses
// its CacheOptions accessor breaks the build here rather than silently
// degrading to "caching requested, nothing sent".

var (
	// Anthropic: standard options, thinking options, and the generic model.
	_ CacheableModel = (*Claude3Opus)(nil)
	_ CacheableModel = (*ClaudeSonnet5)(nil)
	_ CacheableModel = (*ClaudeFable5)(nil)
	_ CacheableModel = (*AnthropicModel)(nil)

	// OpenAI: the standard and reasoning option sets are siblings.
	_ CacheableModel = (*GPT4o)(nil)
	_ CacheableModel = (*GPT5)(nil)
	_ CacheableModel = (*O3)(nil)
	_ CacheableModel = (*GPT56Sol)(nil)
	_ CacheableModel = (*OpenAIModel)(nil)
	_ CacheableModel = (*OpenAIReasoningModel)(nil)

	// Google.
	_ CacheableModel = (*Gemini3Pro)(nil)
	_ CacheableModel = (*Gemini20FlashThinking)(nil)
	_ CacheableModel = (*GoogleModel)(nil)

	// Bedrock: five per-family option structs plus the flat generic model.
	_ CacheableModel = (*BedrockClaudeSonnet5)(nil)
	_ CacheableModel = (*BedrockNovaPro)(nil)
	_ CacheableModel = (*BedrockTitanTextPremier)(nil)
	_ CacheableModel = (*BedrockLlama33Instruct70B)(nil)
	_ CacheableModel = (*BedrockMistralLarge)(nil)
	_ CacheableModel = (*BedrockModel)(nil)

	// The oaicompat family, all sharing oaiOptions.
	_ CacheableModel = (*AzureOpenAIModel)(nil)
	_ CacheableModel = (*AzureOpenAIReasoningModel)(nil)
	_ CacheableModel = (*Grok45)(nil)
	_ CacheableModel = (*Grok420Reasoning)(nil)
	_ CacheableModel = (*XAIModel)(nil)
	_ CacheableModel = (*DeepSeekV4Pro)(nil)
	_ CacheableModel = (*DeepSeekModel)(nil)
	_ CacheableModel = (*OpenRouterModel)(nil)
	_ CacheableModel = (*OpenAICompatibleModel)(nil)

	// Google is also the one provider whose cache is a resource with a
	// lifecycle, so it is the only PromptCacheManager.
	_ PromptCacheManager = (*googleClient)(nil)
)

// ============================================================================
// USAGE NORMALIZATION
// ============================================================================

func TestWithCacheNormalizesBothReportingStyles(t *testing.T) {
	tests := []struct {
		name         string
		in           TokenUsage
		read, write  int
		promptIncl   bool
		want         TokenUsage
		wantUncached int
		wantHit      bool
	}{{
		// Anthropic / Bedrock native Claude: the counters sit alongside the
		// prompt total, so they have to be folded in.
		name: "additive grows prompt and total",
		in:   TokenUsage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
		read: 64, write: 32, promptIncl: false,
		want: TokenUsage{PromptTokens: 106, CompletionTokens: 5, TotalTokens: 111,
			CacheReadTokens: 64, CacheWriteTokens: 32},
		wantUncached: 10, wantHit: true,
	}, {
		// OpenAI-shaped, Google, Cohere: the counters are a breakdown of the
		// prompt total and folding them in would double count.
		name: "subset leaves prompt and total alone",
		in:   TokenUsage{PromptTokens: 100, CompletionTokens: 5, TotalTokens: 105},
		read: 64, write: 8, promptIncl: true,
		want: TokenUsage{PromptTokens: 100, CompletionTokens: 5, TotalTokens: 105,
			CacheReadTokens: 64, CacheWriteTokens: 8},
		wantUncached: 28, wantHit: true,
	}, {
		name: "additive with no cache activity is a no-op",
		in:   TokenUsage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
		read: 0, write: 0, promptIncl: false,
		want:         TokenUsage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
		wantUncached: 10, wantHit: false,
	}, {
		name: "subset with no cache activity is a no-op",
		in:   TokenUsage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
		read: 0, write: 0, promptIncl: true,
		want:         TokenUsage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
		wantUncached: 10, wantHit: false,
	}, {
		name: "write only is not a hit",
		in:   TokenUsage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
		read: 0, write: 40, promptIncl: false,
		want: TokenUsage{PromptTokens: 50, CompletionTokens: 5, TotalTokens: 55,
			CacheWriteTokens: 40},
		wantUncached: 10, wantHit: false,
	}, {
		// A provider reporting nonsense must not push the prompt total below
		// what it already said, and must not produce negative counters.
		name: "negative counters clamp to zero, additive",
		in:   TokenUsage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
		read: -64, write: -1, promptIncl: false,
		want:         TokenUsage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
		wantUncached: 10, wantHit: false,
	}, {
		name: "negative counters clamp to zero, subset",
		in:   TokenUsage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
		read: -1, write: -64, promptIncl: true,
		want:         TokenUsage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
		wantUncached: 10, wantHit: false,
	}, {
		// Counters larger than the prompt total can only come from a provider
		// bug; UncachedPromptTokens floors at zero rather than going negative.
		name: "over-large subset counters floor the uncached remainder",
		in:   TokenUsage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
		read: 64, write: 0, promptIncl: true,
		want: TokenUsage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15,
			CacheReadTokens: 64},
		wantUncached: 0, wantHit: true,
	}}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := tc.in.withCache(tc.read, tc.write, tc.promptIncl)
			if got != tc.want {
				t.Errorf("withCache(%d, %d, %t) = %+v, want %+v",
					tc.read, tc.write, tc.promptIncl, got, tc.want)
			}
			if n := got.UncachedPromptTokens(); n != tc.wantUncached {
				t.Errorf("UncachedPromptTokens() = %d, want %d", n, tc.wantUncached)
			}
			if hit := got.CacheHit(); hit != tc.wantHit {
				t.Errorf("CacheHit() = %t, want %t", hit, tc.wantHit)
			}
		})
	}
}

func TestWithCacheDoesNotMutateTheReceiver(t *testing.T) {
	u := TokenUsage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15}
	_ = u.withCache(64, 32, false)
	if (u != TokenUsage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15}) {
		t.Errorf("receiver mutated: %+v", u)
	}
}

// Cohere is the one usage-only provider with no request-side lane at all, so
// its counters are only ever exercised through cohereUsage. Every field is an
// optional pointer, and the three parts (token block, cached count, neither)
// arrive independently of one another.
func TestCohereUsageCountersAreSubsetsOfThePrompt(t *testing.T) {
	f := func(v float64) *float64 { return &v }

	tests := []struct {
		name         string
		in           *cohere.Usage
		want         TokenUsage
		wantUncached int
	}{{
		name: "cached tokens are part of the input count",
		in: &cohere.Usage{
			Tokens:       &cohere.UsageTokens{InputTokens: f(100), OutputTokens: f(7)},
			CachedTokens: f(64),
		},
		want:         TokenUsage{PromptTokens: 100, CompletionTokens: 7, TotalTokens: 107, CacheReadTokens: 64},
		wantUncached: 36,
	}, {
		name: "no cache activity leaves the counters at zero",
		in: &cohere.Usage{
			Tokens: &cohere.UsageTokens{InputTokens: f(100), OutputTokens: f(7)},
		},
		want:         TokenUsage{PromptTokens: 100, CompletionTokens: 7, TotalTokens: 107},
		wantUncached: 100,
	}, {
		// The counts are independent optional fields, so a cached count can
		// arrive without a token block. Reporting the read is still right; it
		// simply leaves no room for an uncached remainder.
		name:         "a cached count without a token block is still reported",
		in:           &cohere.Usage{CachedTokens: f(64)},
		want:         TokenUsage{CacheReadTokens: 64},
		wantUncached: 0,
	}, {
		name: "no usage at all",
		in:   nil,
		want: TokenUsage{},
	}}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := cohereUsage(tc.in)
			if got != tc.want {
				t.Errorf("cohereUsage() = %+v, want %+v", got, tc.want)
			}
			if n := got.UncachedPromptTokens(); n != tc.wantUncached {
				t.Errorf("UncachedPromptTokens() = %d, want %d", n, tc.wantUncached)
			}
		})
	}
}

// ============================================================================
// OPT-IN GUARANTEE
// ============================================================================

// cacheableModels lists one freshly constructed model per provider. Every entry
// must start with caching neither on nor off, which is what keeps an untouched
// model producing the request it produced before caching existed.
func freshModels() []Model {
	return []Model{
		NewGPT4o(),
		NewGPT5(),
		NewClaudeSonnet5(),
		NewClaude3Opus(),
		NewGemini3Pro(),
		NewBedrockClaudeSonnet5(),
		NewBedrockNovaPro(),
		NewBedrockModel("us.anthropic.claude-opus-5", "claude"),
		NewAzureOpenAIModel("my-deployment"),
		NewAzureOpenAIReasoningModel("my-deployment"),
		NewGrok45(),
		NewDeepSeekV4Pro(),
		NewOpenRouterModel("anthropic/claude-opus-5"),
		NewOpenAICompatibleModel("llama-3.3-70b"),
		NewCommandAPlus(),
		NewSonarPro(),
		NewLlama33(),
	}
}

func TestFreshModelHasCachingUntouched(t *testing.T) {
	for _, m := range freshModels() {
		co := modelCacheOptions(m)
		if co.Enabled() {
			t.Errorf("%s/%s: fresh model reports caching enabled", m.Provider(), m.ModelName())
		}
		if co.Disabled() {
			t.Errorf("%s/%s: fresh model reports caching disabled", m.Provider(), m.ModelName())
		}
		if co.Mode() != CacheModeDefault {
			t.Errorf("%s/%s: mode = %v, want CacheModeDefault", m.Provider(), m.ModelName(), co.Mode())
		}
		if co.SystemPromptCached() || co.PromptCached() {
			t.Errorf("%s/%s: fresh model wants a breakpoint", m.Provider(), m.ModelName())
		}
		if co.TTL() != CacheTTLDefault || co.Key() != "" || co.CachedContent() != "" {
			t.Errorf("%s/%s: fresh model carries cache config: ttl=%q key=%q content=%q",
				m.Provider(), m.ModelName(), co.TTL(), co.Key(), co.CachedContent())
		}
	}
}

// ============================================================================
// Cached / NotCached
// ============================================================================

func TestCachedPreservesConcreteTypeAndChains(t *testing.T) {
	// The point of the generic signature: Cached returns *ClaudeSonnet5, not
	// Model, so the builder chain survives on both sides of it. This has to
	// compile as written.
	m := Cached(NewClaudeSonnet5().WithSystemPrompt("be terse"),
		WithCacheTTL(CacheTTL1h), WithCachePrompt(true)).
		WithMaxTokens(8192).
		WithEffort(EffortHigh)

	var _ *ClaudeSonnet5 = m

	co := m.CacheOptions()
	if !co.Enabled() || co.Disabled() {
		t.Errorf("mode = %v, want CacheModeOn", co.Mode())
	}
	if co.TTL() != CacheTTL1h {
		t.Errorf("ttl = %q", co.TTL())
	}
	if !co.SystemPromptCached() || !co.PromptCached() {
		t.Errorf("breakpoints: system=%t prompt=%t, want both",
			co.SystemPromptCached(), co.PromptCached())
	}
	if m.maxTokens != 8192 || m.systemPrompt != "be terse" || m.thinking.effort != EffortHigh {
		t.Errorf("builder options lost: %+v", m.anthropicThinkingOptions)
	}
}

func TestCachedMutatesInPlace(t *testing.T) {
	m := NewGPT4o()
	if got := Cached(m); got != m {
		t.Error("Cached returned a different pointer; the model must be mutated in place")
	}
	if got := NotCached(m); got != m {
		t.Error("NotCached returned a different pointer")
	}
}

func TestCachedDefaultsToTheSystemPromptBreakpoint(t *testing.T) {
	co := Cached(NewClaudeSonnet5()).CacheOptions()
	if !co.SystemPromptCached() {
		t.Error("bare Cached must mark the system prompt, the only stable prefix a Generate call has")
	}
	if co.PromptCached() {
		t.Error("bare Cached must not mark the user prompt")
	}

	// An explicit choice is not overridden by Enable's default.
	co = Cached(NewClaudeSonnet5(), WithCacheSystemPrompt(false), WithCachePrompt(true)).CacheOptions()
	if co.SystemPromptCached() || !co.PromptCached() {
		t.Errorf("explicit breakpoints overridden: system=%t prompt=%t",
			co.SystemPromptCached(), co.PromptCached())
	}
}

func TestNotCachedSuppressesBreakpoints(t *testing.T) {
	m := NotCached(Cached(NewClaudeSonnet5(), WithCachePrompt(true)))
	co := m.CacheOptions()
	if co.Enabled() || !co.Disabled() || co.Mode() != CacheModeOff {
		t.Errorf("mode = %v, want CacheModeOff", co.Mode())
	}
	if co.SystemPromptCached() || co.PromptCached() {
		t.Error("a disabled model must not want breakpoints")
	}
}

func TestCacheOptionsSettersRoundTrip(t *testing.T) {
	m := NewGemini3Pro()
	m.CacheOptions().Enable().
		WithTTL(CacheTTL5m).
		WithKey("tenant-7").
		WithCachedContent("cachedContents/1234")

	co := modelCacheOptions(m)
	if co.TTL() != CacheTTL5m || co.Key() != "tenant-7" || co.CachedContent() != "cachedContents/1234" {
		t.Errorf("options = ttl:%q key:%q content:%q", co.TTL(), co.Key(), co.CachedContent())
	}
	// The statement form and the functional form reach the same struct.
	if co != m.CacheOptions() {
		t.Error("modelCacheOptions and CacheOptions returned different pointers")
	}
}

func TestCachedOnProviderWithoutCachingIsSilent(t *testing.T) {
	// Cohere, Perplexity and Ollama carry no CacheOptions at all: asking for
	// caching has to fall through the type assertion rather than panic.
	uncacheable := []Model{
		NewCommandAPlus(),
		NewCohereModel("command-a-03-2025"),
		NewSonarPro(),
		NewPerplexityModel("sonar"),
		NewLlama33(),
		NewOllamaModel("qwen3"),
	}

	for _, m := range uncacheable {
		if _, ok := m.(CacheableModel); ok {
			t.Errorf("%s/%s unexpectedly satisfies CacheableModel", m.Provider(), m.ModelName())
		}
		if co := modelCacheOptions(m); co != nil {
			t.Errorf("%s/%s: modelCacheOptions = %+v, want nil", m.Provider(), m.ModelName(), co)
		}
	}

	// The generic helpers return the model untouched, and the concrete type is
	// still there afterwards.
	cohere := NewCommandAPlus()
	if got := Cached(cohere, WithCacheTTL(CacheTTL1h)); got != cohere {
		t.Error("Cached must return the same Cohere model")
	}
	if got := NotCached(cohere).WithMaxTokens(128); got != cohere {
		t.Error("NotCached must return the same Cohere model and keep its concrete type")
	}

	sonar := NewSonarPro()
	if got := Cached(sonar); got != sonar {
		t.Error("Cached must return the same Perplexity model")
	}
	ollama := NewLlama33()
	if got := NotCached(ollama); got != ollama {
		t.Error("NotCached must return the same Ollama model")
	}
}

// ============================================================================
// SUPPORT MATRIX AND NIL SAFETY
// ============================================================================

func TestCachingSupportPerProvider(t *testing.T) {
	tests := []struct {
		provider ProviderType
		want     CacheSupport
		label    string
	}{
		{ProviderAnthropic, CacheSupportExplicit, "explicit"},
		{ProviderGoogle, CacheSupportExplicit, "explicit"},
		{ProviderBedrock, CacheSupportExplicit, "explicit"},
		{ProviderOpenRouter, CacheSupportExplicit, "explicit"},
		{ProviderOpenAI, CacheSupportUsageOnly, "usage-only"},
		{ProviderDeepSeek, CacheSupportUsageOnly, "usage-only"},
		{ProviderXAI, CacheSupportUsageOnly, "usage-only"},
		{ProviderCohere, CacheSupportUsageOnly, "usage-only"},
		{ProviderAzure, CacheSupportUsageOnly, "usage-only"},
		{ProviderOpenAICompatible, CacheSupportUsageOnly, "usage-only"},
		{ProviderPerplexity, CacheSupportNone, "none"},
		{ProviderOllama, CacheSupportNone, "none"},
		{ProviderType("not-a-provider"), CacheSupportNone, "none"},
	}

	for _, tc := range tests {
		if got := CachingSupport(tc.provider); got != tc.want {
			t.Errorf("CachingSupport(%q) = %v, want %v", tc.provider, got, tc.want)
		}
		if got := tc.want.String(); got != tc.label {
			t.Errorf("%v.String() = %q, want %q", tc.want, got, tc.label)
		}
	}
}

func TestNilCacheOptionsAccessorsAreSafe(t *testing.T) {
	// Providers call modelCacheOptions and read straight off the result, which
	// is nil for every model that carries no caching config.
	var co *CacheOptions

	if co.Mode() != CacheModeDefault {
		t.Errorf("Mode() = %v", co.Mode())
	}
	if co.Enabled() {
		t.Error("Enabled() = true")
	}
	if co.Disabled() {
		t.Error("Disabled() = true")
	}
	if co.TTL() != CacheTTLDefault {
		t.Errorf("TTL() = %q", co.TTL())
	}
	if co.Key() != "" {
		t.Errorf("Key() = %q", co.Key())
	}
	if co.CachedContent() != "" {
		t.Errorf("CachedContent() = %q", co.CachedContent())
	}
	if co.SystemPromptCached() {
		t.Error("SystemPromptCached() = true")
	}
	if co.PromptCached() {
		t.Error("PromptCached() = true")
	}
}

// ============================================================================
// WIRE BEHAVIOUR
// ============================================================================

// cacheUsageStub serves an OpenAI-shaped completion whose usage carries cache
// counters, and records the request.
func cacheUsageStub(t *testing.T, c *capture) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		c.record(r, raw)

		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"id":"c1","model":"served-model","object":"chat.completion",
			"choices":[{"index":0,"finish_reason":"stop",
				"message":{"role":"assistant","content":"hi there"}}],
			"usage":{"prompt_tokens":100,"completion_tokens":7,"total_tokens":107,
				"prompt_tokens_details":{"cached_tokens":64,"cache_write_tokens":8}}}`)
	}))
}

func TestCacheUsageIsReportedWithoutOptingIn(t *testing.T) {
	var c capture
	srv := cacheUsageStub(t, &c)
	defer srv.Close()

	// No Cached() anywhere: reporting is unconditional.
	resp := generate(t, &OpenAICompatibleConfig{BaseURL: srv.URL, APIKey: "k"},
		NewOpenAICompatibleModel("llama-3.3-70b"))

	if resp.Usage.CacheReadTokens != 64 || resp.Usage.CacheWriteTokens != 8 {
		t.Errorf("cache counters = %+v", resp.Usage)
	}
	// OpenAI-shaped counters are a subset of prompt_tokens, so the totals stay
	// exactly as the provider reported them.
	if resp.Usage.PromptTokens != 100 || resp.Usage.TotalTokens != 107 {
		t.Errorf("subset counters must not inflate the totals: %+v", resp.Usage)
	}
	if resp.Usage.UncachedPromptTokens() != 28 {
		t.Errorf("UncachedPromptTokens() = %d, want 28", resp.Usage.UncachedPromptTokens())
	}
	if !resp.Usage.CacheHit() {
		t.Error("CacheHit() = false")
	}
}

func TestUntouchedModelSendsNoCacheFields(t *testing.T) {
	var c capture
	srv := oaiStub(t, &c)
	defer srv.Close()

	// OpenRouter is the oaicompat endpoint with the most to send, so it is the
	// strictest check that nothing leaks onto the wire by default.
	generate(t, &OpenRouterConfig{APIKey: "k", BaseURL: srv.URL},
		NewOpenRouterModel("anthropic/claude-opus-5").WithSystemPrompt("be terse"))

	if _, ok := c.body["prompt_cache_key"]; ok {
		t.Errorf("prompt_cache_key sent by default: %v", c.body["prompt_cache_key"])
	}
	for i, msg := range c.body["messages"].([]any) {
		content := msg.(map[string]any)["content"]
		if _, ok := content.(string); !ok {
			t.Errorf("messages[%d].content = %v, want a plain string when caching was never asked for", i, content)
		}
	}
}

func TestOptedInOpenRouterRequestCarriesBreakpoints(t *testing.T) {
	var c capture
	srv := oaiStub(t, &c)
	defer srv.Close()

	generate(t, &OpenRouterConfig{APIKey: "k", BaseURL: srv.URL},
		Cached(NewOpenRouterModel("anthropic/claude-opus-5").WithSystemPrompt("be terse"),
			WithCacheTTL(CacheTTL1h), WithCachePrompt(true), WithCacheKey("tenant-7")))

	if c.body["prompt_cache_key"] != "tenant-7" {
		t.Errorf("prompt_cache_key = %v", c.body["prompt_cache_key"])
	}
	msgs := c.body["messages"].([]any)
	if len(msgs) != 2 {
		t.Fatalf("messages = %v", msgs)
	}
	for i, msg := range msgs {
		parts, ok := msg.(map[string]any)["content"].([]any)
		if !ok || len(parts) != 1 {
			t.Fatalf("messages[%d].content = %v, want one content part", i, msg.(map[string]any)["content"])
		}
		cc, ok := parts[0].(map[string]any)["cache_control"].(map[string]any)
		if !ok {
			t.Fatalf("messages[%d] carries no cache_control: %v", i, parts[0])
		}
		if cc["type"] != "ephemeral" || cc["ttl"] != "1h" {
			t.Errorf("messages[%d].cache_control = %v", i, cc)
		}
	}
}

func TestDisabledModelSendsNoCacheFields(t *testing.T) {
	var c capture
	srv := oaiStub(t, &c)
	defer srv.Close()

	// NotCached wins over a key that was set earlier: CacheModeOff suppresses
	// every cache field lingo would otherwise send.
	generate(t, &OpenRouterConfig{APIKey: "k", BaseURL: srv.URL},
		NotCached(Cached(NewOpenRouterModel("anthropic/claude-opus-5").WithSystemPrompt("be terse"),
			WithCacheKey("tenant-7"))))

	if _, ok := c.body["prompt_cache_key"]; ok {
		t.Errorf("prompt_cache_key sent for a disabled model: %v", c.body["prompt_cache_key"])
	}
	for i, msg := range c.body["messages"].([]any) {
		if _, ok := msg.(map[string]any)["content"].(string); !ok {
			t.Errorf("messages[%d].content = %v, want a plain string", i, msg.(map[string]any)["content"])
		}
	}
}

func TestOpenAIRequestCarriesThePromptCacheKey(t *testing.T) {
	var c capture
	srv := oaiStub(t, &c)
	defer srv.Close()

	// OpenAIConfig already exposes BaseURL for proxies, so the native OpenAI
	// path is reachable by an httptest server without any test-only hook.
	generate(t, &OpenAIConfig{APIKey: "k", BaseURL: srv.URL},
		Cached(NewGPT4o().WithSystemPrompt("be terse"), WithCacheKey("tenant-7")))

	if c.body["prompt_cache_key"] != "tenant-7" {
		t.Errorf("prompt_cache_key = %v", c.body["prompt_cache_key"])
	}
	// OpenAI caches prefixes on its own, so the key is the only field that may
	// change: the messages stay plain strings with no breakpoint markers.
	for i, msg := range c.body["messages"].([]any) {
		if _, ok := msg.(map[string]any)["content"].(string); !ok {
			t.Errorf("messages[%d].content = %v, want a plain string", i, msg.(map[string]any)["content"])
		}
	}

	// A key set on a model that was then disabled must not reach the wire, and
	// a model nobody touched must send nothing either.
	generate(t, &OpenAIConfig{APIKey: "k", BaseURL: srv.URL},
		NotCached(Cached(NewGPT4o(), WithCacheKey("tenant-7"))))
	if _, ok := c.body["prompt_cache_key"]; ok {
		t.Errorf("prompt_cache_key sent for a disabled model: %v", c.body["prompt_cache_key"])
	}

	generate(t, &OpenAIConfig{APIKey: "k", BaseURL: srv.URL}, NewGPT4o().WithSystemPrompt("be terse"))
	if _, ok := c.body["prompt_cache_key"]; ok {
		t.Errorf("prompt_cache_key sent by default: %v", c.body["prompt_cache_key"])
	}
}

// assertOAIBreakpoint fails unless the captured message carries exactly one
// text content part marked with an explicit prompt_cache_breakpoint.
func assertOAIBreakpoint(t *testing.T, msg any, role, text string) {
	t.Helper()
	m, ok := msg.(map[string]any)
	if !ok {
		t.Fatalf("message = %v, want an object", msg)
	}
	if m["role"] != role {
		t.Errorf("role = %v, want %q", m["role"], role)
	}
	parts, ok := m["content"].([]any)
	if !ok || len(parts) != 1 {
		t.Fatalf("content = %v, want one content part", m["content"])
	}
	part := parts[0].(map[string]any)
	if part["text"] != text {
		t.Errorf("text = %v, want %q", part["text"], text)
	}
	bp, ok := part["prompt_cache_breakpoint"].(map[string]any)
	if !ok {
		t.Fatalf("content part carries no prompt_cache_breakpoint: %v", part)
	}
	if bp["mode"] != "explicit" {
		t.Errorf("prompt_cache_breakpoint = %v, want mode explicit", bp)
	}
}

// assertOAIPlainMessages fails unless every captured message uses the plain
// string content form, which is the shape lingo has always sent.
func assertOAIPlainMessages(t *testing.T, c *capture) {
	t.Helper()
	for i, msg := range c.body["messages"].([]any) {
		content := msg.(map[string]any)["content"]
		if _, ok := content.(string); !ok {
			t.Errorf("messages[%d].content = %v, want a plain string", i, content)
		}
	}
}

// assertNoOAICacheRequestOptions guards the two request-level cache fields lingo
// deliberately never sends: prompt_cache_options.mode=explicit would switch off
// OpenAI's own implicit breakpoint, and prompt_cache_retention is deprecated and
// changes data retention for zero-data-retention organizations.
func assertNoOAICacheRequestOptions(t *testing.T, c *capture) {
	t.Helper()
	for _, field := range []string{"prompt_cache_options", "prompt_cache_retention"} {
		if _, ok := c.body[field]; ok {
			t.Errorf("%s sent: %v", field, c.body[field])
		}
	}
}

func TestOpenAIBreakpointsAreGatedToGPT56(t *testing.T) {
	var c capture
	srv := oaiStub(t, &c)
	defer srv.Close()
	cfg := func() ProviderConfig { return &OpenAIConfig{APIKey: "k", BaseURL: srv.URL} }

	// The GPT-5.6 family is the one OpenAI family documented to take an
	// explicit breakpoint, so opting in moves both halves to the array form.
	generate(t, cfg(), Cached(NewGPT56Sol().WithSystemPrompt("be terse"), WithCachePrompt(true)))

	msgs := c.body["messages"].([]any)
	if len(msgs) != 2 {
		t.Fatalf("messages = %v", msgs)
	}
	// Reasoning models keep the "developer" role they have always used.
	assertOAIBreakpoint(t, msgs[0], "developer", "be terse")
	assertOAIBreakpoint(t, msgs[1], "user", "hello")
	assertNoOAICacheRequestOptions(t, &c)

	// The gate holds under maximal opt-in: a model below the gpt-5.6 floor
	// sends exactly the request it sent before breakpoints existed.
	generate(t, cfg(), Cached(NewGPT4o().WithSystemPrompt("be terse"), WithCachePrompt(true)))
	assertOAIPlainMessages(t, &c)
	assertNoOAICacheRequestOptions(t, &c)

	// gpt-5.5 is below the floor too, even though it is a reasoning model.
	generate(t, cfg(), Cached(NewGPT55().WithSystemPrompt("be terse"), WithCachePrompt(true)))
	assertOAIPlainMessages(t, &c)

	// A GPT-5.6 model nobody touched, and one explicitly disabled, both stay
	// on the plain string form.
	generate(t, cfg(), NewGPT56Terra().WithSystemPrompt("be terse"))
	assertOAIPlainMessages(t, &c)
	assertNoOAICacheRequestOptions(t, &c)

	generate(t, cfg(), NotCached(Cached(NewGPT56Luna().WithSystemPrompt("be terse"), WithCachePrompt(true))))
	assertOAIPlainMessages(t, &c)

	// Caching the system prompt only -- the default half -- leaves the user
	// message alone, so a breakpoint lands exactly where it was asked for.
	generate(t, cfg(), Cached(NewGPT56Sol().WithSystemPrompt("be terse")))
	msgs = c.body["messages"].([]any)
	if len(msgs) != 2 {
		t.Fatalf("messages = %v", msgs)
	}
	assertOAIBreakpoint(t, msgs[0], "developer", "be terse")
	if _, ok := msgs[1].(map[string]any)["content"].(string); !ok {
		t.Errorf("user content = %v, want a plain string", msgs[1].(map[string]any)["content"])
	}

	// OpenAI fixes the breakpoint TTL at 30m, so a requested TTL is a silent
	// no-op here rather than a field on the wire.
	generate(t, cfg(), Cached(NewGPT56Sol().WithSystemPrompt("be terse"), WithCacheTTL(CacheTTL1h)))
	assertOAIBreakpoint(t, c.body["messages"].([]any)[0], "developer", "be terse")
	assertNoOAICacheRequestOptions(t, &c)
	if parts, ok := c.body["messages"].([]any)[0].(map[string]any)["content"].([]any); ok {
		if _, sent := parts[0].(map[string]any)["ttl"]; sent {
			t.Errorf("content part carries a ttl: %v", parts[0])
		}
	}
}

func TestOpenAIGenericReasoningModelDeclaresBreakpointSupport(t *testing.T) {
	var c capture
	srv := oaiStub(t, &c)
	defer srv.Close()
	cfg := func() ProviderConfig { return &OpenAIConfig{APIKey: "k", BaseURL: srv.URL} }

	// Reaching a model id lingo has no named type for is the whole point of
	// the generic types, so the caller can vouch for breakpoint support.
	generate(t, cfg(), Cached(
		NewOpenAIReasoningModel("gpt-5.7").WithExplicitPromptCache(true).WithSystemPrompt("be terse"),
		WithCachePrompt(true)))

	msgs := c.body["messages"].([]any)
	if len(msgs) != 2 {
		t.Fatalf("messages = %v", msgs)
	}
	assertOAIBreakpoint(t, msgs[0], "developer", "be terse")
	assertOAIBreakpoint(t, msgs[1], "user", "hello")
	assertNoOAICacheRequestOptions(t, &c)

	// Off by default: without the declaration the same model, fully opted into
	// caching, sends what it always sent.
	generate(t, cfg(), Cached(
		NewOpenAIReasoningModel("gpt-5.7").WithSystemPrompt("be terse"), WithCachePrompt(true)))
	assertOAIPlainMessages(t, &c)
	assertNoOAICacheRequestOptions(t, &c)

	// The declaration is revocable and does not enable caching by itself.
	generate(t, cfg(), NewOpenAIReasoningModel("gpt-5.7").
		WithExplicitPromptCache(true).WithExplicitPromptCache(false).WithSystemPrompt("be terse"))
	assertOAIPlainMessages(t, &c)

	generate(t, cfg(), NewOpenAIReasoningModel("gpt-5.7").
		WithExplicitPromptCache(true).WithSystemPrompt("be terse"))
	assertOAIPlainMessages(t, &c)

	// Standard generic models have no such switch: every non-reasoning OpenAI
	// chat model predates gpt-5.6.
	generate(t, cfg(), Cached(NewOpenAIModel("gpt-4o").WithSystemPrompt("be terse"), WithCachePrompt(true)))
	assertOAIPlainMessages(t, &c)
}

// openAIModelsBelowTheBreakpointFloor returns every OpenAI model type that must
// not carry a prompt_cache_breakpoint, each with a system prompt so a leaking
// gate would have somewhere to place one. It is the whole named catalogue minus
// the GPT-5.6 family, plus both generic types in their default state.
func openAIModelsBelowTheBreakpointFloor() []Model {
	return []Model{
		NewGPT35Turbo().WithSystemPrompt("be terse"),
		NewGPT4().WithSystemPrompt("be terse"),
		NewGPT4Turbo().WithSystemPrompt("be terse"),
		NewGPT4o().WithSystemPrompt("be terse"),
		NewGPT4oMini().WithSystemPrompt("be terse"),
		NewGPT41().WithSystemPrompt("be terse"),
		NewGPT41Mini().WithSystemPrompt("be terse"),
		NewGPT41Nano().WithSystemPrompt("be terse"),
		NewO1().WithSystemPrompt("be terse"),
		NewO1Mini().WithSystemPrompt("be terse"),
		NewO1Pro().WithSystemPrompt("be terse"),
		NewO1Preview().WithSystemPrompt("be terse"),
		NewO3().WithSystemPrompt("be terse"),
		NewO3Mini().WithSystemPrompt("be terse"),
		NewO3Pro().WithSystemPrompt("be terse"),
		NewO4Mini().WithSystemPrompt("be terse"),
		NewGPT5().WithSystemPrompt("be terse"),
		NewGPT5Mini().WithSystemPrompt("be terse"),
		NewGPT5Nano().WithSystemPrompt("be terse"),
		NewGPT5Pro().WithSystemPrompt("be terse"),
		NewGPT51().WithSystemPrompt("be terse"),
		NewGPT51Mini().WithSystemPrompt("be terse"),
		NewGPT51Nano().WithSystemPrompt("be terse"),
		NewGPT51Codex().WithSystemPrompt("be terse"),
		NewGPT51CodexMini().WithSystemPrompt("be terse"),
		NewGPT54().WithSystemPrompt("be terse"),
		NewGPT54Pro().WithSystemPrompt("be terse"),
		NewGPT54Mini().WithSystemPrompt("be terse"),
		NewGPT54Nano().WithSystemPrompt("be terse"),
		NewGPT55().WithSystemPrompt("be terse"),
		NewGPT55Pro().WithSystemPrompt("be terse"),
		NewOpenAIModel("gpt-4o").WithSystemPrompt("be terse"),
		NewOpenAIReasoningModel("gpt-5.7").WithSystemPrompt("be terse"),
	}
}

// TestOpenAIBreakpointGateCoversEveryModelType is the exhaustive half of the
// gate: the wire tests above can only afford a few representatives, but the
// request path consults exactly one predicate, so sweeping every type through
// that predicate covers the whole catalogue. A new named type that picks up the
// marker method by accident fails here rather than at a customer's 400.
func TestOpenAIBreakpointGateCoversEveryModelType(t *testing.T) {
	for _, m := range openAIModelsBelowTheBreakpointFloor() {
		if openAISupportsExplicitCache(m) {
			t.Errorf("%s declares breakpoint support, which is gated to gpt-5.6 and later", m.ModelName())
		}
	}

	for _, m := range []Model{
		NewGPT56Sol(),
		NewGPT56Terra(),
		NewGPT56Luna(),
		NewOpenAIReasoningModel("gpt-5.7").WithExplicitPromptCache(true),
	} {
		if !openAISupportsExplicitCache(m) {
			t.Errorf("%s = false, want the breakpoint gate open", m.ModelName())
		}
	}

	// The predicate takes a Model, so models from every other provider reach it
	// too. They have to fall through the assertion rather than panic.
	for _, m := range []Model{NewClaudeSonnet5(), NewGemini3Pro(), NewCommandAPlus(), NewBedrockNovaPro()} {
		if openAISupportsExplicitCache(m) {
			t.Errorf("%s/%s declares OpenAI breakpoint support", m.Provider(), m.ModelName())
		}
	}
}

// TestOpenAIModelsBelowTheFloorStayPlainUnderMaximalOptIn drives the whole
// pre-5.6 catalogue through the real request path with every cache option set
// at once. Nothing but prompt_cache_key may change, and the message content has
// to stay the plain string form lingo has always sent.
func TestOpenAIModelsBelowTheFloorStayPlainUnderMaximalOptIn(t *testing.T) {
	var c capture
	srv := oaiStub(t, &c)
	defer srv.Close()

	g, err := New([]ProviderConfig{&OpenAIConfig{APIKey: "k", BaseURL: srv.URL}})
	if err != nil {
		t.Fatalf("gateway: %v", err)
	}
	defer g.Close()

	for _, m := range openAIModelsBelowTheBreakpointFloor() {
		m := Cached(m, WithCachePrompt(true), WithCacheSystemPrompt(true),
			WithCacheTTL(CacheTTL1h), WithCacheKey("tenant-7"))
		t.Run(m.ModelName(), func(t *testing.T) {
			if _, err := g.Generate(context.Background(), m, "hello"); err != nil {
				t.Fatalf("generate: %v", err)
			}
			assertOAIPlainMessages(t, &c)
			assertNoOAICacheRequestOptions(t, &c)
			// The routing key is the one field these models do model, so it is
			// the control that proves the opt-in reached the provider at all.
			if c.body["prompt_cache_key"] != "tenant-7" {
				t.Errorf("prompt_cache_key = %v, want the opt-in to have been seen", c.body["prompt_cache_key"])
			}
		})
	}
}

// ============================================================================
// WIRE BEHAVIOUR: AZURE
// ============================================================================
//
// Azure is usage-only with exactly one request-side field, and it exists only
// on the v1 API surface: no dated api-version models prompt_cache_key, so
// sending it there is a 400 rather than the silent no-op the caching contract
// promises. providers_test.go pins which URL each api-version routes to; this
// pins what the body is allowed to carry on each of them.

func TestAzureCacheKeyIsGatedToTheV1Surface(t *testing.T) {
	for _, route := range []struct {
		name       string
		apiVersion string
		path       string
		v1         bool
	}{
		{"unset", "", "/openai/deployments/my-deploy/chat/completions", false},
		{"dated default", AzureAPIVersionDefault, "/openai/deployments/my-deploy/chat/completions", false},
		{"older dated", "2023-05-15", "/openai/deployments/my-deploy/chat/completions", false},
		{"v1", AzureAPIVersionV1, "/openai/v1/chat/completions", true},
		{"v1 preview", AzureAPIVersionV1Preview, "/openai/v1/chat/completions", true},
	} {
		t.Run(route.name, func(t *testing.T) {
			var c capture
			srv := oaiStub(t, &c)
			defer srv.Close()

			cfg := func() ProviderConfig {
				return &AzureOpenAIConfig{Endpoint: srv.URL, APIKey: "azkey", APIVersion: route.apiVersion}
			}
			deployment := func() *AzureOpenAIModel {
				return NewAzureOpenAIModel("my-deploy").WithSystemPrompt("be terse")
			}

			// Untouched: the request lingo has always sent, on every surface.
			generate(t, cfg(), deployment())
			if c.path != route.path {
				t.Errorf("path = %q, want %q", c.path, route.path)
			}
			if _, ok := c.body["prompt_cache_key"]; ok {
				t.Errorf("prompt_cache_key sent by default: %v", c.body["prompt_cache_key"])
			}
			assertOAIPlainMessages(t, &c)

			// Opted in with everything the surface could possibly take. Only the
			// key is modelled, and only on v1; the TTL and both breakpoints are
			// silent no-ops, because Azure decides for itself what to cache.
			generate(t, cfg(), Cached(deployment(),
				WithCacheKey("tenant:acme"), WithCacheTTL(CacheTTL1h), WithCachePrompt(true)))

			want := ""
			if route.v1 {
				want = "tenant:acme"
			}
			if got, _ := c.body["prompt_cache_key"].(string); got != want {
				t.Errorf("prompt_cache_key = %q, want %q", got, want)
			}
			if c.path != route.path {
				t.Errorf("opting in moved the request to %q", c.path)
			}
			assertOAIPlainMessages(t, &c)
			assertNoOAICacheRequestOptions(t, &c)

			// Disabling suppresses the key even where the surface accepts it.
			generate(t, cfg(), NotCached(Cached(deployment(), WithCacheKey("tenant:acme"))))
			if _, ok := c.body["prompt_cache_key"]; ok {
				t.Errorf("prompt_cache_key sent for a disabled model: %v", c.body["prompt_cache_key"])
			}
		})
	}
}

func TestAzureCacheUsageIsReportedOnEveryRoute(t *testing.T) {
	// Reporting is unconditional and independent of the request-side gate: the
	// dated route sends no cache field at all and still reads the counters back.
	for _, apiVersion := range []string{AzureAPIVersionDefault, AzureAPIVersionV1} {
		t.Run(apiVersion, func(t *testing.T) {
			var c capture
			srv := cacheUsageStub(t, &c)
			defer srv.Close()

			resp := generate(t, &AzureOpenAIConfig{Endpoint: srv.URL, APIKey: "azkey", APIVersion: apiVersion},
				NewAzureOpenAIModel("my-deploy"))

			if resp.Usage.CacheReadTokens != 64 || resp.Usage.CacheWriteTokens != 8 {
				t.Errorf("cache counters = %+v", resp.Usage)
			}
			// OpenAI-shaped counters are a breakdown of prompt_tokens, so the
			// totals stay exactly as reported.
			if resp.Usage.PromptTokens != 100 || resp.Usage.TotalTokens != 107 {
				t.Errorf("subset counters must not inflate the totals: %+v", resp.Usage)
			}
			if resp.Usage.UncachedPromptTokens() != 28 {
				t.Errorf("UncachedPromptTokens() = %d, want 28", resp.Usage.UncachedPromptTokens())
			}
		})
	}
}

// ============================================================================
// WIRE BEHAVIOUR: ANTHROPIC
// ============================================================================

// anthropicStub serves a canned Messages response whose usage carries both
// cache counters, records the request, and points the pinned anthropic-sdk-go
// at itself through ANTHROPIC_BASE_URL, which the SDK reads when the client is
// constructed. AnthropicConfig has no BaseURL field, so this env var is the
// only seam that does not mean adding production API surface for a test.
func anthropicStub(t *testing.T, c *capture) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		c.record(r, raw)

		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"msg_1","type":"message","role":"assistant",
			"model":"claude-sonnet-5","stop_reason":"end_turn",
			"content":[{"type":"text","text":"hi there"}],
			"usage":{"input_tokens":10,"output_tokens":7,
				"cache_read_input_tokens":64,"cache_creation_input_tokens":32}}`)
	}))
	t.Setenv("ANTHROPIC_BASE_URL", srv.URL)
	return srv
}

func TestOptedInAnthropicRequestCarriesBreakpoints(t *testing.T) {
	var c capture
	srv := anthropicStub(t, &c)
	defer srv.Close()

	resp := generate(t, &AnthropicConfig{APIKey: "k"},
		Cached(NewClaudeSonnet5().WithSystemPrompt("be terse"),
			WithCacheTTL(CacheTTL1h), WithCachePrompt(true)))

	if c.path != "/v1/messages" {
		t.Errorf("path = %q", c.path)
	}

	// The breakpoint goes on the last system block and the last text block of
	// the last message, which is what makes the prefix before it cacheable.
	sys := c.body["system"].([]any)
	if len(sys) != 1 {
		t.Fatalf("system = %v", c.body["system"])
	}
	cc, ok := sys[0].(map[string]any)["cache_control"].(map[string]any)
	if !ok {
		t.Fatalf("system block carries no cache_control: %v", sys[0])
	}
	if cc["type"] != "ephemeral" || cc["ttl"] != "1h" {
		t.Errorf("system cache_control = %v", cc)
	}

	msgs := c.body["messages"].([]any)
	if len(msgs) != 1 {
		t.Fatalf("messages = %v", msgs)
	}
	blocks := msgs[0].(map[string]any)["content"].([]any)
	cc, ok = blocks[len(blocks)-1].(map[string]any)["cache_control"].(map[string]any)
	if !ok {
		t.Fatalf("last user block carries no cache_control: %v", blocks[len(blocks)-1])
	}
	if cc["type"] != "ephemeral" || cc["ttl"] != "1h" {
		t.Errorf("user cache_control = %v", cc)
	}

	// Anthropic reports the counters alongside input_tokens rather than inside
	// it, so both are folded into the prompt and the total.
	if resp.Usage.CacheReadTokens != 64 || resp.Usage.CacheWriteTokens != 32 {
		t.Errorf("cache counters = %+v", resp.Usage)
	}
	if resp.Usage.PromptTokens != 106 || resp.Usage.TotalTokens != 113 {
		t.Errorf("additive counters must grow the totals: %+v", resp.Usage)
	}
}

func TestUntouchedAnthropicModelSendsNoCacheControl(t *testing.T) {
	var c capture
	srv := anthropicStub(t, &c)
	defer srv.Close()

	generate(t, &AnthropicConfig{APIKey: "k"}, NewClaudeSonnet5().WithSystemPrompt("be terse"))

	for i, block := range c.body["system"].([]any) {
		if _, ok := block.(map[string]any)["cache_control"]; ok {
			t.Errorf("system[%d] carries cache_control by default: %v", i, block)
		}
	}
	for i, block := range c.body["messages"].([]any)[0].(map[string]any)["content"].([]any) {
		if _, ok := block.(map[string]any)["cache_control"]; ok {
			t.Errorf("user block %d carries cache_control by default: %v", i, block)
		}
	}

	// NotCached on a model whose breakpoints were requested earlier is the same
	// wire request as never having asked.
	generate(t, &AnthropicConfig{APIKey: "k"},
		NotCached(Cached(NewClaudeSonnet5().WithSystemPrompt("be terse"), WithCachePrompt(true))))

	if _, ok := c.body["system"].([]any)[0].(map[string]any)["cache_control"]; ok {
		t.Errorf("system carries cache_control for a disabled model: %v", c.body["system"])
	}
	for i, block := range c.body["messages"].([]any)[0].(map[string]any)["content"].([]any) {
		if _, ok := block.(map[string]any)["cache_control"]; ok {
			t.Errorf("user block %d carries cache_control for a disabled model: %v", i, block)
		}
	}
}

// ============================================================================
// WIRE BEHAVIOUR: BEDROCK CLAUDE
// ============================================================================
//
// Bedrock is the second explicit lane and the only one that hand-rolls its JSON,
// so its body is asserted directly off the builder: buildClaudeRequest is pure
// and needs no AWS client. The untouched case is the regression guard that
// matters most -- the request fields are typed `any` so a breakpoint can widen
// them into content blocks, and `omitempty` on an interface only tests nil, so
// nothing but this test would notice the shape changing.

func TestBedrockClaudeRequestBodies(t *testing.T) {
	c := &bedrockClient{}

	build := func(t *testing.T, model Model) (map[string]any, bool, string) {
		t.Helper()
		raw, breakpoint, _, err := c.buildClaudeRequest(model, "hi")
		if err != nil {
			t.Fatal(err)
		}
		var body map[string]any
		if err := json.Unmarshal(raw, &body); err != nil {
			t.Fatal(err)
		}
		return body, breakpoint, string(raw)
	}

	t.Run("untouched model keeps the plain string shape", func(t *testing.T) {
		body, breakpoint, _ := build(t, NewBedrockClaudeSonnet5().WithSystemPrompt("be terse"))
		if breakpoint {
			t.Error("a cache breakpoint was reported for a model nobody opted in")
		}
		if body["system"] != "be terse" {
			t.Errorf("system = %#v, want the plain string it has always been", body["system"])
		}
		msg := body["messages"].([]any)[0].(map[string]any)
		if msg["content"] != "hi" {
			t.Errorf("content = %#v, want the plain string it has always been", msg["content"])
		}
	})

	t.Run("no system prompt places nothing", func(t *testing.T) {
		// Enable() defaults to the system prompt, so a model without one has no
		// prefix to mark: the body must be indistinguishable from untouched.
		body, breakpoint, raw := build(t, Cached(NewBedrockClaudeSonnet5()))
		if breakpoint {
			t.Error("caching was reported as placed with no system prompt to place it on")
		}
		if _, ok := body["system"]; ok {
			t.Errorf("system = %#v, want the field omitted entirely", body["system"])
		}
		if _, _, plain := build(t, NewBedrockClaudeSonnet5()); raw != plain {
			t.Errorf("opted-in body %s differs from the untouched body %s", raw, plain)
		}
	})

	t.Run("opting in marks the system prompt", func(t *testing.T) {
		body, breakpoint, _ := build(t, Cached(
			NewBedrockClaudeSonnet5().WithSystemPrompt("be terse"), WithCacheTTL(CacheTTL1h)))
		if !breakpoint {
			t.Error("a placed cache breakpoint went unreported")
		}
		blocks, ok := body["system"].([]any)
		if !ok || len(blocks) != 1 {
			t.Fatalf("system = %#v, want one content block", body["system"])
		}
		block := blocks[0].(map[string]any)
		if block["type"] != "text" || block["text"] != "be terse" {
			t.Errorf("system block = %v", block)
		}
		cc, ok := block["cache_control"].(map[string]any)
		if !ok || cc["type"] != "ephemeral" || cc["ttl"] != "1h" {
			t.Errorf("cache_control = %#v, want ephemeral at 1h", block["cache_control"])
		}
		// Only the system prompt was asked for, so the message stays a string.
		if msg := body["messages"].([]any)[0].(map[string]any); msg["content"] != "hi" {
			t.Errorf("content = %#v, want the plain string when only the system prompt is cached", msg["content"])
		}
	})
}

// ============================================================================
// WIRE BEHAVIOUR: BEDROCK ROUTING
// ============================================================================
//
// Nova moved to the Converse API; every other family stayed on InvokeModel and
// therefore has to prove it still sends the exact bytes it always did. The
// builder tests above cannot show that, because the routing decision lives in
// Generate and is only visible as the URL the request lands on.
//
// BedrockConfig deliberately exposes no base URL, so the seam is the AWS SDK's
// own BaseEndpoint option applied to a client this package constructs directly.
// Nothing is added to the public API for the test, and unlike a builder probe
// this observes the request after the SDK has serialized it.

type bedrockCall struct {
	path string
	body string // raw, so bodies can be compared byte for byte
}

// bedrockCalls is the request log a Bedrock stub writes and its test reads.
//
// It owns a mutex because the two ends run on different goroutines: an httptest
// server handles each in-flight request on its own goroutine, so two
// overlapping requests appending to a bare slice would race, and a racing
// append does not merely upset the detector -- it drops records, which turns a
// concurrency test into a flake that blames the wrong code. Reads lock too, so
// a test may look while requests are still in flight.
type bedrockCalls struct {
	mu    sync.Mutex
	calls []bedrockCall
}

func (c *bedrockCalls) add(call bedrockCall) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.calls = append(c.calls, call)
}

func (c *bedrockCalls) len() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.calls)
}

// at returns one recorded call. Out of range is a fatal test error rather than
// a panic, so a test that recorded fewer calls than it expected says which.
func (c *bedrockCalls) at(t *testing.T, i int) bedrockCall {
	t.Helper()
	c.mu.Lock()
	defer c.mu.Unlock()
	if i >= len(c.calls) {
		t.Fatalf("call %d was never recorded: the stub saw %d request(s)", i, len(c.calls))
	}
	return c.calls[i]
}

// all returns a copy, so ranging over the log cannot read the slice a handler
// is still appending to.
func (c *bedrockCalls) all() []bedrockCall {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]bedrockCall(nil), c.calls...)
}

// bedrockStub answers the InvokeModel and Converse endpoints with a canned
// response per family and records every request.
// extraHeaders, when given, are set on every response: the InvokeModel
// token-count headers are not part of any response body, so a test that needs
// them has nowhere else to put them.
func bedrockStub(t *testing.T, calls *bedrockCalls, extraHeaders ...map[string]string) *bedrockClient {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		calls.add(bedrockCall{path: r.URL.Path, body: string(raw)})

		w.Header().Set("Content-Type", "application/json")
		for _, hs := range extraHeaders {
			for k, v := range hs {
				w.Header().Set(k, v)
			}
		}
		switch {
		case strings.HasSuffix(r.URL.Path, "/converse"):
			_, _ = io.WriteString(w, `{
				"output":{"message":{"role":"assistant","content":[{"text":"hi there"}]}},
				"stopReason":"end_turn",
				"usage":{"inputTokens":100,"outputTokens":7,"totalTokens":1007,
					"cacheReadInputTokens":900,"cacheWriteInputTokens":0}}`)
		case strings.Contains(r.URL.Path, "anthropic."):
			_, _ = io.WriteString(w, `{"id":"msg_1","type":"message","role":"assistant",
				"stop_reason":"end_turn","content":[{"type":"text","text":"hi there"}],
				"usage":{"input_tokens":10,"output_tokens":7}}`)
		case strings.Contains(r.URL.Path, "meta."):
			_, _ = io.WriteString(w, `{"generation":"hi there","stop_reason":"stop",
				"prompt_token_count":10,"generation_token_count":7}`)
		case strings.Contains(r.URL.Path, "mistral."):
			_, _ = io.WriteString(w, `{"outputs":[{"text":"hi there","stop_reason":"stop"}]}`)
		default:
			// amazon.*, which at this point can only be Titan: Nova is served by
			// the Converse arm above.
			_, _ = io.WriteString(w, `{"results":[{"outputText":"hi there","completionReason":"FINISH","tokenCount":7}]}`)
		}
	}))
	return bedrockClientFor(t, srv)
}

// bedrockClientFor points a real bedrockClient at srv. It is split out of
// bedrockStub so a test that has to serve its own canned response body gets the
// same client the rest of them use -- same middleware registration, same
// credentials -- instead of reassembling one and quietly diverging from it.
func bedrockClientFor(t *testing.T, srv *httptest.Server) *bedrockClient {
	t.Helper()
	t.Cleanup(srv.Close)

	awsCfg := aws.Config{
		Region:      "us-east-1",
		Credentials: credentials.NewStaticCredentialsProvider("id", "secret", ""),
	}
	return &bedrockClient{
		client: newBedrockRuntimeClient(awsCfg, func(o *bedrockruntime.Options) {
			o.BaseEndpoint = aws.String(srv.URL)
		}),
		timeout:     defaultTimeout(),
		logger:      &NopLogger{},
		rateLimiter: newRateLimiter(nil, &NopLogger{}),
	}
}

// TestBedrockInvokeModelFamiliesAreUntouchedByConverse is the blast-radius
// guard for routing Nova through Converse. Each row sends several models that
// must all produce one identical request: the untouched one, the maximally
// opted-in one, and the explicitly disabled one. The recorded body is compared
// against a golden literal as well, so a change in the InvokeModel bodies is
// caught even if it happens to affect all three the same way.
func TestBedrockInvokeModelFamiliesAreUntouchedByConverse(t *testing.T) {
	for _, tc := range []struct {
		name   string
		path   string
		body   string
		models []Model
	}{{
		// Claude is the one InvokeModel family that does place a breakpoint, so
		// its opted-in row is the case where there is nothing to place. The body
		// that does change is pinned by TestBedrockClaudeRequestBodies above.
		name: "claude",
		path: "/model/anthropic.claude-sonnet-5/invoke",
		body: `{"anthropic_version":"bedrock-2023-05-31","max_tokens":8192,"messages":[{"role":"user","content":"hello"}]}`,
		models: []Model{
			NewBedrockClaudeSonnet5(),
			Cached(NewBedrockClaudeSonnet5(), WithCacheTTL(CacheTTL1h)),
			NotCached(Cached(NewBedrockClaudeSonnet5(), WithCachePrompt(true))),
		},
	}, {
		name: "claude with a system prompt",
		path: "/model/anthropic.claude-sonnet-5/invoke",
		body: `{"anthropic_version":"bedrock-2023-05-31","max_tokens":8192,"messages":[{"role":"user","content":"hello"}],"system":"be terse"}`,
		models: []Model{
			NewBedrockClaudeSonnet5().WithSystemPrompt("be terse"),
			NotCached(Cached(NewBedrockClaudeSonnet5().WithSystemPrompt("be terse"),
				WithCachePrompt(true), WithCacheTTL(CacheTTL1h))),
		},
	}, {
		// Titan, Llama and Mistral model no caching at all, so even maximal
		// opt-in has to leave their bodies alone.
		name: "titan",
		path: "/model/amazon.titan-text-premier-v1:0/invoke",
		body: `{"inputText":"be terse\n\nhello","textGenerationConfig":{"maxTokenCount":4096,"temperature":0.7,"topP":0.9}}`,
		models: []Model{
			NewBedrockTitanTextPremier().WithSystemPrompt("be terse"),
			Cached(NewBedrockTitanTextPremier().WithSystemPrompt("be terse"),
				WithCachePrompt(true), WithCacheTTL(CacheTTL1h), WithCacheKey("tenant-7")),
			NotCached(NewBedrockTitanTextPremier().WithSystemPrompt("be terse")),
		},
	}, {
		// The < escapes are what encoding/json emits for the prompt
		// template's angle brackets, so the golden is the literal wire bytes.
		name: "llama",
		path: "/model/meta.llama3-3-70b-instruct-v1:0/invoke",
		body: `{"prompt":"\u003c|begin_of_text|\u003e\u003c|start_header_id|\u003esystem\u003c|end_header_id|\u003e\n\nbe terse\u003c|eot_id|\u003e\u003c|start_header_id|\u003euser\u003c|end_header_id|\u003e\n\nhello\u003c|eot_id|\u003e\u003c|start_header_id|\u003eassistant\u003c|end_header_id|\u003e\n\n","max_gen_len":2048,"temperature":0.6,"top_p":0.9}`,
		models: []Model{
			NewBedrockLlama33Instruct70B().WithSystemPrompt("be terse"),
			Cached(NewBedrockLlama33Instruct70B().WithSystemPrompt("be terse"),
				WithCachePrompt(true), WithCacheTTL(CacheTTL1h), WithCacheKey("tenant-7")),
			NotCached(NewBedrockLlama33Instruct70B().WithSystemPrompt("be terse")),
		},
	}, {
		name: "mistral",
		path: "/model/mistral.mistral-large-2402-v1:0/invoke",
		body: `{"prompt":"\u003cs\u003e[INST] be terse\n\nhello [/INST]","max_tokens":8192,"temperature":0.7,"top_p":0.9}`,
		models: []Model{
			NewBedrockMistralLarge().WithSystemPrompt("be terse"),
			Cached(NewBedrockMistralLarge().WithSystemPrompt("be terse"),
				WithCachePrompt(true), WithCacheTTL(CacheTTL1h), WithCacheKey("tenant-7")),
			NotCached(NewBedrockMistralLarge().WithSystemPrompt("be terse")),
		},
	}} {
		t.Run(tc.name, func(t *testing.T) {
			var calls bedrockCalls
			c := bedrockStub(t, &calls)

			for _, m := range tc.models {
				if _, err := c.Generate(context.Background(), m, "hello"); err != nil {
					t.Fatalf("generate: %v", err)
				}
			}
			if calls.len() != len(tc.models) {
				t.Fatalf("calls = %d, want %d", calls.len(), len(tc.models))
			}
			for i, call := range calls.all() {
				if call.path != tc.path {
					t.Errorf("call %d hit %q, want the InvokeModel path %q", i, call.path, tc.path)
				}
				if call.body != tc.body {
					t.Errorf("call %d body =\n\t%s\nwant\n\t%s", i, call.body, tc.body)
				}
			}
		})
	}
}

// TestBedrockNamedModelsRouteByFamily walks the entire named catalogue through
// the real Generate path. Exactly the four Nova models may reach Converse; a
// model that changes family, or a predicate that widens, shows up here as a
// request landing on the wrong endpoint.
func TestBedrockNamedModelsRouteByFamily(t *testing.T) {
	var calls bedrockCalls
	c := bedrockStub(t, &calls)

	models := []Model{
		NewBedrockClaude35Sonnet(), NewBedrockClaude35Haiku(), NewBedrockClaude37Sonnet(),
		NewBedrockClaudeSonnet4(), NewBedrockClaudeOpus4(), NewBedrockClaudeSonnet45(),
		NewBedrockClaudeOpus45(), NewBedrockClaudeHaiku45(), NewBedrockClaudeOpus46(),
		NewBedrockClaudeSonnet46(), NewBedrockClaudeOpus47(), NewBedrockClaudeOpus48(),
		NewBedrockClaudeFable5(), NewBedrockClaudeOpus5(), NewBedrockClaudeSonnet5(),
		NewBedrockClaude3Sonnet(), NewBedrockClaude3Haiku(), NewBedrockClaude3Opus(),
		NewBedrockTitanTextExpress(), NewBedrockTitanTextLite(), NewBedrockTitanTextPremier(),
		NewBedrockNovaMicro(), NewBedrockNovaLite(), NewBedrockNovaPro(), NewBedrockNovaPremier(),
		NewBedrockLlama31Instruct8B(), NewBedrockLlama31Instruct70B(), NewBedrockLlama31Instruct405B(),
		NewBedrockLlama32Instruct1B(), NewBedrockLlama32Instruct3B(), NewBedrockLlama33Instruct70B(),
		NewBedrockLlama4Scout(), NewBedrockLlama4Maverick(),
		NewBedrockMistral7B(), NewBedrockMixtral8x7B(), NewBedrockMistralLarge(), NewBedrockMistralLarge2407(),
		// The generic model routes on the family the caller declared rather than
		// on the id, which is the one way a non-Nova id can reach Converse.
		NewBedrockModel("us.anthropic.claude-opus-5", "claude"),
		NewBedrockModel("eu.amazon.nova-pro-v1:0", "nova"),
	}

	var converse int
	for i, m := range models {
		if _, err := c.Generate(context.Background(), m, "hello"); err != nil {
			t.Fatalf("%s: %v", m.ModelName(), err)
		}
		call := calls.at(t, i)
		wantInvoke := "/model/" + m.ModelName() + "/invoke"
		wantConverse := "/model/" + m.ModelName() + "/converse"
		switch call.path {
		case wantConverse:
			converse++
			if !strings.Contains(m.ModelName(), "nova") {
				t.Errorf("%s reached Converse; only Nova may", m.ModelName())
			}
		case wantInvoke:
			if strings.Contains(m.ModelName(), "nova") {
				t.Errorf("%s stayed on InvokeModel; Nova is served by Converse", m.ModelName())
			}
		default:
			t.Errorf("%s hit %q, want %q or %q", m.ModelName(), call.path, wantInvoke, wantConverse)
		}
	}
	if converse != 5 {
		t.Errorf("%d requests reached Converse, want 5 (four named Nova models plus the generic one)", converse)
	}
}

// TestBedrockNovaConverseWireShape pins what Converse actually serializes. The
// builder tests in bedrock_test.go assert the input struct; this asserts that
// the cache point survives the SDK and that the counters come back normalized.
func TestBedrockNovaConverseWireShape(t *testing.T) {
	var calls bedrockCalls
	c := bedrockStub(t, &calls)

	// Untouched first: the request a Nova model has to send when nobody asked
	// for caching, on the endpoint it now uses.
	if _, err := c.Generate(context.Background(),
		NewBedrockNovaPro().WithSystemPrompt("be terse"), "hello"); err != nil {
		t.Fatalf("generate: %v", err)
	}
	if calls.at(t, 0).path != "/model/amazon.nova-pro-v1:0/converse" {
		t.Errorf("path = %q", calls.at(t, 0).path)
	}
	if strings.Contains(calls.at(t, 0).body, "cachePoint") {
		t.Errorf("cache point sent without opting in: %s", calls.at(t, 0).body)
	}

	resp, err := c.Generate(context.Background(),
		Cached(NewBedrockNovaPro().WithSystemPrompt("be terse"), WithCachePrompt(true), WithCacheTTL(CacheTTL1h)),
		"hello")
	if err != nil {
		t.Fatalf("generate: %v", err)
	}

	var body struct {
		System []struct {
			Text       string `json:"text"`
			CachePoint *struct {
				Type string `json:"type"`
				TTL  string `json:"ttl"`
			} `json:"cachePoint"`
		} `json:"system"`
		Messages []struct {
			Content []struct {
				Text       string          `json:"text"`
				CachePoint json.RawMessage `json:"cachePoint"`
			} `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal([]byte(calls.at(t, 1).body), &body); err != nil {
		t.Fatalf("converse body %s: %v", calls.at(t, 1).body, err)
	}

	if len(body.System) != 2 || body.System[0].Text != "be terse" {
		t.Fatalf("system = %s, want the text block plus a trailing cache point", calls.at(t, 1).body)
	}
	cp := body.System[1].CachePoint
	if cp == nil || cp.Type != "default" {
		t.Fatalf("system[1] = %s, want a default cache point", calls.at(t, 1).body)
	}
	// Nova documents a five minute lifetime only, so the requested 1h is clamped
	// away rather than sent and rejected.
	if cp.TTL != "" {
		t.Errorf("cache point ttl = %q, want it clamped away", cp.TTL)
	}
	last := body.Messages[0].Content
	if len(last) != 2 || last[1].CachePoint == nil {
		t.Errorf("message content = %s, want a trailing cache point", calls.at(t, 1).body)
	}

	// Converse reports the cache read alongside inputTokens, so PromptTokens has
	// to cover the whole effective prompt with the counter as a subset of it.
	if resp.Usage.PromptTokens != 1000 || resp.Usage.CacheReadTokens != 900 {
		t.Errorf("Usage = %+v, want PromptTokens 1000 and CacheReadTokens 900", resp.Usage)
	}
	if resp.Usage.TotalTokens != 1007 {
		t.Errorf("TotalTokens = %d, want 1007, not a double count", resp.Usage.TotalTokens)
	}
	if resp.Usage.UncachedPromptTokens() != 100 || !resp.Usage.CacheHit() {
		t.Errorf("uncached = %d, hit = %t", resp.Usage.UncachedPromptTokens(), resp.Usage.CacheHit())
	}
	if resp.Text != "hi there" || resp.FinishReason != "end_turn" {
		t.Errorf("response = %q / %q", resp.Text, resp.FinishReason)
	}
}

// ============================================================================
// WIRE BEHAVIOUR: GOOGLE
// ============================================================================

// geminiStub serves a canned generateContent response whose usageMetadata
// carries a cached-token count, records the request, and points the pinned
// genai SDK at itself through GOOGLE_GEMINI_BASE_URL, which the SDK resolves
// when the client is constructed. GoogleConfig has no BaseURL field, so this
// env var is what makes the Gemini request body observable offline.
func geminiStub(t *testing.T, c *capture) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		c.record(r, raw)

		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"candidates":[{"finishReason":"STOP",
				"content":{"role":"model","parts":[{"text":"hi there"}]}}],
			"usageMetadata":{"promptTokenCount":100,"candidatesTokenCount":7,
				"totalTokenCount":107,"cachedContentTokenCount":64}}`)
	}))
	t.Setenv("GOOGLE_GEMINI_BASE_URL", srv.URL)
	return srv
}

func TestGoogleCachedContentReplacesTheSystemInstruction(t *testing.T) {
	var c capture
	srv := geminiStub(t, &c)
	defer srv.Close()

	generate(t, &GoogleConfig{APIKey: "k"},
		Cached(NewGemini3Pro().WithSystemPrompt("be terse"),
			WithCachedContent("cachedContents/1234")))

	if c.path != "/v1beta/models/gemini-3-pro-preview:generateContent" {
		t.Errorf("path = %q", c.path)
	}
	if c.body["cachedContent"] != "cachedContents/1234" {
		t.Errorf("cachedContent = %v", c.body["cachedContent"])
	}
	// Gemini answers 400 to a request that carries both, because the cache
	// resource already owns the system instruction it was created with. The
	// model's own system prompt has to be dropped, not sent alongside.
	if si, ok := c.body["systemInstruction"]; ok {
		t.Errorf("systemInstruction sent alongside cachedContent: %v", si)
	}
}

func TestGoogleSystemInstructionSurvivesWithoutCaching(t *testing.T) {
	var c capture
	srv := geminiStub(t, &c)
	defer srv.Close()

	generate(t, &GoogleConfig{APIKey: "k"}, NewGemini3Pro().WithSystemPrompt("be terse"))

	si, ok := c.body["systemInstruction"].(map[string]any)
	if !ok {
		t.Fatalf("systemInstruction = %v, want the system prompt on an uncached request", c.body["systemInstruction"])
	}
	parts := si["parts"].([]any)
	if len(parts) != 1 || parts[0].(map[string]any)["text"] != "be terse" {
		t.Errorf("systemInstruction.parts = %v", si["parts"])
	}
	if cc, ok := c.body["cachedContent"]; ok {
		t.Errorf("cachedContent sent by default: %v", cc)
	}
}

func TestGoogleDisabledModelSendsNoCachedContent(t *testing.T) {
	var c capture
	srv := geminiStub(t, &c)
	defer srv.Close()

	generate(t, &GoogleConfig{APIKey: "k"},
		NotCached(Cached(NewGemini3Pro(), WithCachedContent("cachedContents/1234"))))

	if cc, ok := c.body["cachedContent"]; ok {
		t.Errorf("cachedContent sent for a disabled model: %v", cc)
	}
	if si, ok := c.body["systemInstruction"]; ok {
		t.Errorf("systemInstruction = %v, want none: the model carries no system prompt", si)
	}

	// Suppressing the cache resource hands the system prompt back rather than
	// leaving the request with neither: disabling is a full revert.
	generate(t, &GoogleConfig{APIKey: "k"},
		NotCached(Cached(NewGemini3Pro().WithSystemPrompt("be terse"),
			WithCachedContent("cachedContents/1234"))))

	if cc, ok := c.body["cachedContent"]; ok {
		t.Errorf("cachedContent sent for a disabled model: %v", cc)
	}
	si, ok := c.body["systemInstruction"].(map[string]any)
	if !ok {
		t.Fatalf("systemInstruction = %v, want it restored once the cache is off", c.body["systemInstruction"])
	}
	if parts := si["parts"].([]any); parts[0].(map[string]any)["text"] != "be terse" {
		t.Errorf("systemInstruction.parts = %v", si["parts"])
	}
}

func TestGoogleCachedTokensAreASubsetOfThePromptTotal(t *testing.T) {
	var c capture
	srv := geminiStub(t, &c)
	defer srv.Close()

	resp := generate(t, &GoogleConfig{APIKey: "k"},
		Cached(NewGemini3Pro(), WithCachedContent("cachedContents/1234")))

	if resp.Usage.CacheReadTokens != 64 {
		t.Errorf("CacheReadTokens = %d, want 64", resp.Usage.CacheReadTokens)
	}
	// Gemini counts cachedContentTokenCount inside promptTokenCount, so the
	// totals must stay exactly as reported. There is no cache-write counter.
	if resp.Usage.PromptTokens != 100 || resp.Usage.TotalTokens != 107 {
		t.Errorf("subset counters must not inflate the totals: %+v", resp.Usage)
	}
	if resp.Usage.CacheWriteTokens != 0 {
		t.Errorf("CacheWriteTokens = %d, want 0: Gemini reports no write counter", resp.Usage.CacheWriteTokens)
	}
	if resp.Usage.UncachedPromptTokens() != 36 {
		t.Errorf("UncachedPromptTokens() = %d, want 36", resp.Usage.UncachedPromptTokens())
	}
	if !resp.Usage.CacheHit() {
		t.Error("CacheHit() = false")
	}
}

// ============================================================================
// WIRE BEHAVIOUR: GOOGLE CACHE RESOURCES
// ============================================================================

// cacheCall is one recorded request to the CachedContents endpoints. The method
// matters here in a way it does not for generateContent: create, refresh and
// delete differ only by verb on nearly the same path.
type cacheCall struct {
	method string
	path   string
	body   map[string]any
}

// cacheCalls is the request log the CachedContents stubs write, locked for the
// same reason bedrockCalls is: the handler goroutine and the test are not the
// same goroutine.
type cacheCalls struct {
	mu    sync.Mutex
	calls []cacheCall
}

func (c *cacheCalls) add(call cacheCall) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.calls = append(c.calls, call)
}

func (c *cacheCalls) len() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.calls)
}

func (c *cacheCalls) at(t *testing.T, i int) cacheCall {
	t.Helper()
	c.mu.Lock()
	defer c.mu.Unlock()
	if i >= len(c.calls) {
		t.Fatalf("call %d was never recorded: the stub saw %d request(s)", i, len(c.calls))
	}
	return c.calls[i]
}

// cacheStub serves the CachedContents resource endpoints, recording every
// request and echoing a canned resource back. It points the pinned genai SDK at
// itself through both base-URL env vars, so one stub covers the Gemini
// Developer API and Vertex AI; a test picks a backend through GoogleConfig and
// only the matching variable is consulted.
func cacheStub(t *testing.T, calls *cacheCalls) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		call := cacheCall{method: r.Method, path: r.URL.Path, body: map[string]any{}}
		_ = json.Unmarshal(raw, &call.body)
		calls.add(call)

		const resource = `{"name":"cachedContents/abc123","displayName":"legal-corpus",
			"model":"models/gemini-3.1-pro-preview",
			"createTime":"2026-08-17T10:00:00Z","expireTime":"2026-08-17T12:00:00Z",
			"usageMetadata":{"totalTokenCount":4096}}`

		w.Header().Set("Content-Type", "application/json")
		switch {
		case r.Method == http.MethodDelete:
			_, _ = io.WriteString(w, `{}`)
		case r.Method == http.MethodGet && strings.HasSuffix(r.URL.Path, "/cachedContents"):
			_, _ = io.WriteString(w, `{"cachedContents":[`+resource+`]}`)
		default:
			_, _ = io.WriteString(w, resource)
		}
	}))
	t.Setenv("GOOGLE_GEMINI_BASE_URL", srv.URL)
	t.Setenv("GOOGLE_VERTEX_BASE_URL", srv.URL)
	return srv
}

// cacheManager builds a gateway over the stub and returns Google's manager.
func cacheManager(t *testing.T, cfg *GoogleConfig) (*LLMGateway, PromptCacheManager) {
	t.Helper()
	g, err := New([]ProviderConfig{cfg})
	if err != nil {
		t.Fatalf("gateway: %v", err)
	}
	mgr, ok := g.CacheManager(ProviderGoogle)
	if !ok {
		g.Close()
		t.Fatal("CacheManager(ProviderGoogle) = false, want the Google manager")
	}
	return g, mgr
}

func TestGoogleCacheCreateHitsTheCachedContentsCollection(t *testing.T) {
	var calls cacheCalls
	srv := cacheStub(t, &calls)
	defer srv.Close()

	g, mgr := cacheManager(t, &GoogleConfig{APIKey: "k"})
	defer g.Close()

	cache, err := mgr.CreateCache(context.Background(), PromptCacheSpec{
		Model:             NewGemini31Pro(),
		Content:           "long document",
		SystemInstruction: "be terse",
		TTL:               2 * time.Hour,
		DisplayName:       "legal-corpus",
	})
	if err != nil {
		t.Fatalf("CreateCache: %v", err)
	}

	if calls.len() != 1 {
		t.Fatalf("calls = %d, want 1", calls.len())
	}
	if calls.at(t, 0).method != http.MethodPost || calls.at(t, 0).path != "/v1beta/cachedContents" {
		t.Errorf("create hit %s %s", calls.at(t, 0).method, calls.at(t, 0).path)
	}
	body := calls.at(t, 0).body
	if body["model"] != "models/gemini-3.1-pro-preview" {
		t.Errorf("model = %v, want the provider-qualified name", body["model"])
	}
	// A Go time.Duration is what the API's own "7200s" encoding is built from,
	// so the TTL needs no conversion on lingo's side.
	if body["ttl"] != "7200s" {
		t.Errorf("ttl = %v, want %q", body["ttl"], "7200s")
	}
	if body["displayName"] != "legal-corpus" {
		t.Errorf("displayName = %v", body["displayName"])
	}
	contents := body["contents"].([]any)
	if len(contents) != 1 {
		t.Fatalf("contents = %v", contents)
	}
	parts := contents[0].(map[string]any)["parts"].([]any)
	if parts[0].(map[string]any)["text"] != "long document" {
		t.Errorf("contents[0].parts = %v", parts)
	}
	// The system instruction has to be baked into the resource: a generate call
	// carrying both a cache and a system instruction is rejected by Gemini, so
	// there is nowhere else for it to go.
	si := body["systemInstruction"].(map[string]any)["parts"].([]any)
	if si[0].(map[string]any)["text"] != "be terse" {
		t.Errorf("systemInstruction.parts = %v", si)
	}

	if cache.Name != "cachedContents/abc123" || cache.DisplayName != "legal-corpus" {
		t.Errorf("cache = %+v", cache)
	}
	// PromptCache.Model is the provider's qualified form, deliberately not
	// normalized back to lingo's ModelName().
	if cache.Model != "models/gemini-3.1-pro-preview" {
		t.Errorf("cache.Model = %q", cache.Model)
	}
	if cache.Tokens != 4096 {
		t.Errorf("cache.Tokens = %d, want 4096", cache.Tokens)
	}
	if cache.CreatedAt.IsZero() || cache.ExpiresAt.IsZero() {
		t.Errorf("timestamps = %v / %v", cache.CreatedAt, cache.ExpiresAt)
	}
	if !cache.ExpiresAt.After(cache.CreatedAt) {
		t.Errorf("ExpiresAt %v is not after CreatedAt %v", cache.ExpiresAt, cache.CreatedAt)
	}
}

func TestGoogleCacheCreateRejectsWhatTheAPICannotAccept(t *testing.T) {
	var calls cacheCalls
	srv := cacheStub(t, &calls)
	defer srv.Close()

	g, mgr := cacheManager(t, &GoogleConfig{APIKey: "k"})
	defer g.Close()

	specs := map[string]PromptCacheSpec{
		"no model":         {Content: "long document"},
		"wrong provider":   {Model: NewClaudeSonnet5(), Content: "long document"},
		"nothing to cache": {Model: NewGemini31Pro()},
	}
	for name, spec := range specs {
		if _, err := mgr.CreateCache(context.Background(), spec); err == nil {
			t.Errorf("CreateCache(%s) = nil error, want a rejection", name)
		}
	}
	// A direct, explicitly addressed resource call reports its own failures
	// rather than no-opping, so none of these reached the wire.
	if calls.len() != 0 {
		t.Errorf("calls = %d, want 0: invalid specs must not reach the provider", calls.len())
	}
}

func TestGoogleCacheRefreshPatchesTheTTL(t *testing.T) {
	var calls cacheCalls
	srv := cacheStub(t, &calls)
	defer srv.Close()

	g, mgr := cacheManager(t, &GoogleConfig{APIKey: "k"})
	defer g.Close()

	if _, err := mgr.RefreshCache(context.Background(), "cachedContents/abc123", 30*time.Minute); err != nil {
		t.Fatalf("RefreshCache: %v", err)
	}
	if calls.len() != 1 {
		t.Fatalf("calls = %d, want 1", calls.len())
	}
	if calls.at(t, 0).method != http.MethodPatch || calls.at(t, 0).path != "/v1beta/cachedContents/abc123" {
		t.Errorf("refresh hit %s %s", calls.at(t, 0).method, calls.at(t, 0).path)
	}
	// Update accepts a lifetime and nothing else: the content, model and system
	// instruction are fixed at creation, so the body carries only the TTL.
	if len(calls.at(t, 0).body) != 1 || calls.at(t, 0).body["ttl"] != "1800s" {
		t.Errorf("refresh body = %v, want only a ttl", calls.at(t, 0).body)
	}

	if _, err := mgr.RefreshCache(context.Background(), "", time.Hour); err == nil {
		t.Error("RefreshCache with no name = nil error")
	}
	if _, err := mgr.RefreshCache(context.Background(), "cachedContents/abc123", 0); err == nil {
		t.Error("RefreshCache with a zero ttl = nil error")
	}
}

func TestGoogleCacheGetAndDeleteAddressOneResource(t *testing.T) {
	var calls cacheCalls
	srv := cacheStub(t, &calls)
	defer srv.Close()

	g, mgr := cacheManager(t, &GoogleConfig{APIKey: "k"})
	defer g.Close()

	cache, err := mgr.GetCache(context.Background(), "cachedContents/abc123")
	if err != nil {
		t.Fatalf("GetCache: %v", err)
	}
	if cache.Name != "cachedContents/abc123" {
		t.Errorf("cache.Name = %q", cache.Name)
	}

	// A bare id resolves to the same resource: the SDK prefixes the collection.
	if _, err := mgr.GetCache(context.Background(), "abc123"); err != nil {
		t.Fatalf("GetCache(bare): %v", err)
	}
	if err := mgr.DeleteCache(context.Background(), "cachedContents/abc123"); err != nil {
		t.Fatalf("DeleteCache: %v", err)
	}

	want := []cacheCall{
		{method: http.MethodGet, path: "/v1beta/cachedContents/abc123"},
		{method: http.MethodGet, path: "/v1beta/cachedContents/abc123"},
		{method: http.MethodDelete, path: "/v1beta/cachedContents/abc123"},
	}
	if calls.len() != len(want) {
		t.Fatalf("calls = %d, want %d", calls.len(), len(want))
	}
	for i, w := range want {
		if calls.at(t, i).method != w.method || calls.at(t, i).path != w.path {
			t.Errorf("call %d = %s %s, want %s %s", i, calls.at(t, i).method, calls.at(t, i).path, w.method, w.path)
		}
	}

	if _, err := mgr.GetCache(context.Background(), ""); err == nil {
		t.Error("GetCache with no name = nil error")
	}
	if err := mgr.DeleteCache(context.Background(), ""); err == nil {
		t.Error("DeleteCache with no name = nil error")
	}
}

func TestGoogleCacheListWalksTheCollection(t *testing.T) {
	var calls cacheCalls
	srv := cacheStub(t, &calls)
	defer srv.Close()

	g, mgr := cacheManager(t, &GoogleConfig{APIKey: "k"})
	defer g.Close()

	caches, err := mgr.ListCaches(context.Background())
	if err != nil {
		t.Fatalf("ListCaches: %v", err)
	}
	if got := calls.at(t, 0); calls.len() != 1 || got.method != http.MethodGet || got.path != "/v1beta/cachedContents" {
		t.Fatalf("list hit %d call(s), first %s %s", calls.len(), got.method, got.path)
	}
	if len(caches) != 1 || caches[0].Name != "cachedContents/abc123" || caches[0].Tokens != 4096 {
		t.Errorf("caches = %+v", caches)
	}
}

func TestGoogleCacheOnVertexUsesQualifiedNames(t *testing.T) {
	var calls cacheCalls
	srv := cacheStub(t, &calls)
	defer srv.Close()

	// Express mode: Vertex with an API key needs no application default
	// credentials, while Project and Location still qualify every path.
	g, mgr := cacheManager(t, &GoogleConfig{
		UseVertexAI: true,
		APIKey:      "k",
		Project:     "p1",
		Location:    "us-central1",
	})
	defer g.Close()

	if _, err := mgr.CreateCache(context.Background(), PromptCacheSpec{
		Model:   NewGemini31Pro(),
		Content: "long document",
	}); err != nil {
		t.Fatalf("CreateCache: %v", err)
	}
	// A bare id is qualified with the project and location on Vertex too, which
	// is why PromptCache.Name is passed through rather than trimmed.
	if _, err := mgr.GetCache(context.Background(), "abc123"); err != nil {
		t.Fatalf("GetCache: %v", err)
	}

	const collection = "/v1beta1/projects/p1/locations/us-central1/cachedContents"
	if calls.len() != 2 {
		t.Fatalf("calls = %d, want 2", calls.len())
	}
	if calls.at(t, 0).method != http.MethodPost || calls.at(t, 0).path != collection {
		t.Errorf("create hit %s %s, want POST %s", calls.at(t, 0).method, calls.at(t, 0).path, collection)
	}
	if want := "projects/p1/locations/us-central1/publishers/google/models/gemini-3.1-pro-preview"; calls.at(t, 0).body["model"] != want {
		t.Errorf("model = %v, want %q", calls.at(t, 0).body["model"], want)
	}
	if calls.at(t, 1).method != http.MethodGet || calls.at(t, 1).path != collection+"/abc123" {
		t.Errorf("get hit %s %s", calls.at(t, 1).method, calls.at(t, 1).path)
	}
}

func TestCacheManagerOnlyExistsForGoogle(t *testing.T) {
	srv := oaiStub(t, &capture{})
	defer srv.Close()

	g, err := New([]ProviderConfig{&OpenAICompatibleConfig{BaseURL: srv.URL, APIKey: "k"}})
	if err != nil {
		t.Fatalf("gateway: %v", err)
	}
	defer g.Close()

	// Discovery never errors: a provider that models caching per request simply
	// reports false, as does one that is not registered at all.
	if mgr, ok := g.CacheManager(ProviderOpenAICompatible); ok || mgr != nil {
		t.Errorf("CacheManager(ProviderOpenAICompatible) = %v, %v; want nil, false", mgr, ok)
	}
	if mgr, ok := g.CacheManager(ProviderGoogle); ok || mgr != nil {
		t.Errorf("CacheManager on an unregistered provider = %v, %v; want nil, false", mgr, ok)
	}
	// The lifecycle surface is discovered separately from the support level, so
	// Google must stay classified exactly as it was.
	if got := CachingSupport(ProviderGoogle); got != CacheSupportExplicit {
		t.Errorf("CachingSupport(ProviderGoogle) = %v, want %v", got, CacheSupportExplicit)
	}
}

// cacheRejectStub refuses every resource call. 400 rather than 500 or 429 so
// nothing in the SDK or the rate limiter treats it as retryable. It records
// through the same locked log cacheStub uses rather than incrementing a bare
// int from the handler goroutine.
func cacheRejectStub(t *testing.T, calls *cacheCalls) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.add(cacheCall{method: r.Method, path: r.URL.Path})
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = io.WriteString(w, `{"error":{"code":400,"status":"INVALID_ARGUMENT",
			"message":"cached content is too small"}}`)
	}))
	t.Setenv("GOOGLE_GEMINI_BASE_URL", srv.URL)
	return srv
}

// Resource calls are the one part of the caching surface that reports failures.
// A request-side cache option a provider cannot honour is a silent no-op, but a
// create the caller explicitly asked for has to say it did not happen, or the
// caller hands a name that does not exist to the next generate call.
func TestGoogleCacheResourceFailuresAreErrors(t *testing.T) {
	var calls cacheCalls
	srv := cacheRejectStub(t, &calls)
	defer srv.Close()

	g, mgr := cacheManager(t, &GoogleConfig{APIKey: "k"})
	defer g.Close()

	ctx := context.Background()
	if _, err := mgr.CreateCache(ctx, PromptCacheSpec{
		Model: NewGemini31Pro(), Content: "too short",
	}); err == nil {
		t.Error("CreateCache = nil error on a rejected create")
	}
	if _, err := mgr.GetCache(ctx, "cachedContents/abc123"); err == nil {
		t.Error("GetCache = nil error on a rejected read")
	}
	if _, err := mgr.ListCaches(ctx); err == nil {
		t.Error("ListCaches = nil error on a rejected list")
	}
	if _, err := mgr.RefreshCache(ctx, "cachedContents/abc123", time.Hour); err == nil {
		t.Error("RefreshCache = nil error on a rejected update")
	}
	if err := mgr.DeleteCache(ctx, "cachedContents/abc123"); err == nil {
		t.Error("DeleteCache = nil error on a rejected delete")
	}
	if calls.len() != 5 {
		t.Errorf("calls = %d, want 5: every method must reach the provider", calls.len())
	}
}

func TestCacheManagerIsDiscoveredAmongOtherProviders(t *testing.T) {
	var c capture
	srv := oaiStub(t, &c)
	defer srv.Close()
	t.Setenv("GOOGLE_GEMINI_BASE_URL", srv.URL)

	g, err := New([]ProviderConfig{
		&GoogleConfig{APIKey: "k"},
		&OpenAIConfig{APIKey: "k", BaseURL: srv.URL},
		&AnthropicConfig{APIKey: "k"},
		&OpenAICompatibleConfig{BaseURL: srv.URL, APIKey: "k"},
	})
	if err != nil {
		t.Fatalf("gateway: %v", err)
	}
	defer g.Close()

	mgr, ok := g.CacheManager(ProviderGoogle)
	if !ok || mgr == nil {
		t.Fatalf("CacheManager(ProviderGoogle) = %v, %v; want the Google manager", mgr, ok)
	}
	// Every other registered provider caches per request, so there is no
	// resource lifecycle to hand back and no error to report either.
	for _, p := range []ProviderType{ProviderOpenAI, ProviderAnthropic, ProviderOpenAICompatible,
		ProviderBedrock, ProviderAzure, ProviderCohere, ProviderPerplexity, ProviderOllama} {
		if mgr, ok := g.CacheManager(p); ok || mgr != nil {
			t.Errorf("CacheManager(%s) = %v, %v; want nil, false", p, mgr, ok)
		}
	}
}

func TestWithPromptCacheRoundTripsIntoTheRequest(t *testing.T) {
	var calls cacheCalls
	cacheSrv := cacheStub(t, &calls)
	defer cacheSrv.Close()

	g, mgr := cacheManager(t, &GoogleConfig{APIKey: "k"})
	cache, err := mgr.CreateCache(context.Background(), PromptCacheSpec{
		Model:             NewGemini31Pro(),
		Content:           "long document",
		SystemInstruction: "be terse",
	})
	g.Close()
	if err != nil {
		t.Fatalf("CreateCache: %v", err)
	}

	// Create to generate with no name plumbing and no genai types in between.
	var c capture
	genSrv := geminiStub(t, &c)
	defer genSrv.Close()

	generate(t, &GoogleConfig{APIKey: "k"},
		Cached(NewGemini31Pro().WithSystemPrompt("be terse"), WithPromptCache(cache)))

	if c.body["cachedContent"] != cache.Name {
		t.Errorf("cachedContent = %v, want %q", c.body["cachedContent"], cache.Name)
	}
	if si, ok := c.body["systemInstruction"]; ok {
		t.Errorf("systemInstruction sent alongside cachedContent: %v", si)
	}

	// A nil cache is inert rather than a panic or an empty resource name, so a
	// create that failed upstream leaves the request exactly as it was.
	generate(t, &GoogleConfig{APIKey: "k"},
		Cached(NewGemini31Pro().WithSystemPrompt("be terse"), WithPromptCache(nil)))

	if cc, ok := c.body["cachedContent"]; ok {
		t.Errorf("cachedContent = %v for a nil cache, want none", cc)
	}
	si, ok := c.body["systemInstruction"].(map[string]any)
	if !ok {
		t.Fatalf("systemInstruction = %v, want the system prompt left in place", c.body["systemInstruction"])
	}
	if parts := si["parts"].([]any); parts[0].(map[string]any)["text"] != "be terse" {
		t.Errorf("systemInstruction.parts = %v", si["parts"])
	}
}

func TestPromptCacheExpiryHelpers(t *testing.T) {
	var zero *PromptCache
	if zero.Expired() || zero.TimeToLive() != 0 {
		t.Error("a nil cache must be inert, not a panic")
	}

	noExpiry := &PromptCache{Name: "cachedContents/abc123"}
	if noExpiry.Expired() {
		t.Error("a cache with no reported expiry is never expired")
	}
	if noExpiry.TimeToLive() != 0 {
		t.Errorf("TimeToLive() = %v, want 0 when no expiry was reported", noExpiry.TimeToLive())
	}

	live := &PromptCache{ExpiresAt: time.Now().Add(time.Hour)}
	if live.Expired() {
		t.Error("Expired() = true for a cache with an hour left")
	}
	if ttl := live.TimeToLive(); ttl <= 59*time.Minute || ttl > time.Hour {
		t.Errorf("TimeToLive() = %v, want just under an hour", ttl)
	}

	dead := &PromptCache{ExpiresAt: time.Now().Add(-time.Minute)}
	if !dead.Expired() {
		t.Error("Expired() = false for a cache that expired a minute ago")
	}
	if dead.TimeToLive() != 0 {
		t.Errorf("TimeToLive() = %v, want 0 once expired", dead.TimeToLive())
	}
}
