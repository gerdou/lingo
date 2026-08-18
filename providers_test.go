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
)

// capture is the last request a stub served, for the test that set the stub up
// to assert against once the exchange has finished.
//
// Every stub in this package writes it through record rather than field by
// field. An httptest server runs one handler goroutine per in-flight request,
// so two overlapping requests would otherwise write these fields -- and build
// that map -- concurrently, which is a data race and not merely a lost record.
// Today's callers are serial and the lock is never contended; it is here so
// that a test which fires two Generates at once is testing what it meant to
// instead of tripping the race detector.
type capture struct {
	mu      sync.Mutex
	path    string
	query   string
	headers http.Header
	body    map[string]any
}

// record stores one request. Readers take no lock: they run after the round
// trip they are asking about has completed, which orders the handler's writes
// ahead of them.
func (c *capture) record(r *http.Request, raw []byte) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.path = r.URL.Path
	c.query = r.URL.RawQuery
	c.headers = r.Header.Clone()
	c.body = map[string]any{}
	_ = json.Unmarshal(raw, &c.body)
}

// oaiStub serves a canned OpenAI chat completion and records the request
func oaiStub(t *testing.T, c *capture) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		c.record(r, raw)

		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"id":"c1","model":"served-model","object":"chat.completion",
			"choices":[{"index":0,"finish_reason":"stop",
				"message":{"role":"assistant","content":"hi there","reasoning_content":"thought"}}],
			"usage":{"prompt_tokens":11,"completion_tokens":7,"total_tokens":18,
				"completion_tokens_details":{"reasoning_tokens":3}}}`)
	}))
}

func generate(t *testing.T, cfg ProviderConfig, m Model) *GenerationResponse {
	t.Helper()
	g, err := New([]ProviderConfig{cfg})
	if err != nil {
		t.Fatalf("gateway: %v", err)
	}
	defer g.Close()

	resp, err := g.Generate(context.Background(), m, "hello")
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	return resp
}

func TestOpenAICompatibleRoundTrip(t *testing.T) {
	var c capture
	srv := oaiStub(t, &c)
	defer srv.Close()

	resp := generate(t,
		&OpenAICompatibleConfig{BaseURL: srv.URL, APIKey: "k", Headers: map[string]string{"X-Custom": "v"}},
		NewOpenAICompatibleModel("llama-3.3-70b").WithMaxTokens(64).WithTemperature(0.5).WithSystemPrompt("be terse"))

	if c.path != "/chat/completions" {
		t.Errorf("path = %q", c.path)
	}
	if got := c.body["model"]; got != "llama-3.3-70b" {
		t.Errorf("model = %v", got)
	}
	if got := c.body["max_tokens"]; got != float64(64) {
		t.Errorf("max_tokens = %v", got)
	}
	if c.headers.Get("Authorization") != "Bearer k" {
		t.Errorf("auth = %q", c.headers.Get("Authorization"))
	}
	if c.headers.Get("X-Custom") != "v" {
		t.Errorf("custom header missing")
	}
	msgs := c.body["messages"].([]any)
	if len(msgs) != 2 || msgs[0].(map[string]any)["role"] != "system" {
		t.Errorf("messages = %v", msgs)
	}

	if resp.Text != "hi there" {
		t.Errorf("text = %q", resp.Text)
	}
	if resp.Provider != ProviderOpenAICompatible {
		t.Errorf("provider = %q", resp.Provider)
	}
	if resp.Usage.PromptTokens != 11 || resp.Usage.CompletionTokens != 7 || resp.Usage.TotalTokens != 18 {
		t.Errorf("usage = %+v", resp.Usage)
	}
	if resp.Metadata["requested_model"] != "llama-3.3-70b" || resp.Model != "served-model" {
		t.Errorf("model echo = %v / %v", resp.Model, resp.Metadata["requested_model"])
	}
	if resp.Metadata["reasoning_tokens"] != "3" {
		t.Errorf("reasoning_tokens = %q", resp.Metadata["reasoning_tokens"])
	}
	if resp.Metadata["reasoning_content"] != "thought" {
		t.Errorf("reasoning_content = %q (must be unwrapped, not a quoted JSON literal)",
			resp.Metadata["reasoning_content"])
	}
}

func TestXAIRoundTrip(t *testing.T) {
	var c capture
	srv := oaiStub(t, &c)
	defer srv.Close()

	generate(t, &XAIConfig{APIKey: "k", BaseURL: srv.URL},
		NewGrok43().WithReasoningEffort(XAIEffortHigh).WithMaxCompletionTokens(99))

	if c.body["model"] != "grok-4.3" {
		t.Errorf("model = %v", c.body["model"])
	}
	if c.body["reasoning_effort"] != "high" {
		t.Errorf("reasoning_effort = %v", c.body["reasoning_effort"])
	}
	if c.body["max_completion_tokens"] != float64(99) {
		t.Errorf("max_completion_tokens = %v", c.body["max_completion_tokens"])
	}
	if _, ok := c.body["max_tokens"]; ok {
		t.Errorf("xAI should not send deprecated max_tokens")
	}
}

func TestDeepSeekThinkingExtraField(t *testing.T) {
	var c capture
	srv := oaiStub(t, &c)
	defer srv.Close()

	// Default: no thinking field, since the API default is enabled
	generate(t, &DeepSeekConfig{APIKey: "k", BaseURL: srv.URL}, NewDeepSeekV4Pro())
	if _, ok := c.body["thinking"]; ok {
		t.Errorf("thinking sent by default: %v", c.body["thinking"])
	}

	// Explicitly disabled
	resp := generate(t, &DeepSeekConfig{APIKey: "k", BaseURL: srv.URL},
		NewDeepSeekV4Flash().WithThinkingDisabled().WithReasoningEffort("high"))
	th, ok := c.body["thinking"].(map[string]any)
	if !ok || th["type"] != "disabled" {
		t.Fatalf("thinking = %v", c.body["thinking"])
	}
	if c.body["reasoning_effort"] != "high" {
		t.Errorf("reasoning_effort = %v", c.body["reasoning_effort"])
	}
	if resp.Metadata["is_reasoning_model"] != "false" {
		t.Errorf("is_reasoning_model = %q", resp.Metadata["is_reasoning_model"])
	}
}

func TestOpenRouterRoutingFields(t *testing.T) {
	var c capture
	srv := oaiStub(t, &c)
	defer srv.Close()

	generate(t, &OpenRouterConfig{APIKey: "k", BaseURL: srv.URL, SiteURL: "https://x.test", AppName: "lingo"},
		NewOpenRouterModel("anthropic/claude-opus-5").
			WithReasoningEffort("high").
			WithReasoningExcluded().
			WithProviderOrder([]string{"anthropic"}).
			WithAllowFallbacks(false).
			WithFallbackModels([]string{"openai/gpt-5.6-sol"}).
			WithTransforms([]string{"middle-out"}))

	if c.headers.Get("HTTP-Referer") != "https://x.test" || c.headers.Get("X-OpenRouter-Title") != "lingo" {
		t.Errorf("attribution headers = %v", c.headers)
	}
	reasoning, ok := c.body["reasoning"].(map[string]any)
	if !ok || reasoning["effort"] != "high" || reasoning["exclude"] != true {
		t.Errorf("reasoning = %v (both setters must merge into one object)", c.body["reasoning"])
	}
	provider, ok := c.body["provider"].(map[string]any)
	if !ok || provider["allow_fallbacks"] != false {
		t.Errorf("provider = %v", c.body["provider"])
	}
	if order := provider["order"].([]any); len(order) != 1 || order[0] != "anthropic" {
		t.Errorf("provider.order = %v", provider["order"])
	}
	if models := c.body["models"].([]any); len(models) != 1 {
		t.Errorf("models = %v", c.body["models"])
	}
	if tr := c.body["transforms"].([]any); tr[0] != "middle-out" {
		t.Errorf("transforms = %v", c.body["transforms"])
	}
}

func TestAzureDeploymentRoutingAndAuth(t *testing.T) {
	var c capture
	srv := oaiStub(t, &c)
	defer srv.Close()

	generate(t, &AzureOpenAIConfig{Endpoint: srv.URL, APIKey: "azkey", APIVersion: "2024-10-21"},
		NewAzureOpenAIReasoningModel("my-gpt5-deployment").WithReasoningEffort("medium"))

	if c.path != "/openai/deployments/my-gpt5-deployment/chat/completions" {
		t.Errorf("azure path = %q", c.path)
	}
	if !strings.Contains(c.query, "api-version=2024-10-21") {
		t.Errorf("azure query = %q", c.query)
	}
	if c.headers.Get("Api-Key") != "azkey" {
		t.Errorf("azure auth header = %v", c.headers)
	}
	if c.headers.Get("Authorization") != "" {
		t.Errorf("azure must not send a bearer token: %q", c.headers.Get("Authorization"))
	}
	if c.body["reasoning_effort"] != "medium" {
		t.Errorf("reasoning_effort = %v", c.body["reasoning_effort"])
	}
}

// TestAzureAPIVersionRouting pins the wire shape of all three api-version
// shapes. The default and dated rows are the byte-identical-by-default
// guarantee: no dated api-version models prompt_cache_key, so the field must
// never leave the process on those routes, however the model is configured.
func TestAzureAPIVersionRouting(t *testing.T) {
	keyed := func() *AzureOpenAIModel {
		return Cached(NewAzureOpenAIModel("my-deploy"), WithCacheKey("tenant:acme"))
	}

	for _, tc := range []struct {
		name       string
		apiVersion string
		path       string
		query      string
		cacheKey   string
	}{
		{"default", "", "/openai/deployments/my-deploy/chat/completions", "api-version=2024-10-21", ""},
		{"dated", AzureAPIVersionDefault, "/openai/deployments/my-deploy/chat/completions", "api-version=2024-10-21", ""},
		{"older dated", "2023-05-15", "/openai/deployments/my-deploy/chat/completions", "api-version=2023-05-15", ""},
		{"v1", AzureAPIVersionV1, "/openai/v1/chat/completions", "", "tenant:acme"},
		{"v1 preview", AzureAPIVersionV1Preview, "/openai/v1/chat/completions", "api-version=preview", "tenant:acme"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var c capture
			srv := oaiStub(t, &c)
			defer srv.Close()

			generate(t, &AzureOpenAIConfig{Endpoint: srv.URL, APIKey: "azkey", APIVersion: tc.apiVersion}, keyed())

			if c.path != tc.path {
				t.Errorf("path = %q, want %q", c.path, tc.path)
			}
			if c.query != tc.query {
				t.Errorf("query = %q, want %q", c.query, tc.query)
			}
			if c.headers.Get("Api-Key") != "azkey" {
				t.Errorf("azure auth header = %v", c.headers)
			}
			if c.body["model"] != "my-deploy" {
				t.Errorf("model = %v (the deployment name routes on both surfaces)", c.body["model"])
			}
			got, _ := c.body["prompt_cache_key"].(string)
			if got != tc.cacheKey {
				t.Errorf("prompt_cache_key = %q, want %q", got, tc.cacheKey)
			}
		})
	}
}

func TestAzureRejectsMissingCredentials(t *testing.T) {
	if _, err := New([]ProviderConfig{&AzureOpenAIConfig{Endpoint: "https://x.test"}}); err == nil {
		t.Error("expected an error when neither APIKey nor TokenCredential is set")
	}
	if _, err := New([]ProviderConfig{&AzureOpenAIConfig{APIKey: "k"}}); err == nil {
		t.Error("expected an error when Endpoint is missing")
	}
}

func TestCohereRoundTrip(t *testing.T) {
	var c capture
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		c.record(r, raw)

		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"r1","finish_reason":"COMPLETE",
			"message":{"role":"assistant","content":[
				{"type":"thinking","thinking":"pondering"},
				{"type":"text","text":"the answer"}]},
			"usage":{"tokens":{"input_tokens":21,"output_tokens":9}}}`)
	}))
	defer srv.Close()

	resp := generate(t, &CohereConfig{APIKey: "k", BaseURL: srv.URL},
		NewCommandAPlus().WithMaxTokens(128).WithTopK(5).WithSystemPrompt("be terse").
			WithSafetyMode(CohereSafetyStrict).WithThinkingBudget(2048))

	if c.body["model"] != "command-a-plus-05-2026" {
		t.Errorf("model = %v", c.body["model"])
	}
	if c.body["max_tokens"] != float64(128) || c.body["k"] != float64(5) {
		t.Errorf("max_tokens/k = %v / %v", c.body["max_tokens"], c.body["k"])
	}
	if c.body["safety_mode"] != "STRICT" {
		t.Errorf("safety_mode = %v", c.body["safety_mode"])
	}
	th, ok := c.body["thinking"].(map[string]any)
	if !ok || th["type"] != "enabled" || th["token_budget"] != float64(2048) {
		t.Errorf("thinking = %v", c.body["thinking"])
	}
	msgs := c.body["messages"].([]any)
	if len(msgs) != 2 {
		t.Fatalf("messages = %v", msgs)
	}
	if m0 := msgs[0].(map[string]any); m0["role"] != "system" || m0["content"] != "be terse" {
		t.Errorf("system message = %v", m0)
	}
	if m1 := msgs[1].(map[string]any); m1["role"] != "user" || m1["content"] != "hello" {
		t.Errorf("user message = %v", m1)
	}

	if resp.Text != "the answer" {
		t.Errorf("text = %q (thinking must not leak into Text)", resp.Text)
	}
	if resp.Metadata["reasoning_content"] != "pondering" {
		t.Errorf("reasoning_content = %q", resp.Metadata["reasoning_content"])
	}
	if resp.FinishReason != "COMPLETE" {
		t.Errorf("finish_reason = %q", resp.FinishReason)
	}
	if resp.Usage.PromptTokens != 21 || resp.Usage.CompletionTokens != 9 || resp.Usage.TotalTokens != 30 {
		t.Errorf("usage = %+v", resp.Usage)
	}
}

func TestCrossProviderModelIsRejected(t *testing.T) {
	var c capture
	srv := oaiStub(t, &c)
	defer srv.Close()

	g, err := New([]ProviderConfig{&XAIConfig{APIKey: "k", BaseURL: srv.URL}})
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()

	if _, err := g.Generate(context.Background(), NewDeepSeekV4Pro(), "hi"); err == nil {
		t.Error("expected DeepSeek model on the xAI provider to be rejected")
	}
}

func TestAllProvidersRegister(t *testing.T) {
	configs := []ProviderConfig{
		&OpenAIConfig{APIKey: "k"},
		&AnthropicConfig{APIKey: "k"},
		&GoogleConfig{APIKey: "k"},
		&PerplexityConfig{APIKey: "k"},
		&OllamaConfig{},
		&XAIConfig{APIKey: "k"},
		&DeepSeekConfig{APIKey: "k"},
		&OpenRouterConfig{APIKey: "k"},
		&CohereConfig{APIKey: "k"},
		&AzureOpenAIConfig{Endpoint: "https://x.test", APIKey: "k"},
		&OpenAICompatibleConfig{BaseURL: BaseURLGroq, APIKey: "k"},
	}

	g, err := New(configs)
	if err != nil {
		t.Fatalf("gateway: %v", err)
	}
	defer g.Close()

	if got := len(g.ListRegisteredProviders()); got != len(configs) {
		t.Errorf("registered %d of %d providers", got, len(configs))
	}
	for _, p := range []ProviderType{ProviderXAI, ProviderDeepSeek, ProviderOpenRouter,
		ProviderCohere, ProviderAzure, ProviderOpenAICompatible} {
		if !g.IsRegistered(p) {
			t.Errorf("%s not registered", p)
		}
	}
}

func TestVertexBackendsValidateConfig(t *testing.T) {
	if _, err := New([]ProviderConfig{&GoogleConfig{UseVertexAI: true}}); err == nil {
		t.Error("expected Vertex without Project or APIKey to fail")
	}
	if _, err := New([]ProviderConfig{&AnthropicConfig{Vertex: &AnthropicVertexConfig{Region: "us-east5"}}}); err == nil {
		t.Error("expected Claude on Vertex without ProjectID to fail")
	}
}
