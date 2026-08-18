package lingo

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

func init() {
	RegisterProvider(ProviderOllama, func(config ProviderConfig, logger Logger) (Provider, error) {
		cfg, ok := config.(*OllamaConfig)
		if !ok {
			return nil, fmt.Errorf("invalid config type for Ollama provider")
		}
		return newOllamaClient(cfg, logger)
	})
}

// ============================================================================
// OLLAMA PROVIDER CONFIG
// ============================================================================

// OllamaConfig contains configuration for the Ollama provider
type OllamaConfig struct {
	// BaseURL is the Ollama server URL (default: http://localhost:11434)
	BaseURL string
	// Timeout is the request timeout (default: 60s)
	Timeout time.Duration
	// RateLimiter is the optional rate limit configuration
	RateLimiter *RateLimitConfig
}

// Implement ProviderConfig interface
func (c *OllamaConfig) providerType() ProviderType        { return ProviderOllama }
func (c *OllamaConfig) apiKey() string                    { return "" } // Ollama doesn't require API key
func (c *OllamaConfig) timeout() time.Duration            { return c.Timeout }
func (c *OllamaConfig) rateLimitConfig() *RateLimitConfig { return c.RateLimiter }

// ============================================================================
// SHARED OPTIONS (embedded in model structs)
// ============================================================================

// ollamaOptions contains options for Ollama models
type ollamaOptions struct {
	modelName    string
	maxTokens    int
	temperature  float64
	topP         float64
	topK         int
	systemPrompt string
	// Ollama-specific options
	numCtx        int     // Context window size
	repeatPenalty float64 // Repetition penalty
	seed          int     // Random seed for reproducibility
	thinking      ThinkingOptions
}

// ThinkingOptions returns the model's thinking configuration. Every Ollama model
// embeds ollamaOptions, so this one declaration makes them all satisfy
// ThinkingModel -- including Llama and Mistral, which carry the configuration
// and send none of it. The wire gate is thinkingDimensions, not the accessor.
//
// Ollama has no thinking setters of its own, so this is the only surface: there
// is nothing here for it to disagree with.
func (o *ollamaOptions) ThinkingOptions() *ThinkingOptions { return &o.thinking }

// thinkingDimensions answers for the models with no thinking capability, which
// is most of the named catalogue. The thinking-capable types override it.
//
// The conservative default is load-bearing rather than tidy: Ollama's server
// answers `think` on a model without the thinking capability with HTTP 400,
// "%q does not support thinking" (server/routes.go), so a dimension claimed too
// widely turns lingo's never-error promise into a failed request.
func (o *ollamaOptions) thinkingDimensions() ThinkingDimension { return 0 }

// ollamaThinkingDims is what a thinking-capable Ollama model honours: `think`
// takes a bool or a level, and the trace comes back on the message. No token
// count is reported -- thinking tokens are folded into eval_count -- so
// ThinkingCanReportTokens is deliberately absent.
const ollamaThinkingDims = ThinkingCanToggle | ThinkingCanSetEffort | ThinkingCanReportTrace

// ollamaThinkingEfforts is the ladder Ollama accepts. It is exactly three rungs:
// api.ThinkValue rejects any other string, so "minimal" clamps up to low and
// "xhigh"/"max" clamp down to high rather than being forwarded to a 400.
var ollamaThinkingEfforts = []ThinkingEffort{
	ThinkingEffortLow,
	ThinkingEffortMedium,
	ThinkingEffortHigh,
}

// ollamaThinkingModels are the model families lingo believes carry Ollama's
// thinking capability. It is deliberately a list of families rather than exact
// tags, so a size or quantization suffix ("qwen3:8b", "deepseek-r1:32b")
// resolves the same as the bare name.
//
// A family name matches only up to Ollama's tag separator, never into a longer
// name (see ollamaThinkingDimensions). "qwen3" is a thinking model and
// "qwen3-coder", "qwen3-embedding" and "qwen3-reranker" are three separate
// models that are not, so a plain prefix test would put `think` on a request
// Ollama answers with a 400.
//
// The list can only ever be incomplete -- NewOllamaModel takes any tag from a
// local registry, including private and renamed ones -- and an unlisted tag is
// treated as unable to think, because asking a model that cannot think to think
// is the one request Ollama answers with an error rather than by ignoring it.
// The same conservatism covers a thinking model that renames rather than tags a
// variant, such as deepseek-v3.1-terminus: it is an unlisted tag, so thinking
// control on it is a silent no-op until the family is added here. The complete
// answer is Ollama's own discovery path, POST /api/show and its "thinking"
// capability, which costs a round trip per model and is left as a follow-up.
var ollamaThinkingModels = []string{
	"deepseek-r1",
	"deepseek-v3.1",
	"qwen3",
	"qwq",
	"gpt-oss",
	"magistral",
	"phi4-reasoning",
	"exaone-deep",
	"smollm3",
	"cogito",
}

// ollamaThinkingDimensions resolves a raw model tag to its thinking dimensions.
//
// An Ollama tag is "<name>:<variant>", and the match respects that boundary: a
// listed family matches the bare name and every variant of it, and nothing else.
// A plain prefix test would let "qwen3" capture qwen3-coder, qwen3-embedding and
// qwen3-reranker -- three models with no thinking capability, whose server
// answers a truthy `think` with HTTP 400 "does not support thinking" rather than
// ignoring it (ollama/ollama server/routes.go, ChatHandler). Erring the other
// way costs only a silent no-op, so the boundary is required, not an
// optimisation.
func ollamaThinkingDimensions(tag string) ThinkingDimension {
	for _, family := range ollamaThinkingModels {
		if tag == family || strings.HasPrefix(tag, family+":") {
			return ollamaThinkingDims
		}
	}
	return 0
}

// ============================================================================
// OLLAMA MODELS
// ============================================================================

// OllamaModel represents a generic Ollama model
// Use this for any model available in your Ollama installation
type OllamaModel struct{ ollamaOptions }

func (m *OllamaModel) ModelName() string      { return m.modelName }
func (m *OllamaModel) Provider() ProviderType { return ProviderOllama }
func (m *OllamaModel) SystemPrompt() string   { return m.systemPrompt }

// thinkingDimensions resolves the local tag, so a thinking model pulled by name
// answers the same as the named type would.
func (m *OllamaModel) thinkingDimensions() ThinkingDimension {
	return ollamaThinkingDimensions(m.modelName)
}

func (m *OllamaModel) WithMaxTokens(n int) *OllamaModel         { m.maxTokens = n; return m }
func (m *OllamaModel) WithTemperature(t float64) *OllamaModel   { m.temperature = t; return m }
func (m *OllamaModel) WithTopP(p float64) *OllamaModel          { m.topP = p; return m }
func (m *OllamaModel) WithTopK(k int) *OllamaModel              { m.topK = k; return m }
func (m *OllamaModel) WithSystemPrompt(s string) *OllamaModel   { m.systemPrompt = s; return m }
func (m *OllamaModel) WithNumCtx(n int) *OllamaModel            { m.numCtx = n; return m }
func (m *OllamaModel) WithRepeatPenalty(p float64) *OllamaModel { m.repeatPenalty = p; return m }
func (m *OllamaModel) WithSeed(s int) *OllamaModel              { m.seed = s; return m }

// NewOllamaModel creates a new Ollama model with the specified model name
func NewOllamaModel(modelName string) *OllamaModel {
	return &OllamaModel{ollamaOptions{
		modelName:   modelName,
		maxTokens:   4096,
		temperature: 0.8,
	}}
}

// Llama3 represents the Llama 3 model
type Llama3 struct{ ollamaOptions }

func (m *Llama3) ModelName() string      { return "llama3" }
func (m *Llama3) Provider() ProviderType { return ProviderOllama }
func (m *Llama3) SystemPrompt() string   { return m.systemPrompt }

func (m *Llama3) WithMaxTokens(n int) *Llama3         { m.maxTokens = n; return m }
func (m *Llama3) WithTemperature(t float64) *Llama3   { m.temperature = t; return m }
func (m *Llama3) WithTopP(p float64) *Llama3          { m.topP = p; return m }
func (m *Llama3) WithTopK(k int) *Llama3              { m.topK = k; return m }
func (m *Llama3) WithSystemPrompt(s string) *Llama3   { m.systemPrompt = s; return m }
func (m *Llama3) WithNumCtx(n int) *Llama3            { m.numCtx = n; return m }
func (m *Llama3) WithRepeatPenalty(p float64) *Llama3 { m.repeatPenalty = p; return m }
func (m *Llama3) WithSeed(s int) *Llama3              { m.seed = s; return m }

// NewLlama3 creates a new Llama 3 model with default options
func NewLlama3() *Llama3 {
	return &Llama3{ollamaOptions{maxTokens: 4096, temperature: 0.8}}
}

// Llama31 represents the Llama 3.1 model
type Llama31 struct{ ollamaOptions }

func (m *Llama31) ModelName() string      { return "llama3.1" }
func (m *Llama31) Provider() ProviderType { return ProviderOllama }
func (m *Llama31) SystemPrompt() string   { return m.systemPrompt }

func (m *Llama31) WithMaxTokens(n int) *Llama31         { m.maxTokens = n; return m }
func (m *Llama31) WithTemperature(t float64) *Llama31   { m.temperature = t; return m }
func (m *Llama31) WithTopP(p float64) *Llama31          { m.topP = p; return m }
func (m *Llama31) WithTopK(k int) *Llama31              { m.topK = k; return m }
func (m *Llama31) WithSystemPrompt(s string) *Llama31   { m.systemPrompt = s; return m }
func (m *Llama31) WithNumCtx(n int) *Llama31            { m.numCtx = n; return m }
func (m *Llama31) WithRepeatPenalty(p float64) *Llama31 { m.repeatPenalty = p; return m }
func (m *Llama31) WithSeed(s int) *Llama31              { m.seed = s; return m }

// NewLlama31 creates a new Llama 3.1 model with default options
func NewLlama31() *Llama31 {
	return &Llama31{ollamaOptions{maxTokens: 4096, temperature: 0.8}}
}

// Llama32 represents the Llama 3.2 model
type Llama32 struct{ ollamaOptions }

func (m *Llama32) ModelName() string      { return "llama3.2" }
func (m *Llama32) Provider() ProviderType { return ProviderOllama }
func (m *Llama32) SystemPrompt() string   { return m.systemPrompt }

func (m *Llama32) WithMaxTokens(n int) *Llama32         { m.maxTokens = n; return m }
func (m *Llama32) WithTemperature(t float64) *Llama32   { m.temperature = t; return m }
func (m *Llama32) WithTopP(p float64) *Llama32          { m.topP = p; return m }
func (m *Llama32) WithTopK(k int) *Llama32              { m.topK = k; return m }
func (m *Llama32) WithSystemPrompt(s string) *Llama32   { m.systemPrompt = s; return m }
func (m *Llama32) WithNumCtx(n int) *Llama32            { m.numCtx = n; return m }
func (m *Llama32) WithRepeatPenalty(p float64) *Llama32 { m.repeatPenalty = p; return m }
func (m *Llama32) WithSeed(s int) *Llama32              { m.seed = s; return m }

// NewLlama32 creates a new Llama 3.2 model with default options
func NewLlama32() *Llama32 {
	return &Llama32{ollamaOptions{maxTokens: 4096, temperature: 0.8}}
}

// Mistral represents the Mistral model
type Mistral struct{ ollamaOptions }

func (m *Mistral) ModelName() string      { return "mistral" }
func (m *Mistral) Provider() ProviderType { return ProviderOllama }
func (m *Mistral) SystemPrompt() string   { return m.systemPrompt }

func (m *Mistral) WithMaxTokens(n int) *Mistral         { m.maxTokens = n; return m }
func (m *Mistral) WithTemperature(t float64) *Mistral   { m.temperature = t; return m }
func (m *Mistral) WithTopP(p float64) *Mistral          { m.topP = p; return m }
func (m *Mistral) WithTopK(k int) *Mistral              { m.topK = k; return m }
func (m *Mistral) WithSystemPrompt(s string) *Mistral   { m.systemPrompt = s; return m }
func (m *Mistral) WithNumCtx(n int) *Mistral            { m.numCtx = n; return m }
func (m *Mistral) WithRepeatPenalty(p float64) *Mistral { m.repeatPenalty = p; return m }
func (m *Mistral) WithSeed(s int) *Mistral              { m.seed = s; return m }

// NewMistral creates a new Mistral model with default options
func NewMistral() *Mistral {
	return &Mistral{ollamaOptions{maxTokens: 4096, temperature: 0.8}}
}

// Mixtral represents the Mixtral model
type Mixtral struct{ ollamaOptions }

func (m *Mixtral) ModelName() string      { return "mixtral" }
func (m *Mixtral) Provider() ProviderType { return ProviderOllama }
func (m *Mixtral) SystemPrompt() string   { return m.systemPrompt }

func (m *Mixtral) WithMaxTokens(n int) *Mixtral         { m.maxTokens = n; return m }
func (m *Mixtral) WithTemperature(t float64) *Mixtral   { m.temperature = t; return m }
func (m *Mixtral) WithTopP(p float64) *Mixtral          { m.topP = p; return m }
func (m *Mixtral) WithTopK(k int) *Mixtral              { m.topK = k; return m }
func (m *Mixtral) WithSystemPrompt(s string) *Mixtral   { m.systemPrompt = s; return m }
func (m *Mixtral) WithNumCtx(n int) *Mixtral            { m.numCtx = n; return m }
func (m *Mixtral) WithRepeatPenalty(p float64) *Mixtral { m.repeatPenalty = p; return m }
func (m *Mixtral) WithSeed(s int) *Mixtral              { m.seed = s; return m }

// NewMixtral creates a new Mixtral model with default options
func NewMixtral() *Mixtral {
	return &Mixtral{ollamaOptions{maxTokens: 4096, temperature: 0.8}}
}

// CodeLlama represents the Code Llama model
type CodeLlama struct{ ollamaOptions }

func (m *CodeLlama) ModelName() string      { return "codellama" }
func (m *CodeLlama) Provider() ProviderType { return ProviderOllama }
func (m *CodeLlama) SystemPrompt() string   { return m.systemPrompt }

func (m *CodeLlama) WithMaxTokens(n int) *CodeLlama         { m.maxTokens = n; return m }
func (m *CodeLlama) WithTemperature(t float64) *CodeLlama   { m.temperature = t; return m }
func (m *CodeLlama) WithTopP(p float64) *CodeLlama          { m.topP = p; return m }
func (m *CodeLlama) WithTopK(k int) *CodeLlama              { m.topK = k; return m }
func (m *CodeLlama) WithSystemPrompt(s string) *CodeLlama   { m.systemPrompt = s; return m }
func (m *CodeLlama) WithNumCtx(n int) *CodeLlama            { m.numCtx = n; return m }
func (m *CodeLlama) WithRepeatPenalty(p float64) *CodeLlama { m.repeatPenalty = p; return m }
func (m *CodeLlama) WithSeed(s int) *CodeLlama              { m.seed = s; return m }

// NewCodeLlama creates a new Code Llama model with default options
func NewCodeLlama() *CodeLlama {
	return &CodeLlama{ollamaOptions{maxTokens: 4096, temperature: 0.8}}
}

// Phi3 represents the Phi-3 model
type Phi3 struct{ ollamaOptions }

func (m *Phi3) ModelName() string      { return "phi3" }
func (m *Phi3) Provider() ProviderType { return ProviderOllama }
func (m *Phi3) SystemPrompt() string   { return m.systemPrompt }

func (m *Phi3) WithMaxTokens(n int) *Phi3         { m.maxTokens = n; return m }
func (m *Phi3) WithTemperature(t float64) *Phi3   { m.temperature = t; return m }
func (m *Phi3) WithTopP(p float64) *Phi3          { m.topP = p; return m }
func (m *Phi3) WithTopK(k int) *Phi3              { m.topK = k; return m }
func (m *Phi3) WithSystemPrompt(s string) *Phi3   { m.systemPrompt = s; return m }
func (m *Phi3) WithNumCtx(n int) *Phi3            { m.numCtx = n; return m }
func (m *Phi3) WithRepeatPenalty(p float64) *Phi3 { m.repeatPenalty = p; return m }
func (m *Phi3) WithSeed(s int) *Phi3              { m.seed = s; return m }

// NewPhi3 creates a new Phi-3 model with default options
func NewPhi3() *Phi3 {
	return &Phi3{ollamaOptions{maxTokens: 4096, temperature: 0.8}}
}

// Gemma2 represents the Gemma 2 model
type Gemma2 struct{ ollamaOptions }

func (m *Gemma2) ModelName() string      { return "gemma2" }
func (m *Gemma2) Provider() ProviderType { return ProviderOllama }
func (m *Gemma2) SystemPrompt() string   { return m.systemPrompt }

func (m *Gemma2) WithMaxTokens(n int) *Gemma2         { m.maxTokens = n; return m }
func (m *Gemma2) WithTemperature(t float64) *Gemma2   { m.temperature = t; return m }
func (m *Gemma2) WithTopP(p float64) *Gemma2          { m.topP = p; return m }
func (m *Gemma2) WithTopK(k int) *Gemma2              { m.topK = k; return m }
func (m *Gemma2) WithSystemPrompt(s string) *Gemma2   { m.systemPrompt = s; return m }
func (m *Gemma2) WithNumCtx(n int) *Gemma2            { m.numCtx = n; return m }
func (m *Gemma2) WithRepeatPenalty(p float64) *Gemma2 { m.repeatPenalty = p; return m }
func (m *Gemma2) WithSeed(s int) *Gemma2              { m.seed = s; return m }

// NewGemma2 creates a new Gemma 2 model with default options
func NewGemma2() *Gemma2 {
	return &Gemma2{ollamaOptions{maxTokens: 4096, temperature: 0.8}}
}

// Qwen2 represents the Qwen 2 model
type Qwen2 struct{ ollamaOptions }

func (m *Qwen2) ModelName() string      { return "qwen2" }
func (m *Qwen2) Provider() ProviderType { return ProviderOllama }
func (m *Qwen2) SystemPrompt() string   { return m.systemPrompt }

func (m *Qwen2) WithMaxTokens(n int) *Qwen2         { m.maxTokens = n; return m }
func (m *Qwen2) WithTemperature(t float64) *Qwen2   { m.temperature = t; return m }
func (m *Qwen2) WithTopP(p float64) *Qwen2          { m.topP = p; return m }
func (m *Qwen2) WithTopK(k int) *Qwen2              { m.topK = k; return m }
func (m *Qwen2) WithSystemPrompt(s string) *Qwen2   { m.systemPrompt = s; return m }
func (m *Qwen2) WithNumCtx(n int) *Qwen2            { m.numCtx = n; return m }
func (m *Qwen2) WithRepeatPenalty(p float64) *Qwen2 { m.repeatPenalty = p; return m }
func (m *Qwen2) WithSeed(s int) *Qwen2              { m.seed = s; return m }

// NewQwen2 creates a new Qwen 2 model with default options
func NewQwen2() *Qwen2 {
	return &Qwen2{ollamaOptions{maxTokens: 4096, temperature: 0.8}}
}

// DeepSeekCoder represents the DeepSeek Coder model
type DeepSeekCoder struct{ ollamaOptions }

func (m *DeepSeekCoder) ModelName() string      { return "deepseek-coder" }
func (m *DeepSeekCoder) Provider() ProviderType { return ProviderOllama }
func (m *DeepSeekCoder) SystemPrompt() string   { return m.systemPrompt }

func (m *DeepSeekCoder) WithMaxTokens(n int) *DeepSeekCoder         { m.maxTokens = n; return m }
func (m *DeepSeekCoder) WithTemperature(t float64) *DeepSeekCoder   { m.temperature = t; return m }
func (m *DeepSeekCoder) WithTopP(p float64) *DeepSeekCoder          { m.topP = p; return m }
func (m *DeepSeekCoder) WithTopK(k int) *DeepSeekCoder              { m.topK = k; return m }
func (m *DeepSeekCoder) WithSystemPrompt(s string) *DeepSeekCoder   { m.systemPrompt = s; return m }
func (m *DeepSeekCoder) WithNumCtx(n int) *DeepSeekCoder            { m.numCtx = n; return m }
func (m *DeepSeekCoder) WithRepeatPenalty(p float64) *DeepSeekCoder { m.repeatPenalty = p; return m }
func (m *DeepSeekCoder) WithSeed(s int) *DeepSeekCoder              { m.seed = s; return m }

// NewDeepSeekCoder creates a new DeepSeek Coder model with default options
func NewDeepSeekCoder() *DeepSeekCoder {
	return &DeepSeekCoder{ollamaOptions{maxTokens: 4096, temperature: 0.8}}
}

// Llama33 represents the Llama 3.3 model
type Llama33 struct{ ollamaOptions }

func (m *Llama33) ModelName() string      { return "llama3.3" }
func (m *Llama33) Provider() ProviderType { return ProviderOllama }
func (m *Llama33) SystemPrompt() string   { return m.systemPrompt }

func (m *Llama33) WithMaxTokens(n int) *Llama33         { m.maxTokens = n; return m }
func (m *Llama33) WithTemperature(t float64) *Llama33   { m.temperature = t; return m }
func (m *Llama33) WithTopP(p float64) *Llama33          { m.topP = p; return m }
func (m *Llama33) WithTopK(k int) *Llama33              { m.topK = k; return m }
func (m *Llama33) WithSystemPrompt(s string) *Llama33   { m.systemPrompt = s; return m }
func (m *Llama33) WithNumCtx(n int) *Llama33            { m.numCtx = n; return m }
func (m *Llama33) WithRepeatPenalty(p float64) *Llama33 { m.repeatPenalty = p; return m }
func (m *Llama33) WithSeed(s int) *Llama33              { m.seed = s; return m }

// NewLlama33 creates a new Llama 3.3 model with default options
func NewLlama33() *Llama33 {
	return &Llama33{ollamaOptions{maxTokens: 4096, temperature: 0.8}}
}

// Gemma3 represents the Gemma 3 model
type Gemma3 struct{ ollamaOptions }

func (m *Gemma3) ModelName() string      { return "gemma3" }
func (m *Gemma3) Provider() ProviderType { return ProviderOllama }
func (m *Gemma3) SystemPrompt() string   { return m.systemPrompt }

func (m *Gemma3) WithMaxTokens(n int) *Gemma3         { m.maxTokens = n; return m }
func (m *Gemma3) WithTemperature(t float64) *Gemma3   { m.temperature = t; return m }
func (m *Gemma3) WithTopP(p float64) *Gemma3          { m.topP = p; return m }
func (m *Gemma3) WithTopK(k int) *Gemma3              { m.topK = k; return m }
func (m *Gemma3) WithSystemPrompt(s string) *Gemma3   { m.systemPrompt = s; return m }
func (m *Gemma3) WithNumCtx(n int) *Gemma3            { m.numCtx = n; return m }
func (m *Gemma3) WithRepeatPenalty(p float64) *Gemma3 { m.repeatPenalty = p; return m }
func (m *Gemma3) WithSeed(s int) *Gemma3              { m.seed = s; return m }

// NewGemma3 creates a new Gemma 3 model with default options
func NewGemma3() *Gemma3 {
	return &Gemma3{ollamaOptions{maxTokens: 4096, temperature: 0.8}}
}

// Qwen3 represents the Qwen 3 model
type Qwen3 struct{ ollamaOptions }

func (m *Qwen3) ModelName() string      { return "qwen3" }
func (m *Qwen3) Provider() ProviderType { return ProviderOllama }
func (m *Qwen3) SystemPrompt() string   { return m.systemPrompt }

func (m *Qwen3) thinkingDimensions() ThinkingDimension { return ollamaThinkingDims }

func (m *Qwen3) WithMaxTokens(n int) *Qwen3         { m.maxTokens = n; return m }
func (m *Qwen3) WithTemperature(t float64) *Qwen3   { m.temperature = t; return m }
func (m *Qwen3) WithTopP(p float64) *Qwen3          { m.topP = p; return m }
func (m *Qwen3) WithTopK(k int) *Qwen3              { m.topK = k; return m }
func (m *Qwen3) WithSystemPrompt(s string) *Qwen3   { m.systemPrompt = s; return m }
func (m *Qwen3) WithNumCtx(n int) *Qwen3            { m.numCtx = n; return m }
func (m *Qwen3) WithRepeatPenalty(p float64) *Qwen3 { m.repeatPenalty = p; return m }
func (m *Qwen3) WithSeed(s int) *Qwen3              { m.seed = s; return m }

// NewQwen3 creates a new Qwen 3 model with default options
func NewQwen3() *Qwen3 {
	return &Qwen3{ollamaOptions{maxTokens: 4096, temperature: 0.8}}
}

// Phi4 represents the Phi 4 model
type Phi4 struct{ ollamaOptions }

func (m *Phi4) ModelName() string      { return "phi4" }
func (m *Phi4) Provider() ProviderType { return ProviderOllama }
func (m *Phi4) SystemPrompt() string   { return m.systemPrompt }

func (m *Phi4) WithMaxTokens(n int) *Phi4         { m.maxTokens = n; return m }
func (m *Phi4) WithTemperature(t float64) *Phi4   { m.temperature = t; return m }
func (m *Phi4) WithTopP(p float64) *Phi4          { m.topP = p; return m }
func (m *Phi4) WithTopK(k int) *Phi4              { m.topK = k; return m }
func (m *Phi4) WithSystemPrompt(s string) *Phi4   { m.systemPrompt = s; return m }
func (m *Phi4) WithNumCtx(n int) *Phi4            { m.numCtx = n; return m }
func (m *Phi4) WithRepeatPenalty(p float64) *Phi4 { m.repeatPenalty = p; return m }
func (m *Phi4) WithSeed(s int) *Phi4              { m.seed = s; return m }

// NewPhi4 creates a new Phi 4 model with default options
func NewPhi4() *Phi4 {
	return &Phi4{ollamaOptions{maxTokens: 4096, temperature: 0.8}}
}

// DeepSeekR1 represents the DeepSeek-R1 reasoning model
type DeepSeekR1 struct{ ollamaOptions }

func (m *DeepSeekR1) ModelName() string      { return "deepseek-r1" }
func (m *DeepSeekR1) Provider() ProviderType { return ProviderOllama }
func (m *DeepSeekR1) SystemPrompt() string   { return m.systemPrompt }

func (m *DeepSeekR1) thinkingDimensions() ThinkingDimension { return ollamaThinkingDims }

func (m *DeepSeekR1) WithMaxTokens(n int) *DeepSeekR1         { m.maxTokens = n; return m }
func (m *DeepSeekR1) WithTemperature(t float64) *DeepSeekR1   { m.temperature = t; return m }
func (m *DeepSeekR1) WithTopP(p float64) *DeepSeekR1          { m.topP = p; return m }
func (m *DeepSeekR1) WithTopK(k int) *DeepSeekR1              { m.topK = k; return m }
func (m *DeepSeekR1) WithSystemPrompt(s string) *DeepSeekR1   { m.systemPrompt = s; return m }
func (m *DeepSeekR1) WithNumCtx(n int) *DeepSeekR1            { m.numCtx = n; return m }
func (m *DeepSeekR1) WithRepeatPenalty(p float64) *DeepSeekR1 { m.repeatPenalty = p; return m }
func (m *DeepSeekR1) WithSeed(s int) *DeepSeekR1              { m.seed = s; return m }

// NewDeepSeekR1 creates a new DeepSeek-R1 model with default options
func NewDeepSeekR1() *DeepSeekR1 {
	return &DeepSeekR1{ollamaOptions{maxTokens: 4096, temperature: 0.8}}
}

// ============================================================================
// OLLAMA PROVIDER CLIENT
// ============================================================================

// ollamaClient implements the Provider interface for Ollama
type ollamaClient struct {
	httpClient  *http.Client
	baseURL     string
	timeout     time.Duration
	logger      Logger
	rateLimiter *rateLimiter
}

// Ollama API request/response types
type ollamaChatRequest struct {
	Model    string              `json:"model"`
	Messages []ollamaChatMessage `json:"messages"`
	Stream   bool                `json:"stream"`
	Options  *ollamaModelOptions `json:"options,omitempty"`
	// Think is Ollama's thinking switch. It is polymorphic on the wire -- the
	// bare JSON value true, false, or one of the level strings "low", "medium",
	// "high", never an object -- so it is typed as any and marshalled as
	// whatever it holds. Ollama's own client models the same union as a wrapper
	// with a custom marshaller; an untyped field keeps that type out of lingo's
	// signatures and out of its dependencies.
	//
	// It is a sibling of model and messages, not one of Options: Ollama's
	// options object has no think field, common advice to put it there
	// notwithstanding.
	//
	// nil leaves the field out, which is what every request lingo sent before
	// thinking control existed does. That is not the same as false: unset lets a
	// thinking-capable model reason as it does by default, while false asks it
	// not to.
	Think any `json:"think,omitempty"`
}

type ollamaChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
	// Thinking is the reasoning trace, returned beside the content rather than
	// inside it. It is response-only: lingo never sets it on an outbound
	// message, and omitempty keeps request bodies byte-identical.
	Thinking string `json:"thinking,omitempty"`
}

type ollamaModelOptions struct {
	NumPredict    int     `json:"num_predict,omitempty"`
	Temperature   float64 `json:"temperature,omitempty"`
	TopP          float64 `json:"top_p,omitempty"`
	TopK          int     `json:"top_k,omitempty"`
	NumCtx        int     `json:"num_ctx,omitempty"`
	RepeatPenalty float64 `json:"repeat_penalty,omitempty"`
	Seed          int     `json:"seed,omitempty"`
}

type ollamaChatResponse struct {
	Model              string            `json:"model"`
	CreatedAt          string            `json:"created_at"`
	Message            ollamaChatMessage `json:"message"`
	Done               bool              `json:"done"`
	DoneReason         string            `json:"done_reason"`
	TotalDuration      int64             `json:"total_duration"`
	LoadDuration       int64             `json:"load_duration"`
	PromptEvalCount    int               `json:"prompt_eval_count"`
	PromptEvalDuration int64             `json:"prompt_eval_duration"`
	EvalCount          int               `json:"eval_count"`
	EvalDuration       int64             `json:"eval_duration"`
}

// newOllamaClient creates a new Ollama client
func newOllamaClient(config *OllamaConfig, logger Logger) (*ollamaClient, error) {
	baseURL := config.BaseURL
	if baseURL == "" {
		baseURL = "http://localhost:11434"
	}

	timeout := config.Timeout
	if timeout == 0 {
		timeout = defaultTimeout()
	}

	return &ollamaClient{
		httpClient: &http.Client{
			Timeout: timeout,
		},
		baseURL:     baseURL,
		timeout:     timeout,
		logger:      logger,
		rateLimiter: newRateLimiter(config.RateLimiter, logger),
	}, nil
}

// getOllamaOptions extracts options from an Ollama model
func getOllamaOptions(model Model) ollamaOptions {
	switch m := model.(type) {
	case *OllamaModel:
		return m.ollamaOptions
	case *Llama3:
		return m.ollamaOptions
	case *Llama31:
		return m.ollamaOptions
	case *Llama32:
		return m.ollamaOptions
	case *Mistral:
		return m.ollamaOptions
	case *Mixtral:
		return m.ollamaOptions
	case *CodeLlama:
		return m.ollamaOptions
	case *Phi3:
		return m.ollamaOptions
	case *Gemma2:
		return m.ollamaOptions
	case *Qwen2:
		return m.ollamaOptions
	case *DeepSeekCoder:
		return m.ollamaOptions
	case *Llama33:
		return m.ollamaOptions
	case *Gemma3:
		return m.ollamaOptions
	case *Qwen3:
		return m.ollamaOptions
	case *Phi4:
		return m.ollamaOptions
	case *DeepSeekR1:
		return m.ollamaOptions
	default:
		return ollamaOptions{}
	}
}

// Generate generates text using Ollama's API
func (c *ollamaClient) Generate(ctx context.Context, model Model, prompt string) (*GenerationResponse, error) {
	// Verify model is for Ollama
	if model.Provider() != ProviderOllama {
		return nil, fmt.Errorf("model %s is not an Ollama model", model.ModelName())
	}

	// Set timeout
	ctx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()

	// Get model options
	opts := getOllamaOptions(model)

	// Build messages
	messages := []ollamaChatMessage{}
	if model.SystemPrompt() != "" {
		messages = append(messages, ollamaChatMessage{
			Role:    "system",
			Content: model.SystemPrompt(),
		})
	}
	messages = append(messages, ollamaChatMessage{
		Role:    "user",
		Content: prompt,
	})

	// Build request
	reqBody := ollamaChatRequest{
		Model:    model.ModelName(),
		Messages: messages,
		Stream:   false,
	}

	// Add options if any are set
	modelOpts := &ollamaModelOptions{}
	hasOpts := false
	if opts.maxTokens > 0 {
		modelOpts.NumPredict = opts.maxTokens
		hasOpts = true
	}
	if opts.temperature > 0 {
		modelOpts.Temperature = opts.temperature
		hasOpts = true
	}
	if opts.topP > 0 {
		modelOpts.TopP = opts.topP
		hasOpts = true
	}
	if opts.topK > 0 {
		modelOpts.TopK = opts.topK
		hasOpts = true
	}
	if opts.numCtx > 0 {
		modelOpts.NumCtx = opts.numCtx
		hasOpts = true
	}
	if opts.repeatPenalty > 0 {
		modelOpts.RepeatPenalty = opts.repeatPenalty
		hasOpts = true
	}
	if opts.seed > 0 {
		modelOpts.Seed = opts.seed
		hasOpts = true
	}
	if hasOpts {
		reqBody.Options = modelOpts
	}

	// Thinking is opt-in and applied once, from a plan built against the model's
	// own capabilities. A model whose ThinkingOptions were never touched produces
	// a zero plan and leaves Think nil, which is the field omitted -- so a
	// thinking-capable model keeps reasoning by default, as Ollama's server
	// arranges when the field is absent.
	//
	// A level implies enabled, so it is sent instead of the bare true, never
	// beside it.
	plan := planThinking(modelThinkingOptions(model), ModelThinkingDimensions(model),
		budgetRange{}, ollamaThinkingEfforts...)
	switch {
	case plan.disable:
		reqBody.Think = false
	case plan.effort != "":
		reqBody.Think = string(plan.effort)
	case plan.enable:
		reqBody.Think = true
	}

	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	c.logger.Debug().
		Str("model", model.ModelName()).
		Str("url", c.baseURL+"/api/chat").
		Bool("has_thinking", reqBody.Think != nil && reqBody.Think != false).
		Str("thinking_translation", plan.translation()).
		Msg("Making Ollama API request")

	// Make request with rate limit handling
	var resp *http.Response
	err = c.rateLimiter.Execute(ctx, func() error {
		req, reqErr := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/api/chat", bytes.NewBuffer(jsonBody))
		if reqErr != nil {
			return reqErr
		}
		req.Header.Set("Content-Type", "application/json")

		resp, reqErr = c.httpClient.Do(req)
		return reqErr
	})
	if err != nil {
		c.logger.Error().
			Err(err).
			Str("model", model.ModelName()).
			Str("prompt_preview", truncateString(prompt, 100)).
			Msg("Ollama generation failed")
		return nil, fmt.Errorf("ollama generation failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("ollama API error: status %d, body: %s", resp.StatusCode, string(body))
	}

	// Parse response
	var ollamaResp ollamaChatResponse
	if err := json.NewDecoder(resp.Body).Decode(&ollamaResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	// Build response.
	//
	// TokenUsage.ThinkingTokens stays zero: Ollama reports prompt_eval_count and
	// eval_count only, and thinking tokens are folded into eval_count with no
	// breakdown to read.
	response := &GenerationResponse{
		Text:         ollamaResp.Message.Content,
		Thinking:     ollamaResp.Message.Thinking,
		Model:        ollamaResp.Model,
		FinishReason: ollamaResp.DoneReason,
		Usage: TokenUsage{
			PromptTokens:     ollamaResp.PromptEvalCount,
			CompletionTokens: ollamaResp.EvalCount,
			TotalTokens:      ollamaResp.PromptEvalCount + ollamaResp.EvalCount,
		},
		Metadata: map[string]string{
			"provider":       "ollama",
			"model":          ollamaResp.Model,
			"total_duration": fmt.Sprintf("%d", ollamaResp.TotalDuration),
			"load_duration":  fmt.Sprintf("%d", ollamaResp.LoadDuration),
		},
	}

	// Whatever lingo had to translate or drop to fit the caller's request onto
	// this model's dialect, so a silent adaptation is never invisible.
	if s := plan.translation(); s != "" {
		response.Metadata["thinking_translation"] = s
	}

	c.logger.Debug().
		Str("model", ollamaResp.Model).
		Int("prompt_tokens", ollamaResp.PromptEvalCount).
		Int("completion_tokens", ollamaResp.EvalCount).
		Int("total_tokens", ollamaResp.PromptEvalCount+ollamaResp.EvalCount).
		Msg("Ollama generation completed")

	return response, nil
}

// Health checks the health of the Ollama client
func (c *ollamaClient) Health(ctx context.Context) error {
	ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, "GET", c.baseURL+"/api/tags", nil)
	if err != nil {
		return fmt.Errorf("ollama health check failed: %w", err)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("ollama health check failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("ollama health check failed: status %d", resp.StatusCode)
	}

	return nil
}

// Close closes the Ollama client (no-op for HTTP client)
func (c *ollamaClient) Close() error {
	return nil
}
