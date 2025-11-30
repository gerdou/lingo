package lingo

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
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
}

type ollamaChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
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

	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	c.logger.Debug().
		Str("model", model.ModelName()).
		Str("url", c.baseURL+"/api/chat").
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

	// Build response
	response := &GenerationResponse{
		Text:         ollamaResp.Message.Content,
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
