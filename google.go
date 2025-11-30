package lingo

import (
	"context"
	"fmt"
	"time"

	"github.com/google/generative-ai-go/genai"
	"google.golang.org/api/option"
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

// GoogleConfig contains configuration for the Google AI provider
type GoogleConfig struct {
	// APIKey is the Google AI API key (required)
	APIKey string
	// Timeout is the request timeout (default: 60s)
	Timeout time.Duration
	// RateLimiter is the optional rate limit configuration
	RateLimiter *RateLimitConfig
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

// Gemini20Flash represents the Gemini 2.0 Flash model
type Gemini20Flash struct{ googleOptions }

func (m *Gemini20Flash) ModelName() string      { return "gemini-2.0-flash" }
func (m *Gemini20Flash) Provider() ProviderType { return ProviderGoogle }
func (m *Gemini20Flash) SystemPrompt() string   { return m.systemPrompt }

func (m *Gemini20Flash) WithMaxTokens(n int) *Gemini20Flash       { m.maxTokens = n; return m }
func (m *Gemini20Flash) WithTemperature(t float64) *Gemini20Flash { m.temperature = t; return m }
func (m *Gemini20Flash) WithTopP(p float64) *Gemini20Flash        { m.topP = p; return m }
func (m *Gemini20Flash) WithTopK(k int) *Gemini20Flash            { m.topK = k; return m }
func (m *Gemini20Flash) WithSystemPrompt(s string) *Gemini20Flash { m.systemPrompt = s; return m }

// NewGemini20Flash creates a new Gemini 2.0 Flash model with default options
func NewGemini20Flash() *Gemini20Flash {
	return &Gemini20Flash{googleOptions{maxTokens: 8192, temperature: 1.0}}
}

// Gemini20FlashLite represents the Gemini 2.0 Flash Lite model
type Gemini20FlashLite struct{ googleOptions }

func (m *Gemini20FlashLite) ModelName() string      { return "gemini-2.0-flash-lite" }
func (m *Gemini20FlashLite) Provider() ProviderType { return ProviderGoogle }
func (m *Gemini20FlashLite) SystemPrompt() string   { return m.systemPrompt }

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
func NewGemini20FlashLite() *Gemini20FlashLite {
	return &Gemini20FlashLite{googleOptions{maxTokens: 4096, temperature: 1.0}}
}

// Gemini15Pro represents the Gemini 1.5 Pro model
// Versions: gemini-1.5-pro, gemini-1.5-pro-latest
type Gemini15Pro struct{ googleOptions }

func (m *Gemini15Pro) ModelName() string {
	if m.modelVersion != "" {
		return m.modelVersion
	}
	return "gemini-1.5-pro"
}
func (m *Gemini15Pro) Provider() ProviderType { return ProviderGoogle }
func (m *Gemini15Pro) SystemPrompt() string   { return m.systemPrompt }

func (m *Gemini15Pro) WithVersion(v string) *Gemini15Pro      { m.modelVersion = v; return m }
func (m *Gemini15Pro) WithMaxTokens(n int) *Gemini15Pro       { m.maxTokens = n; return m }
func (m *Gemini15Pro) WithTemperature(t float64) *Gemini15Pro { m.temperature = t; return m }
func (m *Gemini15Pro) WithTopP(p float64) *Gemini15Pro        { m.topP = p; return m }
func (m *Gemini15Pro) WithTopK(k int) *Gemini15Pro            { m.topK = k; return m }
func (m *Gemini15Pro) WithSystemPrompt(s string) *Gemini15Pro { m.systemPrompt = s; return m }

// NewGemini15Pro creates a new Gemini 1.5 Pro model with default options
func NewGemini15Pro() *Gemini15Pro {
	return &Gemini15Pro{googleOptions{maxTokens: 8192, temperature: 1.0}}
}

// Gemini15Flash represents the Gemini 1.5 Flash model
// Versions: gemini-1.5-flash, gemini-1.5-flash-latest
type Gemini15Flash struct{ googleOptions }

func (m *Gemini15Flash) ModelName() string {
	if m.modelVersion != "" {
		return m.modelVersion
	}
	return "gemini-1.5-flash"
}
func (m *Gemini15Flash) Provider() ProviderType { return ProviderGoogle }
func (m *Gemini15Flash) SystemPrompt() string   { return m.systemPrompt }

func (m *Gemini15Flash) WithVersion(v string) *Gemini15Flash      { m.modelVersion = v; return m }
func (m *Gemini15Flash) WithMaxTokens(n int) *Gemini15Flash       { m.maxTokens = n; return m }
func (m *Gemini15Flash) WithTemperature(t float64) *Gemini15Flash { m.temperature = t; return m }
func (m *Gemini15Flash) WithTopP(p float64) *Gemini15Flash        { m.topP = p; return m }
func (m *Gemini15Flash) WithTopK(k int) *Gemini15Flash            { m.topK = k; return m }
func (m *Gemini15Flash) WithSystemPrompt(s string) *Gemini15Flash { m.systemPrompt = s; return m }

// NewGemini15Flash creates a new Gemini 1.5 Flash model with default options
func NewGemini15Flash() *Gemini15Flash {
	return &Gemini15Flash{googleOptions{maxTokens: 8192, temperature: 1.0}}
}

// Gemini15Flash8b represents the Gemini 1.5 Flash 8B model
type Gemini15Flash8b struct{ googleOptions }

func (m *Gemini15Flash8b) ModelName() string      { return "gemini-1.5-flash-8b" }
func (m *Gemini15Flash8b) Provider() ProviderType { return ProviderGoogle }
func (m *Gemini15Flash8b) SystemPrompt() string   { return m.systemPrompt }

func (m *Gemini15Flash8b) WithMaxTokens(n int) *Gemini15Flash8b       { m.maxTokens = n; return m }
func (m *Gemini15Flash8b) WithTemperature(t float64) *Gemini15Flash8b { m.temperature = t; return m }
func (m *Gemini15Flash8b) WithTopP(p float64) *Gemini15Flash8b        { m.topP = p; return m }
func (m *Gemini15Flash8b) WithTopK(k int) *Gemini15Flash8b            { m.topK = k; return m }
func (m *Gemini15Flash8b) WithSystemPrompt(s string) *Gemini15Flash8b { m.systemPrompt = s; return m }

// NewGemini15Flash8b creates a new Gemini 1.5 Flash 8B model with default options
func NewGemini15Flash8b() *Gemini15Flash8b {
	return &Gemini15Flash8b{googleOptions{maxTokens: 8192, temperature: 1.0}}
}

// Gemini20FlashExp represents the Gemini 2.0 Flash Experimental model
type Gemini20FlashExp struct{ googleOptions }

func (m *Gemini20FlashExp) ModelName() string      { return "gemini-2.0-flash-exp" }
func (m *Gemini20FlashExp) Provider() ProviderType { return ProviderGoogle }
func (m *Gemini20FlashExp) SystemPrompt() string   { return m.systemPrompt }

func (m *Gemini20FlashExp) WithMaxTokens(n int) *Gemini20FlashExp       { m.maxTokens = n; return m }
func (m *Gemini20FlashExp) WithTemperature(t float64) *Gemini20FlashExp { m.temperature = t; return m }
func (m *Gemini20FlashExp) WithTopP(p float64) *Gemini20FlashExp        { m.topP = p; return m }
func (m *Gemini20FlashExp) WithTopK(k int) *Gemini20FlashExp            { m.topK = k; return m }
func (m *Gemini20FlashExp) WithSystemPrompt(s string) *Gemini20FlashExp { m.systemPrompt = s; return m }

// NewGemini20FlashExp creates a new Gemini 2.0 Flash Exp model with default options
func NewGemini20FlashExp() *Gemini20FlashExp {
	return &Gemini20FlashExp{googleOptions{maxTokens: 8192, temperature: 1.0}}
}

// Gemini20FlashThinking represents the Gemini 2.0 Flash Thinking Experimental model
type Gemini20FlashThinking struct{ googleOptions }

func (m *Gemini20FlashThinking) ModelName() string      { return "gemini-2.0-flash-thinking-exp" }
func (m *Gemini20FlashThinking) Provider() ProviderType { return ProviderGoogle }
func (m *Gemini20FlashThinking) SystemPrompt() string   { return m.systemPrompt }

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
func NewGemini20FlashThinking() *Gemini20FlashThinking {
	return &Gemini20FlashThinking{googleOptions{maxTokens: 8192, temperature: 1.0}}
}

// Gemini20ProExp represents the Gemini 2.0 Pro Experimental model
type Gemini20ProExp struct{ googleOptions }

func (m *Gemini20ProExp) ModelName() string      { return "gemini-2.0-pro-exp" }
func (m *Gemini20ProExp) Provider() ProviderType { return ProviderGoogle }
func (m *Gemini20ProExp) SystemPrompt() string   { return m.systemPrompt }

func (m *Gemini20ProExp) WithMaxTokens(n int) *Gemini20ProExp       { m.maxTokens = n; return m }
func (m *Gemini20ProExp) WithTemperature(t float64) *Gemini20ProExp { m.temperature = t; return m }
func (m *Gemini20ProExp) WithTopP(p float64) *Gemini20ProExp        { m.topP = p; return m }
func (m *Gemini20ProExp) WithTopK(k int) *Gemini20ProExp            { m.topK = k; return m }
func (m *Gemini20ProExp) WithSystemPrompt(s string) *Gemini20ProExp { m.systemPrompt = s; return m }

// NewGemini20ProExp creates a new Gemini 2.0 Pro Exp model with default options
func NewGemini20ProExp() *Gemini20ProExp {
	return &Gemini20ProExp{googleOptions{maxTokens: 8192, temperature: 1.0}}
}

// Gemini3Pro represents the Gemini 3 Pro model
// Versions: gemini-3-pro, gemini-3-pro-latest
type Gemini3Pro struct{ googleOptions }

func (m *Gemini3Pro) ModelName() string {
	if m.modelVersion != "" {
		return m.modelVersion
	}
	return "gemini-3-pro"
}
func (m *Gemini3Pro) Provider() ProviderType { return ProviderGoogle }
func (m *Gemini3Pro) SystemPrompt() string   { return m.systemPrompt }

func (m *Gemini3Pro) WithVersion(v string) *Gemini3Pro      { m.modelVersion = v; return m }
func (m *Gemini3Pro) WithMaxTokens(n int) *Gemini3Pro       { m.maxTokens = n; return m }
func (m *Gemini3Pro) WithTemperature(t float64) *Gemini3Pro { m.temperature = t; return m }
func (m *Gemini3Pro) WithTopP(p float64) *Gemini3Pro        { m.topP = p; return m }
func (m *Gemini3Pro) WithTopK(k int) *Gemini3Pro            { m.topK = k; return m }
func (m *Gemini3Pro) WithSystemPrompt(s string) *Gemini3Pro { m.systemPrompt = s; return m }

// NewGemini3Pro creates a new Gemini 3 Pro model with default options
func NewGemini3Pro() *Gemini3Pro {
	return &Gemini3Pro{googleOptions{maxTokens: 8192, temperature: 1.0}}
}

// Gemini3Flash represents the Gemini 3 Flash model
// Versions: gemini-3-flash, gemini-3-flash-latest
type Gemini3Flash struct{ googleOptions }

func (m *Gemini3Flash) ModelName() string {
	if m.modelVersion != "" {
		return m.modelVersion
	}
	return "gemini-3-flash"
}
func (m *Gemini3Flash) Provider() ProviderType { return ProviderGoogle }
func (m *Gemini3Flash) SystemPrompt() string   { return m.systemPrompt }

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

// Gemini3Ultra represents the Gemini 3 Ultra model
type Gemini3Ultra struct{ googleOptions }

func (m *Gemini3Ultra) ModelName() string      { return "gemini-3-ultra" }
func (m *Gemini3Ultra) Provider() ProviderType { return ProviderGoogle }
func (m *Gemini3Ultra) SystemPrompt() string   { return m.systemPrompt }

func (m *Gemini3Ultra) WithMaxTokens(n int) *Gemini3Ultra       { m.maxTokens = n; return m }
func (m *Gemini3Ultra) WithTemperature(t float64) *Gemini3Ultra { m.temperature = t; return m }
func (m *Gemini3Ultra) WithTopP(p float64) *Gemini3Ultra        { m.topP = p; return m }
func (m *Gemini3Ultra) WithTopK(k int) *Gemini3Ultra            { m.topK = k; return m }
func (m *Gemini3Ultra) WithSystemPrompt(s string) *Gemini3Ultra { m.systemPrompt = s; return m }

// NewGemini3Ultra creates a new Gemini 3 Ultra model with default options
func NewGemini3Ultra() *Gemini3Ultra {
	return &Gemini3Ultra{googleOptions{maxTokens: 8192, temperature: 1.0}}
}

// ============================================================================
// GOOGLE PROVIDER CLIENT
// ============================================================================

// googleClient implements the Provider interface for Google AI (Gemini)
type googleClient struct {
	client      *genai.Client
	timeout     time.Duration
	logger      Logger
	rateLimiter *rateLimiter
}

// newGoogleClient creates a new Google AI client using the official Generative AI SDK
func newGoogleClient(config *GoogleConfig, logger Logger) (*googleClient, error) {
	if config.APIKey == "" {
		return nil, fmt.Errorf("google API key is required")
	}

	ctx := context.Background()
	client, err := genai.NewClient(ctx, option.WithAPIKey(config.APIKey))
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

// Generate generates text using Google's Gemini API
func (c *googleClient) Generate(ctx context.Context, model Model, prompt string) (*GenerationResponse, error) {
	// Verify model is for Google
	if model.Provider() != ProviderGoogle {
		return nil, fmt.Errorf("model %s is not a Google model", model.ModelName())
	}

	// Set timeout
	ctx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()

	// Get the generative model
	genModel := c.client.GenerativeModel(model.ModelName())

	// Apply options based on model type
	switch m := model.(type) {
	case *Gemini25Pro:
		if m.temperature > 0 {
			genModel.SetTemperature(float32(m.temperature))
		}
		if m.maxTokens > 0 {
			genModel.SetMaxOutputTokens(int32(m.maxTokens))
		}
		if m.topP > 0 {
			genModel.SetTopP(float32(m.topP))
		}
		if m.topK > 0 {
			genModel.SetTopK(int32(m.topK))
		}
		if m.systemPrompt != "" {
			genModel.SystemInstruction = &genai.Content{
				Parts: []genai.Part{genai.Text(m.systemPrompt)},
			}
		}
	case *Gemini25Flash:
		if m.temperature > 0 {
			genModel.SetTemperature(float32(m.temperature))
		}
		if m.maxTokens > 0 {
			genModel.SetMaxOutputTokens(int32(m.maxTokens))
		}
		if m.topP > 0 {
			genModel.SetTopP(float32(m.topP))
		}
		if m.topK > 0 {
			genModel.SetTopK(int32(m.topK))
		}
		if m.systemPrompt != "" {
			genModel.SystemInstruction = &genai.Content{
				Parts: []genai.Part{genai.Text(m.systemPrompt)},
			}
		}
	case *Gemini20Flash:
		if m.temperature > 0 {
			genModel.SetTemperature(float32(m.temperature))
		}
		if m.maxTokens > 0 {
			genModel.SetMaxOutputTokens(int32(m.maxTokens))
		}
		if m.topP > 0 {
			genModel.SetTopP(float32(m.topP))
		}
		if m.topK > 0 {
			genModel.SetTopK(int32(m.topK))
		}
		if m.systemPrompt != "" {
			genModel.SystemInstruction = &genai.Content{
				Parts: []genai.Part{genai.Text(m.systemPrompt)},
			}
		}
	case *Gemini20FlashLite:
		if m.temperature > 0 {
			genModel.SetTemperature(float32(m.temperature))
		}
		if m.maxTokens > 0 {
			genModel.SetMaxOutputTokens(int32(m.maxTokens))
		}
		if m.topP > 0 {
			genModel.SetTopP(float32(m.topP))
		}
		if m.topK > 0 {
			genModel.SetTopK(int32(m.topK))
		}
		if m.systemPrompt != "" {
			genModel.SystemInstruction = &genai.Content{
				Parts: []genai.Part{genai.Text(m.systemPrompt)},
			}
		}
	case *Gemini15Pro:
		if m.temperature > 0 {
			genModel.SetTemperature(float32(m.temperature))
		}
		if m.maxTokens > 0 {
			genModel.SetMaxOutputTokens(int32(m.maxTokens))
		}
		if m.topP > 0 {
			genModel.SetTopP(float32(m.topP))
		}
		if m.topK > 0 {
			genModel.SetTopK(int32(m.topK))
		}
		if m.systemPrompt != "" {
			genModel.SystemInstruction = &genai.Content{
				Parts: []genai.Part{genai.Text(m.systemPrompt)},
			}
		}
	case *Gemini15Flash:
		if m.temperature > 0 {
			genModel.SetTemperature(float32(m.temperature))
		}
		if m.maxTokens > 0 {
			genModel.SetMaxOutputTokens(int32(m.maxTokens))
		}
		if m.topP > 0 {
			genModel.SetTopP(float32(m.topP))
		}
		if m.topK > 0 {
			genModel.SetTopK(int32(m.topK))
		}
		if m.systemPrompt != "" {
			genModel.SystemInstruction = &genai.Content{
				Parts: []genai.Part{genai.Text(m.systemPrompt)},
			}
		}
	case *Gemini15Flash8b:
		if m.temperature > 0 {
			genModel.SetTemperature(float32(m.temperature))
		}
		if m.maxTokens > 0 {
			genModel.SetMaxOutputTokens(int32(m.maxTokens))
		}
		if m.topP > 0 {
			genModel.SetTopP(float32(m.topP))
		}
		if m.topK > 0 {
			genModel.SetTopK(int32(m.topK))
		}
		if m.systemPrompt != "" {
			genModel.SystemInstruction = &genai.Content{
				Parts: []genai.Part{genai.Text(m.systemPrompt)},
			}
		}
	case *Gemini20FlashExp:
		if m.temperature > 0 {
			genModel.SetTemperature(float32(m.temperature))
		}
		if m.maxTokens > 0 {
			genModel.SetMaxOutputTokens(int32(m.maxTokens))
		}
		if m.topP > 0 {
			genModel.SetTopP(float32(m.topP))
		}
		if m.topK > 0 {
			genModel.SetTopK(int32(m.topK))
		}
		if m.systemPrompt != "" {
			genModel.SystemInstruction = &genai.Content{
				Parts: []genai.Part{genai.Text(m.systemPrompt)},
			}
		}
	case *Gemini20FlashThinking:
		if m.temperature > 0 {
			genModel.SetTemperature(float32(m.temperature))
		}
		if m.maxTokens > 0 {
			genModel.SetMaxOutputTokens(int32(m.maxTokens))
		}
		if m.topP > 0 {
			genModel.SetTopP(float32(m.topP))
		}
		if m.topK > 0 {
			genModel.SetTopK(int32(m.topK))
		}
		if m.systemPrompt != "" {
			genModel.SystemInstruction = &genai.Content{
				Parts: []genai.Part{genai.Text(m.systemPrompt)},
			}
		}
	case *Gemini20ProExp:
		if m.temperature > 0 {
			genModel.SetTemperature(float32(m.temperature))
		}
		if m.maxTokens > 0 {
			genModel.SetMaxOutputTokens(int32(m.maxTokens))
		}
		if m.topP > 0 {
			genModel.SetTopP(float32(m.topP))
		}
		if m.topK > 0 {
			genModel.SetTopK(int32(m.topK))
		}
		if m.systemPrompt != "" {
			genModel.SystemInstruction = &genai.Content{
				Parts: []genai.Part{genai.Text(m.systemPrompt)},
			}
		}
	case *Gemini3Pro:
		if m.temperature > 0 {
			genModel.SetTemperature(float32(m.temperature))
		}
		if m.maxTokens > 0 {
			genModel.SetMaxOutputTokens(int32(m.maxTokens))
		}
		if m.topP > 0 {
			genModel.SetTopP(float32(m.topP))
		}
		if m.topK > 0 {
			genModel.SetTopK(int32(m.topK))
		}
		if m.systemPrompt != "" {
			genModel.SystemInstruction = &genai.Content{
				Parts: []genai.Part{genai.Text(m.systemPrompt)},
			}
		}
	case *Gemini3Flash:
		if m.temperature > 0 {
			genModel.SetTemperature(float32(m.temperature))
		}
		if m.maxTokens > 0 {
			genModel.SetMaxOutputTokens(int32(m.maxTokens))
		}
		if m.topP > 0 {
			genModel.SetTopP(float32(m.topP))
		}
		if m.topK > 0 {
			genModel.SetTopK(int32(m.topK))
		}
		if m.systemPrompt != "" {
			genModel.SystemInstruction = &genai.Content{
				Parts: []genai.Part{genai.Text(m.systemPrompt)},
			}
		}
	case *Gemini3Ultra:
		if m.temperature > 0 {
			genModel.SetTemperature(float32(m.temperature))
		}
		if m.maxTokens > 0 {
			genModel.SetMaxOutputTokens(int32(m.maxTokens))
		}
		if m.topP > 0 {
			genModel.SetTopP(float32(m.topP))
		}
		if m.topK > 0 {
			genModel.SetTopK(int32(m.topK))
		}
		if m.systemPrompt != "" {
			genModel.SystemInstruction = &genai.Content{
				Parts: []genai.Part{genai.Text(m.systemPrompt)},
			}
		}
	}

	c.logger.Debug().
		Str("model", model.ModelName()).
		Msg("Making Google AI API request")

	// Make the request with rate limit handling
	var resp *genai.GenerateContentResponse
	err := c.rateLimiter.Execute(ctx, func() error {
		var reqErr error
		resp, reqErr = genModel.GenerateContent(ctx, genai.Text(prompt))
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

	// Extract text from parts
	var text string
	for _, part := range candidate.Content.Parts {
		if textPart, ok := part.(genai.Text); ok {
			text += string(textPart)
		}
	}

	if text == "" {
		return nil, fmt.Errorf("no text content found in Google AI response")
	}

	// Extract token usage
	var promptTokens, completionTokens, totalTokens int
	if resp.UsageMetadata != nil {
		promptTokens = int(resp.UsageMetadata.PromptTokenCount)
		completionTokens = int(resp.UsageMetadata.CandidatesTokenCount)
		totalTokens = int(resp.UsageMetadata.TotalTokenCount)
	}

	// Determine finish reason
	finishReason := "stop"
	if candidate.FinishReason != genai.FinishReasonUnspecified {
		finishReason = candidate.FinishReason.String()
	}

	// Build response
	response := &GenerationResponse{
		Text:         text,
		Model:        model.ModelName(),
		FinishReason: finishReason,
		Usage: TokenUsage{
			PromptTokens:     promptTokens,
			CompletionTokens: completionTokens,
			TotalTokens:      totalTokens,
		},
		Metadata: map[string]string{
			"provider": "google",
			"model":    model.ModelName(),
		},
	}

	c.logger.Debug().
		Str("model", model.ModelName()).
		Int("prompt_tokens", promptTokens).
		Int("completion_tokens", completionTokens).
		Int("total_tokens", totalTokens).
		Msg("Google AI generation completed")

	return response, nil
}

// Health checks the health of the Google AI client
func (c *googleClient) Health(ctx context.Context) error {
	ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	genModel := c.client.GenerativeModel("gemini-2.0-flash-lite")
	genModel.SetMaxOutputTokens(5)

	_, err := genModel.GenerateContent(ctx, genai.Text("Hello"))
	if err != nil {
		return fmt.Errorf("google AI health check failed: %w", err)
	}

	return nil
}

// Close closes the Google AI client
func (c *googleClient) Close() error {
	if c.client != nil {
		return c.client.Close()
	}
	return nil
}
