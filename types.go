// Package llmux provides a unified gateway for multiple LLM providers.
// It supports OpenAI, Anthropic, Google AI, and Perplexity models with a consistent interface.
package lingo

import (
	"context"
	"time"
)

// ============================================================================
// PROVIDER TYPES
// ============================================================================

// ProviderType identifies the LLM provider
type ProviderType string

const (
	ProviderOpenAI     ProviderType = "openai"
	ProviderAnthropic  ProviderType = "anthropic"
	ProviderGoogle     ProviderType = "google"
	ProviderPerplexity ProviderType = "perplexity"
	ProviderOllama     ProviderType = "ollama"
	ProviderBedrock    ProviderType = "bedrock"
)

// ProviderConfig is the interface that all provider configurations must implement
type ProviderConfig interface {
	providerType() ProviderType
	apiKey() string
	timeout() time.Duration
	rateLimitConfig() *RateLimitConfig
}

// RateLimitConfig contains configuration for rate limit handling
type RateLimitConfig struct {
	// MaxRetries is the maximum number of retry attempts (default: 3)
	MaxRetries int
	// InitialBackoff is the initial backoff duration (default: 1s)
	InitialBackoff time.Duration
	// MaxBackoff is the maximum backoff duration (default: 60s)
	MaxBackoff time.Duration
	// BackoffMultiplier is the multiplier for exponential backoff (default: 2.0)
	BackoffMultiplier float64
}

// DefaultRateLimitConfig returns the default rate limit configuration
func DefaultRateLimitConfig() *RateLimitConfig {
	return &RateLimitConfig{
		MaxRetries:        3,
		InitialBackoff:    1 * time.Second,
		MaxBackoff:        60 * time.Second,
		BackoffMultiplier: 2.0,
	}
}

// ============================================================================
// MODEL INTERFACE
// ============================================================================

// Model is the interface that all model types must implement.
// Each model carries its own generation options with appropriate defaults.
type Model interface {
	// ModelName returns the API model identifier (e.g., "gpt-4o", "claude-3-5-sonnet-20241022")
	ModelName() string
	// Provider returns the provider type for this model
	Provider() ProviderType
	// SystemPrompt returns the system prompt, if set
	SystemPrompt() string
}

// ============================================================================
// GATEWAY INTERFACE
// ============================================================================

// Gateway defines the interface for LLM operations
type Gateway interface {
	// Generate generates text using the specified model
	// The model carries its own generation options
	Generate(ctx context.Context, model Model, prompt string) (*GenerationResponse, error)

	// IsRegistered checks if a provider is registered
	IsRegistered(provider ProviderType) bool

	// ListRegisteredProviders returns a list of registered providers
	ListRegisteredProviders() []ProviderType

	// Health checks the health of a specific provider
	Health(ctx context.Context, provider ProviderType) error

	// Close closes the gateway and all providers
	Close() error
}

// Provider represents a single LLM provider implementation
type Provider interface {
	Generate(ctx context.Context, model Model, prompt string) (*GenerationResponse, error)
	Health(ctx context.Context) error
	Close() error
}

// ============================================================================
// RESPONSE TYPES
// ============================================================================

// GenerationResponse contains the response from text generation
type GenerationResponse struct {
	// Text is the generated text content
	Text string `json:"text"`
	// Provider is the provider that was used
	Provider ProviderType `json:"provider"`
	// Model is the model that was used
	Model string `json:"model"`
	// Usage contains token usage information
	Usage TokenUsage `json:"usage"`
	// FinishReason indicates why generation stopped
	FinishReason string `json:"finish_reason"`
	// Metadata contains additional provider-specific information
	Metadata map[string]string `json:"metadata,omitempty"`
}

// TokenUsage contains token usage information
type TokenUsage struct {
	// PromptTokens is the number of tokens in the prompt
	PromptTokens int `json:"prompt_tokens"`
	// CompletionTokens is the number of tokens in the completion
	CompletionTokens int `json:"completion_tokens"`
	// TotalTokens is the total number of tokens used
	TotalTokens int `json:"total_tokens"`
}

// ============================================================================
// LOGGING INTERFACE
// ============================================================================

// Logger interface for logging - compatible with zerolog and other loggers
type Logger interface {
	Debug() LogEvent
	Info() LogEvent
	Error() LogEvent
}

// LogEvent interface for structured logging
type LogEvent interface {
	Msg(msg string)
	Str(key, val string) LogEvent
	Int(key string, val int) LogEvent
	Int64(key string, val int64) LogEvent
	Bool(key string, val bool) LogEvent
	Err(err error) LogEvent
}
