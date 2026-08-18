// Package lingo provides a unified gateway for multiple LLM providers.
// It supports OpenAI, Anthropic, Google AI, xAI, DeepSeek, Cohere, AWS
// Bedrock, Azure OpenAI, OpenRouter, Perplexity and Ollama models with a
// consistent interface, plus any endpoint speaking the OpenAI chat
// completions dialect.
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
	ProviderAzure      ProviderType = "azure"
	ProviderXAI        ProviderType = "xai"
	ProviderDeepSeek   ProviderType = "deepseek"
	ProviderOpenRouter ProviderType = "openrouter"
	ProviderCohere     ProviderType = "cohere"
	// ProviderOpenAICompatible covers any endpoint speaking the OpenAI
	// chat completions dialect: Groq, Together, Fireworks, Cerebras,
	// DeepInfra, vLLM, LM Studio, llama.cpp, LocalAI and friends.
	ProviderOpenAICompatible ProviderType = "openai-compatible"
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
	// MaxRetries is the maximum number of retry attempts (default: 3).
	// Zero means unset and selects that default, so retries are turned off
	// with a negative value: one attempt is always made either way.
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
	// Thinking is the model's reasoning trace, "" when the provider returned
	// none or the model was asked to withhold it. It is never part of Text:
	// providers that inline the trace in the answer are split before this
	// struct is built.
	//
	// For one release it is also mirrored into the Metadata keys it used to
	// live under -- "thinking" on Anthropic, "reasoning_content" everywhere
	// else -- so existing readers keep working. Those keys are deprecated.
	Thinking string `json:"thinking,omitempty"`
	// Metadata contains additional provider-specific information
	Metadata map[string]string `json:"metadata,omitempty"`
}

// TokenUsage contains token usage information.
//
// Providers disagree about whether cached tokens are counted inside or outside
// the prompt total, so lingo normalizes them: PromptTokens always covers the
// whole effective prompt, and CacheReadTokens and CacheWriteTokens are subsets
// of it. Use UncachedPromptTokens for the portion that was processed fresh.
//
// The completion side is normalized the same way: CompletionTokens always
// covers everything the model generated and ThinkingTokens is a subset of it,
// even on Google, which counts its thoughts outside the candidate total. Use
// AnswerTokens for the part the caller actually received.
//
// These four counters are the portable cache contract. Anything finer grained --
// Anthropic's and Bedrock's per-TTL write split, DeepSeek's miss count -- rides
// in GenerationResponse.Metadata under provider-specific keys, because no two
// providers report it the same way.
type TokenUsage struct {
	// PromptTokens is the number of tokens in the prompt, including any that
	// were read from or written to a prompt cache
	PromptTokens int `json:"prompt_tokens"`
	// CompletionTokens is the number of tokens in the completion
	CompletionTokens int `json:"completion_tokens"`
	// TotalTokens is the total number of tokens used
	TotalTokens int `json:"total_tokens"`
	// CacheReadTokens is the number of prompt tokens served from a provider
	// side prompt cache. These are usually billed at a large discount. It is a
	// subset of PromptTokens and stays zero on providers that do not report it.
	CacheReadTokens int `json:"cache_read_tokens,omitempty"`
	// CacheWriteTokens is the number of prompt tokens written into a provider
	// side prompt cache. Some providers bill these at a premium. It is a subset
	// of PromptTokens and stays zero on providers that do not report it.
	CacheWriteTokens int `json:"cache_write_tokens,omitempty"`
	// ThinkingTokens is the number of completion tokens the model spent
	// reasoning before it answered. It is a subset of CompletionTokens and
	// stays zero on providers that do not report it. Use AnswerTokens for the
	// part of the completion the caller actually received.
	//
	// The OpenAI-dialect providers spell it reasoning_tokens on the wire
	// (completion_tokens_details.reasoning_tokens); lingo uses one name for
	// the concept regardless of which vocabulary the provider speaks.
	ThinkingTokens int `json:"thinking_tokens,omitempty"`
}

// UncachedPromptTokens returns the prompt tokens the provider had to process
// fresh, i.e. those neither read from nor written to the cache.
func (u TokenUsage) UncachedPromptTokens() int {
	n := u.PromptTokens - u.CacheReadTokens - u.CacheWriteTokens
	if n < 0 {
		return 0
	}
	return n
}

// CacheHit reports whether any part of the prompt was served from the cache.
func (u TokenUsage) CacheHit() bool { return u.CacheReadTokens > 0 }

// withCache returns a copy of u with the cache counters filled in, normalized
// to lingo's invariant that PromptTokens covers the whole effective prompt.
//
// promptIncludesCache says whether the provider already counted the cached
// tokens inside its prompt total (OpenAI, Google, Cohere) or reported them
// alongside it (Anthropic, Bedrock). When it did not, the counters are folded
// into PromptTokens and TotalTokens so every provider reads the same way.
func (u TokenUsage) withCache(read, write int, promptIncludesCache bool) TokenUsage {
	if read < 0 {
		read = 0
	}
	if write < 0 {
		write = 0
	}
	u.CacheReadTokens = read
	u.CacheWriteTokens = write
	if !promptIncludesCache {
		u.PromptTokens += read + write
		u.TotalTokens += read + write
	}
	return u
}

// AnswerTokens returns the completion tokens that were not spent thinking,
// i.e. the part of the output the caller received.
func (u TokenUsage) AnswerTokens() int {
	n := u.CompletionTokens - u.ThinkingTokens
	if n < 0 {
		return 0
	}
	return n
}

// withThinking returns a copy of u with the thinking counter filled in,
// normalized to lingo's invariant that CompletionTokens covers everything the
// model generated, thinking included.
//
// completionIncludesThinking says whether the provider already counted the
// thinking tokens inside its completion total. Every provider that reports a
// count at the pinned SDK versions does -- Anthropic documents
// output_tokens_details.thinking_tokens as "Always <= output_tokens", and
// OpenAI, Azure, xAI, DeepSeek and OpenRouter report it as
// completion_tokens_details.reasoning_tokens, a breakdown of completion_tokens
// by construction.
//
// Google is the exception and the reason for the flag: genai counts
// thoughtsTokenCount inside totalTokenCount but NOT inside
// candidatesTokenCount, so passing false folds it into CompletionTokens while
// leaving the provider's own total alone -- the total already covers it, and
// the guard below only raises a total that is demonstrably short.
func (u TokenUsage) withThinking(thinking int, completionIncludesThinking bool) TokenUsage {
	if thinking < 0 {
		thinking = 0
	}
	u.ThinkingTokens = thinking
	if !completionIncludesThinking {
		u.CompletionTokens += thinking
		if total := u.PromptTokens + u.CompletionTokens; u.TotalTokens < total {
			u.TotalTokens = total
		}
	}
	return u
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
