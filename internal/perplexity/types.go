// Package perplexity provides a Go client for the Perplexity API.
// Since there is no official Perplexity Go SDK, this package implements
// the HTTP client from scratch following their API documentation.
//
// Perplexity offers two main APIs:
//   - Search API: Delivers ranked web search results with advanced filtering
//   - Grounded LLM API: AI-powered chat completions with web-grounded knowledge
//
// Reference: https://docs.perplexity.ai/getting-started/overview
package perplexity

import "time"

// BaseURL is the Perplexity API base URL
const BaseURL = "https://api.perplexity.ai"

// ============================================================================
// COMMON TYPES
// ============================================================================

// Message represents a chat message
type Message struct {
	Role    string `json:"role"`    // "system", "user", or "assistant"
	Content string `json:"content"` // The message content
}

// ErrorResponse represents an API error response
type ErrorResponse struct {
	Error ErrorDetail `json:"error"`
}

// ErrorDetail contains error details
type ErrorDetail struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code,omitempty"`
}

// ClientConfig contains configuration for the Perplexity client
type ClientConfig struct {
	// APIKey is the Perplexity API key (required)
	APIKey string

	// BaseURL is the API base URL (defaults to https://api.perplexity.ai)
	BaseURL string

	// Timeout is the HTTP client timeout (default: 30s)
	Timeout time.Duration
}

// ============================================================================
// SEARCH API TYPES
// Reference: https://docs.perplexity.ai/guides/search-quickstart
// ============================================================================

// SearchRequest represents a request to the Search API
type SearchRequest struct {
	// Query is the search query string (required)
	Query string `json:"query"`

	// RecencyFilter filters results by time: "hour", "day", "week", "month", "year"
	RecencyFilter string `json:"recency_filter,omitempty"`

	// DomainFilter limits search to specific domains
	DomainFilter []string `json:"domain_filter,omitempty"`

	// CountryCode filters results by country (e.g., "us", "gb")
	CountryCode string `json:"country_code,omitempty"`

	// LanguageCode filters results by language (e.g., "en", "fr")
	LanguageCode string `json:"language_code,omitempty"`

	// ReturnImages includes image results
	ReturnImages bool `json:"return_images,omitempty"`

	// SafeSearch enables safe search mode
	SafeSearch bool `json:"safe_search,omitempty"`
}

// SearchResponse represents the response from the Search API
type SearchResponse struct {
	// Results contains the search results
	Results []SearchResult `json:"results"`

	// Images contains image results if requested
	Images []ImageResult `json:"images,omitempty"`
}

// SearchResult represents a single search result
type SearchResult struct {
	// Title is the page title
	Title string `json:"title"`

	// URL is the result URL
	URL string `json:"url"`

	// Snippet is the text snippet from the page
	Snippet string `json:"snippet,omitempty"`

	// DatePublished is when the content was published
	DatePublished string `json:"date_published,omitempty"`

	// Author is the content author if available
	Author string `json:"author,omitempty"`
}

// ImageResult represents an image search result
type ImageResult struct {
	// URL is the image URL
	URL string `json:"url"`

	// SourceURL is the page where the image was found
	SourceURL string `json:"source_url,omitempty"`

	// Alt is the image alt text
	Alt string `json:"alt,omitempty"`

	// Width is the image width
	Width int `json:"width,omitempty"`

	// Height is the image height
	Height int `json:"height,omitempty"`
}

// ============================================================================
// GROUNDED LLM (CHAT COMPLETIONS) API TYPES
// Reference: https://docs.perplexity.ai/guides/chat-completions-guide
// ============================================================================

// ChatCompletionRequest represents a request to the chat completions endpoint
type ChatCompletionRequest struct {
	// Model is the name of the model to use (required)
	// Available models: sonar, sonar-pro, sonar-reasoning, sonar-reasoning-pro
	Model string `json:"model"`

	// Messages is the list of messages in the conversation (required)
	Messages []Message `json:"messages"`

	// MaxTokens is the maximum number of tokens to generate
	MaxTokens int `json:"max_tokens,omitempty"`

	// Temperature controls randomness (0-2, default: 0.2)
	Temperature *float64 `json:"temperature,omitempty"`

	// TopP controls nucleus sampling (0-1, default: 0.9)
	TopP *float64 `json:"top_p,omitempty"`

	// TopK controls top-k sampling (default: 0, disabled)
	TopK int `json:"top_k,omitempty"`

	// Stream enables streaming responses
	Stream bool `json:"stream,omitempty"`

	// PresencePenalty penalizes new tokens based on presence in text (-2 to 2)
	PresencePenalty float64 `json:"presence_penalty,omitempty"`

	// FrequencyPenalty penalizes new tokens based on frequency in text (-2 to 2)
	FrequencyPenalty float64 `json:"frequency_penalty,omitempty"`

	// ============================================================================
	// Web Search Options (Sonar models with web grounding)
	// ============================================================================

	// SearchDomainFilter limits search to specific domains
	SearchDomainFilter []string `json:"search_domain_filter,omitempty"`

	// ReturnImages enables image return in responses
	ReturnImages bool `json:"return_images,omitempty"`

	// ReturnRelatedQuestions returns related follow-up questions
	ReturnRelatedQuestions bool `json:"return_related_questions,omitempty"`

	// SearchRecencyFilter filters search by recency: "hour", "day", "week", "month"
	SearchRecencyFilter string `json:"search_recency_filter,omitempty"`
}

// ChatCompletionResponse represents the response from chat completions
type ChatCompletionResponse struct {
	// ID is the unique identifier for the completion
	ID string `json:"id"`

	// Model is the model that was used
	Model string `json:"model"`

	// Object is always "chat.completion"
	Object string `json:"object"`

	// Created is the Unix timestamp of creation
	Created int64 `json:"created"`

	// Choices contains the completion choices
	Choices []Choice `json:"choices"`

	// Usage contains token usage information
	Usage Usage `json:"usage"`

	// Citations contains URLs used for the response (web-grounded responses)
	Citations []string `json:"citations,omitempty"`

	// Images contains images returned if return_images was true
	Images []ImageResult `json:"images,omitempty"`

	// RelatedQuestions contains follow-up questions if requested
	RelatedQuestions []string `json:"related_questions,omitempty"`
}

// Choice represents a single completion choice
type Choice struct {
	// Index is the index of this choice
	Index int `json:"index"`

	// FinishReason indicates why the model stopped generating
	FinishReason string `json:"finish_reason"`

	// Message is the assistant's response message
	Message Message `json:"message"`

	// Delta is used for streaming responses
	Delta *Message `json:"delta,omitempty"`
}

// Usage contains token usage information
type Usage struct {
	// PromptTokens is the number of tokens in the prompt
	PromptTokens int `json:"prompt_tokens"`

	// CompletionTokens is the number of tokens in the completion
	CompletionTokens int `json:"completion_tokens"`

	// TotalTokens is the total number of tokens used
	TotalTokens int `json:"total_tokens"`
}
