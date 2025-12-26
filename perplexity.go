package lingo

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/gerdou/lingo/internal/perplexity"
)

func init() {
	RegisterProvider(ProviderPerplexity, func(config ProviderConfig, logger Logger) (Provider, error) {
		cfg, ok := config.(*PerplexityConfig)
		if !ok {
			return nil, fmt.Errorf("invalid config type for Perplexity provider")
		}
		return newPerplexityClient(cfg, logger)
	})
}

// ============================================================================
// PERPLEXITY PROVIDER CONFIG
// ============================================================================

// PerplexityConfig contains configuration for the Perplexity provider
type PerplexityConfig struct {
	// APIKey is the Perplexity API key (required)
	APIKey string
	// Timeout is the request timeout (default: 60s)
	Timeout time.Duration
	// RateLimiter is the optional rate limit configuration
	RateLimiter *RateLimitConfig
}

// Implement ProviderConfig interface
func (c *PerplexityConfig) providerType() ProviderType        { return ProviderPerplexity }
func (c *PerplexityConfig) apiKey() string                    { return c.APIKey }
func (c *PerplexityConfig) timeout() time.Duration            { return c.Timeout }
func (c *PerplexityConfig) rateLimitConfig() *RateLimitConfig { return c.RateLimiter }

// ============================================================================
// SHARED OPTIONS (embedded in model structs)
// ============================================================================

// perplexityOptions contains options for Perplexity Sonar models
type perplexityOptions struct {
	maxTokens              int
	temperature            float64
	topP                   float64
	topK                   int
	systemPrompt           string
	searchRecencyFilter    string   // "hour", "day", "week", "month"
	searchDomainFilter     []string // Limit search to specific domains
	returnImages           bool
	returnRelatedQuestions bool
}

// ============================================================================
// SONAR MODELS
// ============================================================================

// Sonar represents the Sonar model (lightweight, cost-effective)
type Sonar struct{ perplexityOptions }

func (m *Sonar) ModelName() string      { return "sonar" }
func (m *Sonar) Provider() ProviderType { return ProviderPerplexity }
func (m *Sonar) SystemPrompt() string   { return m.systemPrompt }

func (m *Sonar) WithMaxTokens(n int) *Sonar              { m.maxTokens = n; return m }
func (m *Sonar) WithTemperature(t float64) *Sonar        { m.temperature = t; return m }
func (m *Sonar) WithTopP(p float64) *Sonar               { m.topP = p; return m }
func (m *Sonar) WithTopK(k int) *Sonar                   { m.topK = k; return m }
func (m *Sonar) WithSystemPrompt(s string) *Sonar        { m.systemPrompt = s; return m }
func (m *Sonar) WithSearchRecencyFilter(f string) *Sonar { m.searchRecencyFilter = f; return m }
func (m *Sonar) WithSearchDomainFilter(domains []string) *Sonar {
	m.searchDomainFilter = domains
	return m
}
func (m *Sonar) WithReturnImages(b bool) *Sonar           { m.returnImages = b; return m }
func (m *Sonar) WithReturnRelatedQuestions(b bool) *Sonar { m.returnRelatedQuestions = b; return m }

// NewSonar creates a new Sonar model with default options
func NewSonar() *Sonar {
	return &Sonar{perplexityOptions{maxTokens: 4096, temperature: 0.2}}
}

// SonarPro represents the Sonar Pro model (advanced, complex queries)
type SonarPro struct{ perplexityOptions }

func (m *SonarPro) ModelName() string      { return "sonar-pro" }
func (m *SonarPro) Provider() ProviderType { return ProviderPerplexity }
func (m *SonarPro) SystemPrompt() string   { return m.systemPrompt }

func (m *SonarPro) WithMaxTokens(n int) *SonarPro              { m.maxTokens = n; return m }
func (m *SonarPro) WithTemperature(t float64) *SonarPro        { m.temperature = t; return m }
func (m *SonarPro) WithTopP(p float64) *SonarPro               { m.topP = p; return m }
func (m *SonarPro) WithTopK(k int) *SonarPro                   { m.topK = k; return m }
func (m *SonarPro) WithSystemPrompt(s string) *SonarPro        { m.systemPrompt = s; return m }
func (m *SonarPro) WithSearchRecencyFilter(f string) *SonarPro { m.searchRecencyFilter = f; return m }
func (m *SonarPro) WithSearchDomainFilter(domains []string) *SonarPro {
	m.searchDomainFilter = domains
	return m
}
func (m *SonarPro) WithReturnImages(b bool) *SonarPro { m.returnImages = b; return m }
func (m *SonarPro) WithReturnRelatedQuestions(b bool) *SonarPro {
	m.returnRelatedQuestions = b
	return m
}

// NewSonarPro creates a new Sonar Pro model with default options
func NewSonarPro() *SonarPro {
	return &SonarPro{perplexityOptions{maxTokens: 8192, temperature: 0.2}}
}

// SonarReasoning represents the Sonar Reasoning model (enhanced reasoning)
type SonarReasoning struct{ perplexityOptions }

func (m *SonarReasoning) ModelName() string      { return "sonar-reasoning" }
func (m *SonarReasoning) Provider() ProviderType { return ProviderPerplexity }
func (m *SonarReasoning) SystemPrompt() string   { return m.systemPrompt }

func (m *SonarReasoning) WithMaxTokens(n int) *SonarReasoning       { m.maxTokens = n; return m }
func (m *SonarReasoning) WithTemperature(t float64) *SonarReasoning { m.temperature = t; return m }
func (m *SonarReasoning) WithTopP(p float64) *SonarReasoning        { m.topP = p; return m }
func (m *SonarReasoning) WithTopK(k int) *SonarReasoning            { m.topK = k; return m }
func (m *SonarReasoning) WithSystemPrompt(s string) *SonarReasoning { m.systemPrompt = s; return m }
func (m *SonarReasoning) WithSearchRecencyFilter(f string) *SonarReasoning {
	m.searchRecencyFilter = f
	return m
}
func (m *SonarReasoning) WithSearchDomainFilter(domains []string) *SonarReasoning {
	m.searchDomainFilter = domains
	return m
}
func (m *SonarReasoning) WithReturnImages(b bool) *SonarReasoning { m.returnImages = b; return m }
func (m *SonarReasoning) WithReturnRelatedQuestions(b bool) *SonarReasoning {
	m.returnRelatedQuestions = b
	return m
}

// NewSonarReasoning creates a new Sonar Reasoning model with default options
func NewSonarReasoning() *SonarReasoning {
	return &SonarReasoning{perplexityOptions{maxTokens: 8192, temperature: 0.2}}
}

// SonarReasoningPro represents the Sonar Reasoning Pro model (advanced reasoning)
type SonarReasoningPro struct{ perplexityOptions }

func (m *SonarReasoningPro) ModelName() string      { return "sonar-reasoning-pro" }
func (m *SonarReasoningPro) Provider() ProviderType { return ProviderPerplexity }
func (m *SonarReasoningPro) SystemPrompt() string   { return m.systemPrompt }

func (m *SonarReasoningPro) WithMaxTokens(n int) *SonarReasoningPro { m.maxTokens = n; return m }
func (m *SonarReasoningPro) WithTemperature(t float64) *SonarReasoningPro {
	m.temperature = t
	return m
}
func (m *SonarReasoningPro) WithTopP(p float64) *SonarReasoningPro { m.topP = p; return m }
func (m *SonarReasoningPro) WithTopK(k int) *SonarReasoningPro     { m.topK = k; return m }
func (m *SonarReasoningPro) WithSystemPrompt(s string) *SonarReasoningPro {
	m.systemPrompt = s
	return m
}
func (m *SonarReasoningPro) WithSearchRecencyFilter(f string) *SonarReasoningPro {
	m.searchRecencyFilter = f
	return m
}
func (m *SonarReasoningPro) WithSearchDomainFilter(domains []string) *SonarReasoningPro {
	m.searchDomainFilter = domains
	return m
}
func (m *SonarReasoningPro) WithReturnImages(b bool) *SonarReasoningPro { m.returnImages = b; return m }
func (m *SonarReasoningPro) WithReturnRelatedQuestions(b bool) *SonarReasoningPro {
	m.returnRelatedQuestions = b
	return m
}

// NewSonarReasoningPro creates a new Sonar Reasoning Pro model with default options
func NewSonarReasoningPro() *SonarReasoningPro {
	return &SonarReasoningPro{perplexityOptions{maxTokens: 8192, temperature: 0.2}}
}

// SonarDeepResearch represents the Sonar Deep Research model (in-depth research)
type SonarDeepResearch struct{ perplexityOptions }

func (m *SonarDeepResearch) ModelName() string      { return "sonar-deep-research" }
func (m *SonarDeepResearch) Provider() ProviderType { return ProviderPerplexity }
func (m *SonarDeepResearch) SystemPrompt() string   { return m.systemPrompt }

func (m *SonarDeepResearch) WithMaxTokens(n int) *SonarDeepResearch { m.maxTokens = n; return m }
func (m *SonarDeepResearch) WithTemperature(t float64) *SonarDeepResearch {
	m.temperature = t
	return m
}
func (m *SonarDeepResearch) WithTopP(p float64) *SonarDeepResearch { m.topP = p; return m }
func (m *SonarDeepResearch) WithTopK(k int) *SonarDeepResearch     { m.topK = k; return m }
func (m *SonarDeepResearch) WithSystemPrompt(s string) *SonarDeepResearch {
	m.systemPrompt = s
	return m
}
func (m *SonarDeepResearch) WithSearchRecencyFilter(f string) *SonarDeepResearch {
	m.searchRecencyFilter = f
	return m
}
func (m *SonarDeepResearch) WithSearchDomainFilter(domains []string) *SonarDeepResearch {
	m.searchDomainFilter = domains
	return m
}
func (m *SonarDeepResearch) WithReturnImages(b bool) *SonarDeepResearch { m.returnImages = b; return m }
func (m *SonarDeepResearch) WithReturnRelatedQuestions(b bool) *SonarDeepResearch {
	m.returnRelatedQuestions = b
	return m
}

// NewSonarDeepResearch creates a new Sonar Deep Research model with default options
func NewSonarDeepResearch() *SonarDeepResearch {
	return &SonarDeepResearch{perplexityOptions{maxTokens: 16384, temperature: 0.2}}
}

// ============================================================================
// PERPLEXITY PROVIDER CLIENT
// ============================================================================

// perplexityClient implements the Provider interface for Perplexity
type perplexityClient struct {
	client      *perplexity.Client
	timeout     time.Duration
	logger      Logger
	rateLimiter *rateLimiter
}

// newPerplexityClient creates a new Perplexity client
func newPerplexityClient(config *PerplexityConfig, logger Logger) (*perplexityClient, error) {
	if config.APIKey == "" {
		return nil, fmt.Errorf("perplexity API key is required")
	}

	timeout := config.Timeout
	if timeout == 0 {
		timeout = defaultTimeout()
	}

	client, err := perplexity.NewClient(perplexity.ClientConfig{
		APIKey:  config.APIKey,
		Timeout: timeout,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create perplexity client: %w", err)
	}

	return &perplexityClient{
		client:      client,
		timeout:     timeout,
		logger:      logger,
		rateLimiter: newRateLimiter(config.RateLimiter, logger),
	}, nil
}

// Generate generates text using Perplexity's Grounded LLM API (Chat Completions)
func (c *perplexityClient) Generate(ctx context.Context, model Model, prompt string) (*GenerationResponse, error) {
	// Verify model is for Perplexity
	if model.Provider() != ProviderPerplexity {
		return nil, fmt.Errorf("model %s is not a Perplexity model", model.ModelName())
	}

	// Set timeout
	ctx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()

	// Build messages
	var messages []perplexity.Message

	// Add system message if provided
	if model.SystemPrompt() != "" {
		messages = append(messages, perplexity.Message{
			Role:    "system",
			Content: model.SystemPrompt(),
		})
	}

	// Add user message
	messages = append(messages, perplexity.Message{
		Role:    "user",
		Content: prompt,
	})

	// Build request
	req := perplexity.ChatCompletionRequest{
		Model:    model.ModelName(),
		Messages: messages,
	}

	// Apply options based on model type
	switch m := model.(type) {
	case *Sonar:
		if m.maxTokens > 0 {
			req.MaxTokens = m.maxTokens
		}
		if m.temperature > 0 {
			req.Temperature = &m.temperature
		}
		if m.topP > 0 {
			req.TopP = &m.topP
		}
		if m.topK > 0 {
			req.TopK = m.topK
		}
		if m.searchRecencyFilter != "" {
			req.SearchRecencyFilter = m.searchRecencyFilter
		}
		if len(m.searchDomainFilter) > 0 {
			req.SearchDomainFilter = m.searchDomainFilter
		}
		req.ReturnImages = m.returnImages
		req.ReturnRelatedQuestions = m.returnRelatedQuestions

	case *SonarPro:
		if m.maxTokens > 0 {
			req.MaxTokens = m.maxTokens
		}
		if m.temperature > 0 {
			req.Temperature = &m.temperature
		}
		if m.topP > 0 {
			req.TopP = &m.topP
		}
		if m.topK > 0 {
			req.TopK = m.topK
		}
		if m.searchRecencyFilter != "" {
			req.SearchRecencyFilter = m.searchRecencyFilter
		}
		if len(m.searchDomainFilter) > 0 {
			req.SearchDomainFilter = m.searchDomainFilter
		}
		req.ReturnImages = m.returnImages
		req.ReturnRelatedQuestions = m.returnRelatedQuestions

	case *SonarReasoning:
		if m.maxTokens > 0 {
			req.MaxTokens = m.maxTokens
		}
		if m.temperature > 0 {
			req.Temperature = &m.temperature
		}
		if m.topP > 0 {
			req.TopP = &m.topP
		}
		if m.topK > 0 {
			req.TopK = m.topK
		}
		if m.searchRecencyFilter != "" {
			req.SearchRecencyFilter = m.searchRecencyFilter
		}
		if len(m.searchDomainFilter) > 0 {
			req.SearchDomainFilter = m.searchDomainFilter
		}
		req.ReturnImages = m.returnImages
		req.ReturnRelatedQuestions = m.returnRelatedQuestions

	case *SonarReasoningPro:
		if m.maxTokens > 0 {
			req.MaxTokens = m.maxTokens
		}
		if m.temperature > 0 {
			req.Temperature = &m.temperature
		}
		if m.topP > 0 {
			req.TopP = &m.topP
		}
		if m.topK > 0 {
			req.TopK = m.topK
		}
		if m.searchRecencyFilter != "" {
			req.SearchRecencyFilter = m.searchRecencyFilter
		}
		if len(m.searchDomainFilter) > 0 {
			req.SearchDomainFilter = m.searchDomainFilter
		}
		req.ReturnImages = m.returnImages
		req.ReturnRelatedQuestions = m.returnRelatedQuestions

	case *SonarDeepResearch:
		if m.maxTokens > 0 {
			req.MaxTokens = m.maxTokens
		}
		if m.temperature > 0 {
			req.Temperature = &m.temperature
		}
		if m.topP > 0 {
			req.TopP = &m.topP
		}
		if m.topK > 0 {
			req.TopK = m.topK
		}
		if m.searchRecencyFilter != "" {
			req.SearchRecencyFilter = m.searchRecencyFilter
		}
		if len(m.searchDomainFilter) > 0 {
			req.SearchDomainFilter = m.searchDomainFilter
		}
		req.ReturnImages = m.returnImages
		req.ReturnRelatedQuestions = m.returnRelatedQuestions
	}

	c.logger.Debug().
		Str("model", model.ModelName()).
		Int("message_count", len(messages)).
		Msg("Making Perplexity API request")

	// Make request with rate limit handling
	var resp *perplexity.ChatCompletionResponse
	err := c.rateLimiter.Execute(ctx, func() error {
		var reqErr error
		resp, reqErr = c.client.ChatCompletions(ctx, req)
		return reqErr
	})
	if err != nil {
		c.logger.Error().
			Err(err).
			Str("model", model.ModelName()).
			Str("prompt_preview", truncateString(prompt, 100)).
			Msg("Perplexity generation failed")
		return nil, fmt.Errorf("perplexity generation failed: %w", err)
	}

	if len(resp.Choices) == 0 {
		return nil, fmt.Errorf("no response choices returned from Perplexity")
	}

	choice := resp.Choices[0]

	// Build response
	response := &GenerationResponse{
		Text:         choice.Message.Content,
		Model:        resp.Model,
		FinishReason: choice.FinishReason,
		Usage: TokenUsage{
			PromptTokens:     resp.Usage.PromptTokens,
			CompletionTokens: resp.Usage.CompletionTokens,
			TotalTokens:      resp.Usage.TotalTokens,
		},
		Metadata: map[string]string{
			"provider": "perplexity",
			"model":    resp.Model,
			"id":       resp.ID,
		},
	}

	// Add citations to metadata if present
	if len(resp.Citations) > 0 {
		citationsJSON, _ := json.Marshal(resp.Citations)
		response.Metadata["citations"] = string(citationsJSON)
		response.Metadata["citations_count"] = fmt.Sprintf("%d", len(resp.Citations))
	}

	// Add related questions to metadata if present
	if len(resp.RelatedQuestions) > 0 {
		questionsJSON, _ := json.Marshal(resp.RelatedQuestions)
		response.Metadata["related_questions"] = string(questionsJSON)
	}

	// Add images to metadata if present
	if len(resp.Images) > 0 {
		imagesJSON, _ := json.Marshal(resp.Images)
		response.Metadata["images"] = string(imagesJSON)
	}

	c.logger.Debug().
		Str("model", resp.Model).
		Int("prompt_tokens", resp.Usage.PromptTokens).
		Int("completion_tokens", resp.Usage.CompletionTokens).
		Int("total_tokens", resp.Usage.TotalTokens).
		Int("citations", len(resp.Citations)).
		Msg("Perplexity generation completed")

	return response, nil
}

// Search performs a web search using Perplexity's Search API
func (c *perplexityClient) Search(ctx context.Context, query string, options *SearchOptions) (*SearchResponse, error) {
	ctx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()

	req := perplexity.SearchRequest{
		Query: query,
	}

	if options != nil {
		req.RecencyFilter = options.RecencyFilter
		req.DomainFilter = options.DomainFilter
		req.CountryCode = options.CountryCode
		req.LanguageCode = options.LanguageCode
		req.ReturnImages = options.ReturnImages
		req.SafeSearch = options.SafeSearch
	}

	c.logger.Debug().
		Str("query", truncateString(query, 100)).
		Msg("Making Perplexity Search API request")

	var resp *perplexity.SearchResponse
	err := c.rateLimiter.Execute(ctx, func() error {
		var reqErr error
		resp, reqErr = c.client.Search(ctx, req)
		return reqErr
	})
	if err != nil {
		c.logger.Error().
			Err(err).
			Str("query", truncateString(query, 100)).
			Msg("Perplexity search failed")
		return nil, fmt.Errorf("perplexity search failed: %w", err)
	}

	// Convert to our SearchResponse type
	result := &SearchResponse{
		Results: make([]SearchResult, len(resp.Results)),
	}

	for i, r := range resp.Results {
		result.Results[i] = SearchResult{
			Title:         r.Title,
			URL:           r.URL,
			Snippet:       r.Snippet,
			DatePublished: r.DatePublished,
			Author:        r.Author,
		}
	}

	if len(resp.Images) > 0 {
		result.Images = make([]ImageResult, len(resp.Images))
		for i, img := range resp.Images {
			result.Images[i] = ImageResult{
				URL:       img.URL,
				SourceURL: img.SourceURL,
				Alt:       img.Alt,
				Width:     img.Width,
				Height:    img.Height,
			}
		}
	}

	c.logger.Debug().
		Int("results", len(result.Results)).
		Int("images", len(result.Images)).
		Msg("Perplexity search completed")

	return result, nil
}

// Health checks the health of the Perplexity client
func (c *perplexityClient) Health(ctx context.Context) error {
	ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	return c.client.Ping(ctx, "sonar")
}

// Close closes the Perplexity client (no-op as HTTP client doesn't need closing)
func (c *perplexityClient) Close() error {
	return nil
}

// ============================================================================
// SEARCH API TYPES
// ============================================================================

// SearchOptions contains options for Perplexity Search API
type SearchOptions struct {
	// RecencyFilter filters results by time: "hour", "day", "week", "month", "year"
	RecencyFilter string
	// DomainFilter limits search to specific domains
	DomainFilter []string
	// CountryCode filters results by country (e.g., "us", "gb")
	CountryCode string
	// LanguageCode filters results by language (e.g., "en", "fr")
	LanguageCode string
	// ReturnImages includes image results
	ReturnImages bool
	// SafeSearch enables safe search mode
	SafeSearch bool
}

// SearchResponse contains the response from Perplexity Search API
type SearchResponse struct {
	// Results contains the search results
	Results []SearchResult
	// Images contains image results if requested
	Images []ImageResult
}

// SearchResult represents a single search result
type SearchResult struct {
	// Title is the page title
	Title string
	// URL is the result URL
	URL string
	// Snippet is the text snippet from the page
	Snippet string
	// DatePublished is when the content was published
	DatePublished string
	// Author is the content author if available
	Author string
}

// ImageResult represents an image search result
type ImageResult struct {
	// URL is the image URL
	URL string
	// SourceURL is the page where the image was found
	SourceURL string
	// Alt is the image alt text
	Alt string
	// Width is the image width
	Width int
	// Height is the image height
	Height int
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

// GetPerplexityClient returns the underlying Perplexity client for Search API access
func GetPerplexityClient(g *LLMGateway) (*perplexityClient, error) {
	g.mu.RLock()
	provider, exists := g.providers[ProviderPerplexity]
	g.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("perplexity provider is not registered")
	}

	client, ok := provider.(*perplexityClient)
	if !ok {
		return nil, fmt.Errorf("invalid perplexity provider type")
	}

	return client, nil
}
