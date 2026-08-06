package lingo

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/shared"
)

// ============================================================================
// SHARED OPENAI-COMPATIBLE CORE
// ============================================================================
//
// Most providers speak the OpenAI /chat/completions dialect, so xAI, DeepSeek,
// OpenRouter, Azure OpenAI and the generic OpenAI-compatible provider all share
// the client below.
//
// Unlike openai.go, which applies options through a per-model type switch,
// these providers dispatch through the oaiCompatibleModel interface: a new
// model is a struct embedding oaiOptions and needs no edit to the client.

// oaiOptions is the generation option set shared by models served over the
// OpenAI chat completions API. Providers embed it and expose only the setters
// their own API actually honours.
type oaiOptions struct {
	modelVersion        string // Optional: override model name with specific version
	maxTokens           int
	maxCompletionTokens int
	temperature         float64
	topP                float64
	reasoningEffort     string
	systemPrompt        string
	reasoning           bool           // Model reasons before answering; reported in metadata
	extraFields         map[string]any // Provider-specific body fields (e.g. DeepSeek "thinking")
}

// SystemPrompt satisfies Model for every type embedding oaiOptions.
func (o *oaiOptions) SystemPrompt() string { return o.systemPrompt }

// chatOptions exposes the embedded option set to the shared client.
func (o *oaiOptions) chatOptions() *oaiOptions { return o }

// setExtra records a provider-specific field to merge into the request body.
func (o *oaiOptions) setExtra(key string, value any) {
	if o.extraFields == nil {
		o.extraFields = make(map[string]any)
	}
	o.extraFields[key] = value
}

// oaiCompatibleModel is implemented by every model routed through the shared
// OpenAI-compatible client.
type oaiCompatibleModel interface {
	Model
	chatOptions() *oaiOptions
}

// resolveModelName returns the version override when one is set.
func resolveModelName(o *oaiOptions, defaultName string) string {
	if o.modelVersion != "" {
		return o.modelVersion
	}
	return defaultName
}

// ============================================================================
// SHARED CLIENT
// ============================================================================

// oaiCompatClient implements Provider for any endpoint speaking the OpenAI
// chat completions dialect.
type oaiCompatClient struct {
	client   openai.Client
	provider ProviderType
	// label is the human-readable provider name used in errors and logs
	label string
	// healthModel is generated against by Health; when empty Health lists
	// models instead, which every compliant endpoint supports
	healthModel string
	timeout     time.Duration
	logger      Logger
	rateLimiter *rateLimiter
}

// newOAICompatClient builds a shared client from a base set of request options.
func newOAICompatClient(provider ProviderType, label string, healthModel string, timeout time.Duration, rl *RateLimitConfig, logger Logger, opts ...option.RequestOption) *oaiCompatClient {
	if timeout == 0 {
		timeout = defaultTimeout()
	}
	return &oaiCompatClient{
		client:      openai.NewClient(opts...),
		provider:    provider,
		label:       label,
		healthModel: healthModel,
		timeout:     timeout,
		logger:      logger,
		rateLimiter: newRateLimiter(rl, logger),
	}
}

// Generate generates text over the OpenAI chat completions API
func (c *oaiCompatClient) Generate(ctx context.Context, model Model, prompt string) (*GenerationResponse, error) {
	// Verify model belongs to this provider
	if model.Provider() != c.provider {
		return nil, fmt.Errorf("model %s is not a %s model", model.ModelName(), c.label)
	}

	m, ok := model.(oaiCompatibleModel)
	if !ok {
		return nil, fmt.Errorf("model %s does not carry %s generation options", model.ModelName(), c.label)
	}
	opts := m.chatOptions()

	// Set timeout
	ctx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()

	// Build messages with optional system prompt. Third-party endpoints
	// accept "system" universally; "developer" is OpenAI-only.
	var messages []openai.ChatCompletionMessageParamUnion
	if s := model.SystemPrompt(); s != "" {
		messages = append(messages, openai.SystemMessage(s))
	}
	messages = append(messages, openai.UserMessage(prompt))

	params := openai.ChatCompletionNewParams{
		Model:    openai.ChatModel(model.ModelName()),
		Messages: messages,
	}
	if opts.maxTokens > 0 {
		params.MaxTokens = openai.Int(int64(opts.maxTokens))
	}
	if opts.maxCompletionTokens > 0 {
		params.MaxCompletionTokens = openai.Int(int64(opts.maxCompletionTokens))
	}
	if opts.temperature > 0 {
		params.Temperature = openai.Float(opts.temperature)
	}
	if opts.topP > 0 {
		params.TopP = openai.Float(opts.topP)
	}
	if opts.reasoningEffort != "" {
		params.ReasoningEffort = shared.ReasoningEffort(opts.reasoningEffort)
	}

	// Merge provider-specific body fields the typed params don't model
	var reqOpts []option.RequestOption
	for k, v := range opts.extraFields {
		reqOpts = append(reqOpts, option.WithJSONSet(k, v))
	}

	c.logger.Debug().
		Str("provider", string(c.provider)).
		Str("model", model.ModelName()).
		Bool("is_reasoning_model", opts.reasoning).
		Msg("Making " + c.label + " API request")

	// Make request with rate limit handling
	var resp *openai.ChatCompletion
	err := c.rateLimiter.Execute(ctx, func() error {
		var reqErr error
		resp, reqErr = c.client.Chat.Completions.New(ctx, params, reqOpts...)
		return reqErr
	})
	if err != nil {
		c.logger.Error().
			Err(err).
			Str("provider", string(c.provider)).
			Str("model", model.ModelName()).
			Str("prompt_preview", truncateString(prompt, 100)).
			Msg(c.label + " generation failed")
		return nil, fmt.Errorf("%s generation failed: %w", c.label, err)
	}

	if len(resp.Choices) == 0 {
		return nil, fmt.Errorf("no response choices returned from %s", c.label)
	}

	choice := resp.Choices[0]

	// Model echoed by the API can differ from the one requested (OpenRouter
	// routes to a backing provider, Azure echoes the deployment name)
	modelName := resp.Model
	if modelName == "" {
		modelName = model.ModelName()
	}

	response := &GenerationResponse{
		Text:         choice.Message.Content,
		Model:        modelName,
		FinishReason: string(choice.FinishReason),
		Usage: TokenUsage{
			PromptTokens:     int(resp.Usage.PromptTokens),
			CompletionTokens: int(resp.Usage.CompletionTokens),
			TotalTokens:      int(resp.Usage.TotalTokens),
		},
		Metadata: map[string]string{
			"provider":           string(c.provider),
			"model":              modelName,
			"is_reasoning_model": fmt.Sprintf("%t", opts.reasoning),
		},
	}

	// Requested model is worth keeping when the API rewrote it
	if modelName != model.ModelName() {
		response.Metadata["requested_model"] = model.ModelName()
	}

	// Reasoning traces are returned out of band by most of these providers
	if resp.Usage.CompletionTokensDetails.ReasoningTokens > 0 {
		response.Metadata["reasoning_tokens"] = fmt.Sprintf("%d", resp.Usage.CompletionTokensDetails.ReasoningTokens)
	}
	if reasoning := extractReasoningContent(choice.Message); reasoning != "" {
		response.Metadata["reasoning_content"] = reasoning
	}

	c.logger.Debug().
		Str("provider", string(c.provider)).
		Str("model", modelName).
		Bool("is_reasoning_model", opts.reasoning).
		Int64("prompt_tokens", resp.Usage.PromptTokens).
		Int64("completion_tokens", resp.Usage.CompletionTokens).
		Int64("total_tokens", resp.Usage.TotalTokens).
		Msg(c.label + " generation completed")

	return response, nil
}

// extractReasoningContent pulls the non-standard reasoning field that DeepSeek
// and several OpenRouter backends return alongside the answer.
func extractReasoningContent(msg openai.ChatCompletionMessage) string {
	for _, key := range []string{"reasoning_content", "reasoning"} {
		raw, ok := msg.JSON.ExtraFields[key]
		if !ok {
			continue
		}

		value := raw.Raw()
		if value == "" || value == "null" {
			continue
		}

		// The field arrives as raw JSON; unwrap the common string case so
		// callers get the text rather than a quoted literal
		var text string
		if err := json.Unmarshal([]byte(value), &text); err == nil {
			if text == "" {
				continue
			}
			return text
		}
		return value
	}
	return ""
}

// Health checks the health of the endpoint
func (c *oaiCompatClient) Health(ctx context.Context) error {
	ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	// Without a known-good model, listing models is the cheapest call that
	// still proves credentials and connectivity
	if c.healthModel == "" {
		if _, err := c.client.Models.List(ctx); err != nil {
			return fmt.Errorf("%s health check failed: %w", c.label, err)
		}
		return nil
	}

	params := openai.ChatCompletionNewParams{
		Model:     openai.ChatModel(c.healthModel),
		Messages:  []openai.ChatCompletionMessageParamUnion{openai.UserMessage("Hello")},
		MaxTokens: openai.Int(5),
	}

	if _, err := c.client.Chat.Completions.New(ctx, params); err != nil {
		return fmt.Errorf("%s health check failed: %w", c.label, err)
	}

	return nil
}

// Close closes the client (no-op for HTTP-backed providers)
func (c *oaiCompatClient) Close() error {
	return nil
}
