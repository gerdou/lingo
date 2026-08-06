package lingo

import (
	"context"
	"fmt"
	"strings"
	"time"

	cohere "github.com/cohere-ai/cohere-go/v2"
	cohereclient "github.com/cohere-ai/cohere-go/v2/client"
	cohereoption "github.com/cohere-ai/cohere-go/v2/option"
)

func init() {
	RegisterProvider(ProviderCohere, func(config ProviderConfig, logger Logger) (Provider, error) {
		cfg, ok := config.(*CohereConfig)
		if !ok {
			return nil, fmt.Errorf("invalid config type for Cohere provider")
		}
		return newCohereClient(cfg, logger)
	})
}

// ============================================================================
// COHERE PROVIDER CONFIG
// ============================================================================

// CohereConfig contains configuration for the Cohere provider
type CohereConfig struct {
	// APIKey is the Cohere API key (required)
	APIKey string
	// Timeout is the request timeout (default: 60s)
	Timeout time.Duration
	// RateLimiter is the optional rate limit configuration
	RateLimiter *RateLimitConfig
	// BaseURL is an optional custom base URL, for private deployments
	BaseURL string
}

// Implement ProviderConfig interface
func (c *CohereConfig) providerType() ProviderType        { return ProviderCohere }
func (c *CohereConfig) apiKey() string                    { return c.APIKey }
func (c *CohereConfig) timeout() time.Duration            { return c.Timeout }
func (c *CohereConfig) rateLimitConfig() *RateLimitConfig { return c.RateLimiter }

// Safety modes accepted by Cohere's chat API. CONTEXTUAL is the API default;
// OFF omits the safety instruction entirely and is unavailable on the newer
// models, which take CONTEXTUAL or STRICT only.
const (
	CohereSafetyContextual = string(cohere.V2ChatRequestSafetyModeContextual)
	CohereSafetyStrict     = string(cohere.V2ChatRequestSafetyModeStrict)
	CohereSafetyOff        = string(cohere.V2ChatRequestSafetyModeOff)
)

// ============================================================================
// SHARED OPTIONS (embedded in model structs)
// ============================================================================

// cohereOptions contains options for Cohere Command models
type cohereOptions struct {
	modelVersion  string // Optional: override model name with specific version
	maxTokens     int
	temperature   float64
	topP          float64 // Sent as "p"
	topK          int     // Sent as "k"
	seed          int
	systemPrompt  string
	stopSequences []string
	safetyMode    string
	// Thinking is on by default on models that support it, so lingo sends the
	// field only once a setter has asked for a change
	thinkingSet     bool
	thinkingEnabled bool
	thinkingBudget  int
	reasoning       bool
}

// SystemPrompt satisfies Model for every type embedding cohereOptions.
func (o *cohereOptions) SystemPrompt() string { return o.systemPrompt }

// cohereOpts exposes the embedded option set to the client.
func (o *cohereOptions) cohereOpts() *cohereOptions { return o }

// cohereModel is implemented by every model routed through the Cohere client.
type cohereModel interface {
	Model
	cohereOpts() *cohereOptions
}

// resolveCohereModelName returns the version override when one is set.
func resolveCohereModelName(o *cohereOptions, defaultName string) string {
	if o.modelVersion != "" {
		return o.modelVersion
	}
	return defaultName
}

// ============================================================================
// COMMAND MODELS
// ============================================================================

// CommandAPlus represents command-a-plus-05-2026, Cohere's flagship and its
// first mixture-of-experts model, combining vision, reasoning and translation.
// 128K token context window, 64K max output tokens.
type CommandAPlus struct{ cohereOptions }

func (m *CommandAPlus) ModelName() string {
	return resolveCohereModelName(&m.cohereOptions, "command-a-plus-05-2026")
}
func (m *CommandAPlus) Provider() ProviderType { return ProviderCohere }

func (m *CommandAPlus) WithVersion(v string) *CommandAPlus      { m.modelVersion = v; return m }
func (m *CommandAPlus) WithMaxTokens(n int) *CommandAPlus       { m.maxTokens = n; return m }
func (m *CommandAPlus) WithTemperature(t float64) *CommandAPlus { m.temperature = t; return m }
func (m *CommandAPlus) WithTopP(p float64) *CommandAPlus        { m.topP = p; return m }
func (m *CommandAPlus) WithTopK(k int) *CommandAPlus            { m.topK = k; return m }
func (m *CommandAPlus) WithSeed(s int) *CommandAPlus            { m.seed = s; return m }
func (m *CommandAPlus) WithSystemPrompt(s string) *CommandAPlus { m.systemPrompt = s; return m }
func (m *CommandAPlus) WithStopSequences(s []string) *CommandAPlus {
	m.stopSequences = s
	return m
}
func (m *CommandAPlus) WithSafetyMode(mode string) *CommandAPlus { m.safetyMode = mode; return m }

// WithThinkingDisabled turns reasoning off, trading depth for latency
func (m *CommandAPlus) WithThinkingDisabled() *CommandAPlus {
	m.thinkingSet, m.thinkingEnabled, m.reasoning = true, false, false
	return m
}

// WithThinkingBudget caps the tokens spent on reasoning and enables thinking
func (m *CommandAPlus) WithThinkingBudget(tokens int) *CommandAPlus {
	m.thinkingSet, m.thinkingEnabled, m.reasoning = true, true, true
	m.thinkingBudget = tokens
	return m
}

// NewCommandAPlus creates a new Command A+ model with default options
func NewCommandAPlus() *CommandAPlus {
	return &CommandAPlus{cohereOptions{maxTokens: 4096, reasoning: true}}
}

// CommandA represents command-a-03-2025, the previous flagship and the widest
// context in the family. 256K token context window, 8K max output tokens.
type CommandA struct{ cohereOptions }

func (m *CommandA) ModelName() string {
	return resolveCohereModelName(&m.cohereOptions, "command-a-03-2025")
}
func (m *CommandA) Provider() ProviderType { return ProviderCohere }

func (m *CommandA) WithVersion(v string) *CommandA         { m.modelVersion = v; return m }
func (m *CommandA) WithMaxTokens(n int) *CommandA          { m.maxTokens = n; return m }
func (m *CommandA) WithTemperature(t float64) *CommandA    { m.temperature = t; return m }
func (m *CommandA) WithTopP(p float64) *CommandA           { m.topP = p; return m }
func (m *CommandA) WithTopK(k int) *CommandA               { m.topK = k; return m }
func (m *CommandA) WithSeed(s int) *CommandA               { m.seed = s; return m }
func (m *CommandA) WithSystemPrompt(s string) *CommandA    { m.systemPrompt = s; return m }
func (m *CommandA) WithStopSequences(s []string) *CommandA { m.stopSequences = s; return m }
func (m *CommandA) WithSafetyMode(mode string) *CommandA   { m.safetyMode = mode; return m }

// NewCommandA creates a new Command A model with default options
func NewCommandA() *CommandA {
	return &CommandA{cohereOptions{maxTokens: 4096}}
}

// CommandAReasoning represents command-a-reasoning-08-2025, tuned for
// multi-step reasoning. 256K token context window, 32K max output tokens.
type CommandAReasoning struct{ cohereOptions }

func (m *CommandAReasoning) ModelName() string {
	return resolveCohereModelName(&m.cohereOptions, "command-a-reasoning-08-2025")
}
func (m *CommandAReasoning) Provider() ProviderType { return ProviderCohere }

func (m *CommandAReasoning) WithVersion(v string) *CommandAReasoning { m.modelVersion = v; return m }
func (m *CommandAReasoning) WithMaxTokens(n int) *CommandAReasoning  { m.maxTokens = n; return m }
func (m *CommandAReasoning) WithTemperature(t float64) *CommandAReasoning {
	m.temperature = t
	return m
}
func (m *CommandAReasoning) WithTopP(p float64) *CommandAReasoning { m.topP = p; return m }
func (m *CommandAReasoning) WithTopK(k int) *CommandAReasoning     { m.topK = k; return m }
func (m *CommandAReasoning) WithSeed(s int) *CommandAReasoning     { m.seed = s; return m }
func (m *CommandAReasoning) WithSystemPrompt(s string) *CommandAReasoning {
	m.systemPrompt = s
	return m
}
func (m *CommandAReasoning) WithStopSequences(s []string) *CommandAReasoning {
	m.stopSequences = s
	return m
}
func (m *CommandAReasoning) WithSafetyMode(mode string) *CommandAReasoning {
	m.safetyMode = mode
	return m
}

// WithThinkingDisabled turns reasoning off, trading depth for latency
func (m *CommandAReasoning) WithThinkingDisabled() *CommandAReasoning {
	m.thinkingSet, m.thinkingEnabled, m.reasoning = true, false, false
	return m
}

// WithThinkingBudget caps the tokens spent on reasoning and enables thinking
func (m *CommandAReasoning) WithThinkingBudget(tokens int) *CommandAReasoning {
	m.thinkingSet, m.thinkingEnabled, m.reasoning = true, true, true
	m.thinkingBudget = tokens
	return m
}

// NewCommandAReasoning creates a new Command A Reasoning model with default options
func NewCommandAReasoning() *CommandAReasoning {
	return &CommandAReasoning{cohereOptions{maxTokens: 8192, reasoning: true}}
}

// CommandAVision represents command-a-vision-07-2025, tuned for image
// understanding. 128K token context window, 8K max output tokens.
type CommandAVision struct{ cohereOptions }

func (m *CommandAVision) ModelName() string {
	return resolveCohereModelName(&m.cohereOptions, "command-a-vision-07-2025")
}
func (m *CommandAVision) Provider() ProviderType { return ProviderCohere }

func (m *CommandAVision) WithVersion(v string) *CommandAVision      { m.modelVersion = v; return m }
func (m *CommandAVision) WithMaxTokens(n int) *CommandAVision       { m.maxTokens = n; return m }
func (m *CommandAVision) WithTemperature(t float64) *CommandAVision { m.temperature = t; return m }
func (m *CommandAVision) WithTopP(p float64) *CommandAVision        { m.topP = p; return m }
func (m *CommandAVision) WithTopK(k int) *CommandAVision            { m.topK = k; return m }
func (m *CommandAVision) WithSeed(s int) *CommandAVision            { m.seed = s; return m }
func (m *CommandAVision) WithSystemPrompt(s string) *CommandAVision {
	m.systemPrompt = s
	return m
}
func (m *CommandAVision) WithStopSequences(s []string) *CommandAVision {
	m.stopSequences = s
	return m
}
func (m *CommandAVision) WithSafetyMode(mode string) *CommandAVision { m.safetyMode = mode; return m }

// NewCommandAVision creates a new Command A Vision model with default options
func NewCommandAVision() *CommandAVision {
	return &CommandAVision{cohereOptions{maxTokens: 4096}}
}

// CommandATranslate represents command-a-translate-08-2025, tuned for
// translation. 8K token context window, 8K max output tokens.
type CommandATranslate struct{ cohereOptions }

func (m *CommandATranslate) ModelName() string {
	return resolveCohereModelName(&m.cohereOptions, "command-a-translate-08-2025")
}
func (m *CommandATranslate) Provider() ProviderType { return ProviderCohere }

func (m *CommandATranslate) WithVersion(v string) *CommandATranslate { m.modelVersion = v; return m }
func (m *CommandATranslate) WithMaxTokens(n int) *CommandATranslate  { m.maxTokens = n; return m }
func (m *CommandATranslate) WithTemperature(t float64) *CommandATranslate {
	m.temperature = t
	return m
}
func (m *CommandATranslate) WithTopP(p float64) *CommandATranslate { m.topP = p; return m }
func (m *CommandATranslate) WithTopK(k int) *CommandATranslate     { m.topK = k; return m }
func (m *CommandATranslate) WithSeed(s int) *CommandATranslate     { m.seed = s; return m }
func (m *CommandATranslate) WithSystemPrompt(s string) *CommandATranslate {
	m.systemPrompt = s
	return m
}
func (m *CommandATranslate) WithStopSequences(s []string) *CommandATranslate {
	m.stopSequences = s
	return m
}

// NewCommandATranslate creates a new Command A Translate model with default options
func NewCommandATranslate() *CommandATranslate {
	return &CommandATranslate{cohereOptions{maxTokens: 4096}}
}

// CommandR7B represents command-r7b-12-2024, the smallest and cheapest model
// in the family. 128K token context window, 4K max output tokens.
type CommandR7B struct{ cohereOptions }

func (m *CommandR7B) ModelName() string {
	return resolveCohereModelName(&m.cohereOptions, "command-r7b-12-2024")
}
func (m *CommandR7B) Provider() ProviderType { return ProviderCohere }

func (m *CommandR7B) WithVersion(v string) *CommandR7B         { m.modelVersion = v; return m }
func (m *CommandR7B) WithMaxTokens(n int) *CommandR7B          { m.maxTokens = n; return m }
func (m *CommandR7B) WithTemperature(t float64) *CommandR7B    { m.temperature = t; return m }
func (m *CommandR7B) WithTopP(p float64) *CommandR7B           { m.topP = p; return m }
func (m *CommandR7B) WithTopK(k int) *CommandR7B               { m.topK = k; return m }
func (m *CommandR7B) WithSeed(s int) *CommandR7B               { m.seed = s; return m }
func (m *CommandR7B) WithSystemPrompt(s string) *CommandR7B    { m.systemPrompt = s; return m }
func (m *CommandR7B) WithStopSequences(s []string) *CommandR7B { m.stopSequences = s; return m }
func (m *CommandR7B) WithSafetyMode(mode string) *CommandR7B   { m.safetyMode = mode; return m }

// NewCommandR7B creates a new Command R7B model with default options
func NewCommandR7B() *CommandR7B {
	return &CommandR7B{cohereOptions{maxTokens: 4096}}
}

// CommandR represents command-r-08-2024.
// 128K token context window, 4K max output tokens.
type CommandR struct{ cohereOptions }

func (m *CommandR) ModelName() string {
	return resolveCohereModelName(&m.cohereOptions, "command-r-08-2024")
}
func (m *CommandR) Provider() ProviderType { return ProviderCohere }

func (m *CommandR) WithVersion(v string) *CommandR         { m.modelVersion = v; return m }
func (m *CommandR) WithMaxTokens(n int) *CommandR          { m.maxTokens = n; return m }
func (m *CommandR) WithTemperature(t float64) *CommandR    { m.temperature = t; return m }
func (m *CommandR) WithTopP(p float64) *CommandR           { m.topP = p; return m }
func (m *CommandR) WithTopK(k int) *CommandR               { m.topK = k; return m }
func (m *CommandR) WithSeed(s int) *CommandR               { m.seed = s; return m }
func (m *CommandR) WithSystemPrompt(s string) *CommandR    { m.systemPrompt = s; return m }
func (m *CommandR) WithStopSequences(s []string) *CommandR { m.stopSequences = s; return m }
func (m *CommandR) WithSafetyMode(mode string) *CommandR   { m.safetyMode = mode; return m }

// NewCommandR creates a new Command R model with default options
func NewCommandR() *CommandR {
	return &CommandR{cohereOptions{maxTokens: 4096}}
}

// CommandRPlus represents command-r-plus-08-2024.
// 128K token context window, 4K max output tokens.
type CommandRPlus struct{ cohereOptions }

func (m *CommandRPlus) ModelName() string {
	return resolveCohereModelName(&m.cohereOptions, "command-r-plus-08-2024")
}
func (m *CommandRPlus) Provider() ProviderType { return ProviderCohere }

func (m *CommandRPlus) WithVersion(v string) *CommandRPlus      { m.modelVersion = v; return m }
func (m *CommandRPlus) WithMaxTokens(n int) *CommandRPlus       { m.maxTokens = n; return m }
func (m *CommandRPlus) WithTemperature(t float64) *CommandRPlus { m.temperature = t; return m }
func (m *CommandRPlus) WithTopP(p float64) *CommandRPlus        { m.topP = p; return m }
func (m *CommandRPlus) WithTopK(k int) *CommandRPlus            { m.topK = k; return m }
func (m *CommandRPlus) WithSeed(s int) *CommandRPlus            { m.seed = s; return m }
func (m *CommandRPlus) WithSystemPrompt(s string) *CommandRPlus { m.systemPrompt = s; return m }
func (m *CommandRPlus) WithStopSequences(s []string) *CommandRPlus {
	m.stopSequences = s
	return m
}
func (m *CommandRPlus) WithSafetyMode(mode string) *CommandRPlus { m.safetyMode = mode; return m }

// NewCommandRPlus creates a new Command R+ model with default options
func NewCommandRPlus() *CommandRPlus {
	return &CommandRPlus{cohereOptions{maxTokens: 4096}}
}

// CohereModel is any Cohere model addressed by its raw ID, for models newer
// than this package or for fine-tuned model IDs.
type CohereModel struct {
	cohereOptions
	modelID string
}

func (m *CohereModel) ModelName() string      { return m.modelID }
func (m *CohereModel) Provider() ProviderType { return ProviderCohere }

func (m *CohereModel) WithMaxTokens(n int) *CohereModel          { m.maxTokens = n; return m }
func (m *CohereModel) WithTemperature(t float64) *CohereModel    { m.temperature = t; return m }
func (m *CohereModel) WithTopP(p float64) *CohereModel           { m.topP = p; return m }
func (m *CohereModel) WithTopK(k int) *CohereModel               { m.topK = k; return m }
func (m *CohereModel) WithSeed(s int) *CohereModel               { m.seed = s; return m }
func (m *CohereModel) WithSystemPrompt(s string) *CohereModel    { m.systemPrompt = s; return m }
func (m *CohereModel) WithStopSequences(s []string) *CohereModel { m.stopSequences = s; return m }
func (m *CohereModel) WithSafetyMode(mode string) *CohereModel   { m.safetyMode = mode; return m }
func (m *CohereModel) WithThinkingDisabled() *CohereModel {
	m.thinkingSet, m.thinkingEnabled, m.reasoning = true, false, false
	return m
}
func (m *CohereModel) WithThinkingBudget(tokens int) *CohereModel {
	m.thinkingSet, m.thinkingEnabled, m.reasoning = true, true, true
	m.thinkingBudget = tokens
	return m
}

// NewCohereModel creates a Cohere model by ID, e.g. "command-a-plus-05-2026"
func NewCohereModel(modelID string) *CohereModel {
	return &CohereModel{modelID: modelID}
}

// Compile-time check that every Cohere model routes through the client
var (
	_ cohereModel = (*CommandAPlus)(nil)
	_ cohereModel = (*CommandA)(nil)
	_ cohereModel = (*CommandAReasoning)(nil)
	_ cohereModel = (*CommandAVision)(nil)
	_ cohereModel = (*CommandATranslate)(nil)
	_ cohereModel = (*CommandR7B)(nil)
	_ cohereModel = (*CommandR)(nil)
	_ cohereModel = (*CommandRPlus)(nil)
	_ cohereModel = (*CohereModel)(nil)
)

// ============================================================================
// CLIENT
// ============================================================================

// cohereClient implements the Provider interface for Cohere
type cohereClient struct {
	client      *cohereclient.Client
	timeout     time.Duration
	logger      Logger
	rateLimiter *rateLimiter
}

// newCohereClient creates a new Cohere client using the official SDK
func newCohereClient(config *CohereConfig, logger Logger) (*cohereClient, error) {
	if config.APIKey == "" {
		return nil, fmt.Errorf("cohere API key is required")
	}

	opts := []cohereoption.RequestOption{cohereclient.WithToken(config.APIKey)}
	if config.BaseURL != "" {
		opts = append(opts, cohereclient.WithBaseURL(config.BaseURL))
	}

	timeout := config.Timeout
	if timeout == 0 {
		timeout = defaultTimeout()
	}

	return &cohereClient{
		client:      cohereclient.NewClient(opts...),
		timeout:     timeout,
		logger:      logger,
		rateLimiter: newRateLimiter(config.RateLimiter, logger),
	}, nil
}

// Generate generates text using Cohere's v2 Chat API
func (c *cohereClient) Generate(ctx context.Context, model Model, prompt string) (*GenerationResponse, error) {
	// Verify model is for Cohere
	if model.Provider() != ProviderCohere {
		return nil, fmt.Errorf("model %s is not a Cohere model", model.ModelName())
	}

	m, ok := model.(cohereModel)
	if !ok {
		return nil, fmt.Errorf("model %s does not carry Cohere generation options", model.ModelName())
	}
	opts := m.cohereOpts()

	// Set timeout
	ctx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()

	// Build messages with optional system prompt
	var messages cohere.ChatMessages
	if s := model.SystemPrompt(); s != "" {
		messages = append(messages, &cohere.ChatMessageV2{
			Role:   "system",
			System: &cohere.SystemMessageV2{Content: &cohere.SystemMessageV2Content{String: s}},
		})
	}
	messages = append(messages, &cohere.ChatMessageV2{
		Role: "user",
		User: &cohere.UserMessageV2{Content: &cohere.UserMessageV2Content{String: prompt}},
	})

	req := &cohere.V2ChatRequest{
		Model:    model.ModelName(),
		Messages: messages,
	}
	if opts.maxTokens > 0 {
		req.MaxTokens = &opts.maxTokens
	}
	if opts.temperature > 0 {
		req.Temperature = &opts.temperature
	}
	if opts.topP > 0 {
		req.P = &opts.topP
	}
	if opts.topK > 0 {
		req.K = &opts.topK
	}
	if opts.seed > 0 {
		req.Seed = &opts.seed
	}
	if len(opts.stopSequences) > 0 {
		req.StopSequences = opts.stopSequences
	}
	if opts.safetyMode != "" {
		mode := cohere.V2ChatRequestSafetyMode(opts.safetyMode)
		req.SafetyMode = &mode
	}
	if opts.thinkingSet {
		thinking := &cohere.Thinking{Type: cohere.ThinkingTypeDisabled}
		if opts.thinkingEnabled {
			thinking.Type = cohere.ThinkingTypeEnabled
			if opts.thinkingBudget > 0 {
				thinking.TokenBudget = &opts.thinkingBudget
			}
		}
		req.Thinking = thinking
	}

	c.logger.Debug().
		Str("model", model.ModelName()).
		Bool("is_reasoning_model", opts.reasoning).
		Msg("Making Cohere API request")

	// Make request with rate limit handling
	var resp *cohere.V2ChatResponse
	err := c.rateLimiter.Execute(ctx, func() error {
		var reqErr error
		resp, reqErr = c.client.V2.Chat(ctx, req)
		return reqErr
	})
	if err != nil {
		c.logger.Error().
			Err(err).
			Str("model", model.ModelName()).
			Str("prompt_preview", truncateString(prompt, 100)).
			Msg("Cohere generation failed")
		return nil, fmt.Errorf("cohere generation failed: %w", err)
	}

	if resp.Message == nil {
		return nil, fmt.Errorf("no message returned from Cohere")
	}

	// Content arrives as a list of text and thinking blocks
	var text, thinking strings.Builder
	for _, item := range resp.Message.Content {
		if item == nil {
			continue
		}
		if item.Text != nil {
			text.WriteString(item.Text.Text)
		}
		if item.Thinking != nil {
			thinking.WriteString(item.Thinking.Thinking)
		}
	}

	response := &GenerationResponse{
		Text:         text.String(),
		Model:        model.ModelName(),
		FinishReason: string(resp.FinishReason),
		Usage:        cohereUsage(resp.Usage),
		Metadata: map[string]string{
			"provider":           "cohere",
			"model":              model.ModelName(),
			"is_reasoning_model": fmt.Sprintf("%t", opts.reasoning),
		},
	}
	if resp.Id != "" {
		response.Metadata["response_id"] = resp.Id
	}
	if thinking.Len() > 0 {
		response.Metadata["reasoning_content"] = thinking.String()
	}
	if resp.Message.ToolPlan != nil && *resp.Message.ToolPlan != "" {
		response.Metadata["tool_plan"] = *resp.Message.ToolPlan
	}

	c.logger.Debug().
		Str("model", model.ModelName()).
		Bool("is_reasoning_model", opts.reasoning).
		Int("prompt_tokens", response.Usage.PromptTokens).
		Int("completion_tokens", response.Usage.CompletionTokens).
		Int("total_tokens", response.Usage.TotalTokens).
		Msg("Cohere generation completed")

	return response, nil
}

// cohereUsage converts Cohere's float token counts into TokenUsage
func cohereUsage(usage *cohere.Usage) TokenUsage {
	if usage == nil || usage.Tokens == nil {
		return TokenUsage{}
	}

	var in, out int
	if usage.Tokens.InputTokens != nil {
		in = int(*usage.Tokens.InputTokens)
	}
	if usage.Tokens.OutputTokens != nil {
		out = int(*usage.Tokens.OutputTokens)
	}

	return TokenUsage{
		PromptTokens:     in,
		CompletionTokens: out,
		TotalTokens:      in + out,
	}
}

// Health checks the health of the Cohere client
func (c *cohereClient) Health(ctx context.Context) error {
	ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	// Validating the key is cheaper than generating
	if _, err := c.client.CheckApiKey(ctx); err != nil {
		return fmt.Errorf("cohere health check failed: %w", err)
	}

	return nil
}

// Close closes the Cohere client (no-op for Cohere)
func (c *cohereClient) Close() error {
	return nil
}
