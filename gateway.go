package lingo

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/rs/zerolog"
)

// ProviderFactory creates a new provider instance from a provider config
type ProviderFactory func(config ProviderConfig, logger Logger) (Provider, error)

// providerFactories maps provider types to their factory functions
var (
	providerFactories   = make(map[ProviderType]ProviderFactory)
	providerFactoriesMu sync.RWMutex
)

// RegisterProvider registers a provider factory. Called by provider implementations in their init().
func RegisterProvider(providerType ProviderType, factory ProviderFactory) {
	providerFactoriesMu.Lock()
	defer providerFactoriesMu.Unlock()
	providerFactories[providerType] = factory
}

// LLMGateway implements the Gateway interface and manages multiple LLM providers
type LLMGateway struct {
	providers map[ProviderType]Provider
	mu        sync.RWMutex
	logger    Logger
}

// Option is a functional option for configuring the gateway
type Option func(*LLMGateway)

// WithLogger sets the logger for the gateway
func WithLogger(logger Logger) Option {
	return func(g *LLMGateway) {
		g.logger = logger
	}
}

// WithZerolog sets a zerolog logger for the gateway
func WithZerolog(logger zerolog.Logger) Option {
	return func(g *LLMGateway) {
		g.logger = NewZerologAdapter(logger)
	}
}

// New creates a new LLM gateway with the provided provider configurations.
// Each ProviderConfig in the slice will be used to initialize its corresponding provider.
// Returns an error if any provider fails to initialize.
func New(configs []ProviderConfig, opts ...Option) (*LLMGateway, error) {
	g := &LLMGateway{
		providers: make(map[ProviderType]Provider),
		logger:    &NopLogger{},
	}

	// Apply options first so logger is available during registration
	for _, opt := range opts {
		opt(g)
	}

	// Register each configured provider
	for _, config := range configs {
		if config == nil {
			continue
		}

		providerType := config.providerType()

		providerFactoriesMu.RLock()
		factory, exists := providerFactories[providerType]
		providerFactoriesMu.RUnlock()
		if !exists {
			return nil, fmt.Errorf("unknown provider type: %s", providerType)
		}

		client, err := factory(config, g.logger)
		if err != nil {
			return nil, fmt.Errorf("failed to initialize %s: %w", providerType, err)
		}

		g.providers[providerType] = client
		g.logger.Info().Str("provider", string(providerType)).Msg("Provider registered")
	}

	if len(g.providers) == 0 {
		return nil, fmt.Errorf("at least one provider must be configured")
	}

	return g, nil
}

// Generate generates text using the specified model.
// The model carries its own generation options and knows which provider to use.
func (g *LLMGateway) Generate(ctx context.Context, model Model, prompt string) (*GenerationResponse, error) {
	provider := model.Provider()

	g.mu.RLock()
	client, exists := g.providers[provider]
	g.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("provider %s is not registered", provider)
	}

	resp, err := client.Generate(ctx, model, prompt)
	if err != nil {
		return nil, err
	}

	// Set provider in response
	resp.Provider = provider
	return resp, nil
}

// IsRegistered checks if a provider is registered
func (g *LLMGateway) IsRegistered(provider ProviderType) bool {
	g.mu.RLock()
	defer g.mu.RUnlock()
	_, exists := g.providers[provider]
	return exists
}

// ListRegisteredProviders returns a list of registered providers
func (g *LLMGateway) ListRegisteredProviders() []ProviderType {
	g.mu.RLock()
	defer g.mu.RUnlock()

	providers := make([]ProviderType, 0, len(g.providers))
	for p := range g.providers {
		providers = append(providers, p)
	}
	return providers
}

// Health checks the health of a specific provider
func (g *LLMGateway) Health(ctx context.Context, provider ProviderType) error {
	g.mu.RLock()
	client, exists := g.providers[provider]
	g.mu.RUnlock()

	if !exists {
		return fmt.Errorf("provider %s is not registered", provider)
	}

	return client.Health(ctx)
}

// Close closes all registered providers
func (g *LLMGateway) Close() error {
	g.mu.Lock()
	defer g.mu.Unlock()

	var errors []error
	for name, provider := range g.providers {
		if err := provider.Close(); err != nil {
			errors = append(errors, fmt.Errorf("provider %s: %w", name, err))
		}
	}

	if len(errors) > 0 {
		return fmt.Errorf("failed to close providers: %v", errors)
	}

	return nil
}

// truncateString truncates a string to the specified length
func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

// defaultTimeout returns the default timeout for providers
func defaultTimeout() time.Duration {
	return 60 * time.Second
}
