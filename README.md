# Lingo

[![Go Reference](https://pkg.go.dev/badge/github.com/gerdou/lingo.svg)](https://pkg.go.dev/github.com/gerdou/lingo)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A unified Go gateway for multiple LLM providers. Lingo provides a consistent interface to interact with various Large Language Model APIs including OpenAI, Anthropic, Google Gemini, AWS Bedrock, Perplexity, and Ollama.

## Features

- **Unified Interface**: Single API to interact with multiple LLM providers
- **Type-Safe Models**: Strongly typed model configurations with fluent builder pattern
- **Built-in Rate Limiting**: Automatic retry with exponential backoff for rate-limited requests
- **Provider Health Checks**: Monitor the health of your LLM providers
- **Extensible Logging**: Pluggable logging interface with zerolog adapter included
- **Official SDKs**: Uses official SDKs where available for maximum compatibility

## Supported Providers

| Provider | Models |
|----------|--------|
| **OpenAI** | GPT-4o, GPT-4o-mini, GPT-4 Turbo, o1, o1-mini, o3-mini |
| **Anthropic** | Claude 3.5 Sonnet, Claude 3.5 Haiku, Claude 3 Opus |
| **Google Gemini** | Gemini 2.5 Pro/Flash, Gemini 2.0 Flash, Gemini 1.5 Pro/Flash |
| **AWS Bedrock** | Claude, Llama, Titan, and other Bedrock models |
| **Perplexity** | Sonar, Sonar Pro, Sonar Reasoning |
| **Ollama** | Any locally running Ollama model |

## Installation

```bash
go get github.com/gerdou/lingo
```

## Quick Start

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/gerdou/lingo"
)

func main() {
    // Create a gateway with multiple providers
    gateway, err := lingo.New([]lingo.ProviderConfig{
        &lingo.OpenAIConfig{APIKey: "your-openai-key"},
        &lingo.AnthropicConfig{APIKey: "your-anthropic-key"},
        &lingo.GoogleConfig{APIKey: "your-google-key"},
    })
    if err != nil {
        log.Fatal(err)
    }
    defer gateway.Close()

    // Use OpenAI
    response, err := gateway.Generate(
        context.Background(),
        lingo.NewGPT4o().WithMaxTokens(1000).WithTemperature(0.7),
        "Explain quantum computing in simple terms",
    )
    if err != nil {
        log.Fatal(err)
    }
    fmt.Println(response.Text)

    // Use Anthropic with the same gateway
    response, err = gateway.Generate(
        context.Background(),
        lingo.NewClaude35Sonnet().WithMaxTokens(1000),
        "Write a haiku about programming",
    )
    if err != nil {
        log.Fatal(err)
    }
    fmt.Println(response.Text)
}
```

## Provider Configuration

### OpenAI

```go
config := &lingo.OpenAIConfig{
    APIKey:      "your-api-key",
    Timeout:     60 * time.Second,
    RateLimiter: lingo.DefaultRateLimitConfig(),
}

// Available models
model := lingo.NewGPT4o()
model := lingo.NewGPT4oMini()
model := lingo.NewO1()
model := lingo.NewO1Mini()
model := lingo.NewO3Mini()
```

### Anthropic

```go
config := &lingo.AnthropicConfig{
    APIKey:  "your-api-key",
    Timeout: 60 * time.Second,
}

// Available models
model := lingo.NewClaude35Sonnet()
model := lingo.NewClaude35Haiku()
model := lingo.NewClaude3Opus()
```

### Google Gemini

```go
config := &lingo.GoogleConfig{
    APIKey:  "your-api-key",
    Timeout: 60 * time.Second,
}

// Available models
model := lingo.NewGemini25Pro()
model := lingo.NewGemini25Flash()
model := lingo.NewGemini20Flash()
model := lingo.NewGemini15Pro()
```

### AWS Bedrock

```go
config := &lingo.BedrockConfig{
    Region: "us-east-1",
    // Uses default AWS credentials chain
}

// Or with explicit credentials
config := &lingo.BedrockConfig{
    Region:          "us-east-1",
    AccessKeyID:     "your-access-key",
    SecretAccessKey: "your-secret-key",
}
```

### Perplexity

```go
config := &lingo.PerplexityConfig{
    APIKey: "your-api-key",
}

// Available models
model := lingo.NewSonar()
model := lingo.NewSonarPro()
model := lingo.NewSonarReasoning()
```

### Ollama

```go
config := &lingo.OllamaConfig{
    BaseURL: "http://localhost:11434", // default
}

// Use any model running in Ollama
model := lingo.NewOllamaModel("llama2")
model := lingo.NewOllamaModel("mistral")
```

## Model Configuration

All models support a fluent builder pattern for configuration:

```go
model := lingo.NewGPT4o().
    WithMaxTokens(2000).
    WithTemperature(0.8).
    WithTopP(0.9).
    WithSystemPrompt("You are a helpful assistant")
```

## Logging

Lingo supports pluggable logging. Use the built-in zerolog adapter:

```go
import "github.com/rs/zerolog"

logger := zerolog.New(os.Stdout).With().Timestamp().Logger()

gateway, err := lingo.New(
    configs,
    lingo.WithZerolog(logger),
)
```

Or implement your own logger:

```go
type MyLogger struct{}

func (l *MyLogger) Debug() lingo.LogEvent { /* ... */ }
func (l *MyLogger) Info() lingo.LogEvent  { /* ... */ }
func (l *MyLogger) Error() lingo.LogEvent { /* ... */ }

gateway, err := lingo.New(configs, lingo.WithLogger(&MyLogger{}))
```

## Rate Limiting

Built-in rate limit handling with exponential backoff:

```go
config := &lingo.OpenAIConfig{
    APIKey: "your-api-key",
    RateLimiter: &lingo.RateLimitConfig{
        MaxRetries:        5,
        InitialBackoff:    1 * time.Second,
        MaxBackoff:        60 * time.Second,
        BackoffMultiplier: 2.0,
    },
}
```

## Health Checks

Monitor provider availability:

```go
// Check specific provider
err := gateway.Health(ctx, lingo.ProviderOpenAI)

// List all registered providers
providers := gateway.ListRegisteredProviders()

// Check if provider is registered
if gateway.IsRegistered(lingo.ProviderAnthropic) {
    // Use Anthropic
}
```

## Response Structure

```go
type GenerationResponse struct {
    Text         string            // Generated text
    Provider     ProviderType      // Provider used
    Model        string            // Model used
    Usage        TokenUsage        // Token counts
    FinishReason string            // Why generation stopped
    Metadata     map[string]string // Provider-specific data
}

type TokenUsage struct {
    PromptTokens     int
    CompletionTokens int
    TotalTokens      int
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

