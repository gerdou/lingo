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
| **OpenAI** | GPT-5.5/5.5 Pro, GPT-5.4 (Pro/mini/nano), GPT-5.1 (incl. Codex), GPT-5, o-series, GPT-4o |
| **Anthropic** | Claude Fable 5, Claude Opus 4.8/4.7/4.6, Claude Sonnet 4.6, Claude Haiku 4.5 (+ earlier Claude 4/3.x) |
| **Google Gemini** | Gemini 3.1 Pro, Gemini 3.5 Flash, Gemini 3.1 Flash-Lite, Gemini 3 Pro/Flash, Gemini 2.5 Pro/Flash |
| **AWS Bedrock** | Claude, Amazon Nova, Llama, Mistral, Titan, and other Bedrock models |
| **Perplexity** | Sonar, Sonar Pro, Sonar Reasoning Pro, Sonar Deep Research |
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
        lingo.NewClaudeSonnet46().WithMaxTokens(1000),
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

// Available models (latest first)
model := lingo.NewGPT55()     // GPT-5.5 (frontier)
model := lingo.NewGPT55Pro()  // GPT-5.5 Pro
model := lingo.NewGPT54()      // GPT-5.4
model := lingo.NewGPT54Mini()
model := lingo.NewGPT54Nano()
model := lingo.NewGPT51()      // GPT-5.1
model := lingo.NewGPT4o()      // legacy
```

### Anthropic

```go
config := &lingo.AnthropicConfig{
    APIKey:  "your-api-key",
    Timeout: 60 * time.Second,
}

// Available models (latest first)
model := lingo.NewClaudeFable5()  // most capable; thinking always on
model := lingo.NewClaudeOpus48()  // current recommended Opus
model := lingo.NewClaudeOpus47()
model := lingo.NewClaudeOpus46()
model := lingo.NewClaudeSonnet46()
model := lingo.NewClaudeHaiku45()

// Adaptive thinking (recommended on Claude 4.6+; required form on 4.7/4.8)
model := lingo.NewClaudeOpus48().WithAdaptiveThinking()

// Any other Claude model by ID
model := lingo.NewAnthropicModel("claude-opus-4-5-20251101")
```

Note: Claude Opus 4.7/4.8 and Fable 5 reject sampling parameters
(temperature/topP/topK) and fixed thinking budgets, so those setters are not
available on their model types.

### Google Gemini

```go
config := &lingo.GoogleConfig{
    APIKey:  "your-api-key",
    Timeout: 60 * time.Second,
}

// Available models (latest first)
model := lingo.NewGemini31Pro()        // Gemini 3.1 Pro (preview)
model := lingo.NewGemini35Flash()      // Gemini 3.5 Flash
model := lingo.NewGemini31FlashLite()  // Gemini 3.1 Flash-Lite
model := lingo.NewGemini3Pro()
model := lingo.NewGemini25Pro()
model := lingo.NewGemini25Flash()
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

// Available models (examples)
model := lingo.NewBedrockClaudeFable5()
model := lingo.NewBedrockClaudeOpus48()
model := lingo.NewBedrockNovaPro()
model := lingo.NewBedrockLlama4Maverick()

// Any Bedrock model by ID, including cross-region inference profiles
// (required for many newer models outside their home regions)
model := lingo.NewBedrockModel("us.anthropic.claude-opus-4-8", "claude")
```

### Perplexity

```go
config := &lingo.PerplexityConfig{
    APIKey: "your-api-key",
}

// Available models
model := lingo.NewSonar()
model := lingo.NewSonarPro()
model := lingo.NewSonarReasoningPro()
model := lingo.NewSonarDeepResearch()
```

### Ollama

```go
config := &lingo.OllamaConfig{
    BaseURL: "http://localhost:11434", // default
}

// Use any model running in Ollama
model := lingo.NewOllamaModel("llama3.3")
model := lingo.NewOllamaModel("mistral")

// Or use a preset
model := lingo.NewDeepSeekR1()
model := lingo.NewQwen3()
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

