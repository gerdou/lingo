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
| **OpenAI** | GPT-5.6 (Sol/Terra/Luna), GPT-5.5/5.5 Pro, GPT-5.4 (Pro/mini/nano), GPT-5.1, GPT-5, o-series, GPT-4o |
| **Anthropic** | Claude Fable 5, Claude Opus 5, Claude Sonnet 5, Claude Opus 4.8/4.7/4.6, Claude Sonnet 4.6, Claude Haiku 4.5 (+ earlier Claude 4/3.x) |
| **Google Gemini** | Gemini 3.6 Flash, Gemini 3.5 Flash / Flash-Lite, Gemini 3.1 Pro, Gemini 3.1 Flash-Lite, Gemini 2.5 Pro/Flash |
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
        lingo.NewGPT56Terra().WithMaxCompletionTokens(1000),
        "Explain quantum computing in simple terms",
    )
    if err != nil {
        log.Fatal(err)
    }
    fmt.Println(response.Text)

    // Use Anthropic with the same gateway
    response, err = gateway.Generate(
        context.Background(),
        lingo.NewClaudeSonnet5().WithMaxTokens(1000),
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
model := lingo.NewGPT56Sol()    // GPT-5.6 Sol — current frontier (alias: gpt-5.6)
model := lingo.NewGPT56Terra()  // GPT-5.6 Terra — balances intelligence and cost
model := lingo.NewGPT56Luna()   // GPT-5.6 Luna — cost-sensitive, high volume
model := lingo.NewGPT55()       // GPT-5.5
model := lingo.NewGPT55Pro()    // GPT-5.5 Pro
model := lingo.NewGPT54()       // GPT-5.4
model := lingo.NewGPT51()       // GPT-5.1
model := lingo.NewGPT4o()       // legacy
```

The GPT-5.6 family has a 1.05M token context window and 128K max output tokens.

Note: `gpt-5.1-codex` and `gpt-5.1-codex-mini` were retired on Jul 23, 2026 and
now 404; `NewGPT51Codex()`/`NewGPT51CodexMini()` are kept only for source
compatibility. The GPT-5 and o3 families shut down Dec 11, 2026, and GPT-4,
GPT-4-turbo, GPT-3.5-turbo and o1 shut down Oct 23, 2026.

### Anthropic

```go
config := &lingo.AnthropicConfig{
    APIKey:  "your-api-key",
    Timeout: 60 * time.Second,
}

// Available models (latest first)
model := lingo.NewClaudeFable5()   // most capable; thinking always on
model := lingo.NewClaudeOpus5()    // current recommended Opus
model := lingo.NewClaudeSonnet5()  // current recommended speed/intelligence balance
model := lingo.NewClaudeOpus48()
model := lingo.NewClaudeOpus47()
model := lingo.NewClaudeOpus46()
model := lingo.NewClaudeSonnet46()
model := lingo.NewClaudeHaiku45()

// Effort controls thinking depth and overall token spend (Claude 4.6+).
// Start at EffortXHigh for coding/agentic work, EffortHigh elsewhere.
model := lingo.NewClaudeOpus5().WithEffort(lingo.EffortXHigh)

// On the Claude 5 series thinking is adaptive and on by default; opt out with:
model := lingo.NewClaudeOpus5().WithThinkingDisabled()

// On Claude 4.6-4.8, adaptive thinking is opt-in
model := lingo.NewClaudeOpus48().WithAdaptiveThinking()

// Any other Claude model by ID
model := lingo.NewAnthropicModel("claude-mythos-5")
```

Notes:

- Claude Opus 5/Sonnet 5, Opus 4.7/4.8 and Fable 5 reject sampling parameters
  (temperature/topP/topK) and fixed thinking budgets, so those setters are not
  available on their model types. Use `WithEffort` instead.
- `WithThinkingDisabled` on Opus 5 is only accepted at `EffortHigh` or below;
  pairing it with `EffortXHigh`/`EffortMax` returns a 400. Prefer a lower effort
  level over disabling thinking.
- `EffortXHigh` requires Claude 4.7 or later.
- Claude Opus 4.1 was retired on Aug 5, 2026; `NewClaudeOpus41()` now 404s.

### Google Gemini

```go
config := &lingo.GoogleConfig{
    APIKey:  "your-api-key",
    Timeout: 60 * time.Second,
}

// Available models (latest first)
model := lingo.NewGemini36Flash()      // Gemini 3.6 Flash — current Flash flagship
model := lingo.NewGemini35Flash()      // Gemini 3.5 Flash
model := lingo.NewGemini35FlashLite()  // Gemini 3.5 Flash-Lite
model := lingo.NewGemini31Pro()        // Gemini 3.1 Pro (preview)
model := lingo.NewGemini31FlashLite()  // Gemini 3.1 Flash-Lite
model := lingo.NewGemini25Pro()
model := lingo.NewGemini25Flash()
```

Note: `gemini-3-pro-preview` now redirects to `gemini-3.1-pro-preview`; use
`NewGemini31Pro()` to target it directly. The Gemini 2.0 Flash models were shut
down on June 1, 2026.

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
model := lingo.NewBedrockClaudeOpus5()
model := lingo.NewBedrockClaudeSonnet5()
model := lingo.NewBedrockClaudeOpus48()
model := lingo.NewBedrockNovaPro()
model := lingo.NewBedrockLlama4Maverick()

// Any Bedrock model by ID, including cross-region inference profiles
// (required for many newer models outside their home regions)
model := lingo.NewBedrockModel("us.anthropic.claude-opus-5", "claude")
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

