# Lingo

[![Go Reference](https://pkg.go.dev/badge/github.com/gerdou/lingo.svg)](https://pkg.go.dev/github.com/gerdou/lingo)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A unified Go gateway for multiple LLM providers. Lingo provides a consistent interface to interact with various Large Language Model APIs including OpenAI, Anthropic, Google Gemini, xAI, DeepSeek, Cohere, AWS Bedrock, Azure OpenAI, OpenRouter, Perplexity, and Ollama — plus any endpoint that speaks the OpenAI chat completions dialect.

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
| **xAI** | Grok 4.5, Grok 4.3, Grok 4.20 (reasoning / non-reasoning / multi-agent), grok-build-0.1 |
| **DeepSeek** | DeepSeek V4 Pro, DeepSeek V4 Flash |
| **Cohere** | Command A+, Command A, Command A Reasoning / Vision / Translate, Command R7B, Command R / R+ |
| **AWS Bedrock** | Claude, Amazon Nova, Llama, Mistral, Titan, and other Bedrock models |
| **Azure OpenAI** | Any chat or reasoning deployment in your Azure OpenAI resource |
| **Google Vertex AI** | Gemini and Claude through Google Cloud (see the Gemini and Anthropic sections) |
| **OpenRouter** | 400+ models from many vendors behind one key |
| **Perplexity** | Sonar, Sonar Pro, Sonar Reasoning Pro, Sonar Deep Research |
| **Ollama** | Any locally running Ollama model |
| **OpenAI-compatible** | Groq, Together, Fireworks, Cerebras, DeepInfra, SambaNova, Mistral, Z.ai, NVIDIA NIM, vLLM, LM Studio, llama.cpp, LocalAI, … |

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

#### Claude on Google Cloud (Vertex AI)

Set `Vertex` to reach Claude through Vertex AI instead of the Anthropic API —
the GCP counterpart to the Bedrock provider. Authentication uses Google
application default credentials, so no API key is needed.

```go
config := &lingo.AnthropicConfig{
    Vertex: &lingo.AnthropicVertexConfig{
        ProjectID: "my-gcp-project",
        Region:    "us-east5", // or "global"
    },
    HealthCheckModel: "claude-opus-4-5@20251101",
}

// Vertex publishes Claude under @-versioned ids that differ from the ones the
// Anthropic API uses, so address models by id rather than by constructor
model := lingo.NewAnthropicModel("claude-opus-4-5@20251101")
```

`Health` needs `HealthCheckModel` on Vertex, since the available ids are
project- and region-specific.

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

#### Gemini on Vertex AI

Set `UseVertexAI` to bill and authenticate through Google Cloud instead of the
Gemini Developer API. The models and builders are unchanged.

```go
config := &lingo.GoogleConfig{
    UseVertexAI: true,
    Project:     "my-gcp-project",
    Location:    "us-central1", // or "global"
}

// Or Vertex express mode, which authenticates with an API key
config := &lingo.GoogleConfig{UseVertexAI: true, APIKey: "your-vertex-api-key"}
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
model := lingo.NewBedrockClaudeOpus5()
model := lingo.NewBedrockClaudeSonnet5()
model := lingo.NewBedrockClaudeOpus48()
model := lingo.NewBedrockNovaPro()
model := lingo.NewBedrockLlama4Maverick()

// Any Bedrock model by ID, including cross-region inference profiles
// (required for many newer models outside their home regions)
model := lingo.NewBedrockModel("us.anthropic.claude-opus-5", "claude")
```

### Azure OpenAI

Azure is not reachable by pointing `OpenAIConfig.BaseURL` at an Azure resource:
it authenticates with an `api-key` header rather than a bearer token, requires
an `api-version` query parameter, and routes by deployment name instead of
model name. `AzureOpenAIConfig` handles all three.

```go
config := &lingo.AzureOpenAIConfig{
    Endpoint:   "https://my-resource.openai.azure.com",
    APIKey:     "your-azure-key",
    APIVersion: lingo.AzureAPIVersionDefault, // "2024-10-21"; override for preview features
}

// Or authenticate with Microsoft Entra ID (managed identity, workload identity, …)
cred, _ := azidentity.NewDefaultAzureCredential(nil)
config := &lingo.AzureOpenAIConfig{
    Endpoint:        "https://my-resource.openai.azure.com",
    TokenCredential: cred,
}

// Models are addressed by deployment name, which you choose when deploying
model := lingo.NewAzureOpenAIModel("my-gpt4o-deployment").WithMaxTokens(1000)

// Reasoning deployments take max_completion_tokens and reasoning_effort
model := lingo.NewAzureOpenAIReasoningModel("my-gpt5-deployment").
    WithReasoningEffort("high")
```

Azure and OpenAI are separate providers, so a single gateway can hold both.

### xAI

```go
config := &lingo.XAIConfig{
    APIKey: "your-xai-key",
}

// Available models (latest first)
model := lingo.NewGrok45()               // grok-4.5 — latest and fastest, 500K context
model := lingo.NewGrok43()               // grok-4.3 — 1M context, accepts reasoning_effort
model := lingo.NewGrok420Reasoning()     // grok-4.20-0309-reasoning
model := lingo.NewGrok420NonReasoning()  // grok-4.20-0309-non-reasoning
model := lingo.NewGrok420MultiAgent()    // grok-4.20-multi-agent-0309
model := lingo.NewGrokBuild01()          // grok-build-0.1 — 256K context

// Grok 4.3 accepts reasoning_effort; XAIEffortNone switches reasoning off
model := lingo.NewGrok43().WithReasoningEffort(lingo.XAIEffortHigh)

// Any other Grok model by ID, including "-latest" and date-pinned aliases
model := lingo.NewXAIModel("grok-4.5-latest")
```

Notes:

- xAI takes `max_completion_tokens`; `max_tokens` is deprecated upstream, so
  lingo only sends the former.
- Reasoning models reject `frequency_penalty`, `presence_penalty` and `stop`,
  and models from grok-4.20 onward silently ignore `logprobs`.

### DeepSeek

```go
config := &lingo.DeepSeekConfig{
    APIKey:  "your-deepseek-key",
    Timeout: 300 * time.Second, // thinking mode can run long
}

// Available models — 1M context, 384K max output
model := lingo.NewDeepSeekV4Pro()    // deepseek-v4-pro — most capable
model := lingo.NewDeepSeekV4Flash()  // deepseek-v4-flash — cost-efficient

// Thinking is on by default on V4; trade depth for latency by turning it off
model := lingo.NewDeepSeekV4Flash().WithThinkingDisabled()

// Or tune how hard it thinks
model := lingo.NewDeepSeekV4Pro().WithReasoningEffort("high")

// Any other DeepSeek model by ID
model := lingo.NewDeepSeekModel("deepseek-v4-flash")
```

Reasoning traces come back in `response.Metadata["reasoning_content"]`.

### Cohere

```go
config := &lingo.CohereConfig{
    APIKey: "your-cohere-key",
}

// Available models (latest first)
model := lingo.NewCommandAPlus()       // command-a-plus-05-2026 — flagship, 128K context
model := lingo.NewCommandA()           // command-a-03-2025 — 256K context
model := lingo.NewCommandAReasoning()  // command-a-reasoning-08-2025
model := lingo.NewCommandAVision()     // command-a-vision-07-2025
model := lingo.NewCommandATranslate()  // command-a-translate-08-2025
model := lingo.NewCommandR7B()         // command-r7b-12-2024 — smallest and cheapest
model := lingo.NewCommandR()           // command-r-08-2024
model := lingo.NewCommandRPlus()       // command-r-plus-08-2024

// Cohere-specific options
model := lingo.NewCommandAPlus().
    WithSafetyMode(lingo.CohereSafetyStrict).
    WithThinkingBudget(4096).
    WithStopSequences([]string{"END"})

// Any other Cohere model by ID, including fine-tunes
model := lingo.NewCohereModel("command-a-plus-05-2026")
```

Note: `WithTopP` maps to Cohere's `p` and `WithTopK` to `k`. Reasoning traces
come back in `response.Metadata["reasoning_content"]`.

### OpenRouter

One key reaches hundreds of models from many vendors. The catalogue changes
constantly, so models are addressed by ID rather than typed constructors.

```go
config := &lingo.OpenRouterConfig{
    APIKey:  "your-openrouter-key",
    SiteURL: "https://myapp.example",  // optional, for openrouter.ai rankings
    AppName: "My App",                 // optional, for openrouter.ai rankings
}

model := lingo.NewOpenRouterModel("anthropic/claude-opus-5").WithMaxTokens(1000)

// Reasoning is normalised across vendors
model := lingo.NewOpenRouterModel("openai/gpt-5.6-sol").WithReasoningEffort("high")

// Routing control: pin providers, fail rather than fall back, avoid providers
// that retain prompts, and name backup models
model := lingo.NewOpenRouterModel("meta-llama/llama-4-maverick").
    WithProviderOrder([]string{"fireworks", "together"}).
    WithAllowFallbacks(false).
    WithDataCollection("deny").
    WithFallbackModels([]string{"meta-llama/llama-4-scout"})
```

When OpenRouter routes to a different model than requested, `response.Model`
holds the model actually served and `response.Metadata["requested_model"]` the
one you asked for.

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

### Any OpenAI-compatible endpoint

Most inference services speak the OpenAI chat completions dialect. Point
`OpenAICompatibleConfig` at one and address models by ID — this covers Groq,
Together, Fireworks, Cerebras, DeepInfra, SambaNova, Hyperbolic, Nebius,
NVIDIA NIM, Hugging Face, Mistral, Z.ai, and local vLLM / LM Studio /
llama.cpp / LocalAI servers.

```go
config := &lingo.OpenAICompatibleConfig{
    BaseURL: lingo.BaseURLGroq, // or any other URL, e.g. "http://localhost:8000/v1"
    APIKey:  "your-key",        // optional: local servers rarely need one
}

model := lingo.NewOpenAICompatibleModel("llama-3.3-70b-versatile").
    WithMaxTokens(1000).
    WithTemperature(0.7)

// Options the OpenAI schema does not model can be set directly on the body
model := lingo.NewOpenAICompatibleModel("qwen3-coder").
    WithExtraField("chat_template_kwargs", map[string]any{"enable_thinking": true})
```

Base URL shortcuts: `BaseURLGroq`, `BaseURLTogether`, `BaseURLFireworks`,
`BaseURLCerebras`, `BaseURLDeepInfra`, `BaseURLSambaNova`, `BaseURLHyperbolic`,
`BaseURLNebius`, `BaseURLNvidiaNIM`, `BaseURLHuggingFace`, `BaseURLMistral`,
`BaseURLZAI`, `BaseURLVLLM`, `BaseURLLMStudio`, `BaseURLLlamaCPP`,
`BaseURLLocalAI`, `BaseURLOllamaOAI`.

Use a dedicated provider where one exists (OpenAI, Azure, xAI, DeepSeek,
OpenRouter): those ship typed models and provider-specific options.

Because the gateway keys providers by type, one gateway holds one
OpenAI-compatible endpoint. Construct a second gateway to reach another.

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

What a health check costs varies by provider. OpenAI, Anthropic and Bedrock
generate a few tokens against a small model. xAI, DeepSeek, OpenRouter, Azure
and OpenAI-compatible endpoints list models instead, which proves credentials
and connectivity without spending tokens; Cohere validates the key directly.

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

