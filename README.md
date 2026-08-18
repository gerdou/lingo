# Lingo

[![Go Reference](https://pkg.go.dev/badge/github.com/gerdou/lingo.svg)](https://pkg.go.dev/github.com/gerdou/lingo)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A unified Go gateway for multiple LLM providers. Lingo provides a consistent interface to interact with various Large Language Model APIs including OpenAI, Anthropic, Google Gemini, xAI, DeepSeek, Cohere, AWS Bedrock, Azure OpenAI, OpenRouter, Perplexity, and Ollama — plus any endpoint that speaks the OpenAI chat completions dialect.

## Features

- **Unified Interface**: Single API to interact with multiple LLM providers
- **Type-Safe Models**: Strongly typed model configurations with fluent builder pattern
- **Built-in Rate Limiting**: Automatic retry with exponential backoff for rate-limited requests
- **Prompt Caching**: One opt-in API over explicit breakpoints, cache keys and provider-side cache resources — including their create/refresh/delete lifecycle on Google — with normalized cache token accounting
- **Thinking Control**: One opt-in API over token budgets, effort ladders and on/off switches, translated per model and never an error where a provider cannot honour it, with the reasoning trace and normalized thinking token accounting on the way back
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

Lingo needs Go 1.25 or newer. It uses each vendor's official SDK where one
exists — `anthropics/anthropic-sdk-go`, `openai/openai-go/v3` (which also serves
Azure and every OpenAI-dialect endpoint), `google.golang.org/genai`,
`aws-sdk-go-v2/service/bedrockruntime` and `cohere-ai/cohere-go/v2` — and speaks
HTTP directly to Perplexity and Ollama. `go.mod` is the authoritative list of
pinned versions. Prompt caching in particular tracks those SDKs closely: which
cache fields exist at all is a function of the pinned version, so upgrade them
as a set rather than one at a time.

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
    APIVersion: lingo.AzureAPIVersionDefault, // "2024-10-21", the newest dated GA version
}

// Or the v1 API surface: no api-version, plain OpenAI paths under /openai/v1,
// and the only Azure route that accepts a prompt cache key
config := &lingo.AzureOpenAIConfig{
    Endpoint:   "https://my-resource.openai.azure.com",
    APIKey:     "your-azure-key",
    APIVersion: lingo.AzureAPIVersionV1, // or AzureAPIVersionV1Preview
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

Reasoning traces come back in `response.Thinking` (and, deprecated, in
`response.Metadata["reasoning_content"]`). See
[Thinking Control](#thinking-control) for the portable way to ask.

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
come back in `response.Thinking` (and, deprecated, in
`response.Metadata["reasoning_content"]`). See
[Thinking Control](#thinking-control) for the portable way to ask.

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

## Prompt Caching

Providers can reuse the work they already did on a stable prefix of a request —
usually a long system prompt — instead of re-processing it on every call. Reads
from the cache are billed at a steep discount; on some providers the write that
fills it costs a premium.

They disagree about how you ask. Lingo sorts them into three levels, which
`CachingSupport` reports:

- **explicit** — the provider caches nothing unless the request marks a
  breakpoint. Lingo places the breakpoint for you when you opt in.
- **usage-only** — the provider decides for itself what to cache. There is
  nothing to ask for, though some accept a routing key.
- **none** — no caching, or none the API reports.

```go
switch lingo.CachingSupport(lingo.ProviderAnthropic) {
case lingo.CacheSupportExplicit:  // breakpoints available
case lingo.CacheSupportUsageOnly: // automatic; counters only
case lingo.CacheSupportNone:      // nothing to report
}
```

Caching is opt-in on the request side: a model whose cache options you never
touched sends byte-for-byte the request it sent before this feature existed.
Reporting is not opt-in — `response.Usage.CacheReadTokens` and `CacheWriteTokens`
are filled in whenever the provider reports them, whether or not you asked for
anything. Asking for caching where a provider cannot honour it is a silent
no-op, never an error, so `Cached` is safe to call generically.

### What each provider does

| Provider | Level | What lingo sends when you opt in |
|----------|-------|----------------------------------|
| **Anthropic** | explicit | `cache_control: {"type": "ephemeral"}` after the system prompt, the user prompt, or both, with an optional `ttl` of `5m` or `1h` |
| **AWS Bedrock** | explicit | Claude, on `InvokeModel`: the same Anthropic-dialect `cache_control` markers. Nova, on the Converse API: a `cachePoint`, at a fixed 5-minute lifetime. Llama, Mistral and Titan ignore the setting |
| **OpenRouter** | explicit | Anthropic-dialect `cache_control` markers, plus `prompt_cache_key` when you set one |
| **Google Gemini / Vertex** | explicit | `cachedContent` — the resource name of a cache you created yourself, through `gateway.CacheManager(lingo.ProviderGoogle)` or otherwise. There is no per-request breakpoint |
| **OpenAI** | usage-only | `prompt_cache_key` when you set one; OpenAI caches prefixes on its own. The GPT-5.6 models are the exception: they also take a `prompt_cache_breakpoint` marking the exact end of the prefix, at OpenAI's fixed 30-minute lifetime |
| **Azure OpenAI** | usage-only | `prompt_cache_key`, on the v1 API surface only (`APIVersion: lingo.AzureAPIVersionV1`); nothing at all on the dated api-versions |
| **OpenAI-compatible** | usage-only | `prompt_cache_key` when you set one |
| **DeepSeek** | usage-only | nothing — DeepSeek caches server-side |
| **xAI** | usage-only | nothing — xAI caches server-side |
| **Cohere** | usage-only | nothing — Cohere caches server-side and its chat request has no cache field |
| **Perplexity**, **Ollama** | none | nothing — neither API has a cache field to send |

What comes back differs too:

- Anthropic and Bedrock Claude report reads and writes, and split the write by
  lifetime into `response.Metadata["cache_write_tokens_5m"]` and
  `["cache_write_tokens_1h"]`.
- OpenAI, Azure and OpenAI-compatible endpoints report reads, plus writes where
  the endpoint sends them.
- DeepSeek reports reads, with the miss count in
  `response.Metadata["prompt_cache_miss_tokens"]`.
- Google and Cohere report reads only; neither exposes a cache-write counter.
- Bedrock Nova reports reads and writes, and splits the write by lifetime into
  the same `cache_write_tokens_*` metadata keys, using whatever lifetime label
  Converse returns. See [Bedrock's two APIs](#bedrock-nova-on-converse-everything-else-on-invokemodel).
- Bedrock's Llama, Mistral and Titan families, Perplexity and Ollama report
  neither, so both counters stay zero.

### Explicit breakpoints (Anthropic, Bedrock, OpenRouter)

`Cached` turns caching on and returns the same model, so it slots into the
builder chain and keeps the concrete type:

```go
instructions := loadStyleGuide() // long, and identical on every call

model := lingo.Cached(lingo.NewClaudeSonnet5(), lingo.WithCacheTTL(lingo.CacheTTL1h)).
    WithMaxTokens(4096).
    WithSystemPrompt(instructions)

// The first call writes the cache; calls within the TTL read it back.
response, err := gateway.Generate(ctx, model, "First question")
```

The breakpoint lands at the end of the system prompt, the only stable prefix a
single-turn `Generate` has. Cache the user prompt as well when the same long
document is sent with different instructions appended downstream:

```go
model := lingo.Cached(lingo.NewClaudeOpus5(),
    lingo.WithCacheTTL(lingo.CacheTTL1h),
    lingo.WithCachePrompt(true),
)
```

The statement form is equivalent, and is how you reach the options on a model
you already built:

```go
model := lingo.NewClaudeOpus5().WithSystemPrompt(instructions)
model.CacheOptions().Enable().WithTTL(lingo.CacheTTL1h).WithPrompt(true)
```

Bedrock and OpenRouter take the same calls:

```go
model := lingo.Cached(lingo.NewBedrockClaudeSonnet5(), lingo.WithCacheTTL(lingo.CacheTTL5m)).
    WithSystemPrompt(instructions)
```

`NotCached(model)` suppresses every cache field lingo would otherwise send. It
cannot switch off caching a provider performs on its own — none of the supported
providers offer an opt-out that is safe to send to every model — so on
usage-only providers it behaves like the default.

### Bedrock: Nova on Converse, everything else on InvokeModel

Bedrock is one provider with two request formats behind it, and lingo picks per
model family. Claude, Llama, Mistral and Titan go out on `InvokeModel` with the
vendor's own JSON body, exactly as they always have. Amazon Nova — and only
Nova — goes out on the Converse API instead, because `cachePoint` markers and
the cache token counters live there and not in Nova's `InvokeModel` body.

For a caller this is meant to be invisible. Model builders, prompts, options and
`GenerationResponse` are identical either way; nothing about `NewBedrockNovaPro`
or `NewBedrockModel(id, "nova")` changes at the call site:

```go
model := lingo.Cached(lingo.NewBedrockNovaPro()).WithSystemPrompt(instructions)
```

Two things are worth knowing anyway:

- A Nova model whose cache options you never touched sends no `cachePoint`, and
  is priced exactly as it was before. The switch to Converse is what makes the
  markers reachable; it does not turn caching on.
- On a cached Nova call `Usage.PromptTokens` covers the whole effective prompt,
  so it is larger than the raw `inputTokens` Converse reports — the same
  normalization Anthropic and Bedrock Claude get. `TotalTokens` is derived from
  the normalized prompt and completion counts rather than read from Converse,
  which reports a total that already folds the cache reads back in.

`WithCacheTTL` is clamped away on Nova: Nova documents a 5-minute cache lifetime
only, so lingo sends the default lifetime rather than a `1h` the model may
reject. Bedrock Claude honours `5m` and `1h` as usual.

### Cache keys (OpenAI, Azure, OpenAI-compatible)

OpenAI caches long prefixes automatically, so on most models there is no
breakpoint to place. What you can do is partition its cache, routing requests
that share a prefix to the same one. A key is a routing hint rather than an
opt-in, so setting one is enough — `Cached` is not required:

```go
model := lingo.NewGPT56Terra().
    WithMaxCompletionTokens(1000).
    WithSystemPrompt(instructions)
model.CacheOptions().WithKey("tenant-42")

// Or in a chain
model := lingo.Cached(lingo.NewGPT56Sol(), lingo.WithCacheKey("tenant-42"))
```

TTLs are ignored on all three: none of them lets you choose a cache lifetime.
Breakpoints are ignored too, with one exception — OpenAI's GPT-5.6 models, which
is the next section. On Azure the key itself needs the v1 surface; see
[Azure: cache keys need the v1 surface](#azure-cache-keys-need-the-v1-surface).

### Explicit breakpoints (OpenAI GPT-5.6)

The GPT-5.6 family — `NewGPT56Sol`, `NewGPT56Terra`, `NewGPT56Luna` — is the one
OpenAI family that accepts an explicit breakpoint. It marks the exact end of the
reusable prefix instead of letting OpenAI round the boundary to a token block,
so opting in is worth it when the prefix is stable:

```go
model := lingo.Cached(lingo.NewGPT56Sol()).WithSystemPrompt(instructions)
```

A `prompt_cache_breakpoint` reaches the wire only when **both** halves of a gate
are true:

1. **The model accepts one.** The three GPT-5.6 constructors above say yes
   permanently. A model id you reach through `NewOpenAIReasoningModel` says yes
   only if you declare it, because lingo cannot know what an arbitrary id
   accepts. Every other OpenAI model — GPT-5.5, GPT-5.4, GPT-5.1, GPT-5, GPT-4.1,
   GPT-4o, the o-series, and anything from `NewOpenAIModel` — says no.
2. **You opted in.** `Cached` (or `CacheOptions().Enable()`) marks the system
   prompt; add `WithCachePrompt(true)` to mark the user prompt as well. An
   untouched or `NotCached` GPT-5.6 model sends nothing.

```go
// Declaring a newer id yourself
model := lingo.Cached(lingo.NewOpenAIReasoningModel("gpt-5.7").
    WithExplicitPromptCache(true)).WithSystemPrompt(instructions)
```

Everything the gate turns away falls back to OpenAI's implicit caching, which is
the default behaviour and needs nothing from you: OpenAI still caches long stable
prefixes on its own and still reports `CacheReadTokens`. So a `Cached` call on a
GPT-4o model is not a failure to cache — it is caching without a breakpoint, and
that model sends byte-for-byte the request it sent before. The same is true of
GPT-5.6 itself when you do not opt in; the breakpoint sharpens the prefix
boundary rather than switching caching on.

Two details of the opt-in path:

- `WithCacheTTL` is ignored — OpenAI fixes the breakpoint lifetime at 30 minutes.
- The marked message switches from `"content": "..."` to
  `"content": [{"type": "text", ...}]` so the marker has somewhere to live.
  Semantically identical, but visible if you diff request bodies.

Lingo never sends `prompt_cache_options` — its `explicit` mode would switch off
OpenAI's own implicit breakpoint — nor the deprecated `prompt_cache_retention`,
whose `24h` value changes data retention for zero-data-retention organizations.

Azure is deliberately excluded from the gate, on every api-version: its models
are addressed by deployment name, so lingo cannot tell which model a deployment
serves and cannot know whether a breakpoint would be accepted or rejected.

### Azure: cache keys need the v1 surface

Azure exposes two different API surfaces, and only one of them models a prompt
cache key.

**The dated surface** is the default. `APIVersion` is a frozen date —
`AzureAPIVersionDefault` is `"2024-10-21"`, the newest dated GA version, and the
dated line stops there — and requests go to
`/openai/deployments/<name>/chat/completions?api-version=<date>`. No dated
api-version, GA or preview, has `prompt_cache_key` in its request schema, and
Azure answers a body field its api-version does not model with a 400 rather than
ignoring it. So on any dated route lingo drops the key instead of sending a
request it knows will fail. That is the silent no-op rule doing its job: setting
a key there is not an error, it simply has nowhere to go.

**The v1 surface** is Azure's next-generation route: plain OpenAI paths under
`/openai/v1`, no dated api-version to pick, and `prompt_cache_key` present in the
schema. Opt into it on the config, then set a key as you would on OpenAI:

```go
config := &lingo.AzureOpenAIConfig{
    Endpoint:   "https://my-resource.openai.azure.com",
    APIKey:     "your-azure-key",
    APIVersion: lingo.AzureAPIVersionV1, // or AzureAPIVersionV1Preview
}

model := lingo.NewAzureOpenAIModel("my-gpt4o-deployment").WithMaxTokens(1000)
model.CacheOptions().WithKey("tenant-42")
```

Three independent opt-ins have to line up: the key rides the wire only when
`APIVersion` selects v1, **and** you set a key, **and** you did not `NotCached`
the model. Any one of them missing means nothing is sent.

`AzureAPIVersionV1Preview` is the same route with preview features on — it pins
`api-version=preview` on the query string and is otherwise identical. Two other
things live on v1 and not on the dated line: `cached_tokens` reporting, and
`reasoning_effort` for reasoning deployments. Azure's implicit caching happens on
either surface, but only v1 is documented to report it, so a dated route can
legitimately show zero reads on a request that was in fact cached.

If you authenticate with Microsoft Entra ID rather than a key, note that v1 may
want a different token audience than the dated surface;
`AzureOpenAIConfig.TokenCredentialScopes` is the override.

### Cache resources (Google)

Gemini's explicit cache is a resource with its own lifecycle: you create it, it
holds your content, and you reference it by name. If you already have a resource
name, point a model at it:

```go
model := lingo.Cached(lingo.NewGemini31Pro(),
    lingo.WithCachedContent("cachedContents/1234567890")).
    WithMaxTokens(1000)
```

To create and own the resource, ask the gateway for Google's cache manager.
Google is the only provider that has one — every other provider caches per
request, with nothing to create or delete — so the accessor reports `false`
rather than failing. The whole lifecycle, end to end:

```go
package main

import (
    "context"
    "fmt"
    "log"
    "os"
    "time"

    "github.com/gerdou/lingo"
)

func main() {
    gateway, err := lingo.New([]lingo.ProviderConfig{
        &lingo.GoogleConfig{APIKey: "your-google-key"},
    })
    if err != nil {
        log.Fatal(err)
    }
    defer gateway.Close()

    ctx := context.Background()

    // Discovery, not a capability check that can fail: false means Google is
    // not registered, or the provider has no cache resources to manage.
    mgr, ok := gateway.CacheManager(lingo.ProviderGoogle)
    if !ok {
        log.Fatal("no cache resource manager for this provider")
    }

    corpus, err := os.ReadFile("contracts.txt") // long, and identical every call
    if err != nil {
        log.Fatal(err)
    }

    model := lingo.NewGemini31Pro().WithMaxTokens(1000)

    // 1. Create the resource. The system instruction belongs here, not on the
    //    model — see below.
    cache, err := mgr.CreateCache(ctx, lingo.PromptCacheSpec{
        Model:             model,
        Content:           string(corpus),
        SystemInstruction: "Answer only from the documents.",
        TTL:               2 * time.Hour,
        DisplayName:       "legal-corpus",
    })
    if err != nil {
        log.Fatal(err)
    }
    fmt.Printf("cached %d tokens as %s\n", cache.Tokens, cache.Name)

    // 3. Drop it when you are done. A resource you forget lapses on its own
    //    TTL, but you pay storage until it does.
    defer func() {
        if err := mgr.DeleteCache(context.Background(), cache.Name); err != nil {
            log.Printf("cache %s left behind: %v", cache.Name, err)
        }
    }()

    // 2. Generate against it as often as you like. Cached mutates and returns
    //    the same model, so this is the model above, now pointed at the cache.
    cached := lingo.Cached(model, lingo.WithPromptCache(cache))

    for _, question := range []string{"Who signs first?", "What ends the term?"} {
        // Extend the lease before it lapses. The name does not change, so the
        // model stays pointed at the same resource.
        if cache.TimeToLive() < 15*time.Minute {
            if cache, err = mgr.RefreshCache(ctx, cache.Name, 2*time.Hour); err != nil {
                log.Fatal(err)
            }
        }

        resp, err := gateway.Generate(ctx, cached, question)
        if err != nil {
            log.Fatal(err)
        }
        fmt.Println(resp.Text)
        fmt.Printf("%d of %d prompt tokens came from the cache\n",
            resp.Usage.CacheReadTokens, resp.Usage.PromptTokens)
    }
}
```

`PromptCache` is provider-neutral, so nothing in the flow requires the
`google.golang.org/genai` types. `GetCache` and `ListCaches` read resources back;
only metadata comes back, never the content a resource was created with, and
`ListCaches` walks pagination for you inside a single request timeout. Unlike the
request-side options, these are direct resource calls and report their failures
as errors — what stays a silent no-op is the discovery above.

`CreateCache` rejects a spec with no model, a model from another provider, or
neither content nor a system instruction, before anything reaches the network.
Everything else — a corpus below Gemini's per-model minimum, most obviously — is
the provider's 400, wrapped and returned.

`WithCacheTTL` does nothing on Google: the lifetime belongs to the resource, set
at creation and changed only through `RefreshCache`. `Cached` on its own does
nothing either, since there is no per-request breakpoint to place. A cache
resource is bound to one model at creation time, so the name must match the model
you generate with or the API rejects the call. Both the Gemini Developer API and
Vertex AI support it; Vertex returns fully qualified resource names
(`projects/p/locations/l/cachedContents/…`) and either form round-trips through
`WithCachedContent` unchanged.

`PromptCache.Model` is the provider's own qualified form of the model name —
`models/gemini-3.1-pro-preview` on the Gemini Developer API, a
`projects/…/publishers/google/models/…` path on Vertex — not what
`Model.ModelName()` returns. Comparing the two directly will not match; lingo
passes the provider's form through rather than trimming it, because on Vertex the
prefix carries the project and location.

The system instruction belongs to the cache resource too. Gemini rejects a
request that carries both a cached content name and a system instruction, so
when you set `WithCachedContent` lingo drops the model's own `WithSystemPrompt`
and logs that it did. Put the system prompt in `PromptCacheSpec.SystemInstruction`
so it is baked into the resource instead.

### Nothing to ask for (Cohere, Perplexity, Ollama)

Three providers have no request-side caching surface at all, and it is worth
being plain about why: this is not a gap in lingo waiting to be filled, it is
absent upstream.

- **Cohere** caches on its own and reports a hit count. Its Chat v2 request has
  no cache field of any kind, so there is nothing for `Cached` to send — which is
  why Cohere models deliberately do not carry `CacheOptions` at all, even though
  `CachingSupport(lingo.ProviderCohere)` reports usage-only. Reads still land in
  `Usage.CacheReadTokens`.
- **Perplexity** exposes no cached-token field on the Sonar chat completions API
  lingo uses, in either the request or the usage block. (Cache counters do appear
  on Perplexity's separate Agent API, which is a different product with a
  different request shape and is not what lingo talks to.) Both counters stay
  zero.
- **Ollama** does keep a KV cache, and its runner does receive a cache-hit count
  from llama.cpp — then folds it into `prompt_eval_count` before the HTTP layer
  sees it. No client can separate the two back out. So `Usage.PromptTokens` is
  the full prompt length, correct and not under-reported on a warm context, and
  both cache counters stay zero.

Calling `Cached` on any of them is a silent no-op, exactly as on a provider that
does support it but cannot honour a particular option.

### Reading the counters

```go
u := response.Usage

fmt.Println(u.PromptTokens)            // whole effective prompt, cached tokens included
fmt.Println(u.CacheReadTokens)         // subset served from the cache, billed cheaply
fmt.Println(u.CacheWriteTokens)        // subset written into the cache
fmt.Println(u.UncachedPromptTokens())  // PromptTokens - reads - writes
fmt.Println(u.CacheHit())              // CacheReadTokens > 0

// Anthropic and Bedrock Claude also split the write by lifetime
fmt.Println(response.Metadata["cache_write_tokens_5m"])
fmt.Println(response.Metadata["cache_write_tokens_1h"])
```

Caching is working when the first call reports a non-zero `CacheWriteTokens` and
later calls report a `CacheReadTokens` of roughly the same size.

When both stay zero, check the request first. Every provider that places a
breakpoint logs at `Debug` whether one actually landed:

```
cache_breakpoint=false  Making Anthropic API request
```

That field reports what the request carries, not what you asked for, so it is
`false` on the most common misconfiguration — `Cached` on a model with no system
prompt, which has no stable prefix to mark. On the completion side the same logs
carry `cache_read_tokens` and `cache_write_tokens`, taken from the normalized
`response.Usage` so they are the numbers your caller sees.

### Traps

**`PromptTokens` covers the whole prompt; the cache counters are subsets of it.**
Providers disagree: OpenAI, Google and Cohere already count cached tokens inside
their prompt total, while Anthropic and Bedrock report them alongside it. Lingo
normalizes to the first convention so `PromptTokens` means the same thing
everywhere and `UncachedPromptTokens()` gives the part processed fresh. The
consequence is that **on Anthropic and Bedrock, `PromptTokens` and `TotalTokens`
grow by reads + writes once caching is enabled.** No existing number moves — the
counters cannot be non-zero until you opt in — but cost dashboards that diff
against history will see a step on the call where caching goes on.

**A prefix below the provider's minimum is silently not cached.** Anthropic's
documented minimum ranges from 512 to 4096 tokens depending on the model, and it
is not monotonic across generations (1024 on Claude Sonnet 5, 512 on Opus 5 and
Fable 5, 4096 on Haiku 4.5); OpenAI's is 1024. Neither SDK encodes these
thresholds, so lingo cannot warn you. Enabling caching on a 200-token system
prompt produces `CacheWriteTokens == 0` and no error — that is the provider
declining, not a bug. Check the provider's current documentation for the exact
figure, and treat the reported counters as ground truth.

**Azure sends a cache key only on the v1 API surface.** `prompt_cache_key` is
absent from every dated api-version, GA and preview alike, so lingo suppresses it
there rather than send a request Azure will answer with a 400. A key you set on a
dated route silently does nothing. See
[Azure: cache keys need the v1 surface](#azure-cache-keys-need-the-v1-surface).

**Google needs a cache resource you created yourself.** `Cached` alone changes
nothing about a Gemini request. Reads are still reported, because Gemini also
caches implicitly, so Google is the one explicit-support provider where
`CacheHit()` can be true on a model you never opted in.

**Rate-limit retries re-send the breakpoint.** The markers are idempotent, so
this is harmless, but a retried attempt usually reads the cache the first attempt
wrote — a call you expected to report a write can report a read instead.

**Opting in on OpenRouter or GPT-5.6 changes the message shape.** The body
switches from `"content": "..."` to `"content": [{"type": "text", ...}]` so the
marker has somewhere to live. It is semantically identical, and only happens on
the opt-in path, but it is visible if you diff request bodies — or if a proxy or
log scraper in front of the provider assumes string content.

**Cohere's counter is unverified against its prompt total.** Cohere reports a
cache-hit count but does not document whether it sits inside or outside
`input_tokens`. Lingo assumes inside, which is the conservative choice: if it is
wrong, `PromptTokens` still matches what Cohere billed and only
`UncachedPromptTokens()` under-reports.

## Thinking Control

Reasoning models spend tokens working a problem out before they answer. How much
they spend is the main lever you have over cost, latency and accuracy — and every
vendor spells it differently.

They disagree in three vocabularies:

- **budget** — a ceiling in thinking tokens: Anthropic's `thinking.budget_tokens`,
  Google's `thinkingBudget`, Cohere's `thinking.token_budget`, OpenRouter's
  `reasoning.max_tokens`.
- **effort** — an ordinal ladder: `reasoning_effort` on OpenAI, Azure, xAI,
  DeepSeek and Perplexity, `output_config.effort` on Claude 4.6+, `thinkingLevel`
  on Gemini 3.x, `think: "low"` on Ollama.
- **toggle** — on or off, with the depth left to the model: DeepSeek's and
  Cohere's `thinking` object, Anthropic's adaptive and disabled configs, Ollama's
  `think: true`, Gemini 2.5's budget of `0`.

`ThinkingOptions` carries all three, plus a trace-visibility knob, and each
provider projects it onto whatever its API actually models. `ThinkingSupport`
reports how far a provider goes:

```go
switch lingo.ThinkingSupport(lingo.ProviderAnthropic) {
case lingo.ThinkSupportControl:   // the request carries a thinking instruction
case lingo.ThinkSupportUsageOnly: // the model reasons on its own terms; counters only
case lingo.ThinkSupportNone:      // nothing to ask for and nothing reported
}
```

That ladder is coarse on purpose — every provider except the generic
OpenAI-compatible endpoint is `control` — because unlike caching, thinking
support varies **per model** inside Anthropic, OpenAI, Google, xAI, Cohere,
Bedrock, Perplexity and Ollama. The precise question is
`ModelThinkingDimensions`, which answers for one model rather than a catalogue:

```go
if lingo.ModelThinkingDimensions(model).Has(lingo.ThinkingCanSetBudget) {
    model = lingo.Thinking(model, lingo.WithThinkingBudget(16000))
}
```

The dimensions are `ThinkingCanToggle`, `ThinkingCanSetEffort`,
`ThinkingCanSetBudget`, `ThinkingCanHideTrace`, `ThinkingCanReportTokens` and
`ThinkingCanReportTrace`.

Thinking is opt-in on the request side: a model whose thinking options you never
touched sends byte-for-byte the request it sent before this feature existed —
including the `reasoning_effort` that OpenAI's and xAI's constructors have always
seeded. Reporting is not opt-in — `response.Thinking` and
`response.Usage.ThinkingTokens` are filled in whenever the provider returns them,
whether or not you asked for anything. Asking for something a provider cannot do
is a silent no-op, never an error, so `Thinking` and `NoThinking` are safe to
call generically.

### What each provider does

| Provider | Level | What lingo sends when you opt in |
|----------|-------|----------------------------------|
| **Anthropic** | control | Per generation: `thinking: {"type": "enabled", "budget_tokens": N}` on Claude 3.7–4.6, `{"type": "adaptive"}` on 4.6 and later, `{"type": "disabled"}` wherever there is an off switch, `output_config.effort` (`low`…`max`; `xhigh` on 4.7+) on 4.6 and later, and `thinking.display` of `summarized` or `omitted` when you ask about the trace. Claude Fable 5 takes effort only; Claude 3.5 and earlier take nothing at all |
| **Google Gemini / Vertex** | control | Gemini 2.5: `thinkingConfig.thinkingBudget` — a number, `-1` for dynamic, or `0` to switch thinking off on Flash and Flash-Lite. Gemini 3.x: `thinkingConfig.thinkingLevel` (`minimal`…`high`). Either generation: `includeThoughts` when you ask for the trace. Gemini 1.5 and 2.0 take nothing |
| **OpenRouter** | control | One `reasoning` object — `{effort, max_tokens, exclude, enabled}` — the only endpoint here that models every dimension lingo has, normalized by OpenRouter onto whatever the model behind the id speaks |
| **DeepSeek** | control | `thinking: {"type": "enabled"}` or `{"type": "disabled"}`, and `reasoning_effort` (`low`, `high`, `max`). Both together when you ask for both |
| **Cohere** | control | `thinking: {"type": "enabled", "token_budget": N}` or `{"type": "disabled"}`, on Command A+ and Command A Reasoning. The other Command models send nothing |
| **Ollama** | control | `think`: `true`, `false`, or a level — `"low"`, `"medium"`, `"high"` — on the model families known to carry Ollama's thinking capability |
| **AWS Bedrock** | control | Claude only, on `InvokeModel`: the Anthropic-dialect `thinking: {"type": "enabled", "budget_tokens": N}` on 3.7–4.6 and `{"type": "disabled"}` wherever the generation has an off switch. No `output_config.effort` and no adaptive config. Nova, Llama, Mistral and Titan send nothing |
| **OpenAI** | control | `reasoning_effort`, on reasoning models only, clamped to the ladder that model id accepts. GPT-4o and the other standard chat models carry no thinking storage at all |
| **Azure OpenAI** | control | `reasoning_effort`, on a deployment you built with `NewAzureOpenAIReasoningModel`; nothing on a standard deployment |
| **xAI** | control | `reasoning_effort` on grok-4.3 (`none`, `low`, `medium`, `high`), grok-4.5 (`low`…`high`) and raw ids through `NewXAIModel`. The Grok 4.20 family and grok-build-0.1 send nothing |
| **Perplexity** | control | `reasoning_effort` (`minimal`…`high`) on Sonar Deep Research. The other Sonar models send nothing |
| **OpenAI-compatible** | usage-only | Only the `reasoning_effort` you set yourself with `WithReasoningEffort`; lingo synthesizes nothing, because the endpoint behind `BaseURL` decides what thinking means and local servers commonly reject a field they do not know. Anything richer goes through `WithExtraField` |

What comes back differs too:

- Anthropic and Bedrock Claude return the trace in `response.Thinking`, with
  `Metadata["thinking_signature"]` and, for encrypted blocks,
  `Metadata["thinking_redacted"]`. Anthropic reports `ThinkingTokens`; Bedrock
  does not report one on either of its request paths, so there the counter stays
  zero.
- Google returns its thought parts in `response.Thinking` — but only when you ask
  with `ThinkingTraceInclude` — and reports `ThinkingTokens` either way. The
  thought signature is base64 in `Metadata["thinking_signature"]`.
- DeepSeek, xAI, OpenRouter and OpenAI-compatible endpoints return
  `reasoning_content` in `response.Thinking`, and report `ThinkingTokens` where
  the endpoint sends `completion_tokens_details.reasoning_tokens`.
- OpenAI and Azure report `ThinkingTokens` and nothing else: chat completions
  never returns a trace.
- Cohere returns its thinking blocks in `response.Thinking` and reports no
  thinking token count — Cohere folds them into `output_tokens`.
- Perplexity splits the leading `<think>…</think>` block out of the answer into
  `response.Thinking`, and reports `reasoning_tokens`.
- Ollama returns `message.thinking` in `response.Thinking`. Thinking tokens are
  folded into `eval_count` before the HTTP layer sees them, so that counter stays
  zero as well.

### Your existing code keeps working

The neutral surface is not a replacement. It is a second view of the same
storage: every per-model thinking setter lingo already shipped now reads and
writes the same `ThinkingOptions` the portable API does, so the two can never
disagree about what the request will carry. Nothing below changed its signature
or its bytes:

```go
adaptive := lingo.NewClaudeOpus48().WithAdaptiveThinking()
effort   := lingo.NewClaudeOpus5().WithEffort(lingo.EffortXHigh)
budget   := lingo.NewClaudeSonnet4().WithThinkingBudget(4096)
gpt      := lingo.NewGPT56Sol().WithReasoningEffort("xhigh")
grok     := lingo.NewGrok43().WithReasoningEffort(lingo.XAIEffortNone)
deepseek := lingo.NewDeepSeekV4Flash().WithThinkingDisabled()
command  := lingo.NewCommandAReasoning().WithThinkingBudget(4096)
router   := lingo.NewOpenRouterModel("anthropic/claude-sonnet-5").WithReasoningMaxTokens(8000)
```

Two rules keep that true, and they are worth knowing because they are also the
two places the portable API deliberately behaves differently:

1. **A value a per-model setter set is pinned: forwarded verbatim, never clamped,
   translated or dropped.** `WithReasoningEffort("obsessive")` reaches the wire as
   `"obsessive"`. `WithThinkingBudget(500)` on Claude reaches it as `500`, even
   though the API's floor is 1024 and lingo knows it. You named a knob on a
   concrete model type, so lingo assumes you meant that model's dialect and lets
   the API answer. The same protection covers constructor-seeded defaults:
   `NewGPT5()` has always sent `reasoning_effort: "medium"` and `NewGrok43()`
   `"low"`, and both still do.
2. **A value the portable API set is adapted.** `WithThinkingEffort` and
   `WithThinkingBudget` promise that your intent survives, not that your number
   does: they clamp to the model's ladder or budget window and translate across
   vocabularies. Writing a dimension through the portable API also clears the pin
   on it, which is why
   `lingo.Thinking(lingo.NewGPT56Sol(), lingo.WithThinkingEffort(lingo.ThinkingEffortMax))`
   sends `"xhigh"` rather than a `"max"` that chat completions would reject.

Both views read the same fields:

```go
model := lingo.NewClaudeSonnet46().WithAdaptiveThinking()

model.ThinkingOptions().DynamicBudget() // true
model.ThinkingOptions().Enabled()       // true
```

One asymmetry to keep in mind on Anthropic: `WithEffort` sets
`output_config.effort` and nothing else, because on Claude that field caps
overall spend rather than switching reasoning on — `NewClaudeOpus46().WithEffort(lingo.EffortHigh)`
sends no `thinking` key at all. The portable `WithThinkingEffort` does turn
thinking on, as it does everywhere else.

### One API, three vocabularies

When you ask for a knob a model does not have, lingo translates rather than
drops — losing your intent entirely is worse — and the translation is
deterministic, published here, applied only to unpinned values, and reported
back in `response.Metadata["thinking_translation"]`.

**An effort on a model that budgets in tokens** — Cohere, Gemini 2.5, Claude
3.7–4.5, Bedrock Claude — becomes a fraction of that model's usable thinking
window, through `ThinkingBudgetForEffort`:

| effort | share of the window |
|--------|---------------------|
| `minimal` | 5% |
| `low` | 12% |
| `medium` | 30% |
| `high` | 60% |
| `xhigh` | 85% |
| `max` | 100% |

The ladder is geometric rather than linear because thinking quality is roughly
logarithmic in budget: low to medium buys far more than xhigh to max. So
`WithThinkingEffort(ThinkingEffortLow)` on Command A Reasoning, whose window
ceiling is its own 8192-token `max_tokens`, sends `token_budget: 983`.

**A budget on a model that only takes an effort** — OpenAI, Azure, xAI, DeepSeek,
Ollama, Gemini 3.x — becomes a level, through `ThinkingEffortForBudget`:

| budget | effort |
|--------|--------|
| 1 – 2048 | `minimal` |
| 2049 – 8192 | `low` |
| 8193 – 24576 | `medium` |
| 24577 – 65536 | `high` |
| 65537 and above | `xhigh` |

This is the lossy direction to be explicit about: **the number is not honoured.**
`lingo.Thinking(lingo.NewGPT56Sol(), lingo.WithThinkingBudget(30000))` does not
cap anything at 30000 tokens. It sends `reasoning_effort: "high"`, and what
OpenAI actually spends is OpenAI's business. If a hard ceiling is what you need,
check `ModelThinkingDimensions(model).Has(lingo.ThinkingCanSetBudget)` first and
pick a model that has one.

**An effort off the model's ladder** clamps to the nearest rung the model does
accept, preferring to go down: `ThinkingEffortMax` becomes `xhigh` on GPT-5.6 and
`high` on Gemini 3 Pro. One exception keeps a small request from becoming no
request: an effort that asks the model to think a little never clamps down to
`none`, so `ThinkingEffortMinimal` on a ladder that has `none` but no `minimal`
(gpt-5.1, grok-4.3) becomes `low`.

**A budget outside the model's window** is clamped into it — 1024 to
`max_tokens - 1` on Claude, 128 to 32768 on Gemini 2.5 Pro, 1 to the model's own
`max_tokens` on Cohere.

**Off, on a model with no off switch,** is sent as `reasoning_effort: "none"`
where that rung exists. Where it does not, the off — and only the off — is
dropped: a depth the model was already carrying still goes out, because sending
nothing would leave the model at a server-side default that reasons harder than
you asked for. See [Turning thinking off](#turning-thinking-off).

Every one of those decisions leaves a breadcrumb:

```go
fmt.Println(response.Metadata["thinking_translation"])
// budget 30000 mapped to effort high
// effort max clamped to xhigh; trace omission dropped: model always returns its trace
```

Every provider also logs it at `Debug` as `thinking_translation`, beside the
`reasoning_effort` or budget the request actually carries, so you can see what
went out without diffing bodies.

### Budget style (Anthropic, Google, Cohere)

`Thinking` turns thinking on and returns the same model, so it slots into the
builder chain and keeps the concrete type — exactly like `Cached`:

```go
// Claude 3.7 through 4.6 take a fixed budget: at least 1024 tokens, and
// strictly below max_tokens.
model := lingo.Thinking(lingo.NewClaudeSonnet46().WithMaxTokens(16000),
    lingo.WithThinkingBudget(8000))
// thinking: {"type": "enabled", "budget_tokens": 8000}

// Gemini 2.5 budgets in tokens too, in a per-model window.
gemini := lingo.Thinking(lingo.NewGemini25Pro(), lingo.WithThinkingBudget(8192))
// thinkingConfig: {"thinkingBudget": 8192}

// "Think, and decide for yourself how much" is a budget of its own.
dynamic := lingo.Thinking(lingo.NewGemini25Flash(), lingo.WithDynamicThinking())
// thinkingConfig: {"thinkingBudget": -1}

// Cohere pairs its toggle with a ceiling.
command := lingo.Thinking(lingo.NewCommandAReasoning(), lingo.WithThinkingBudget(2048))
// thinking: {"type": "enabled", "token_budget": 2048}
```

The statement form is equivalent, and is how you reach the options on a model you
already built:

```go
model := lingo.NewClaudeOpus5().WithMaxTokens(8192)
model.ThinkingOptions().Enable().WithEffort(lingo.ThinkingEffortXHigh)
```

On Claude, `WithDynamicThinking` is the adaptive config on 4.6 and later, and on
3.7–4.5 — which predate it — a plain `Thinking(model)` becomes a fixed budget of
60% of the room `max_tokens` leaves, with a note saying so. That is the whole
translation contract in one place: the request is never a no-op just because the
generation is old, and it never lies about what it did.

### Effort style (OpenAI, Azure, xAI, DeepSeek, Perplexity, Claude 4.6+)

```go
model := lingo.Thinking(lingo.NewGPT56Sol().WithMaxCompletionTokens(4096),
    lingo.WithThinkingEffort(lingo.ThinkingEffortXHigh))
// reasoning_effort: "xhigh"

// The ladder is per model id, not per provider: gpt-5.1 has "none" and no
// "minimal", the original GPT-5 family has the reverse, gpt-5-pro takes only
// "high", and o1-mini takes no effort at all.
shallow := lingo.Thinking(lingo.NewGPT5(), lingo.WithThinkingEffort(lingo.ThinkingEffortMinimal))
// reasoning_effort: "minimal"

// Claude spells depth as output_config.effort from 4.6 on, beside the thinking
// config rather than inside it — so the portable call sets both.
claude := lingo.Thinking(lingo.NewClaudeOpus5(), lingo.WithThinkingEffort(lingo.ThinkingEffortXHigh))
// output_config: {"effort": "xhigh"}, thinking: {"type": "adaptive"}

// Gemini 3.x has its own four-rung ladder and no budget at all.
gemini := lingo.Thinking(lingo.NewGemini3Pro(), lingo.WithThinkingEffort(lingo.ThinkingEffortHigh))
// thinkingConfig: {"thinkingLevel": "HIGH"}
```

### On and off (DeepSeek, Ollama, Cohere, Claude 5)

Several models reason by default, so the switch is there to say stop as often as
start:

```go
// DeepSeek V4 reasons unless told not to.
fast := lingo.NoThinking(lingo.NewDeepSeekV4Flash())
// thinking: {"type": "disabled"}

// The toggle and the depth are independent fields here, so both go out.
deep := lingo.Thinking(lingo.NewDeepSeekV4Pro(), lingo.WithThinkingEffort(lingo.ThinkingEffortMax))
// thinking: {"type": "enabled"}, reasoning_effort: "max"

// Ollama takes a bool or a level in one field, so a level is sent instead of
// the bare true, never beside it.
local := lingo.Thinking(lingo.NewQwen3(), lingo.WithThinkingEffort(lingo.ThinkingEffortLow))
// think: "low"

// Claude 5 reasons by default as well, so an explicit enable spells out what
// the model was already doing.
claude := lingo.Thinking(lingo.NewClaudeSonnet5())
// thinking: {"type": "adaptive"}
```

### Turning thinking off

```go
lingo.NoThinking(lingo.NewClaudeOpus5())   // thinking: {"type": "disabled"}
lingo.NoThinking(lingo.NewGemini25Flash()) // thinkingConfig: {"thinkingBudget": 0}
lingo.NoThinking(lingo.NewGPT51())         // reasoning_effort: "none"
lingo.NoThinking(lingo.NewQwen3())         // think: false

lingo.NoThinking(lingo.NewGemini3Pro())    // nothing — 3.x cannot be switched off
lingo.NoThinking(lingo.NewGrok45())        // nothing — no documented off switch
```

`NoThinking` cannot silence a model that reasons unconditionally — Gemini 3.x,
Claude Fable 5, Gemini 2.5 Pro, Sonar Reasoning, the Grok 4.20 family. There it
is a documented no-op, and the reason lands in
`Metadata["thinking_translation"]` as `thinking off dropped: model has no off
switch`. On Claude Opus 5 the API accepts `disabled` only at `EffortHigh` or
below, so pair it with a lower effort rather than `xhigh`.

**A no-op leaves the depth alone.** Where there is no off switch, `NoThinking`
changes nothing at all — it does not also drop a depth the model was already
carrying:

```go
// o3 has no "none" rung, so the off is dropped and the pinned effort stands.
lingo.NoThinking(lingo.NewO3().WithReasoningEffort("high")) // reasoning_effort: "high"

// The seed 26 of the 27 OpenAI reasoning constructors ship with survives too.
lingo.NoThinking(lingo.NewGPT5())                           // reasoning_effort: "medium"

// gpt-5.1 onward do have the rung, so there the off is real and replaces it.
lingo.NoThinking(lingo.NewGPT51())                          // reasoning_effort: "none"
```

Dropping the depth would look like the cautious choice and is the opposite of
one: `o3` with no `reasoning_effort` on the wire is `o3` at OpenAI's own
server-side default, which reasons *harder* than the `"high"` you pinned and
bills accordingly. So "I cannot switch it off" never turns into "so I turned it
up". The rule applies to a portable depth as well, which is still clamped to the
model's ladder as usual — it is simply not thrown away.

### Reading the trace and the counters

```go
response, err := gateway.Generate(ctx, model, "Explain the proof")

fmt.Println(response.Text)     // the answer
fmt.Println(response.Thinking) // the reasoning trace, "" when none came back

u := response.Usage
fmt.Println(u.CompletionTokens) // everything the model generated
fmt.Println(u.ThinkingTokens)   // the subset spent reasoning
fmt.Println(u.AnswerTokens())   // CompletionTokens - ThinkingTokens

fmt.Println(response.Metadata["thinking_translation"]) // what lingo adapted, if anything
fmt.Println(response.Metadata["thinking_signature"])   // Anthropic, Bedrock, Google
```

`Thinking` is never part of `Text`. Providers that return the trace as its own
content block are read block by block, and Perplexity, which inlines a leading
`<think>…</think>` in the message itself, is split before the response struct is
built — so an answer no longer arrives with its reasoning glued to the front.

`ThinkingTokens` is a subset of `CompletionTokens` everywhere, including on
Google, which is the one provider that counts thoughts outside its candidate
total. Lingo folds them in so the invariant holds, which means **on Google,
`CompletionTokens` grows by the thought count relative to the raw API response**;
`TotalTokens` does not move, because Google's own total already included them.
Everywhere else the provider already reported the subset relationship and no
number changes.

Three metadata keys are kept for one release and are deprecated:
`Metadata["thinking"]` (Anthropic, Bedrock) and `Metadata["reasoning_content"]`
(DeepSeek, xAI, OpenRouter, Cohere, Perplexity, OpenAI-compatible) still mirror
`response.Thinking`, and `Metadata["reasoning_tokens"]` still mirrors
`Usage.ThinkingTokens` on the OpenAI dialect. Read the typed field and the
counter instead.

### Traps

**A budget on an effort-only provider is adapted, not honoured.** It becomes the
nearest effort level by the table above, and the model spends whatever it spends.
Only Anthropic 3.7–4.6, Google 2.5, Cohere, OpenRouter and Bedrock Claude take a
real ceiling; ask `ModelThinkingDimensions` before you rely on one.

**The knobs are per model, not per provider.** `ThinkingSupport` says `control`
for Anthropic, and `NewClaude35Sonnet()` still takes nothing at all. The same
holds for GPT-4o, Command R, standard Azure deployments, Bedrock's Titan, Llama
and Mistral families, the Grok 4.20 models, Sonar and Sonar Pro, and most of
Ollama's catalogue. Where lingo can arrange it — Claude 3.5 and earlier, GPT-4o
and the other standard OpenAI chat models — those types do not carry
`ThinkingOptions` at all, so `model.ThinkingOptions()` does not compile on them.
`Thinking(model)` stays a silent no-op either way.

**Trace visibility is a no-op on OpenAI and Azure.** `ThinkingTraceInclude` asks
for the reasoning to come back with the answer; over chat completions, the
dialect lingo speaks, there is no field for it, and reasoning summaries belong to
the Responses API, which lingo does not use. It is also a no-op on xAI, DeepSeek,
Cohere, Perplexity and Ollama — they return their trace whether or not you ask.
Only Anthropic (`thinking.display`), Google (`includeThoughts`) and OpenRouter
(`reasoning.exclude`) model it. Google is the one where it matters in the other
direction: no thoughts come back unless you ask for them.

**Bedrock Claude is a narrower surface than first-party Anthropic.** It sends
token budgets and the off switch and nothing else: neither `output_config.effort`
nor the adaptive config is verifiable against the `InvokeModel` body at the
pinned SDK version, and an unknown field there is a 400 rather than an ignored
key. So depth is settable on Bedrock's 3.7 through 4.6, only the off switch is
reachable on 4.7/4.8 and the 5 series, and Bedrock reports no thinking token
count at all — `ThinkingTokens` is always zero there, on both request paths.

**Bedrock Nova is gated off.** Nova models store thinking configuration and send
none of it: `bedrockruntime`'s `InferenceConfiguration` has no reasoning field at
the pinned version, and the `additionalModelRequestFields` key Nova would need is
not verifiable from any source lingo pins, so lingo declines to guess.
`Thinking(NewBedrockNovaPro())` is a silent no-op. Reasoning blocks that come
back on the Converse response are read normally.

**Ollama refuses to think on a model that cannot.** Ollama answers `think` on a
model without the thinking capability with a 400 rather than ignoring it — the
one place where being optimistic would break the never-error rule. So lingo gates
on a list of known thinking families (deepseek-r1, qwen3, qwq, gpt-oss,
magistral, phi4-reasoning, exaone-deep, smollm3, cogito, deepseek-v3.1) and
treats an unlisted tag — including your own private one — as unable to think:
`Thinking(NewOllamaModel("my-finetune"))` sends nothing. Ollama's own discovery
path, `POST /api/show`, would answer precisely; it costs a round trip per model
and is not wired up.

A family matches only up to Ollama's tag separator, so it covers the bare name
and every `:variant` of it and nothing else. `qwen3` and `qwen3:30b-a3b` think;
`qwen3-coder:30b`, `qwen3-embedding` and `qwen3-reranker` are separate models
that do not, and lingo sends them nothing rather than a 400. The same
conservatism catches a thinking model that renames rather than tags its variant —
`deepseek-v3.1-terminus` is an unlisted tag and therefore a silent no-op — which
is the safe direction to be wrong in.

**A pinned value can still be rejected.** Verbatim forwarding is the price of not
changing anyone's bytes, and it cuts both ways: `NewO1Mini()` has always seeded
`reasoning_effort: "medium"` even though o1-mini accepts no effort at all, and
lingo keeps sending it. `WithThinkingBudget(500)` on Claude is still the 400 it
always was. The portable surface is the one that protects you from those; the
per-model setters are the one that reproduces them.

**An unknown model id resolves differently on purpose.** A `claude-*` id lingo
has not seen gets the current generation's dialect, because no knob has ever been
withdrawn from it. A Gemini id gets nothing, because Google's generations
disagree about the vocabulary itself and guessing would send a budget where a
level belongs. A Cohere id is taken at its word, an Ollama tag is assumed not to
think, and an OpenAI id falls back to the current family's ladder. The asymmetry
is deliberate: it always resolves toward the silent no-op rather than the 400.

**Unverified: Anthropic's trace default.** The SDK documents `thinking.display`
as defaulting to `summarized`; Anthropic's platform documentation says the newer
models omit the trace by default. Lingo sends no `display` unless you ask, so if
the documentation is right, `response.Thinking` can be empty on Claude 4.7+ until
you pass `WithThinkingTrace(lingo.ThinkingTraceInclude)`.

**Unverified: Gemini 2.5's budget windows.** `genai` models `thinkingBudget` as a
bare integer with no range, so the numbers lingo clamps into (Pro 128–32768 and
no off switch, Flash 1–24576, Flash-Lite 512–24576 and off by default) come from
Google's model documentation rather than from anything the SDK encodes. They
affect only an unpinned budget: a stale figure means lingo clamps to a slightly
wrong value, never that it sends an illegal one.

**Unverified: where Perplexity counts its reasoning tokens.** The API reference
names `usage.reasoning_tokens` but does not say whether it sits inside
`completion_tokens`. Lingo treats it as a subset, because the `<think>` block is
part of the message content and is therefore already counted there. If Perplexity
is in fact reporting it additively, `AnswerTokens()` under-reports by that much.

**Azure sends `reasoning_effort` on every api-version.** Unlike the cache key,
lingo does not gate it: Microsoft publishes no minimum api-version for it, and
lingo has always sent it once a caller asked. Azure deployments are also
addressed by name, so lingo cannot tell which model is behind one and cannot
clamp the ladder per model — the per-model legality of a rung is yours to check.

**A trace signature holds only the last block.** `Metadata` is
`map[string]string`, so a response split across several thinking blocks leaves
just the final signature in `Metadata["thinking_signature"]`. That is enough for
logging and not enough for faithful multi-turn replay, which needs a typed
content API that single-turn `Generate` does not have.

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
    Thinking     string            // Reasoning trace, "" when none came back
    Metadata     map[string]string // Provider-specific data
}

type TokenUsage struct {
    PromptTokens     int // whole effective prompt, cached tokens included
    CompletionTokens int // everything generated, thinking included
    TotalTokens      int
    CacheReadTokens  int // subset of PromptTokens served from a prompt cache
    CacheWriteTokens int // subset of PromptTokens written into a prompt cache
    ThinkingTokens   int // subset of CompletionTokens spent reasoning
}

func (u TokenUsage) UncachedPromptTokens() int // PromptTokens - reads - writes
func (u TokenUsage) CacheHit() bool            // CacheReadTokens > 0
func (u TokenUsage) AnswerTokens() int         // CompletionTokens - ThinkingTokens
```

The two cache counters stay zero on providers that do not report them, and are
filled in whether or not you opted into caching. See
[Prompt Caching](#prompt-caching) for what each provider reports and how the
normalization affects `PromptTokens`.

`Thinking` and `ThinkingTokens` behave the same way: both are filled in whenever
the provider returns them, opted in or not, and both stay empty on providers that
report neither. `Thinking` is never part of `Text`, and `ThinkingTokens` is
always a subset of `CompletionTokens` — including on Google, where lingo folds
the thought count in to keep that true. See
[Thinking Control](#thinking-control) for what each provider returns and which
metadata keys it deprecates.

## License

MIT License - see [LICENSE](LICENSE) for details.

