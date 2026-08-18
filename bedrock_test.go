package lingo

import (
	"encoding/json"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	brtypes "github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

// ============================================================================
// CONVERSE ROUTING
// ============================================================================
//
// Only Nova is served by Converse. Every other family stays on InvokeModel and
// therefore keeps the exact request bytes it produced before Converse existed;
// the guard in Generate is a single early return keyed on this predicate, so
// this test is what pins the blast radius to one family.

func TestBedrockUsesConverseIsNovaOnly(t *testing.T) {
	for _, family := range []string{"claude", "titan", "llama", "mistral", "unknown", ""} {
		if bedrockUsesConverse(family) {
			t.Errorf("bedrockUsesConverse(%q) = true, want false: %s must stay on InvokeModel", family, family)
		}
	}
	if !bedrockUsesConverse("nova") {
		t.Error(`bedrockUsesConverse("nova") = false, want true`)
	}
	// The routing key comes from getModelFamily for every model but BedrockModel.
	for _, modelID := range []string{"amazon.nova-pro-v1:0", "us.amazon.nova-premier-v1:0"} {
		if got := getModelFamily(modelID); !bedrockUsesConverse(got) {
			t.Errorf("getModelFamily(%q) = %q, which does not route to Converse", modelID, got)
		}
	}
	// Titan shares the "amazon." prefix and must not be dragged along.
	if got := getModelFamily("amazon.titan-text-premier-v1:0"); bedrockUsesConverse(got) {
		t.Errorf("Titan routed to Converse as family %q", got)
	}
}

// ============================================================================
// CONVERSE REQUEST SHAPE
// ============================================================================

// bedrockCachePoints counts the cache checkpoints in a Converse input, split by
// the union they sit in.
func bedrockCachePoints(in *bedrockruntime.ConverseInput) (system, message int) {
	for _, b := range in.System {
		if _, ok := b.(*brtypes.SystemContentBlockMemberCachePoint); ok {
			system++
		}
	}
	for _, m := range in.Messages {
		for _, b := range m.Content {
			if _, ok := b.(*brtypes.ContentBlockMemberCachePoint); ok {
				message++
			}
		}
	}
	return system, message
}

func TestUntouchedNovaConverseInputCarriesNoCachePoint(t *testing.T) {
	c := &bedrockClient{}

	for _, model := range []Model{
		NewBedrockNovaMicro(),
		NewBedrockNovaLite(),
		NewBedrockNovaPro().WithSystemPrompt("be terse"),
		NewBedrockNovaPremier().WithSystemPrompt("be terse"),
		NewBedrockModel("amazon.nova-pro-v1:0", "nova").WithSystemPrompt("be terse"),
		// Explicitly disabled must look exactly like never-touched.
		NotCached(NewBedrockNovaPro().WithSystemPrompt("be terse")),
	} {
		in, breakpoint := c.buildConverseInput(model, "hello", model.ModelName())
		if sys, msg := bedrockCachePoints(in); sys != 0 || msg != 0 {
			t.Errorf("%s: cache points sent without opting in (system=%d, message=%d)",
				model.ModelName(), sys, msg)
		}
		if breakpoint {
			t.Errorf("%s: reported a cache point it never placed", model.ModelName())
		}
		if len(in.Messages) != 1 || len(in.Messages[0].Content) != 1 {
			t.Errorf("%s: messages = %+v, want one message with one text block", model.ModelName(), in.Messages)
		}
		if in.Messages[0].Role != brtypes.ConversationRoleUser {
			t.Errorf("%s: role = %q", model.ModelName(), in.Messages[0].Role)
		}
		if in.AdditionalModelRequestFields != nil {
			t.Errorf("%s: additionalModelRequestFields sent without WithTopK", model.ModelName())
		}
	}
}

func TestOptedInNovaConverseInputCarriesCachePoints(t *testing.T) {
	c := &bedrockClient{}

	// System only, which is what Enable() defaults to.
	in, breakpoint := c.buildConverseInput(Cached(NewBedrockNovaPro().WithSystemPrompt("be terse")), "hello", "amazon.nova-pro-v1:0")
	if !breakpoint {
		t.Error("a placed system cache point went unreported")
	}
	sys, msg := bedrockCachePoints(in)
	if sys != 1 || msg != 0 {
		t.Fatalf("system-only caching produced system=%d message=%d cache points", sys, msg)
	}
	if len(in.System) != 2 {
		t.Fatalf("system = %+v, want the text block plus a trailing cache point", in.System)
	}
	cp, ok := in.System[1].(*brtypes.SystemContentBlockMemberCachePoint)
	if !ok {
		t.Fatalf("system[1] = %T, want a cache point last", in.System[1])
	}
	if cp.Value.Type != brtypes.CachePointTypeDefault {
		t.Errorf("cache point type = %q, want %q", cp.Value.Type, brtypes.CachePointTypeDefault)
	}

	// Both breakpoints, with a TTL Nova does not model: it is clamped away
	// rather than rejected, so the request stays valid.
	in, _ = c.buildConverseInput(
		Cached(NewBedrockNovaPro().WithSystemPrompt("be terse"), WithCachePrompt(true), WithCacheTTL(CacheTTL1h)),
		"hello", "amazon.nova-pro-v1:0")
	if sys, msg = bedrockCachePoints(in); sys != 1 || msg != 1 {
		t.Fatalf("system+prompt caching produced system=%d message=%d cache points", sys, msg)
	}
	if cp := in.System[1].(*brtypes.SystemContentBlockMemberCachePoint); cp.Value.Ttl != "" {
		t.Errorf("system cache point Ttl = %q, want it clamped away", cp.Value.Ttl)
	}
	last := in.Messages[0].Content
	if _, ok := last[len(last)-1].(*brtypes.ContentBlockMemberCachePoint); !ok {
		t.Errorf("message content = %T..., want a trailing cache point", last[len(last)-1])
	}

	// Asking for a system breakpoint on a model with no system prompt is a
	// no-op rather than an empty system block.
	in, breakpoint = c.buildConverseInput(Cached(NewBedrockNovaPro()), "hello", "amazon.nova-pro-v1:0")
	if len(in.System) != 0 {
		t.Errorf("system = %+v, want nothing when there is no system prompt to cache", in.System)
	}
	if breakpoint {
		t.Error("opting in with no system prompt reported a cache point that was never placed")
	}
}

func TestNovaConverseInputCarriesInferenceOptions(t *testing.T) {
	c := &bedrockClient{}

	in, _ := c.buildConverseInput(
		NewBedrockNovaPro().WithMaxTokens(1234).WithTemperature(0.42).WithTopP(0.77).WithTopK(20),
		"hello", "amazon.nova-pro-v1:0")

	if got := aws.ToInt32(in.InferenceConfig.MaxTokens); got != 1234 {
		t.Errorf("maxTokens = %d, want 1234", got)
	}
	if got := aws.ToFloat32(in.InferenceConfig.Temperature); got != 0.42 {
		t.Errorf("temperature = %v, want 0.42", got)
	}
	if got := aws.ToFloat32(in.InferenceConfig.TopP); got != 0.77 {
		t.Errorf("topP = %v, want 0.77", got)
	}
	// InferenceConfiguration models no topK, so it rides along under Nova's own
	// inferenceConfig path in additionalModelRequestFields.
	if in.AdditionalModelRequestFields == nil {
		t.Fatal("topK was dropped: additionalModelRequestFields is nil")
	}
	raw, err := in.AdditionalModelRequestFields.MarshalSmithyDocument()
	if err != nil {
		t.Fatal(err)
	}
	var extra map[string]map[string]int
	if err := json.Unmarshal(raw, &extra); err != nil {
		t.Fatal(err)
	}
	if extra["inferenceConfig"]["topK"] != 20 {
		t.Errorf("additionalModelRequestFields = %s, want inferenceConfig.topK 20", raw)
	}
}

// ============================================================================
// CONVERSE RESPONSE ACCOUNTING
// ============================================================================

func TestParseConverseOutputNormalizesCacheCounters(t *testing.T) {
	c := &bedrockClient{}

	out := &bedrockruntime.ConverseOutput{
		Output: &brtypes.ConverseOutputMemberMessage{Value: brtypes.Message{
			Role: brtypes.ConversationRoleAssistant,
			Content: []brtypes.ContentBlock{
				&brtypes.ContentBlockMemberText{Value: "hel"},
				&brtypes.ContentBlockMemberText{Value: "lo"},
			},
		}},
		StopReason: brtypes.StopReasonEndTurn,
		Usage: &brtypes.TokenUsage{
			InputTokens:  aws.Int32(100),
			OutputTokens: aws.Int32(7),
			// Converse reports its own totalTokens with the cache tokens
			// already folded in; taking it at face value and then folding
			// again would double count.
			TotalTokens:           aws.Int32(1007),
			CacheReadInputTokens:  aws.Int32(900),
			CacheWriteInputTokens: aws.Int32(0),
			CacheDetails: []brtypes.CacheDetail{
				{InputTokens: aws.Int32(64), Ttl: brtypes.CacheTTLFiveMinutes},
				{InputTokens: aws.Int32(0), Ttl: brtypes.CacheTTLOneHour},
			},
		},
	}

	resp, err := c.parseConverseOutput(out, "amazon.nova-pro-v1:0", "nova")
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text != "hello" {
		t.Errorf("Text = %q, want the concatenated text blocks", resp.Text)
	}
	if resp.FinishReason != "end_turn" {
		t.Errorf("FinishReason = %q, want end_turn", resp.FinishReason)
	}
	// PromptTokens covers the whole effective prompt; the counters are subsets.
	if resp.Usage.PromptTokens != 1000 || resp.Usage.CacheReadTokens != 900 {
		t.Errorf("Usage = %+v, want PromptTokens 1000 and CacheReadTokens 900", resp.Usage)
	}
	if resp.Usage.TotalTokens != 1007 {
		t.Errorf("TotalTokens = %d, want 1007 (prompt 1000 + completion 7), not a double count", resp.Usage.TotalTokens)
	}
	if resp.Usage.UncachedPromptTokens() != 100 {
		t.Errorf("UncachedPromptTokens() = %d, want 100", resp.Usage.UncachedPromptTokens())
	}
	if !resp.Usage.CacheHit() {
		t.Error("CacheHit() = false")
	}
	if resp.Metadata["family"] != "nova" || resp.Metadata["provider"] != "bedrock" {
		t.Errorf("Metadata = %v", resp.Metadata)
	}
	if resp.Metadata["cache_write_tokens_5m"] != "64" {
		t.Errorf("cache_write_tokens_5m = %q, want 64", resp.Metadata["cache_write_tokens_5m"])
	}
	if _, ok := resp.Metadata["cache_write_tokens_1h"]; ok {
		t.Errorf("an empty TTL bucket was recorded: %v", resp.Metadata)
	}
}

func TestParseConverseOutputWithoutCaching(t *testing.T) {
	c := &bedrockClient{}

	out := &bedrockruntime.ConverseOutput{
		Output: &brtypes.ConverseOutputMemberMessage{Value: brtypes.Message{
			Content: []brtypes.ContentBlock{&brtypes.ContentBlockMemberText{Value: "hi"}},
		}},
		StopReason: brtypes.StopReasonMaxTokens,
		Usage: &brtypes.TokenUsage{
			InputTokens:  aws.Int32(12),
			OutputTokens: aws.Int32(3),
			TotalTokens:  aws.Int32(15),
		},
	}

	resp, err := c.parseConverseOutput(out, "amazon.nova-lite-v1:0", "nova")
	if err != nil {
		t.Fatal(err)
	}
	want := TokenUsage{PromptTokens: 12, CompletionTokens: 3, TotalTokens: 15}
	if resp.Usage != want {
		t.Errorf("Usage = %+v, want %+v unchanged when nothing was cached", resp.Usage, want)
	}
	if resp.FinishReason != "max_tokens" {
		t.Errorf("FinishReason = %q", resp.FinishReason)
	}
}

func TestParseConverseOutputRejectsUnusableResponses(t *testing.T) {
	c := &bedrockClient{}

	if _, err := c.parseConverseOutput(nil, "amazon.nova-pro-v1:0", "nova"); err == nil {
		t.Error("a nil ConverseOutput must be an error, not a panic")
	}
	empty := &bedrockruntime.ConverseOutput{
		Output: &brtypes.ConverseOutputMemberMessage{Value: brtypes.Message{}},
	}
	if _, err := c.parseConverseOutput(empty, "amazon.nova-pro-v1:0", "nova"); err == nil {
		t.Error("a response with no text content must be an error")
	}
}
