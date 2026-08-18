package lingo

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
)

// ============================================================================
// ARN MODEL IDS
// ============================================================================
//
// InvokeModel's ModelId takes an ARN as readily as it takes a bare id: an
// inference profile ARN, or a provisioned-throughput one. Everything lingo
// decides about a Bedrock request is decided from that string -- which family
// builds the body, which Llama template it is written in, which Claude thinking
// dialect it can carry -- so an ARN that classifies as nothing is not cosmetic.
// It is the Llama 4 that gets sent the 3.x template, and the Claude that gets
// its thinking config dropped.

const (
	arnLlama4 = "arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.meta.llama4-scout-17b-instruct-v1:0"
	arnLlama3 = "arn:aws:bedrock:eu-west-1:123456789012:inference-profile/eu.meta.llama3-3-70b-instruct-v1:0"
	arnClaude = "arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.anthropic.claude-sonnet-4-20250514-v1:0"
	arnNova   = "arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.amazon.nova-pro-v1:0"
	// A provisioned-throughput ARN names an opaque id. There is nothing in it
	// to classify, and that has to stay a safe default rather than a panic.
	arnOpaque = "arn:aws:bedrock:us-east-1:123456789012:provisioned-model/abc123"
)

func TestBedrockResolveModelIDUnwrapsARNs(t *testing.T) {
	for id, want := range map[string]string{
		arnLlama4: "meta.llama4-scout-17b-instruct-v1:0",
		arnLlama3: "meta.llama3-3-70b-instruct-v1:0",
		arnClaude: "anthropic.claude-sonnet-4-20250514-v1:0",
		arnNova:   "amazon.nova-pro-v1:0",
		arnOpaque: "abc123",
		// A foundation-model ARN carries an unscoped id in the same place.
		"arn:aws:bedrock:us-east-1::foundation-model/meta.llama3-3-70b-instruct-v1:0": "meta.llama3-3-70b-instruct-v1:0",
		// The two non-ARN forms still resolve to exactly what they always did.
		"meta.llama4-scout-17b-instruct-v1:0":    "meta.llama4-scout-17b-instruct-v1:0",
		"us.meta.llama4-scout-17b-instruct-v1:0": "meta.llama4-scout-17b-instruct-v1:0",
		"anthropic.claude-opus-5":                "anthropic.claude-opus-5",
		// Degenerate shapes resolve to something empty or meaningless, which is
		// a classifier's default rather than an error.
		"":  "",
		"/": "",
		"arn:aws:bedrock:us-east-1:123456789012:provisioned-model/": "",
	} {
		if got := bedrockResolveModelID(id); got != want {
			t.Errorf("bedrockResolveModelID(%q) = %q, want %q", id, got, want)
		}
	}
}

// TestBedrockARNClassifiesLikeTheIDItNames is the finding itself: an ARN and the
// id it points at are the same model, so every classifier keyed on the model id
// has to answer the same for both. The absolute answers are spelled out too, so
// two sides that agree on a wrong answer still fails.
func TestBedrockARNClassifiesLikeTheIDItNames(t *testing.T) {
	for _, tc := range []struct {
		arn, id string
		family  string
		gen     int
	}{
		{arnLlama4, "meta.llama4-scout-17b-instruct-v1:0", "llama", 4},
		{arnLlama3, "meta.llama3-3-70b-instruct-v1:0", "llama", 3},
		{arnClaude, "anthropic.claude-sonnet-4-20250514-v1:0", "claude", 0},
		{arnNova, "amazon.nova-pro-v1:0", "nova", 0},
	} {
		t.Run(tc.family, func(t *testing.T) {
			if got, want := getModelFamily(tc.arn), getModelFamily(tc.id); got != want {
				t.Errorf("getModelFamily(ARN) = %q, but the id it names is %q", got, want)
			}
			if got := getModelFamily(tc.arn); got != tc.family {
				t.Errorf("getModelFamily(%q) = %q, want %q", tc.arn, got, tc.family)
			}
			if got, want := bedrockLlamaGeneration(tc.arn), bedrockLlamaGeneration(tc.id); got != want {
				t.Errorf("bedrockLlamaGeneration(ARN) = %d, but the id it names is %d", got, want)
			}
			if got := bedrockLlamaGeneration(tc.arn); got != tc.gen {
				t.Errorf("bedrockLlamaGeneration(%q) = %d, want %d", tc.arn, got, tc.gen)
			}
			// The prompt template is the thing the generation decides, so it is
			// checked as a whole rather than inferred from the number above.
			if got, want := bedrockLlamaPrompt(tc.arn, "be terse", "hi"), bedrockLlamaPrompt(tc.id, "be terse", "hi"); got != want {
				t.Errorf("prompt for ARN\n got %q\nwant %q (what the id it names renders)", got, want)
			}
			// Thinking is resolved from the same string, and an ARN that says
			// "not a Claude" says "cannot think" -- which is the bigger half of
			// this bug, since it silently drops a caller's thinking config.
			if got, want := bedrockThinkingDimensions(tc.arn), bedrockThinkingDimensions(tc.id); got != want {
				t.Errorf("bedrockThinkingDimensions(ARN) = %06b, but the id it names is %06b", got, want)
			}
		})
	}
}

// TestBedrockLlama4ARNGetsTheLlama4TemplateOnTheWire is the same finding one
// layer out: not what the classifier answers, but what Bedrock receives.
func TestBedrockLlama4ARNGetsTheLlama4TemplateOnTheWire(t *testing.T) {
	var calls bedrockCalls
	c := bedrockStub(t, &calls)

	if _, err := c.Generate(context.Background(),
		NewBedrockModel(arnLlama4, "llama").WithSystemPrompt("be terse"), "hello"); err != nil {
		t.Fatalf("generate: %v", err)
	}
	if calls.len() != 1 {
		t.Fatalf("calls = %d, want 1", calls.len())
	}

	var body bedrockLlamaRequest
	if err := json.Unmarshal([]byte(calls.at(t, 0).body), &body); err != nil {
		t.Fatal(err)
	}
	if want := llama4Prompt("be terse", "hello"); body.Prompt != want {
		t.Errorf("prompt\n got %q\nwant %q", body.Prompt, want)
	}
	// Llama 4 does not carry the 3.x spellings in its vocabulary, so each of
	// these reaching it is a junk token where a turn boundary should be.
	for _, dead := range []string{"<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"} {
		if strings.Contains(body.Prompt, dead) {
			t.Errorf("a Llama 4 ARN was sent the 3.x marker %q", dead)
		}
	}
}

// TestBedrockClaudeARNStillSpeaksItsThinkingDialect covers the wider half of the
// blind spot: the family routing for an ARN was a loud error, but the thinking
// era was a silent "this model cannot think", which drops a configured budget
// without sending anything.
func TestBedrockClaudeARNStillSpeaksItsThinkingDialect(t *testing.T) {
	if got, want := ModelThinkingDimensions(NewBedrockModel(arnClaude, "claude")),
		ModelThinkingDimensions(NewBedrockClaudeSonnet4()); got != want {
		t.Errorf("ModelThinkingDimensions(ARN) = %06b, want %06b -- the same model by another name", got, want)
	}

	var calls bedrockCalls
	c := bedrockStub(t, &calls)
	if _, err := c.Generate(context.Background(),
		Thinking(NewBedrockModel(arnClaude, "claude"), WithThinkingBudget(2048)), "hello"); err != nil {
		t.Fatalf("generate: %v", err)
	}

	var body struct {
		Thinking json.RawMessage `json:"thinking"`
	}
	if err := json.Unmarshal([]byte(calls.at(t, 0).body), &body); err != nil {
		t.Fatal(err)
	}
	if want := `{"type":"enabled","budget_tokens":2048}`; string(body.Thinking) != want {
		t.Errorf("thinking = %s, want %s", body.Thinking, want)
	}
}

// TestBedrockUnclassifiableARNDegradesSafely pins the other half of the
// contract: an ARN with no model id in it must land on the defaults that were
// already there, and no id shape may panic on the way.
func TestBedrockUnclassifiableARNDegradesSafely(t *testing.T) {
	if got := getModelFamily(arnOpaque); got != "unknown" {
		t.Errorf("getModelFamily(%q) = %q, want unknown: there is no model id in it to read", arnOpaque, got)
	}
	if got := bedrockLlamaGeneration(arnOpaque); got != 0 {
		t.Errorf("bedrockLlamaGeneration(%q) = %d, want 0", arnOpaque, got)
	}
	if got := bedrockThinkingDimensions(arnOpaque); got != 0 {
		t.Errorf("bedrockThinkingDimensions(%q) = %06b, want 0", arnOpaque, got)
	}
	// An unrecognised id keeps the 3.x template, which is what every Llama on
	// Bedrock but Llama 4 speaks and what this defaulted to before ARNs were
	// understood at all.
	if got, want := bedrockLlamaPrompt(arnOpaque, "", "hi"), llama3Prompt("", "hi"); got != want {
		t.Errorf("prompt for an opaque ARN\n got %q\nwant the 3.x default %q", got, want)
	}

	for _, id := range []string{
		"", "/", "//", "arn:", "arn:aws:bedrock:us-east-1:1:provisioned-model/",
		"arn:aws:bedrock:us-east-1:1:inference-profile/us.", "us.", "meta.", "/meta.llama4-scout",
	} {
		func() {
			defer func() {
				if r := recover(); r != nil {
					t.Errorf("classifying %q panicked: %v", id, r)
				}
			}()
			getModelFamily(id)
			bedrockLlamaGeneration(id)
			bedrockThinkingDimensions(id)
			bedrockLlamaPrompt(id, "be terse", "hi")
		}()
	}
}
