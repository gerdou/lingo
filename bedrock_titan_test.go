package lingo

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"
)

// ============================================================================
// ARRAY-SHAPED RESPONSES
// ============================================================================
//
// Titan's results and Mistral's outputs are arrays and lingo returns the first
// element of each. That is correct -- AWS documents Titan's results as "an
// array of one item", and neither request type this package builds has a
// numResults or n field for anything to set -- but "correct because nobody can
// ask for more" only holds while the parser cannot quietly drop what it did
// receive. A second result used to take its token count with it, which is a
// count the caller was billed for and never saw.

func TestTitanSumsTheTokenCountsOfEveryResult(t *testing.T) {
	c := bedrockCannedStub(t, `{"inputTextTokenCount":11,"results":[
		{"outputText":"first","completionReason":"FINISH","tokenCount":7},
		{"outputText":"second","completionReason":"FINISH","tokenCount":5}]}`, nil)

	resp, err := c.Generate(context.Background(), NewBedrockTitanTextPremier(), "hello")
	if err != nil {
		t.Fatalf("generate: %v", err)
	}

	// The answer is still the first completion: GenerationResponse holds one
	// Text and concatenating alternatives would invent a document nobody wrote.
	if resp.Text != "first" {
		t.Errorf("Text = %q, want the first result", resp.Text)
	}
	// The billing is all of them.
	want := TokenUsage{PromptTokens: 11, CompletionTokens: 12, TotalTokens: 23}
	if resp.Usage != want {
		t.Errorf("usage = %+v, want %+v: every result generated is a result billed", resp.Usage, want)
	}
	// And the drop is visible rather than silent.
	if got := resp.Metadata["results"]; got != "2" {
		t.Errorf("Metadata[\"results\"] = %q, want 2: a discarded completion has to leave a trace", got)
	}
}

// The documented single-result case is untouched, metadata included: a response
// shaped the way every real one is must not grow a key.
func TestTitanSingleResultIsUnchanged(t *testing.T) {
	c := bedrockCannedStub(t,
		`{"inputTextTokenCount":11,"results":[{"outputText":"hi there","completionReason":"FINISH","tokenCount":7}]}`, nil)

	resp, err := c.Generate(context.Background(), NewBedrockTitanTextPremier(), "hello")
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	want := TokenUsage{PromptTokens: 11, CompletionTokens: 7, TotalTokens: 18}
	if resp.Usage != want {
		t.Errorf("usage = %+v, want %+v", resp.Usage, want)
	}
	if _, ok := resp.Metadata["results"]; ok {
		t.Errorf("Metadata = %v, want no results key when there is nothing to report", resp.Metadata)
	}
}

// Mistral's outputs carry no token counts at all -- the response headers cover
// the whole call however many outputs it produced -- so nothing billed can go
// missing here. Only alternative text can, and that still has to be said out
// loud rather than dropped on the floor.
func TestMistralRecordsExtraOutputs(t *testing.T) {
	c := bedrockCannedStub(t, `{"outputs":[
		{"text":"first","stop_reason":"stop"},
		{"text":"second","stop_reason":"stop"}]}`, bedrockTokenHeaders)

	resp, err := c.Generate(context.Background(), NewBedrockMistralLarge2407(), "hello")
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	if resp.Text != "first" {
		t.Errorf("Text = %q, want the first output", resp.Text)
	}
	if want := (TokenUsage{PromptTokens: 42, CompletionTokens: 7, TotalTokens: 49}); resp.Usage != want {
		t.Errorf("usage = %+v, want %+v: the headers count the whole call", resp.Usage, want)
	}
	if got := resp.Metadata["outputs"]; got != "2" {
		t.Errorf("Metadata[\"outputs\"] = %q, want 2", got)
	}
}

func TestMistralSingleOutputIsUnchanged(t *testing.T) {
	c := bedrockCannedStub(t, `{"outputs":[{"text":"hi there","stop_reason":"stop"}]}`, nil)

	resp, err := c.Generate(context.Background(), NewBedrockMistral7B(), "hello")
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	if _, ok := resp.Metadata["outputs"]; ok {
		t.Errorf("Metadata = %v, want no outputs key when there is nothing to report", resp.Metadata)
	}
}

// ============================================================================
// TITAN HAS NO SYSTEM PROMPT
// ============================================================================
//
// Titan's whole request body is inputText plus textGenerationConfig -- no
// system field, no instructions field, no roles -- so WithSystemPrompt on a
// Titan model can only concatenate. That is not a defect to fix, since there is
// nothing on this API to fix it with, but it is a difference from every other
// family lingo serves, and a caller who assumes a system prompt outranks the
// user prompt is assuming something Titan never promised.
//
// The two tests below are what stop that limitation from being lost: the first
// pins the wire shape so the concatenation cannot change into something that
// merely looks like a boundary, and the second pins the fact that it is written
// down where a caller will meet it.

func TestTitanSystemPromptIsPlainConcatenation(t *testing.T) {
	var calls bedrockCalls
	c := bedrockStub(t, &calls)

	if _, err := c.Generate(context.Background(),
		NewBedrockTitanTextPremier().WithSystemPrompt("be terse"), "hello"); err != nil {
		t.Fatalf("generate: %v", err)
	}

	var body map[string]any
	if err := json.Unmarshal([]byte(calls.at(t, 0).body), &body); err != nil {
		t.Fatal(err)
	}
	if got := body["inputText"]; got != "be terse\n\nhello" {
		t.Errorf("inputText = %q, want the system prompt, a blank line and the prompt", got)
	}
	// There is no second place for it to be, and inventing one would be
	// inventing a field Bedrock does not model.
	for _, key := range []string{"system", "systemPrompt", "instructions", "messages"} {
		if _, ok := body[key]; ok {
			t.Errorf("request carries a %q field: Titan's body is inputText and textGenerationConfig only", key)
		}
	}
	if len(body) != 2 {
		t.Errorf("request has keys %v, want inputText and textGenerationConfig only", body)
	}
}

// A limitation that exists only in the maintainer's head is one callers meet in
// production, so it is documented in the builder and in the README, and this is
// what keeps both from being deleted as noise.
func TestTitanSystemPromptLimitationIsDocumented(t *testing.T) {
	for _, tc := range []struct {
		file  string
		wants []string
	}{{
		file: "bedrock.go",
		wants: []string{
			// The doc comment on buildTitanRequest.
			"Titan has no system prompt",
			"the blank line is the entire boundary",
		},
	}, {
		file: "README.md",
		wants: []string{
			"Titan has no system prompt",
			"There is no `system` field in it",
		},
	}} {
		raw, err := os.ReadFile(tc.file)
		if err != nil {
			t.Fatal(err)
		}
		for _, want := range tc.wants {
			if !strings.Contains(string(raw), want) {
				t.Errorf("%s no longer documents Titan's system-prompt limitation: %q is missing", tc.file, want)
			}
		}
	}
}
