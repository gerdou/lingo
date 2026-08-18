package lingo

import (
	"context"
	"strings"
	"testing"
)

// ============================================================================
// THE ESCAPE HATCH'S MODEL FAMILY
// ============================================================================
//
// NewBedrockModel takes a family beside the model id, and an empty one used to
// be a dead end. Generate read BedrockModel.modelFamily verbatim and never fell
// back to the classifier, so
//
//	NewBedrockModel("us.anthropic.claude-opus-5", "")
//
// came back "unsupported model family: " for an id that classifies perfectly.
// The id is the thing that knows -- a base id, a cross-region inference profile
// id and an inference-profile ARN all reduce to the same vendor prefix -- so an
// unset family now means "work it out from the id".
//
// The two halves of the contract are tested together, because either alone is
// the wrong fix: a declared family must still be forwarded verbatim, and an id
// that classifies to nothing must still be a loud error rather than a request
// built by whichever builder happened to be first.

// bedrockBuilderFor names the family builder that produced one recorded
// request, read from the shape of the body rather than from the model id, so
// the assertion is about what Bedrock received.
func bedrockBuilderFor(t *testing.T, call bedrockCall) string {
	t.Helper()
	switch {
	case strings.HasSuffix(call.path, "/converse"):
		return "nova"
	case strings.Contains(call.body, `"anthropic_version"`):
		return "claude"
	case strings.Contains(call.body, `"inputText"`):
		return "titan"
	case strings.Contains(call.body, `"max_gen_len"`):
		return "llama"
	case strings.Contains(call.body, `"prompt"`):
		return "mistral"
	default:
		t.Fatalf("unrecognised request body: %s", call.body)
		return ""
	}
}

func TestBedrockEmptyFamilyClassifiesTheModelID(t *testing.T) {
	for _, tc := range []struct {
		name    string
		modelID string
		family  string
	}{
		{"base id", "anthropic.claude-opus-5", "claude"},
		{"cross-region profile id", "us.anthropic.claude-opus-5", "claude"},
		{"inference-profile ARN", arnClaude, "claude"},
		{"titan", "amazon.titan-text-express-v1", "titan"},
		{"llama", "meta.llama3-3-70b-instruct-v1:0", "llama"},
		{"mistral", "mistral.mistral-large-2402-v1:0", "mistral"},
		// Nova is the family that leaves InvokeModel entirely, so classifying
		// it wrong is a different URL and not just a different body.
		{"nova", "us.amazon.nova-pro-v1:0", "nova"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var calls bedrockCalls
			c := bedrockStub(t, &calls)

			if _, err := c.Generate(context.Background(),
				NewBedrockModel(tc.modelID, ""), "hello"); err != nil {
				t.Fatalf("generate with an unset family: %v", err)
			}
			if calls.len() != 1 {
				t.Fatalf("calls = %d, want 1", calls.len())
			}
			if got := bedrockBuilderFor(t, calls.at(t, 0)); got != tc.family {
				t.Errorf("%q with an unset family was built by the %s builder, want %s",
					tc.modelID, got, tc.family)
			}
		})
	}
}

// A declared family is a caller telling lingo something the id may not say, and
// it has to keep winning: the fallback is for the empty string only. Titan is
// asked for on a Nova id here because getting it wrong is visible as a whole
// different API, not a subtly different body.
func TestBedrockDeclaredFamilyStillWinsOverTheModelID(t *testing.T) {
	var calls bedrockCalls
	c := bedrockStub(t, &calls)

	if _, err := c.Generate(context.Background(),
		NewBedrockModel("amazon.nova-pro-v1:0", "titan"), "hello"); err != nil {
		t.Fatalf("generate: %v", err)
	}
	if got := bedrockBuilderFor(t, calls.at(t, 0)); got != "titan" {
		t.Errorf("a declared family of %q was overruled: the request was built by the %s builder", "titan", got)
	}

	// And clearing it hands the decision back to the id.
	var cleared bedrockCalls
	c2 := bedrockStub(t, &cleared)
	if _, err := c2.Generate(context.Background(),
		NewBedrockModel("amazon.nova-pro-v1:0", "titan").WithModelFamily(""), "hello"); err != nil {
		t.Fatalf("generate: %v", err)
	}
	if got := bedrockBuilderFor(t, cleared.at(t, 0)); got != "nova" {
		t.Errorf("WithModelFamily(\"\") left the request on the %s builder, want the id's own answer, nova", got)
	}
}

// The other half: an unset family may not turn an unknown model into a silent
// success. A provisioned-throughput ARN names an opaque id -- no vendor, no
// generation, nothing to classify -- and there is no safe guess to make, so it
// stays the loud error it always was.
func TestBedrockUnclassifiableIDWithNoFamilyIsStillAnError(t *testing.T) {
	var calls bedrockCalls
	c := bedrockStub(t, &calls)

	for _, modelID := range []string{arnOpaque, "", "not-a-bedrock-id"} {
		_, err := c.Generate(context.Background(), NewBedrockModel(modelID, ""), "hello")
		if err == nil {
			t.Errorf("Generate(%q with no family) = nil error, want a refusal", modelID)
			continue
		}
		if !strings.Contains(err.Error(), "unsupported model family") {
			t.Errorf("Generate(%q) = %v, want the unsupported-family error", modelID, err)
		}
		// The id is worth naming: it is what tells "I typed a family lingo does
		// not have" apart from "my ARN carries nothing to classify".
		if !strings.Contains(err.Error(), modelID) {
			t.Errorf("Generate(%q) = %v, which does not name the model", modelID, err)
		}
	}
	if calls.len() != 0 {
		t.Errorf("%d request(s) reached Bedrock for a model no builder claims", calls.len())
	}
}
