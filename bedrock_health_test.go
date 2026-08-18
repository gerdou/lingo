package lingo

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// ============================================================================
// HEALTH PROBE
// ============================================================================
//
// Health is asked one question -- can this client reach Bedrock with usable
// credentials -- and it used to answer a narrower one, by invoking
// amazon.titan-text-lite-v1 as "most widely available". Titan Text is past
// end-of-life on Bedrock, so on an account without Titan access the check
// failed for a reason that had nothing to do with health, and the same trap is
// waiting for any other model the library picks on the caller's behalf.
//
// So the default probe names no model at all. ListAsyncInvokes is documented as
// "GET /async-invoke" with no request body and no model parameter, which is the
// cheapest thing the runtime API offers -- the same reasoning cohere.go,
// ollama.go and oaicompat.go already use when the provider has a call cheaper
// than a generation. A caller who does want a specific model proved sets
// BedrockConfig.HealthCheckModel.

// bedrockHealthStub answers the async-invoke listing and the InvokeModel
// endpoints, recording every request path. bedrockStub cannot be reused: it has
// no arm for /async-invoke and would answer the probe with a Titan body.
func bedrockHealthStub(t *testing.T, calls *bedrockCalls, status int) *bedrockClient {
	t.Helper()
	return bedrockClientFor(t, httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		calls.add(bedrockCall{path: r.URL.Path, body: string(raw)})

		w.Header().Set("Content-Type", "application/json")
		if status != http.StatusOK {
			w.Header().Set("X-Amzn-Errortype", "AccessDeniedException")
			w.WriteHeader(status)
			_, _ = io.WriteString(w, `{"message":"no"}`)
			return
		}
		switch {
		case strings.HasSuffix(r.URL.Path, "/async-invoke"):
			_, _ = io.WriteString(w, `{"asyncInvokeSummaries":[]}`)
		default:
			_, _ = io.WriteString(w, `{"id":"msg_1","type":"message","role":"assistant",
				"stop_reason":"end_turn","content":[{"type":"text","text":"hi"}],
				"usage":{"input_tokens":3,"output_tokens":1}}`)
		}
	})))
}

// TestBedrockHealthInvokesNoModelByDefault is the finding: a health check that
// depends on one model's availability reports that model, not the provider.
func TestBedrockHealthInvokesNoModelByDefault(t *testing.T) {
	var calls bedrockCalls
	c := bedrockHealthStub(t, &calls, http.StatusOK)

	if err := c.Health(context.Background()); err != nil {
		t.Fatalf("Health: %v", err)
	}
	if calls.len() != 1 {
		t.Fatalf("calls = %d, want 1", calls.len())
	}

	call := calls.at(t, 0)
	if !strings.HasSuffix(call.path, "/async-invoke") {
		t.Errorf("health probe hit %q, want the async-invoke listing", call.path)
	}
	// The specifics that make it a liveness call rather than a generation.
	if strings.Contains(call.path, "/model/") || strings.Contains(call.path, "/invoke") {
		t.Errorf("health probe invoked a model: %q", call.path)
	}
	if strings.Contains(call.path, "titan") {
		t.Errorf("health probe still depends on the retired Titan models: %q", call.path)
	}
	if call.body != "" {
		t.Errorf("health probe sent a body: %q", call.body)
	}
}

// A caller who wants "is this model servable" answered can ask for it, and then
// the probe goes down the ordinary Generate path -- same family routing, same
// builders -- rather than a second, differently-shaped request.
func TestBedrockHealthGeneratesAgainstAConfiguredModel(t *testing.T) {
	var calls bedrockCalls
	c := bedrockHealthStub(t, &calls, http.StatusOK)
	c.healthModel = "us.anthropic.claude-opus-5"

	if err := c.Health(context.Background()); err != nil {
		t.Fatalf("Health: %v", err)
	}
	if calls.len() != 1 {
		t.Fatalf("calls = %d, want 1", calls.len())
	}

	call := calls.at(t, 0)
	if call.path != "/model/us.anthropic.claude-opus-5/invoke" {
		t.Errorf("health probe hit %q, want the configured model", call.path)
	}
	// Five tokens, and the Claude builder -- which is only reachable because an
	// unset family classifies the id.
	if want := `"max_tokens":5`; !strings.Contains(call.body, want) {
		t.Errorf("health body = %s, want %s", call.body, want)
	}
	if !strings.Contains(call.body, `"anthropic_version"`) {
		t.Errorf("health body = %s, want the Claude builder's body", call.body)
	}
}

// The knob is only reachable through the config, so the wiring is worth one
// line: a field the constructor drops leaves Health silently on the default.
func TestBedrockConfigCarriesTheHealthCheckModel(t *testing.T) {
	c, err := newBedrockClient(&BedrockConfig{
		Region:           "us-east-1",
		AccessKeyID:      "id",
		SecretAccessKey:  "secret",
		HealthCheckModel: "us.anthropic.claude-opus-5",
	}, &NopLogger{})
	if err != nil {
		t.Fatalf("newBedrockClient: %v", err)
	}
	if c.healthModel != "us.anthropic.claude-opus-5" {
		t.Errorf("healthModel = %q, want the configured id", c.healthModel)
	}
}

// Whatever it probes with, a refusal is still a failure and still says so.
func TestBedrockHealthReportsFailures(t *testing.T) {
	var calls bedrockCalls
	c := bedrockHealthStub(t, &calls, http.StatusForbidden)

	err := c.Health(context.Background())
	if err == nil {
		t.Fatal("Health = nil error on a 403")
	}
	if !strings.Contains(err.Error(), "bedrock health check failed") {
		t.Errorf("Health = %v, want the health-check wrapper", err)
	}
}
