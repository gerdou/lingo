package lingo

import (
	"path/filepath"
	"strings"
	"testing"
)

// ============================================================================
// ANTHROPIC ON VERTEX AI: CREDENTIAL FAILURE
// ============================================================================
//
// New documents an error when a provider fails to initialize. The Vertex
// helpers in anthropic-sdk-go report every credential failure by panicking
// (vertex/vertex.go:45-53 and :91-93 at v1.63.1) and offer no variant that
// returns, so an absent or rotated credentials file would otherwise take the
// caller's process down at construction rather than fail one provider.

// missingCredentials points GOOGLE_APPLICATION_CREDENTIALS at a file that does
// not exist. The variable takes precedence over every other ADC source, so the
// lookup fails the same way on a developer laptop, in CI and on a GCE host.
func missingCredentials(t *testing.T) {
	t.Helper()
	t.Setenv("GOOGLE_APPLICATION_CREDENTIALS", filepath.Join(t.TempDir(), "rotated-away.json"))
}

// TestAnthropicVertexMissingCredentialsReturnsError is the regression: the
// gateway must come back with an error in hand, not unwind the stack.
func TestAnthropicVertexMissingCredentialsReturnsError(t *testing.T) {
	missingCredentials(t)

	g, err := New([]ProviderConfig{&AnthropicConfig{
		Vertex: &AnthropicVertexConfig{ProjectID: "p", Region: "us-east5"},
	}})
	if err == nil {
		g.Close()
		t.Fatal("New succeeded with no application default credentials")
	}
	if !strings.Contains(err.Error(), "Vertex AI credentials") {
		t.Errorf("err = %q, want it to name the Vertex credential failure", err.Error())
	}
	if !strings.Contains(err.Error(), "default credentials") {
		t.Errorf("err = %q, want it to carry the SDK's own reason", err.Error())
	}
}

// TestAnthropicVertexAuthRecoversSDKPanic pins the guarantee at the point that
// depends on it, for a panic lingo's own argument checks do not pre-empt: the
// SDK panics on an empty region before it ever looks at credentials, and that
// too has to come back as a value.
func TestAnthropicVertexAuthRecoversSDKPanic(t *testing.T) {
	opt, err := anthropicVertexAuth(t.Context(), &AnthropicVertexConfig{ProjectID: "p"})
	if err == nil {
		t.Fatal("anthropicVertexAuth returned no error for an empty region")
	}
	if opt != nil {
		t.Error("anthropicVertexAuth returned an option beside its error")
	}
	if !strings.Contains(err.Error(), "region must be provided") {
		t.Errorf("err = %q, want it to carry the SDK's panic value", err.Error())
	}
}

// TestAnthropicAPIKeyUnaffectedByMissingADC keeps the guard narrow: the direct
// API path never asks Google for anything, so a broken ADC environment must
// leave it alone.
func TestAnthropicAPIKeyUnaffectedByMissingADC(t *testing.T) {
	missingCredentials(t)

	g, err := New([]ProviderConfig{&AnthropicConfig{APIKey: "k"}})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer g.Close()
}
