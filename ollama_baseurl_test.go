package lingo

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
)

// ============================================================================
// OLLAMA BASE URL
// ============================================================================
//
// Ollama is the one provider that builds its request URLs by concatenation:
// every other provider hands its base to openai-go, which resolves the path
// relative to a normalized base, or trims at its own construction site the way
// Azure does. So Ollama is the one provider where a trailing slash -- the form
// an env var or a settings field usually yields -- would otherwise reach the
// wire as a doubled path segment.

// ollamaPaths records the request path of every call an Ollama stub serves.
// It locks because httptest handles each request on its own goroutine.
type ollamaPaths struct {
	mu    sync.Mutex
	paths []string
}

func (p *ollamaPaths) add(path string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.paths = append(p.paths, path)
}

func (p *ollamaPaths) all() []string {
	p.mu.Lock()
	defer p.mu.Unlock()
	return append([]string(nil), p.paths...)
}

// ollamaPathStub answers /api/chat and /api/tags, and records the path it was
// actually asked for -- including a path it does not serve, which is the whole
// point: a stub that answered everything would hide the bug.
func ollamaPathStub(t *testing.T, seen *ollamaPaths) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.ReadAll(r.Body)
		seen.add(r.URL.Path)

		switch r.URL.Path {
		case "/api/chat":
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{
				"model":"qwen3","created_at":"2026-08-17T00:00:00Z",
				"message":{"role":"assistant","content":"hi there"},
				"done":true,"done_reason":"stop",
				"prompt_eval_count":11,"eval_count":7}`)
		case "/api/tags":
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{"models":[]}`)
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(srv.Close)
	return srv
}

// TestOllamaBaseURLIsNormalized is the regression: newOllamaClient stored the
// configured base verbatim, so "http://host:11434/" produced "//api/chat".
// Ollama's router is gin.Default(), which leaves RedirectFixedPath off and so
// does not collapse the doubled segment -- every generation would 404.
//
// Trimming happens before the default is applied, which is what makes a base of
// "/" mean "unset" rather than "", so the last row is part of the fix too.
func TestOllamaBaseURLIsNormalized(t *testing.T) {
	tests := []struct {
		name string
		base string
		want string
	}{
		{"unset", "", "http://localhost:11434"},
		{"plain host", "http://ollama.internal:11434", "http://ollama.internal:11434"},
		{"trailing slash", "http://ollama.internal:11434/", "http://ollama.internal:11434"},
		{"several trailing slashes", "http://ollama.internal:11434///", "http://ollama.internal:11434"},
		{"path with trailing slash", "http://gw.internal/ollama/", "http://gw.internal/ollama"},
		{"slash only", "/", "http://localhost:11434"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c, err := newOllamaClient(&OllamaConfig{BaseURL: tt.base}, &NopLogger{})
			if err != nil {
				t.Fatalf("newOllamaClient: %v", err)
			}
			if c.baseURL != tt.want {
				t.Errorf("baseURL = %q, want %q", c.baseURL, tt.want)
			}
		})
	}
}

// TestOllamaTrailingSlashReachesTheSamePaths is the same regression read off
// the wire, where it actually bites: net/http preserves a doubled slash in the
// request-URI, so both call sites -- Generate's /api/chat and Health's
// /api/tags -- have to be checked against a server that serves only the real
// paths.
func TestOllamaTrailingSlashReachesTheSamePaths(t *testing.T) {
	for _, suffix := range []string{"", "/"} {
		name := "no trailing slash"
		if suffix != "" {
			name = "trailing slash"
		}
		t.Run(name, func(t *testing.T) {
			var seen ollamaPaths
			srv := ollamaPathStub(t, &seen)

			g, err := New([]ProviderConfig{&OllamaConfig{BaseURL: srv.URL + suffix}})
			if err != nil {
				t.Fatalf("gateway: %v", err)
			}
			defer g.Close()

			resp, err := g.Generate(context.Background(), NewQwen3(), "hello")
			if err != nil {
				t.Fatalf("generate: %v", err)
			}
			if resp.Text != "hi there" {
				t.Errorf("text = %q", resp.Text)
			}
			if err := g.Health(context.Background(), ProviderOllama); err != nil {
				t.Fatalf("health: %v", err)
			}

			want := []string{"/api/chat", "/api/tags"}
			got := seen.all()
			if len(got) != len(want) {
				t.Fatalf("paths = %q, want %q", got, want)
			}
			for i := range want {
				if got[i] != want[i] {
					t.Errorf("path %d = %q, want %q", i, got[i], want[i])
				}
			}
		})
	}
}
