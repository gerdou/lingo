package lingo

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// ============================================================================
// GOOGLE THINKING: GOLDEN REQUESTS
// ============================================================================
//
// Google is the one provider where thinking is entirely new: googleOptions
// carried no thinking field before this change and no Gemini had a thinking
// setter, so there is no legacy vocabulary to preserve. What there is instead is
// a generation split with no safe default -- 1.5 and 2.0 reject every shape of
// thinkingConfig, 2.5 takes a token budget, 3.x takes a level, and asking for
// both a level and a budget in one request is a hard error.
//
// The table below is the whole contract: every untouched model must send no
// thinkingConfig at all, and every configured one must send exactly the one
// field its generation speaks.

// geminiThinkingResponse is the default stub body: an answer, a thought part
// ahead of it, a thought signature, and a usage block whose totalTokenCount
// already includes the thoughts while candidatesTokenCount does not.
const geminiThinkingResponse = `{
	"candidates":[{"finishReason":"STOP","content":{"role":"model","parts":[
		{"text":"weighing the options","thought":true,"thoughtSignature":"c2ln"},
		{"text":"the answer"},
		{"text":" continues"}]}}],
	"usageMetadata":{"promptTokenCount":100,"candidatesTokenCount":7,
		"totalTokenCount":150,"thoughtsTokenCount":43}}`

// geminiPlainResponse is the same shape with no thinking at all, for the golden
// request rows where only the outbound body matters.
const geminiPlainResponse = `{
	"candidates":[{"finishReason":"STOP","content":{"role":"model","parts":[
		{"text":"hi there"}]}}],
	"usageMetadata":{"promptTokenCount":11,"candidatesTokenCount":7,"totalTokenCount":18}}`

// geminiThinkingStub serves a caller-supplied generateContent response, records
// the request, and points the pinned genai SDK at itself through
// GOOGLE_GEMINI_BASE_URL, which the SDK resolves when the client is built.
func geminiThinkingStub(t *testing.T, c *capture, body string) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		c.record(r, raw)

		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, body)
	}))
	t.Setenv("GOOGLE_GEMINI_BASE_URL", srv.URL)
	return srv
}

// geminiThinkingWire runs one Generate against the stub and returns the
// generationConfig.thinkingConfig the SDK put on the wire, nil when the request
// carried none.
func geminiThinkingWire(t *testing.T, m Model) any {
	t.Helper()
	var c capture
	srv := geminiThinkingStub(t, &c, geminiPlainResponse)
	defer srv.Close()
	generate(t, &GoogleConfig{APIKey: "k"}, m)

	gc, ok := c.body["generationConfig"].(map[string]any)
	if !ok {
		return nil
	}
	return gc["thinkingConfig"]
}

// wantThinkingConfig compares the wire config against a JSON literal, treating a
// "" want as "no thinkingConfig may be sent".
func wantThinkingConfig(t *testing.T, got any, want string) {
	t.Helper()
	if want == "" {
		if got != nil {
			raw, _ := json.Marshal(got)
			t.Errorf("thinkingConfig = %s, want the key to be absent", raw)
		}
		return
	}
	if got == nil {
		t.Errorf("thinkingConfig is absent, want %s", want)
		return
	}
	raw, _ := json.Marshal(got)
	var wantAny any
	if err := json.Unmarshal([]byte(want), &wantAny); err != nil {
		t.Fatalf("bad want literal %q: %v", want, err)
	}
	wantRaw, _ := json.Marshal(wantAny)
	if string(raw) != string(wantRaw) {
		t.Errorf("thinkingConfig = %s, want %s", raw, wantRaw)
	}
}

func TestGoogleThinkingGoldenRequests(t *testing.T) {
	tests := []struct {
		name  string
		model Model
		want  string // "" means no thinkingConfig may be sent
	}{
		// THE OPT-IN GUARANTEE. Every Gemini now carries a ThinkingOptions, and
		// an untouched one must leave the request exactly as it was before the
		// field existed -- on every generation, including the two that would
		// answer 400 to a config of any shape.
		{"1.5 pro untouched", NewGemini15Pro(), ""},
		{"2.0 flash untouched", NewGemini20Flash(), ""},
		{"2.0 flash thinking untouched", NewGemini20FlashThinking(), ""},
		{"2.5 pro untouched", NewGemini25Pro(), ""},
		{"2.5 flash untouched", NewGemini25Flash(), ""},
		{"2.5 flash-lite untouched", NewGemini25FlashLite(), ""},
		{"3 pro untouched", NewGemini3Pro(), ""},
		{"3.6 flash untouched", NewGemini36Flash(), ""},
		{"generic untouched", NewGoogleModel("gemini-3.1-pro-preview"), ""},

		// 2.5: depth is a token budget, and -1 is how "you decide" is spelled.
		// Flash-Lite defaults to thinking off, so a bare enable has real work to
		// do there; Flash already reasons dynamically and gets the same field.
		{"2.5 flash enabled", Thinking(NewGemini25Flash()), `{"thinkingBudget":-1}`},
		{"2.5 flash-lite enabled", Thinking(NewGemini25FlashLite()), `{"thinkingBudget":-1}`},
		{"2.5 flash dynamic", Thinking(NewGemini25Flash(), WithDynamicThinking()), `{"thinkingBudget":-1}`},
		{"2.5 flash budget", Thinking(NewGemini25Flash(), WithThinkingBudget(8000)), `{"thinkingBudget":8000}`},
		{"2.5 pro dynamic", Thinking(NewGemini25Pro(), WithDynamicThinking()), `{"thinkingBudget":-1}`},
		{"2.5 pro budget", Thinking(NewGemini25Pro(), WithThinkingBudget(30000)), `{"thinkingBudget":30000}`},

		// 2.5 Pro reasons unconditionally: it has no toggle, so a bare enable is
		// a no-op rather than a field, and it rejects a budget of 0.
		{"2.5 pro enabled", Thinking(NewGemini25Pro()), ""},
		{"2.5 pro disabled", NoThinking(NewGemini25Pro()), ""},

		// The off switch that does exist is a budget of 0.
		{"2.5 flash disabled", NoThinking(NewGemini25Flash()), `{"thinkingBudget":0}`},
		{"2.5 flash-lite disabled", NoThinking(NewGemini25FlashLite()), `{"thinkingBudget":0}`},

		// Budgets outside a model's window are clamped rather than forwarded to
		// be rejected: the portable surface promises adaptation, and no Google
		// setter ever pinned a value that would have to be preserved verbatim.
		{"2.5 flash budget above the ceiling", Thinking(NewGemini25Flash(), WithThinkingBudget(99999)),
			`{"thinkingBudget":24576}`},
		{"2.5 flash-lite budget below the floor", Thinking(NewGemini25FlashLite(), WithThinkingBudget(100)),
			`{"thinkingBudget":512}`},
		{"2.5 pro budget below the floor", Thinking(NewGemini25Pro(), WithThinkingBudget(10)),
			`{"thinkingBudget":128}`},

		// 3.x: depth is a level, and never a budget -- sending both is a hard
		// error, so a caller who asks in tokens gets the nearest level instead.
		{"3 pro high", Thinking(NewGemini3Pro(), WithThinkingEffort(ThinkingEffortHigh)),
			`{"thinkingLevel":"HIGH"}`},
		{"3 flash minimal", Thinking(NewGemini3Flash(), WithThinkingEffort(ThinkingEffortMinimal)),
			`{"thinkingLevel":"MINIMAL"}`},
		{"3.6 flash medium", Thinking(NewGemini36Flash(), WithThinkingEffort(ThinkingEffortMedium)),
			`{"thinkingLevel":"MEDIUM"}`},
		{"3 pro above the ladder", Thinking(NewGemini3Pro(), WithThinkingEffort(ThinkingEffortMax)),
			`{"thinkingLevel":"HIGH"}`},
		{"3 pro budget becomes a level", Thinking(NewGemini31Pro(), WithThinkingBudget(30000)),
			`{"thinkingLevel":"HIGH"}`},
		{"2.5 flash effort becomes a budget", Thinking(NewGemini25Flash(), WithThinkingEffort(ThinkingEffortLow)),
			`{"thinkingBudget":2949}`},

		// 3.x cannot be switched off and has no dynamic setting; both requests
		// are dropped rather than translated into something else.
		{"3 pro disabled", NoThinking(NewGemini3Pro()), ""},
		{"3 pro dynamic", Thinking(NewGemini3Pro(), WithDynamicThinking()), ""},
		{"3 pro enabled", Thinking(NewGemini3Pro()), ""},

		// Trace visibility is orthogonal to depth. Withholding it is already the
		// API default, so lingo has nothing to send for it.
		{"3 pro trace included", Thinking(NewGemini3Pro(), WithThinkingTrace(ThinkingTraceInclude)),
			`{"includeThoughts":true}`},
		{"3 pro trace omitted", Thinking(NewGemini3Pro(), WithThinkingTrace(ThinkingTraceOmit)), ""},
		{"2.5 flash budget and trace",
			Thinking(NewGemini25Flash(), WithThinkingBudget(8000), WithThinkingTrace(ThinkingTraceInclude)),
			`{"thinkingBudget":8000,"includeThoughts":true}`},

		// Generations with no thinking field of any kind: every knob is a silent
		// no-op, because sending one is a 400 rather than a wasted parameter.
		{"1.5 pro effort", Thinking(NewGemini15Pro(), WithThinkingEffort(ThinkingEffortHigh)), ""},
		{"2.0 flash budget", Thinking(NewGemini20Flash(), WithThinkingBudget(4096)), ""},
		{"2.0 flash thinking enabled", Thinking(NewGemini20FlashThinking()), ""},
		{"2.0 flash disabled", NoThinking(NewGemini20Flash()), ""},

		// The generic model resolves its dialect from the id it was handed, so a
		// preview this build has never heard of still reaches the right knob --
		// and an id from a generation this build cannot place reaches none.
		{"generic 3.x preview", Thinking(NewGoogleModel("gemini-3.7-pro-preview"), WithThinkingEffort(ThinkingEffortMedium)),
			`{"thinkingLevel":"MEDIUM"}`},
		{"generic 2.5 alias", Thinking(NewGoogleModel("gemini-2.5-flash-preview-05-20"), WithThinkingBudget(4096)),
			`{"thinkingBudget":4096}`},
		{"generic unknown generation", Thinking(NewGoogleModel("gemini-4-pro"), WithThinkingEffort(ThinkingEffortHigh)), ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			wantThinkingConfig(t, geminiThinkingWire(t, tt.model), tt.want)
		})
	}
}

// ============================================================================
// GOOGLE THINKING: CAPABILITIES
// ============================================================================

func TestGoogleThinkingDimensionsPerGeneration(t *testing.T) {
	const report = ThinkingCanReportTokens | ThinkingCanReportTrace
	tests := []struct {
		model Model
		want  ThinkingDimension
	}{
		// 1.5 and 2.0 model nothing: no knobs, and no counters either.
		{NewGemini15Pro(), 0},
		{NewGemini15Flash(), 0},
		{NewGemini15Flash8b(), 0},
		{NewGemini20Flash(), 0},
		{NewGemini20FlashLite(), 0},
		{NewGemini20FlashExp(), 0},
		{NewGemini20ProExp(), 0},
		{NewGemini20FlashThinking(), 0},

		// 2.5 budgets in tokens. Pro is the one that cannot be switched off.
		{NewGemini25Pro(), ThinkingCanSetBudget | ThinkingCanHideTrace | report},
		{NewGemini25Flash(), ThinkingCanToggle | ThinkingCanSetBudget | ThinkingCanHideTrace | report},
		{NewGemini25FlashLite(), ThinkingCanToggle | ThinkingCanSetBudget | ThinkingCanHideTrace | report},

		// 3.x takes a level, and cannot be switched off at all.
		{NewGemini3Pro(), ThinkingCanSetEffort | ThinkingCanHideTrace | report},
		{NewGemini3Flash(), ThinkingCanSetEffort | ThinkingCanHideTrace | report},
		{NewGemini31Pro(), ThinkingCanSetEffort | ThinkingCanHideTrace | report},
		{NewGemini35Flash(), ThinkingCanSetEffort | ThinkingCanHideTrace | report},
		{NewGemini31FlashLite(), ThinkingCanSetEffort | ThinkingCanHideTrace | report},
		{NewGemini36Flash(), ThinkingCanSetEffort | ThinkingCanHideTrace | report},
		{NewGemini35FlashLite(), ThinkingCanSetEffort | ThinkingCanHideTrace | report},

		// The generic type answers from its id, not from its Go type.
		{NewGoogleModel("gemini-2.5-flash"), ThinkingCanToggle | ThinkingCanSetBudget | ThinkingCanHideTrace | report},
		{NewGoogleModel("gemini-3.1-flash-lite"), ThinkingCanSetEffort | ThinkingCanHideTrace | report},
		{NewGoogleModel("gemini-1.5-pro"), 0},
		{NewGoogleModel("gemini-4-pro"), 0},
		{NewGoogleModel(""), 0},
	}

	for _, tt := range tests {
		if got := ModelThinkingDimensions(tt.model); got != tt.want {
			t.Errorf("ModelThinkingDimensions(%s) = %06b, want %06b", tt.model.ModelName(), got, tt.want)
		}
	}
}

// A Vertex AI deployment addresses the same model through a resource path. The
// dialect lookup has to see through it, or every Vertex caller silently loses
// the whole thinking surface.
func TestGoogleThinkingDimensionsSeeThroughAVertexResourcePath(t *testing.T) {
	tests := map[string]ThinkingDimension{
		"gemini-3-pro-preview": ThinkingCanSetEffort | ThinkingCanHideTrace |
			ThinkingCanReportTokens | ThinkingCanReportTrace,
		"publishers/google/models/gemini-3-pro-preview": ThinkingCanSetEffort | ThinkingCanHideTrace |
			ThinkingCanReportTokens | ThinkingCanReportTrace,
		"projects/p/locations/l/publishers/google/models/gemini-2.5-pro": ThinkingCanSetBudget | ThinkingCanHideTrace |
			ThinkingCanReportTokens | ThinkingCanReportTrace,
	}
	for id, want := range tests {
		if got := googleThinkingDimensions(id); got != want {
			t.Errorf("googleThinkingDimensions(%q) = %06b, want %06b", id, got, want)
		}
	}
}

// The capability answer is resolved from the model id, so a zero-value literal
// -- constructible outside the package, since every Gemini type is a struct with
// one embedded field -- keeps the capabilities its constructor would have given
// it. A stored caps field would have silently answered "nothing" here.
func TestZeroValueGeminiKeepsItsCapabilities(t *testing.T) {
	if got, want := ModelThinkingDimensions(&Gemini3Pro{}), ModelThinkingDimensions(NewGemini3Pro()); got != want {
		t.Errorf("&Gemini3Pro{} dimensions = %06b, want %06b", got, want)
	}
	if got, want := ModelThinkingDimensions(&Gemini25Flash{}), ModelThinkingDimensions(NewGemini25Flash()); got != want {
		t.Errorf("&Gemini25Flash{} dimensions = %06b, want %06b", got, want)
	}
}

// A model version override changes the id, and the id is what decides the
// dialect -- so pinning a 2.5 Flash to a dated preview must not lose its budget
// knob, and a Gemini 3 Pro pinned to a 3.x preview must keep its level knob.
func TestGoogleThinkingFollowsTheVersionOverride(t *testing.T) {
	m := NewGemini25Flash().WithVersion("gemini-2.5-flash-preview-05-20")
	want := ThinkingCanToggle | ThinkingCanSetBudget | ThinkingCanHideTrace |
		ThinkingCanReportTokens | ThinkingCanReportTrace
	if got := ModelThinkingDimensions(m); got != want {
		t.Errorf("versioned 2.5 flash dimensions = %06b, want %06b", got, want)
	}
}

// Every Gemini carries thinking configuration, including the generations that
// send none of it. That is the documented separation: the accessor says where
// configuration can be stored, thinkingDimensions says what reaches the wire.
func TestEveryGeminiSatisfiesThinkingModel(t *testing.T) {
	models := []Model{
		NewGemini25Pro(), NewGemini25Flash(), NewGemini25FlashLite(),
		NewGemini20Flash(), NewGemini20FlashLite(), NewGemini20FlashExp(),
		NewGemini20FlashThinking(), NewGemini20ProExp(),
		NewGemini15Pro(), NewGemini15Flash(), NewGemini15Flash8b(),
		NewGemini3Pro(), NewGemini3Flash(), NewGemini31Pro(), NewGemini35Flash(),
		NewGemini31FlashLite(), NewGemini36Flash(), NewGemini35FlashLite(),
		NewGoogleModel("gemini-3.1-pro-preview"),
	}
	for _, m := range models {
		to := modelThinkingOptions(m)
		if to == nil {
			t.Errorf("%T does not satisfy ThinkingModel", m)
			continue
		}
		if to.Mode() != ThinkingModeDefault || to.Effort() != "" || to.Budget() != 0 {
			t.Errorf("%T starts with a non-default thinking configuration: %+v", m, to)
		}
	}
}

// ============================================================================
// GOOGLE THINKING: THE READ SIDE
// ============================================================================

// The bug this change had to fix before it could wire includeThoughts up: a
// thought part carries its text in the same field as the answer, and the old
// loop tested only Text. Turning the trace on would have concatenated the
// model's reasoning into GenerationResponse.Text and returned it as the answer.
func TestGoogleThoughtPartsNeverLeakIntoTheAnswer(t *testing.T) {
	var c capture
	srv := geminiThinkingStub(t, &c, geminiThinkingResponse)
	defer srv.Close()

	resp := generate(t, &GoogleConfig{APIKey: "k"},
		Thinking(NewGemini3Pro(), WithThinkingTrace(ThinkingTraceInclude)))

	if resp.Text != "the answer continues" {
		t.Errorf("Text = %q, want the answer parts only, concatenated in order", resp.Text)
	}
	if strings.Contains(resp.Text, "weighing the options") {
		t.Errorf("thought text leaked into the answer: %q", resp.Text)
	}
	if resp.Thinking != "weighing the options" {
		t.Errorf("Thinking = %q, want the thought part", resp.Thinking)
	}
	// The signature is opaque bytes meant to be replayed, so it rides in
	// metadata base64-encoded rather than in a typed field.
	if got := resp.Metadata["thinking_signature"]; got != "c2ln" {
		t.Errorf("Metadata[thinking_signature] = %q, want %q", got, "c2ln")
	}
}

// Google is the only provider in the library that counts its thinking tokens
// outside the completion total. lingo normalizes that away so CompletionTokens
// means the same thing here as everywhere else -- and must not inflate a total
// that already covered them.
func TestGoogleThoughtsTokensAreFoldedIntoTheCompletionTotal(t *testing.T) {
	var c capture
	srv := geminiThinkingStub(t, &c, geminiThinkingResponse)
	defer srv.Close()

	resp := generate(t, &GoogleConfig{APIKey: "k"}, Thinking(NewGemini3Pro()))

	if resp.Usage.ThinkingTokens != 43 {
		t.Errorf("ThinkingTokens = %d, want 43", resp.Usage.ThinkingTokens)
	}
	// candidatesTokenCount 7 excludes the 43 thoughts, so the completion total
	// has to absorb them.
	if resp.Usage.CompletionTokens != 50 {
		t.Errorf("CompletionTokens = %d, want 50 (7 answer + 43 thoughts)", resp.Usage.CompletionTokens)
	}
	// totalTokenCount already covered them, so it must be reported unchanged.
	if resp.Usage.TotalTokens != 150 {
		t.Errorf("TotalTokens = %d, want 150 unchanged: Gemini's total already covers the thoughts", resp.Usage.TotalTokens)
	}
	if resp.Usage.PromptTokens != 100 {
		t.Errorf("PromptTokens = %d, want 100", resp.Usage.PromptTokens)
	}
	if resp.Usage.AnswerTokens() != 7 {
		t.Errorf("AnswerTokens() = %d, want 7", resp.Usage.AnswerTokens())
	}
}

// Reporting is unconditional: a Gemini that reasons on its own terms has its
// counters and its trace read back whether or not the caller opted in.
func TestGoogleReportsThinkingWithoutBeingAsked(t *testing.T) {
	var c capture
	srv := geminiThinkingStub(t, &c, geminiThinkingResponse)
	defer srv.Close()

	resp := generate(t, &GoogleConfig{APIKey: "k"}, NewGemini3Pro())

	if gc, ok := c.body["generationConfig"].(map[string]any); ok {
		if tc, ok := gc["thinkingConfig"]; ok {
			t.Errorf("thinkingConfig = %v, want none on an untouched model", tc)
		}
	}
	if resp.Thinking != "weighing the options" || resp.Usage.ThinkingTokens != 43 {
		t.Errorf("untouched model dropped what came back: Thinking=%q ThinkingTokens=%d",
			resp.Thinking, resp.Usage.ThinkingTokens)
	}
}

// A response with no thoughts leaves the thinking fields at their zero values
// rather than inventing them, and the metadata key stays absent.
func TestGoogleWithoutThoughtsReportsNothing(t *testing.T) {
	var c capture
	srv := geminiThinkingStub(t, &c, geminiPlainResponse)
	defer srv.Close()

	resp := generate(t, &GoogleConfig{APIKey: "k"}, NewGemini3Pro())

	if resp.Thinking != "" || resp.Usage.ThinkingTokens != 0 {
		t.Errorf("Thinking=%q ThinkingTokens=%d, want both empty", resp.Thinking, resp.Usage.ThinkingTokens)
	}
	if _, ok := resp.Metadata["thinking_signature"]; ok {
		t.Error("thinking_signature written for a response that carried none")
	}
	if resp.Usage.CompletionTokens != 7 || resp.Usage.TotalTokens != 18 {
		t.Errorf("usage disturbed by the thinking normalizer: %+v", resp.Usage)
	}
}

// ============================================================================
// GOOGLE THINKING: TRANSLATION AND THE NEVER-ERROR RULE
// ============================================================================

// Whatever lingo had to translate or drop is reported back, so an adaptation is
// never silent.
func TestGoogleRecordsWhatItTranslated(t *testing.T) {
	tests := []struct {
		name  string
		model Model
		want  string
	}{
		{"budget becomes a level", Thinking(NewGemini3Pro(), WithThinkingBudget(30000)),
			"budget 30000 mapped to effort high"},
		{"effort becomes a budget", Thinking(NewGemini25Flash(), WithThinkingEffort(ThinkingEffortLow)),
			"effort low mapped to budget 2949 tokens"},
		{"effort above the ladder", Thinking(NewGemini3Pro(), WithThinkingEffort(ThinkingEffortMax)),
			"effort max clamped to high"},
		{"budget above the ceiling", Thinking(NewGemini25Flash(), WithThinkingBudget(99999)),
			"budget 99999 clamped to 24576"},
		{"no off switch", NoThinking(NewGemini3Pro()),
			"thinking off dropped: model has no off switch"},
		{"no dynamic setting", Thinking(NewGemini3Pro(), WithDynamicThinking()),
			"dynamic thinking dropped: model has no dynamic setting"},
		{"no depth setting at all", Thinking(NewGemini20Flash(), WithThinkingEffort(ThinkingEffortHigh)),
			"effort high dropped: model takes no depth setting"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var c capture
			srv := geminiThinkingStub(t, &c, geminiPlainResponse)
			defer srv.Close()

			resp := generate(t, &GoogleConfig{APIKey: "k"}, tt.model)
			if got := resp.Metadata["thinking_translation"]; !strings.Contains(got, tt.want) {
				t.Errorf("thinking_translation = %q, want it to contain %q", got, tt.want)
			}
		})
	}
}

// A model that translated nothing writes no breadcrumb at all.
func TestGoogleWritesNoTranslationWhenNothingWasAdapted(t *testing.T) {
	var c capture
	srv := geminiThinkingStub(t, &c, geminiPlainResponse)
	defer srv.Close()

	for _, m := range []Model{
		NewGemini3Pro(),
		Thinking(NewGemini3Pro(), WithThinkingEffort(ThinkingEffortHigh)),
		Thinking(NewGemini25Flash(), WithThinkingBudget(8000)),
	} {
		resp := generate(t, &GoogleConfig{APIKey: "k"}, m)
		if s, ok := resp.Metadata["thinking_translation"]; ok {
			t.Errorf("%s wrote thinking_translation = %q, want none", m.ModelName(), s)
		}
	}
}

// Asking for something a Gemini cannot do is a no-op, never an error and never a
// panic -- including on the generations whose API would reject the field, which
// is exactly why lingo does not send it.
func TestGoogleThinkingIsNeverAnError(t *testing.T) {
	var c capture
	srv := geminiThinkingStub(t, &c, geminiPlainResponse)
	defer srv.Close()

	models := []Model{
		Thinking(NewGemini15Pro(), WithThinkingBudget(4096)),
		Thinking(NewGemini20Flash(), WithThinkingEffort(ThinkingEffortHigh)),
		NoThinking(NewGemini20FlashThinking()),
		NoThinking(NewGemini25Pro()),
		NoThinking(NewGemini3Pro()),
		Thinking(NewGemini3Pro(), WithThinkingEffort("no-such-level")),
		Thinking(NewGemini25Flash(), WithThinkingBudget(-99)),
		Thinking(NewGoogleModel("gemini-4-pro"), WithDynamicThinking()),
	}
	for _, m := range models {
		if resp := generate(t, &GoogleConfig{APIKey: "k"}, m); resp.Text != "hi there" {
			t.Errorf("%s: Text = %q", m.ModelName(), resp.Text)
		}
	}
}

// An off-ladder effort string is dropped rather than forwarded: Google's
// thinkingLevel is a closed enum, and THINKING_LEVEL_UNSPECIFIED is a value in
// its own right rather than a way to say "nothing".
func TestGoogleDropsAnOffLadderEffort(t *testing.T) {
	wantThinkingConfig(t, geminiThinkingWire(t, Thinking(NewGemini3Pro(), WithThinkingEffort("no-such-level"))), "")
}

// flagLogger records the Bool fields of every log event, so a debug field that
// makes a claim about the request can be asserted rather than argued about. It
// satisfies Logger and LogEvent in one type; every setter but Bool discards.
type flagLogger struct{ flags map[string]bool }

func newFlagLogger() *flagLogger { return &flagLogger{flags: map[string]bool{}} }

func (l *flagLogger) Debug() LogEvent { return l }
func (l *flagLogger) Info() LogEvent  { return l }
func (l *flagLogger) Error() LogEvent { return l }

func (l *flagLogger) Msg(string)                   {}
func (l *flagLogger) Str(string, string) LogEvent  { return l }
func (l *flagLogger) Int(string, int) LogEvent     { return l }
func (l *flagLogger) Int64(string, int64) LogEvent { return l }
func (l *flagLogger) Err(error) LogEvent           { return l }
func (l *flagLogger) Bool(k string, v bool) LogEvent {
	l.flags[k] = v
	return l
}

// TestGoogleHasThinkingIsFalseWhenTheRequestSwitchedThinkingOff covers the shape
// that makes Google's debug field easy to get wrong. Gemini spells off as
// thinkingBudget 0, so a request that switched thinking OFF carries a
// thinkingConfig just as one that switched it on does — the presence of the
// config says only that lingo had something to say, not what it said. A bare
// `thinkingConfig != nil` therefore logs a disabled request as a thinking one,
// which is the same claim-versus-body mismatch as is_reasoning_model on the
// OpenAI-compatible lane, in the one field Google makes the claim in.
func TestGoogleHasThinkingIsFalseWhenTheRequestSwitchedThinkingOff(t *testing.T) {
	tests := []struct {
		name  string
		model Model
		want  bool
	}{
		{"off", NoThinking(NewGemini25Flash()), false},
		{"on", Thinking(NewGemini25Flash()), true},
		{"a budget", Thinking(NewGemini25Flash(), WithThinkingBudget(4096)), true},
		{"untouched", NewGemini25Flash(), false},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var c capture
			srv := geminiThinkingStub(t, &c, geminiPlainResponse)
			defer srv.Close()

			log := newFlagLogger()
			g, err := New([]ProviderConfig{&GoogleConfig{APIKey: "k"}}, WithLogger(log))
			if err != nil {
				t.Fatalf("gateway: %v", err)
			}
			defer g.Close()
			if _, err := g.Generate(context.Background(), tc.model, "hello"); err != nil {
				t.Fatalf("generate: %v", err)
			}

			if got := log.flags["has_thinking"]; got != tc.want {
				t.Errorf("has_thinking = %t, want %t: the field means \"this request asks the "+
					"model to think\", and off is carried BY a thinkingConfig here, not by its absence",
					got, tc.want)
			}
		})
	}

	// And the fact that makes the guard necessary: an off really does build a
	// config, so the nil check alone cannot tell the two apart.
	t.Run("off builds a config, so nil-ness cannot answer", func(t *testing.T) {
		cfg, plan := googleThinkingConfig(NoThinking(NewGemini25Flash()))
		if cfg == nil || !plan.disable {
			t.Fatalf("expected a disabling thinkingConfig: cfg=%v disable=%t", cfg, plan.disable)
		}
	})
}

// The portable surface and the model's own storage are the same fields, so
// reading a model back reports exactly what was asked for.
func TestGoogleThinkingOptionsRoundTrip(t *testing.T) {
	m := NewGemini25Flash()
	m.ThinkingOptions().WithBudget(4096).WithTrace(ThinkingTraceInclude)

	if got := m.ThinkingOptions().Budget(); got != 4096 {
		t.Errorf("Budget() = %d, want 4096", got)
	}
	if !m.ThinkingOptions().Enabled() {
		t.Error("Enabled() = false, want a budget to have turned thinking on")
	}
	if got := m.ThinkingOptions().Trace(); got != ThinkingTraceInclude {
		t.Errorf("Trace() = %v, want ThinkingTraceInclude", got)
	}

	NoThinking(m)
	if !m.ThinkingOptions().Disabled() {
		t.Error("NoThinking left the model enabled")
	}
}
