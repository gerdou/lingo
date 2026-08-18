package lingo

import (
	"encoding/json"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	brtypes "github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

// ============================================================================
// BEDROCK THINKING: GOLDEN REQUESTS
// ============================================================================
//
// Bedrock hand-rolls its JSON, so the opt-in guarantee is checkable exactly: the
// bodies below were recorded off the builders before the thinking field existed
// and must keep marshalling byte for byte. Every family is here, not just the
// one that grew a knob -- Claude's request struct gained a field, and the four
// families that did not are what proves the change stayed inside its lane.
//
// A row that changes is a behaviour change for callers who wrote their code
// before the portable thinking surface existed, and has to be argued for rather
// than discovered.

// bedrockClaudeBody builds a Claude InvokeModel body and returns it as raw JSON
// alongside the notes lingo recorded about what it had to adapt.
func bedrockClaudeBody(t *testing.T, m Model) (string, string) {
	t.Helper()
	raw, _, plan, err := (&bedrockClient{}).buildClaudeRequest(m, "hi")
	if err != nil {
		t.Fatal(err)
	}
	return string(raw), plan.translation()
}

func TestBedrockDefaultClaudeBodiesAreUnchanged(t *testing.T) {
	const v = `"anthropic_version":"bedrock-2023-05-31"`

	for _, tc := range []struct {
		name  string
		model func() Model
		want  string
	}{
		{"claude 3.5 sonnet", func() Model { return NewBedrockClaude35Sonnet() },
			`{` + v + `,"max_tokens":4096,"messages":[{"role":"user","content":"hi"}],"temperature":1}`},
		{"claude 3.5 haiku", func() Model { return NewBedrockClaude35Haiku() },
			`{` + v + `,"max_tokens":4096,"messages":[{"role":"user","content":"hi"}],"temperature":1}`},
		{"claude 3.7 sonnet", func() Model { return NewBedrockClaude37Sonnet() },
			`{` + v + `,"max_tokens":8192,"messages":[{"role":"user","content":"hi"}],"temperature":1}`},
		{"claude sonnet 4", func() Model { return NewBedrockClaudeSonnet4() },
			`{` + v + `,"max_tokens":8192,"messages":[{"role":"user","content":"hi"}],"temperature":1}`},
		{"claude opus 4", func() Model { return NewBedrockClaudeOpus4() },
			`{` + v + `,"max_tokens":8192,"messages":[{"role":"user","content":"hi"}],"temperature":1}`},
		{"claude sonnet 4.5", func() Model { return NewBedrockClaudeSonnet45() },
			`{` + v + `,"max_tokens":8192,"messages":[{"role":"user","content":"hi"}],"temperature":1}`},
		{"claude opus 4.5", func() Model { return NewBedrockClaudeOpus45() },
			`{` + v + `,"max_tokens":8192,"messages":[{"role":"user","content":"hi"}],"temperature":1}`},
		{"claude haiku 4.5", func() Model { return NewBedrockClaudeHaiku45() },
			`{` + v + `,"max_tokens":8192,"messages":[{"role":"user","content":"hi"}],"temperature":1}`},
		{"claude opus 4.6", func() Model { return NewBedrockClaudeOpus46() },
			`{` + v + `,"max_tokens":8192,"messages":[{"role":"user","content":"hi"}],"temperature":1}`},
		{"claude sonnet 4.6", func() Model { return NewBedrockClaudeSonnet46() },
			`{` + v + `,"max_tokens":8192,"messages":[{"role":"user","content":"hi"}],"temperature":1}`},
		{"claude opus 4.7", func() Model { return NewBedrockClaudeOpus47() },
			`{` + v + `,"max_tokens":8192,"messages":[{"role":"user","content":"hi"}]}`},
		{"claude opus 4.8", func() Model { return NewBedrockClaudeOpus48() },
			`{` + v + `,"max_tokens":8192,"messages":[{"role":"user","content":"hi"}]}`},
		{"claude fable 5", func() Model { return NewBedrockClaudeFable5() },
			`{` + v + `,"max_tokens":8192,"messages":[{"role":"user","content":"hi"}]}`},
		{"claude opus 5", func() Model { return NewBedrockClaudeOpus5() },
			`{` + v + `,"max_tokens":8192,"messages":[{"role":"user","content":"hi"}]}`},
		{"claude sonnet 5", func() Model { return NewBedrockClaudeSonnet5() },
			`{` + v + `,"max_tokens":8192,"messages":[{"role":"user","content":"hi"}]}`},
		{"claude 3 sonnet", func() Model { return NewBedrockClaude3Sonnet() },
			`{` + v + `,"max_tokens":4096,"messages":[{"role":"user","content":"hi"}],"temperature":1}`},
		{"claude 3 haiku", func() Model { return NewBedrockClaude3Haiku() },
			`{` + v + `,"max_tokens":4096,"messages":[{"role":"user","content":"hi"}],"temperature":1}`},
		{"claude 3 opus", func() Model { return NewBedrockClaude3Opus() },
			`{` + v + `,"max_tokens":4096,"messages":[{"role":"user","content":"hi"}],"temperature":1}`},
		{"generic claude", func() Model { return NewBedrockModel("anthropic.claude-opus-5", "claude") },
			`{` + v + `,"max_tokens":4096,"messages":[{"role":"user","content":"hi"}],"temperature":0.7}`},
		{"every sampling option set", func() Model {
			return NewBedrockClaudeSonnet4().WithSystemPrompt("be terse").WithMaxTokens(1000).
				WithTemperature(0.3).WithTopP(0.9).WithTopK(5)
		},
			`{` + v + `,"max_tokens":1000,"messages":[{"role":"user","content":"hi"}],"system":"be terse","temperature":0.3,"top_p":0.9,"top_k":5}`},
		{"caching still marks the system prompt", func() Model {
			return Cached(NewBedrockClaudeSonnet5().WithSystemPrompt("be terse"), WithCacheTTL(CacheTTL1h))
		},
			`{` + v + `,"max_tokens":8192,"messages":[{"role":"user","content":"hi"}],"system":[{"type":"text","text":"be terse","cache_control":{"type":"ephemeral","ttl":"1h"}}]}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, notes := bedrockClaudeBody(t, tc.model())
			if got != tc.want {
				t.Errorf("body changed\n got %s\nwant %s", got, tc.want)
			}
			if notes != "" {
				t.Errorf("a model nobody touched recorded a translation: %q", notes)
			}
		})
	}
}

func TestBedrockOtherFamilyBodiesAreUnchanged(t *testing.T) {
	c := &bedrockClient{}

	t.Run("titan", func(t *testing.T) {
		for _, tc := range []struct {
			model Model
			want  string
		}{
			{NewBedrockTitanTextExpress(), `{"inputText":"hi","textGenerationConfig":{"maxTokenCount":4096,"temperature":0.7,"topP":0.9}}`},
			{NewBedrockTitanTextLite(), `{"inputText":"hi","textGenerationConfig":{"maxTokenCount":4096,"temperature":0.7,"topP":0.9}}`},
			{NewBedrockTitanTextPremier().WithSystemPrompt("be terse").WithMaxTokens(99),
				`{"inputText":"be terse\n\nhi","textGenerationConfig":{"maxTokenCount":99,"temperature":0.7,"topP":0.9}}`},
			{NewBedrockModel("amazon.titan-text-premier-v1:0", "titan"),
				`{"inputText":"hi","textGenerationConfig":{"maxTokenCount":4096,"temperature":0.7,"topP":0.9}}`},
		} {
			raw, err := c.buildTitanRequest(tc.model, "hi")
			if err != nil {
				t.Fatal(err)
			}
			if string(raw) != tc.want {
				t.Errorf("%s body changed\n got %s\nwant %s", tc.model.ModelName(), raw, tc.want)
			}
		}
	})

	t.Run("llama", func(t *testing.T) {
		for _, tc := range []struct {
			model Model
			want  string
		}{
			{NewBedrockLlama33Instruct70B(), `{"prompt":"\u003c|begin_of_text|\u003e\u003c|start_header_id|\u003euser\u003c|end_header_id|\u003e\n\nhi\u003c|eot_id|\u003e\u003c|start_header_id|\u003eassistant\u003c|end_header_id|\u003e\n\n","max_gen_len":2048,"temperature":0.6,"top_p":0.9}`},
			{NewBedrockLlama4Scout().WithSystemPrompt("be terse").WithMaxTokens(77),
				`{"prompt":"\u003c|begin_of_text|\u003e\u003c|header_start|\u003esystem\u003c|header_end|\u003e\n\nbe terse\u003c|eot|\u003e\u003c|header_start|\u003euser\u003c|header_end|\u003e\n\nhi\u003c|eot|\u003e\u003c|header_start|\u003eassistant\u003c|header_end|\u003e\n\n","max_gen_len":77,"temperature":0.6,"top_p":0.9}`},
			{NewBedrockModel("meta.llama3-3-70b-instruct-v1:0", "llama"),
				`{"prompt":"\u003c|begin_of_text|\u003e\u003c|start_header_id|\u003euser\u003c|end_header_id|\u003e\n\nhi\u003c|eot_id|\u003e\u003c|start_header_id|\u003eassistant\u003c|end_header_id|\u003e\n\n","max_gen_len":4096,"temperature":0.7,"top_p":0.9}`},
		} {
			raw, err := c.buildLlamaRequest(tc.model, "hi")
			if err != nil {
				t.Fatal(err)
			}
			if string(raw) != tc.want {
				t.Errorf("%s body changed\n got %s\nwant %s", tc.model.ModelName(), raw, tc.want)
			}
		}
	})

	t.Run("mistral", func(t *testing.T) {
		for _, tc := range []struct {
			model Model
			want  string
		}{
			{NewBedrockMistral7B(), `{"prompt":"\u003cs\u003e[INST] hi [/INST]","max_tokens":4096,"temperature":0.7,"top_p":0.9}`},
			{NewBedrockMistralLarge().WithSystemPrompt("be terse").WithMaxTokens(55).WithTopK(3),
				`{"prompt":"\u003cs\u003e[INST] be terse\n\nhi [/INST]","max_tokens":55,"temperature":0.7,"top_p":0.9,"top_k":3}`},
			{NewBedrockModel("mistral.mistral-large-2407-v1:0", "mistral"),
				`{"prompt":"\u003cs\u003e[INST] hi [/INST]","max_tokens":4096,"temperature":0.7,"top_p":0.9}`},
		} {
			raw, err := c.buildMistralRequest(tc.model, "hi")
			if err != nil {
				t.Fatal(err)
			}
			if string(raw) != tc.want {
				t.Errorf("%s body changed\n got %s\nwant %s", tc.model.ModelName(), raw, tc.want)
			}
		}
	})

	// Nova is served by Converse, but buildNovaRequest is deliberately kept as
	// the rollback path, so its body is pinned too.
	t.Run("nova invokemodel", func(t *testing.T) {
		for _, tc := range []struct {
			model Model
			want  string
		}{
			{NewBedrockNovaMicro(), `{"schemaVersion":"messages-v1","messages":[{"role":"user","content":[{"text":"hi"}]}],"inferenceConfig":{"maxTokens":4096,"temperature":0.7}}`},
			{NewBedrockNovaPro().WithSystemPrompt("be terse").WithMaxTokens(100).WithTemperature(0.2).WithTopP(0.8).WithTopK(7),
				`{"schemaVersion":"messages-v1","messages":[{"role":"user","content":[{"text":"hi"}]}],"system":[{"text":"be terse"}],"inferenceConfig":{"maxTokens":100,"temperature":0.2,"topP":0.8,"topK":7}}`},
		} {
			raw, err := c.buildNovaRequest(tc.model, "hi")
			if err != nil {
				t.Fatal(err)
			}
			if string(raw) != tc.want {
				t.Errorf("%s body changed\n got %s\nwant %s", tc.model.ModelName(), raw, tc.want)
			}
		}
	})
}

// ============================================================================
// BEDROCK THINKING: CAPABILITIES
// ============================================================================

func TestBedrockThinkingDimensionsResolveFromModelID(t *testing.T) {
	const budgetEra = ThinkingCanToggle | ThinkingCanSetBudget | ThinkingCanReportTrace
	const toggleEra = ThinkingCanToggle | ThinkingCanReportTrace

	for _, tc := range []struct {
		modelID string
		want    ThinkingDimension
	}{
		// 3.5 and earlier have no thinking field of any kind.
		{"anthropic.claude-3-5-sonnet-20241022-v2:0", 0},
		{"anthropic.claude-3-opus-20240229-v1:0", 0},
		// 3.7 through 4.6 take the fixed thinking config lingo writes.
		{"anthropic.claude-3-7-sonnet-20250219-v1:0", budgetEra},
		{"anthropic.claude-sonnet-4-20250514-v1:0", budgetEra},
		{"anthropic.claude-opus-4-5-20251101-v1:0", budgetEra},
		{"anthropic.claude-haiku-4-5-20251001-v1:0", budgetEra},
		{"anthropic.claude-opus-4-6-v1", budgetEra},
		{"anthropic.claude-sonnet-4-6", budgetEra},
		// 4.7 onwards reject a fixed budget, so only the off switch is reachable.
		{"anthropic.claude-opus-4-7", toggleEra},
		{"anthropic.claude-opus-4-8", toggleEra},
		{"anthropic.claude-opus-5", toggleEra},
		{"anthropic.claude-sonnet-5", toggleEra},
		// Fable 5 reasons server-side and rejects any thinking config.
		{"anthropic.claude-fable-5", ThinkingCanReportTrace},
		// A cross-region inference profile is the same model.
		{"us.anthropic.claude-sonnet-4-20250514-v1:0", budgetEra},
		{"eu.anthropic.claude-opus-5", toggleEra},
		{"global.anthropic.claude-3-5-sonnet-20241022-v2:0", 0},
		// Other vendors have no thinking dialect lingo can write.
		{"amazon.nova-pro-v1:0", 0},
		{"us.amazon.nova-premier-v1:0", 0},
		{"amazon.titan-text-premier-v1:0", 0},
		{"meta.llama3-3-70b-instruct-v1:0", 0},
		{"mistral.mistral-large-2407-v1:0", 0},
		{"", 0},
	} {
		if got := bedrockThinkingDimensions(tc.modelID); got != tc.want {
			t.Errorf("bedrockThinkingDimensions(%q) = %06b, want %06b", tc.modelID, got, tc.want)
		}
		// The generic model resolves from the id it was handed, which is what
		// makes a Claude released after this build reachable.
		generic := NewBedrockModel(tc.modelID, getModelFamily(tc.modelID))
		if got := ModelThinkingDimensions(generic); got != tc.want {
			t.Errorf("ModelThinkingDimensions(BedrockModel %q) = %06b, want %06b", tc.modelID, got, tc.want)
		}
	}
}

func TestBedrockModelThinkingDimensionsPerType(t *testing.T) {
	for _, tc := range []struct {
		model Model
		want  ThinkingDimension
	}{
		{NewBedrockClaudeSonnet4(), ThinkingCanToggle | ThinkingCanSetBudget | ThinkingCanReportTrace},
		{NewBedrockClaudeOpus46(), ThinkingCanToggle | ThinkingCanSetBudget | ThinkingCanReportTrace},
		{NewBedrockClaudeOpus47(), ThinkingCanToggle | ThinkingCanReportTrace},
		{NewBedrockClaudeSonnet5(), ThinkingCanToggle | ThinkingCanReportTrace},
		{NewBedrockClaudeFable5(), ThinkingCanReportTrace},
		{NewBedrockClaude35Sonnet(), 0},
		// Nova reasons upstream but lingo asks it for nothing, and the coarse
		// provider answer must not be allowed to imply otherwise.
		{NewBedrockNovaPro(), 0},
		{NewBedrockNovaMicro(), 0},
		// The families with no thinking field at all fall through to the same 0,
		// rather than inheriting the provider-wide answer.
		{NewBedrockTitanTextPremier(), 0},
		{NewBedrockLlama4Scout(), 0},
		{NewBedrockMistralLarge(), 0},
	} {
		if got := ModelThinkingDimensions(tc.model); got != tc.want {
			t.Errorf("ModelThinkingDimensions(%s) = %06b, want %06b", tc.model.ModelName(), got, tc.want)
		}
	}

	// A zero-value literal is constructible outside this package and must answer
	// the same as its constructor, which is why dimensions are resolved from the
	// model id rather than stored by the constructor.
	if got, want := ModelThinkingDimensions(&BedrockClaudeSonnet4{}), ModelThinkingDimensions(NewBedrockClaudeSonnet4()); got != want {
		t.Errorf("&BedrockClaudeSonnet4{} reports %06b, want %06b", got, want)
	}
}

func TestBedrockThinkingModelMembership(t *testing.T) {
	// Claude carries thinking configuration because its body has a field for it;
	// Nova carries it so the surface is there the day the wire key is verified.
	for _, m := range []Model{
		NewBedrockClaudeSonnet5(), NewBedrockClaude35Sonnet(), NewBedrockClaudeFable5(),
		NewBedrockNovaPro(), NewBedrockModel("anthropic.claude-opus-5", "claude"),
	} {
		if _, ok := m.(ThinkingModel); !ok {
			t.Errorf("%T does not satisfy ThinkingModel", m)
		}
	}
	// Titan, Llama and Mistral do not embed the options struct at all, so the
	// exclusion is structural rather than a runtime check that could be forgotten.
	for _, m := range []Model{
		NewBedrockTitanTextPremier(), NewBedrockLlama4Scout(), NewBedrockMistralLarge(),
	} {
		if _, ok := m.(ThinkingModel); ok {
			t.Errorf("%T satisfies ThinkingModel, but its API has no thinking field", m)
		}
	}
}

// ============================================================================
// BEDROCK THINKING: WIRE
// ============================================================================

func TestBedrockClaudeThinkingWire(t *testing.T) {
	// The thinking object is the only thing under test, so the rest of the body
	// is compared as a whole against the model's untouched form once per case.
	thinkingOf := func(t *testing.T, body string) string {
		t.Helper()
		var parsed struct {
			Thinking json.RawMessage `json:"thinking"`
		}
		if err := json.Unmarshal([]byte(body), &parsed); err != nil {
			t.Fatalf("bad body %s: %v", body, err)
		}
		return string(parsed.Thinking)
	}

	for _, tc := range []struct {
		name  string
		model func() Model
		want  string // "" means the key must be absent
		notes string
	}{
		{"a budget-era model takes an explicit budget verbatim",
			func() Model { return Thinking(NewBedrockClaudeSonnet4(), WithThinkingBudget(4000)) },
			`{"type":"enabled","budget_tokens":4000}`, ""},
		{"a budget below the API floor is clamped rather than rejected",
			func() Model { return Thinking(NewBedrockClaudeSonnet4(), WithThinkingBudget(500)) },
			`{"type":"enabled","budget_tokens":1024}`, "budget 500 clamped to 1024"},
		{"a budget at or above max_tokens is clamped below it",
			func() Model { return Thinking(NewBedrockClaudeSonnet4(), WithThinkingBudget(99999)) },
			`{"type":"enabled","budget_tokens":8191}`, "budget 99999 clamped to 8191"},
		{"an effort becomes a budget, because Bedrock takes no effort ladder",
			func() Model { return Thinking(NewBedrockClaudeSonnet4(), WithThinkingEffort(ThinkingEffortMax)) },
			`{"type":"enabled","budget_tokens":8191}`, "effort max mapped to budget 8191 tokens"},
		{"plain enable becomes a fixed budget on a generation with no adaptive config",
			func() Model { return Thinking(NewBedrockClaudeSonnet4()) },
			`{"type":"enabled","budget_tokens":4914}`,
			"thinking enabled as a fixed budget of 4914 tokens: lingo sends no adaptive thinking config on Bedrock"},
		{"a dynamic budget degrades the same way",
			func() Model { return Thinking(NewBedrockClaudeSonnet4(), WithDynamicThinking()) },
			`{"type":"enabled","budget_tokens":4914}`,
			"thinking enabled as a fixed budget of 4914 tokens: lingo sends no adaptive thinking config on Bedrock"},
		{"max_tokens too small for a legal budget drops the field instead of sending an illegal one",
			func() Model { return Thinking(NewBedrockClaudeSonnet4().WithMaxTokens(1024)) },
			"", "thinking enabled but dropped: max_tokens leaves no room for a legal budget"},
		{"switching thinking off is the one knob every thinking generation has",
			func() Model { return NoThinking(NewBedrockClaudeSonnet4()) },
			`{"type":"disabled"}`, ""},
		{"4.6 still takes the fixed budget",
			func() Model { return Thinking(NewBedrockClaudeOpus46(), WithThinkingBudget(2048)) },
			`{"type":"enabled","budget_tokens":2048}`, ""},
		{"4.7 rejects a fixed budget, so a budget is dropped rather than sent",
			func() Model { return Thinking(NewBedrockClaudeOpus47(), WithThinkingBudget(2048)) },
			"", "budget 2048 dropped: model takes no token budget; thinking enabled but dropped: this generation takes only an adaptive thinking config, which lingo does not send on Bedrock"},
		{"4.7 can still be switched off",
			func() Model { return NoThinking(NewBedrockClaudeOpus47()) },
			`{"type":"disabled"}`, ""},
		{"claude 5 reasons by default, so asking for it changes no bytes",
			func() Model { return Thinking(NewBedrockClaudeSonnet5()) },
			"", ""},
		{"claude 5 can be told not to reason",
			func() Model { return NoThinking(NewBedrockClaudeSonnet5()) },
			`{"type":"disabled"}`, ""},
		{"fable 5 reasons server-side and takes no config at all",
			func() Model { return NoThinking(NewBedrockClaudeFable5()) },
			"", "thinking off dropped: model has no off switch"},
		{"claude 3.5 has no thinking field, so every knob is dropped rather than sent",
			func() Model { return Thinking(NewBedrockClaude35Sonnet(), WithThinkingBudget(4000)) },
			"", "budget 4000 dropped: model takes no token budget"},
		{"the generic model resolves its dialect from the id it was handed",
			func() Model {
				return Thinking(NewBedrockModel("eu.anthropic.claude-3-7-sonnet-20250219-v1:0", "claude"),
					WithThinkingBudget(2000))
			},
			`{"type":"enabled","budget_tokens":2000}`, ""},
		{"trace visibility is not something this path can ask for",
			func() Model { return Thinking(NewBedrockClaudeSonnet5(), WithThinkingTrace(ThinkingTraceOmit)) },
			"", "trace omission dropped: model always returns its trace"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			body, notes := bedrockClaudeBody(t, tc.model())
			if got := thinkingOf(t, body); got != tc.want {
				t.Errorf("thinking = %s, want %s\nbody: %s", or(got, "<absent>"), or(tc.want, "<absent>"), body)
			}
			if notes != tc.notes {
				t.Errorf("translation = %q, want %q", notes, tc.notes)
			}
		})
	}
}

// or returns b when a is empty, for readable "absent" diffs.
func or(a, b string) string {
	if a == "" {
		return b
	}
	return a
}

func TestBedrockClaudeThinkingLeavesTheRestOfTheBodyAlone(t *testing.T) {
	// Enabling thinking must add one key and touch nothing else, including the
	// cache breakpoint that shares the tail of the builder.
	plain, _ := bedrockClaudeBody(t, NewBedrockClaudeSonnet4().WithSystemPrompt("be terse"))
	thinking, _ := bedrockClaudeBody(t,
		Thinking(NewBedrockClaudeSonnet4().WithSystemPrompt("be terse"), WithThinkingBudget(2000)))

	var a, b map[string]any
	if err := json.Unmarshal([]byte(plain), &a); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal([]byte(thinking), &b); err != nil {
		t.Fatal(err)
	}
	if _, ok := b["thinking"]; !ok {
		t.Fatalf("thinking key missing from %s", thinking)
	}
	delete(b, "thinking")
	if len(a) != len(b) {
		t.Fatalf("key sets differ\n plain %v\nthinking %v", a, b)
	}
	for k, want := range a {
		got, _ := json.Marshal(b[k])
		wantRaw, _ := json.Marshal(want)
		if string(got) != string(wantRaw) {
			t.Errorf("%s = %s, want %s", k, got, wantRaw)
		}
	}

	// And caching still works alongside it.
	both, _ := bedrockClaudeBody(t, Thinking(
		Cached(NewBedrockClaudeSonnet4().WithSystemPrompt("be terse")), WithThinkingBudget(2000)))
	var parsed map[string]any
	if err := json.Unmarshal([]byte(both), &parsed); err != nil {
		t.Fatal(err)
	}
	if _, ok := parsed["thinking"]; !ok {
		t.Errorf("thinking dropped when caching is on: %s", both)
	}
	blocks, ok := parsed["system"].([]any)
	if !ok || len(blocks) != 1 {
		t.Fatalf("system = %#v, want the cached content block", parsed["system"])
	}
	if _, ok := blocks[0].(map[string]any)["cache_control"]; !ok {
		t.Errorf("cache breakpoint dropped when thinking is on: %s", both)
	}
}

// Nova stores whatever it is given and sends none of it, because the Converse
// wire key for a reasoning config could not be verified from any pinned source.
func TestBedrockNovaStoresThinkingAndSendsNothing(t *testing.T) {
	c := &bedrockClient{}

	m := Thinking(NewBedrockNovaPro().WithSystemPrompt("be terse").WithTopK(7), WithThinkingBudget(4000))
	if !m.ThinkingOptions().Enabled() {
		t.Error("Nova did not store the thinking configuration it was given")
	}

	in, _ := c.buildConverseInput(m, "hi", m.ModelName())
	if in.AdditionalModelRequestFields == nil {
		t.Fatal("topK was dropped")
	}
	raw, err := in.AdditionalModelRequestFields.MarshalSmithyDocument()
	if err != nil {
		t.Fatal(err)
	}
	if string(raw) != `{"inferenceConfig":{"topK":7}}` {
		t.Errorf("additionalModelRequestFields = %s, want topK alone: nothing about thinking may be sent", raw)
	}

	// And a Nova model nobody touched still sends no document at all.
	untouched, _ := c.buildConverseInput(Thinking(NewBedrockNovaPro()), "hi", "amazon.nova-pro-v1:0")
	if untouched.AdditionalModelRequestFields != nil {
		t.Error("opting into thinking put a document on a Nova request")
	}
}

// ============================================================================
// BEDROCK THINKING: RESPONSE
// ============================================================================

func TestParseClaudeResponseExtractsThinking(t *testing.T) {
	c := &bedrockClient{}

	body := []byte(`{
		"content": [
			{"type": "thinking", "thinking": "first ", "signature": "sig-1"},
			{"type": "text", "text": "hel"},
			{"type": "thinking", "thinking": "second", "signature": "sig-2"},
			{"type": "redacted_thinking", "data": "encrypted"},
			{"type": "text", "text": "lo"}
		],
		"stop_reason": "end_turn",
		"usage": {"input_tokens": 10, "output_tokens": 40, "output_tokens_details": {"thinking_tokens": 25}}
	}`)

	resp, err := c.parseClaudeResponse(body, "anthropic.claude-sonnet-4-20250514-v1:0")
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text != "hello" {
		t.Errorf("Text = %q, want the text blocks accumulated and the trace kept out of them", resp.Text)
	}
	if resp.Thinking != "first second" {
		t.Errorf("Thinking = %q, want every thinking block accumulated", resp.Thinking)
	}
	if resp.Metadata["thinking"] != resp.Thinking {
		t.Errorf("Metadata[thinking] = %q, want it mirrored for readers written before the typed field",
			resp.Metadata["thinking"])
	}
	if resp.Metadata["thinking_signature"] != "sig-2" {
		t.Errorf("Metadata[thinking_signature] = %q, want the last block's", resp.Metadata["thinking_signature"])
	}
	if resp.Metadata["thinking_redacted"] != "encrypted" {
		t.Errorf("Metadata[thinking_redacted] = %q", resp.Metadata["thinking_redacted"])
	}
	// thinking_tokens is a subset of output_tokens, so the totals do not move.
	want := TokenUsage{PromptTokens: 10, CompletionTokens: 40, TotalTokens: 50, ThinkingTokens: 25}
	if resp.Usage != want {
		t.Errorf("Usage = %+v, want %+v", resp.Usage, want)
	}
	if resp.Usage.AnswerTokens() != 15 {
		t.Errorf("AnswerTokens() = %d, want 15", resp.Usage.AnswerTokens())
	}
}

func TestParseClaudeResponseWithoutThinking(t *testing.T) {
	c := &bedrockClient{}

	body := []byte(`{"content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn",
		"usage":{"input_tokens":5,"output_tokens":2}}`)

	resp, err := c.parseClaudeResponse(body, "anthropic.claude-sonnet-5")
	if err != nil {
		t.Fatal(err)
	}
	if resp.Thinking != "" {
		t.Errorf("Thinking = %q, want empty", resp.Thinking)
	}
	for _, key := range []string{"thinking", "thinking_signature", "thinking_redacted", "thinking_translation"} {
		if v, ok := resp.Metadata[key]; ok {
			t.Errorf("Metadata[%s] = %q, want the key absent when nothing was reported", key, v)
		}
	}
	// Bedrock reports no thinking token count, so the counter stays 0 rather
	// than being inferred from anything.
	want := TokenUsage{PromptTokens: 5, CompletionTokens: 2, TotalTokens: 7}
	if resp.Usage != want {
		t.Errorf("Usage = %+v, want %+v", resp.Usage, want)
	}
}

func TestParseConverseOutputExtractsReasoning(t *testing.T) {
	c := &bedrockClient{}

	out := &bedrockruntime.ConverseOutput{
		Output: &brtypes.ConverseOutputMemberMessage{Value: brtypes.Message{
			Role: brtypes.ConversationRoleAssistant,
			Content: []brtypes.ContentBlock{
				&brtypes.ContentBlockMemberReasoningContent{
					Value: &brtypes.ReasoningContentBlockMemberReasoningText{
						Value: brtypes.ReasoningTextBlock{
							Text:      aws.String("thought hard"),
							Signature: aws.String("sig"),
						},
					},
				},
				&brtypes.ContentBlockMemberText{Value: "hello"},
				&brtypes.ContentBlockMemberReasoningContent{
					Value: &brtypes.ReasoningContentBlockMemberRedactedContent{Value: []byte("secret")},
				},
			},
		}},
		StopReason: brtypes.StopReasonEndTurn,
		Usage: &brtypes.TokenUsage{
			InputTokens: aws.Int32(9), OutputTokens: aws.Int32(4), TotalTokens: aws.Int32(13),
		},
	}

	resp, err := c.parseConverseOutput(out, "amazon.nova-pro-v1:0", "nova")
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text != "hello" {
		t.Errorf("Text = %q, want the reasoning kept out of the answer", resp.Text)
	}
	if resp.Thinking != "thought hard" {
		t.Errorf("Thinking = %q", resp.Thinking)
	}
	if resp.Metadata["thinking_signature"] != "sig" {
		t.Errorf("Metadata[thinking_signature] = %q", resp.Metadata["thinking_signature"])
	}
	// The SDK decodes the blob, so it is re-encoded into the same form the
	// Claude InvokeModel path records.
	if got := resp.Metadata["thinking_redacted"]; got != "c2VjcmV0" {
		t.Errorf("Metadata[thinking_redacted] = %q, want the base64 of the encrypted blob", got)
	}
	// Converse reports no thinking token count at any version this is built
	// against, so the counter stays 0 even with a trace in hand.
	if resp.Usage.ThinkingTokens != 0 {
		t.Errorf("ThinkingTokens = %d, want 0: Converse reports no such count", resp.Usage.ThinkingTokens)
	}
}
