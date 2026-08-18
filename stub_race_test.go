package lingo

import (
	"context"
	"fmt"
	"sync"
	"testing"
)

// ============================================================================
// THE STUBS THEMSELVES
// ============================================================================
//
// An httptest server runs one handler goroutine per in-flight request, so every
// stub in this package writes its record from a goroutine that is not the
// test's. That was fine only because every caller happened to be serial:
// bedrockStub appended to a shared slice with no lock, and the capture stubs
// assigned four fields and built a map the same way. The first test to fire two
// requests at once would have got a data race, and in the slice case a lost
// record too -- which is worse than a detector complaint, because it fails as a
// wrong count and sends the reader looking at the code under test.
//
// The recorders now own a mutex. These tests are what says so: they run under
// -race, where an unlocked recorder is reported rather than tolerated, and the
// bedrock one also checks the count, which an unlocked append drops even
// without the detector.

const stubRaceCalls = 64

// TestBedrockStubRecordsEveryConcurrentRequest is the finding. A racing append
// loses records: two goroutines that read the same slice header both write to
// the same index, and one of the two calls vanishes.
func TestBedrockStubRecordsEveryConcurrentRequest(t *testing.T) {
	var calls bedrockCalls
	c := bedrockStub(t, &calls)

	var wg sync.WaitGroup
	problems := make(chan string, stubRaceCalls)
	for i := 0; i < stubRaceCalls; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			if _, err := c.Generate(context.Background(), NewBedrockClaudeSonnet5(),
				fmt.Sprintf("prompt-%d", i)); err != nil {
				problems <- fmt.Sprintf("request %d: %v", i, err)
			}
		}(i)
	}
	wg.Wait()
	close(problems)
	for p := range problems {
		t.Error(p)
	}

	if calls.len() != stubRaceCalls {
		t.Fatalf("recorded %d of %d concurrent requests: the log dropped records",
			calls.len(), stubRaceCalls)
	}
	// Every prompt exactly once, so a log that happens to be the right length
	// because one record was written twice still fails.
	seen := map[string]int{}
	for _, call := range calls.all() {
		seen[call.body]++
	}
	if len(seen) != stubRaceCalls {
		t.Errorf("recorded %d distinct bodies for %d distinct prompts", len(seen), stubRaceCalls)
	}
}

// The capture stubs lose nothing -- one field assignment overwrites another and
// the last writer legitimately wins -- so what is at stake there is the race
// itself: concurrent writes to the same string fields, and to the same map
// while json.Unmarshal is filling it. Only -race can see it, which is why this
// test asserts so little and matters anyway.
func TestCaptureStubSurvivesConcurrentRequests(t *testing.T) {
	var c capture
	srv := oaiStub(t, &c)
	defer srv.Close()

	g, err := New([]ProviderConfig{&OpenAICompatibleConfig{BaseURL: srv.URL, APIKey: "k"}})
	if err != nil {
		t.Fatalf("gateway: %v", err)
	}
	defer g.Close()

	model := NewOpenAICompatibleModel("llama-3.3-70b")
	var wg sync.WaitGroup
	problems := make(chan string, stubRaceCalls)
	for i := 0; i < stubRaceCalls; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			if _, err := g.Generate(context.Background(), model, fmt.Sprintf("prompt-%d", i)); err != nil {
				problems <- fmt.Sprintf("request %d: %v", i, err)
			}
		}(i)
	}
	wg.Wait()
	close(problems)
	for p := range problems {
		t.Error(p)
	}

	// Whichever request landed last, the capture is a whole record of one
	// request rather than a mixture of several.
	if c.path != "/chat/completions" {
		t.Errorf("path = %q", c.path)
	}
	if got := c.body["model"]; got != "llama-3.3-70b" {
		t.Errorf("model = %v", got)
	}
}

// The Google resource stubs share the same shape and the same fix.
func TestCacheStubRecordsEveryConcurrentRequest(t *testing.T) {
	var calls cacheCalls
	srv := cacheStub(t, &calls)
	defer srv.Close()

	g, mgr := cacheManager(t, &GoogleConfig{APIKey: "k"})
	defer g.Close()

	var wg sync.WaitGroup
	problems := make(chan string, stubRaceCalls)
	for i := 0; i < stubRaceCalls; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			if _, err := mgr.GetCache(context.Background(), "cachedContents/abc123"); err != nil {
				problems <- fmt.Sprintf("request %d: %v", i, err)
			}
		}(i)
	}
	wg.Wait()
	close(problems)
	for p := range problems {
		t.Error(p)
	}

	if calls.len() != stubRaceCalls {
		t.Fatalf("recorded %d of %d concurrent requests: the log dropped records",
			calls.len(), stubRaceCalls)
	}
}
