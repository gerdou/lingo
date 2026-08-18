package lingo

import (
	"context"
	"errors"
	"fmt"
	"math/rand"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/aws/aws-sdk-go-v2/aws"
	cohereCore "github.com/cohere-ai/cohere-go/v2/core"
	"github.com/gerdou/lingo/internal/perplexity"
	"github.com/openai/openai-go/v3"
	"google.golang.org/genai"
)

// rateLimiter handles rate limit detection and retry logic
type rateLimiter struct {
	config *RateLimitConfig
	logger Logger
}

// newRateLimiter creates a new rate limiter with the given config
func newRateLimiter(config *RateLimitConfig, logger Logger) *rateLimiter {
	if config == nil {
		config = DefaultRateLimitConfig()
	}
	// Apply defaults for zero values
	if config.MaxRetries == 0 {
		config.MaxRetries = 3
	}
	// A negative count is the natural way to ask for retries off, since zero is
	// spoken for by the default above. It means one attempt and no retry -- it
	// must never mean no attempt at all.
	if config.MaxRetries < 0 {
		config.MaxRetries = 0
	}
	if config.InitialBackoff == 0 {
		config.InitialBackoff = 1 * time.Second
	}
	if config.MaxBackoff == 0 {
		config.MaxBackoff = 60 * time.Second
	}
	if config.BackoffMultiplier == 0 {
		config.BackoffMultiplier = 2.0
	}
	return &rateLimiter{
		config: config,
		logger: logger,
	}
}

// RetryFunc is a function that can be retried
type RetryFunc func() error

// Execute executes the given function with retry logic for rate limits
func (r *rateLimiter) Execute(ctx context.Context, fn RetryFunc) error {
	var lastErr error
	backoff := r.config.InitialBackoff

	// The loop always runs at least once. Every caller reads a nil error as
	// "the closure ran and assigned my response", so a bound that skipped the
	// body -- which a negative MaxRetries on a hand-built rateLimiter does --
	// would hand back nil beside a nil response and panic the caller rather
	// than fail it. newRateLimiter clamps this too; this is the guarantee at
	// the point that depends on it.
	maxRetries := r.config.MaxRetries
	if maxRetries < 0 {
		maxRetries = 0
	}

	for attempt := 0; attempt <= maxRetries; attempt++ {
		// Check if context is cancelled before attempting
		select {
		case <-ctx.Done():
			return retryCtxErr(ctx, lastErr)
		default:
		}

		err := fn()
		if err == nil {
			return nil
		}

		lastErr = err

		// Check if this is a rate limit error
		if !isRateLimitError(err) {
			return err // Not a rate limit error, don't retry
		}

		// Check if we've exhausted retries
		if attempt >= maxRetries {
			r.logger.Error().
				Int("attempts", attempt+1).
				Err(err).
				Msg("Rate limit retries exhausted")
			return err
		}

		// Calculate backoff with jitter
		waitDuration := r.calculateBackoff(backoff, err)

		r.logger.Debug().
			Int("attempt", attempt+1).
			Int("max_retries", maxRetries).
			Str("wait_duration", waitDuration.String()).
			Msg("Rate limited, waiting before retry")

		// Wait with context cancellation support
		select {
		case <-ctx.Done():
			return retryCtxErr(ctx, lastErr)
		case <-time.After(waitDuration):
		}

		// Increase backoff for next iteration
		backoff = time.Duration(float64(backoff) * r.config.BackoffMultiplier)
		if backoff > r.config.MaxBackoff {
			backoff = r.config.MaxBackoff
		}
	}

	return lastErr
}

// retryCtxErr reports a context that ended while Execute was retrying, without
// throwing away the error that made it retry.
//
// Every provider wraps this loop in the caller's own timeout, so a context that
// ends in here has almost always ended because the attempts outlasted that
// budget -- and the half of the story worth reading is the 429 or the 503 that
// caused the waiting, not the deadline that ended it. Returning ctx.Err() by
// itself hands someone debugging a throttled account a bare "context deadline
// exceeded" and points them at their network.
//
// Both errors are wrapped, so errors.Is still finds context.DeadlineExceeded or
// context.Canceled and errors.As still finds the provider's own typed error
// underneath. Before the first attempt there is nothing to add and the context
// error stands alone.
func retryCtxErr(ctx context.Context, lastErr error) error {
	if lastErr == nil {
		return ctx.Err()
	}
	return fmt.Errorf("%w after %w", ctx.Err(), lastErr)
}

// calculateBackoff calculates the wait duration, potentially using Retry-After header
func (r *rateLimiter) calculateBackoff(baseBackoff time.Duration, err error) time.Duration {
	// Try to extract Retry-After from error if available
	if retryAfter := extractRetryAfter(err); retryAfter > 0 {
		return retryAfter
	}

	// Add jitter (±25% of backoff)
	jitter := float64(baseBackoff) * 0.25 * (rand.Float64()*2 - 1)
	return baseBackoff + time.Duration(jitter)
}

// retryableStatuses are the HTTP statuses whose only sane answer is to wait and
// send the request again: the server has said "not now", not "not ever".
//
//   - 429 is every provider's rate limit.
//   - 503 is a service that is briefly unavailable -- Bedrock's
//     ServiceUnavailableException, Google's UNAVAILABLE.
//   - 529 is Anthropic's overloaded_error, which has no net/http constant.
//
// Anything else a server states is a verdict about this request, and repeating
// it only spends money and time to be told the same thing again.
var retryableStatuses = map[int]bool{
	http.StatusTooManyRequests:    true,
	http.StatusServiceUnavailable: true,
	529:                           true,
}

// errorHTTPStatus reports the HTTP status an error carries, if it carries one.
//
// A status is a fact the server stated. The wording scan in isRateLimitError is
// a guess about prose, and a guess that reads an error message looking for the
// digits 429 finds them inside "142900 tokens" in a 400 for an over-long prompt.
// So anything that can be asked for its status is asked, and the guess is never
// reached for it.
//
// Each SDK spells the field differently and none of them share a method, so the
// types are named here rather than sniffed. Smithy is the exception and matches
// structurally, which also covers every AWS service error that wraps in it.
func errorHTTPStatus(err error) (int, bool) {
	// A zero is one of these types carrying no status at all rather than a
	// status of zero, so it is reported as absent and the wording is read after
	// all -- the same treatment an error of an unknown type gets.
	code := errorStatusCode(err)
	return code, code > 0
}

// errorStatusCode returns the HTTP status an error carries, or zero.
func errorStatusCode(err error) int {
	// The one that exposes a method: AWS smithy, under Bedrock. Matching it
	// structurally also covers every service error that wraps in it.
	var withMethod interface{ HTTPStatusCode() int }
	if errors.As(err, &withMethod) {
		return withMethod.HTTPStatusCode()
	}
	// Providers that drive raw HTTP through lingo itself.
	var own *HTTPStatusError
	if errors.As(err, &own) {
		return own.StatusCode
	}
	var pplx *perplexity.APIError
	if errors.As(err, &pplx) {
		return pplx.StatusCode
	}
	// Vendor SDK error types.
	var oai *openai.Error
	if errors.As(err, &oai) {
		return oai.StatusCode
	}
	var ant *anthropic.Error
	if errors.As(err, &ant) {
		return ant.StatusCode
	}
	var coh *cohereCore.APIError
	if errors.As(err, &coh) {
		return coh.StatusCode
	}
	// genai returns its APIError by value, and spells the status Code.
	var gen genai.APIError
	if errors.As(err, &gen) {
		return gen.Code
	}
	return 0
}

// containsIsolatedNumber reports whether s contains num as a run of digits that
// is not part of a longer number. "status 429," contains 429; "142900 tokens",
// "must be <= 4296" and "ft:gpt-4o:acme:run-4299" do not.
func containsIsolatedNumber(s, num string) bool {
	for i := 0; i+len(num) <= len(s); {
		idx := strings.Index(s[i:], num)
		if idx == -1 {
			return false
		}
		start := i + idx
		end := start + len(num)
		beforeOK := start == 0 || s[start-1] < '0' || s[start-1] > '9'
		afterOK := end == len(s) || s[end] < '0' || s[end] > '9'
		if beforeOK && afterOK {
			return true
		}
		i = start + 1
	}
	return false
}

// isRateLimitError reports whether an error is worth sending the same request
// for a second time.
//
// The order is deliberate: an error that classifies itself is believed, then a
// status the server stated decides on its own, and only an error that offers
// neither is read for its wording. Generation requests are neither free nor
// idempotent, so a guess that says "retry" costs the caller four full prompts
// and four charges to arrive at the 400 the first attempt already had.
func isRateLimitError(err error) bool {
	if err == nil {
		return false
	}

	// An error that classifies itself is believed before its wording is read:
	// *HTTPStatusError carries the status a provider that drives raw HTTP saw,
	// which is a fact, where the scan below is a guess about prose.
	var classified interface{ IsRateLimited() bool }
	if errors.As(err, &classified) && classified.IsRateLimited() {
		return true
	}

	// A stated status is the whole answer, in both directions. Falling through
	// to the prose scan after reading a 400 would put the guess back in charge
	// of the case it gets wrong.
	if status, ok := errorHTTPStatus(err); ok {
		return retryableStatuses[status]
	}

	errStr := strings.ToLower(err.Error())

	// Reached only by errors that carry no status: a transport failure, a
	// gateway that answered in prose, a provider yet to be typed above. These
	// are phrases rather than fragments -- a bare number is checked separately
	// so that it has to stand alone.
	rateLimitIndicators := []string{
		"rate limit",
		"rate_limit",
		"ratelimit",
		"too many requests",
		"quota exceeded",
		"quota_exceeded",
		"overloaded",
		"throttl",
	}

	for _, indicator := range rateLimitIndicators {
		if strings.Contains(errStr, indicator) {
			return true
		}
	}

	return containsIsolatedNumber(errStr, "429")
}

// extractRetryAfter attempts to extract a Retry-After duration from an error
func extractRetryAfter(err error) time.Duration {
	if err == nil {
		return 0
	}

	errStr := err.Error()

	// Look for patterns like "retry after X seconds" or "retry-after: X"
	patterns := []string{
		"retry after ",
		"retry-after: ",
		"retry_after=",
		"retry_after_ms=",
	}

	for _, pattern := range patterns {
		idx := strings.Index(strings.ToLower(errStr), pattern)
		if idx == -1 {
			continue
		}

		// Extract the number after the pattern
		start := idx + len(pattern)
		end := start
		for end < len(errStr) && (errStr[end] >= '0' && errStr[end] <= '9' || errStr[end] == '.') {
			end++
		}

		if end > start {
			if val, parseErr := strconv.ParseFloat(errStr[start:end], 64); parseErr == nil {
				// Check if it's milliseconds
				if strings.Contains(pattern, "ms") {
					return time.Duration(val) * time.Millisecond
				}
				return time.Duration(val) * time.Second
			}
		}
	}

	return 0
}

// HTTPStatusError wraps an HTTP status code error
type HTTPStatusError struct {
	StatusCode int
	Message    string
}

func (e *HTTPStatusError) Error() string {
	return e.Message
}

// IsRateLimited returns true if the status code indicates rate limiting
func (e *HTTPStatusError) IsRateLimited() bool {
	return e.StatusCode == http.StatusTooManyRequests
}

// ============================================================================
// SDK-LEVEL RETRY
// ============================================================================
//
// Three of the vendored SDKs retry on their own before lingo ever sees an
// error. openai-go and anthropic-sdk-go both default to MaxRetries: 2, which is
// three attempts (openai-go@v3.52.0 internal/requestconfig/requestconfig.go:269
// and anthropic-sdk-go@v1.63.1 internal/requestconfig/requestconfig.go:173),
// and aws-sdk-go-v2@v1.43.6 defaults to DefaultMaxAttempts = 3
// (aws/retry/standard.go:29).
//
// Left alone those multiply with the loop above instead of adding to it: three
// SDK attempts inside each of lingo's four is twelve upstream requests for one
// Generate. The extra delay is the SDK's own Retry-After backoff, which
// RateLimitConfig cannot reach, and it is spent inside a timeout every provider
// applies once around the whole retry sequence -- so a throttled account
// usually surfaces as a deadline rather than as the 429 that caused it.
//
// So retry is split rather than stacked. lingo owns retryableStatuses, because
// RateLimitConfig is the caller's only handle on them and the loop above is
// where the caller's backoff, jitter and Retry-After live. The SDK keeps
// everything else it retries -- connection failures, 408, 409, 500, 502, 504 --
// which lingo does not retry at all and which would simply stop being retried
// if the SDK's own retrying were switched off wholesale.

// suppressStainlessRetry stops openai-go and anthropic-sdk-go from retrying the
// statuses lingo retries itself.
//
// Both are Stainless-generated and both consult the x-should-retry response
// header ahead of their own status rules (openai-go requestconfig.go:385-390,
// anthropic-sdk-go requestconfig.go:262-267), so writing the header on the way
// out is how a caller declines a retry without also declining the ones it
// wants kept. Their option.Middleware and option.MiddlewareNext are aliases for
// the same plain function types, so this one value satisfies both.
//
// Nothing about the request or the response body changes: the header is
// metadata each SDK reads and drops before the caller sees the response, and a
// status lingo does not retry is left entirely to the SDK. The header is
// written unconditionally for those statuses rather than only when absent,
// because on them lingo is the layer that decides.
func suppressStainlessRetry(req *http.Request, next func(*http.Request) (*http.Response, error)) (*http.Response, error) {
	res, err := next(req)
	if res != nil && retryableStatuses[res.StatusCode] {
		res.Header.Set("x-should-retry", "false")
	}
	return res, err
}

// lingoOwnedRetryer is the same split for aws-sdk-go-v2, which has no header
// convention but does let the retryer be replaced.
//
// It defers to the SDK for every question except whether an error is worth
// another attempt, and answers that one "no" exactly where lingo's own loop
// would answer "yes" -- a Bedrock ThrottlingException is a 429 and a
// ServiceUnavailableException a 503, both of which isRateLimitError reads off
// the smithy status rather than out of the message.
type lingoOwnedRetryer struct {
	aws.Retryer
}

// IsErrorRetryable declines the errors lingo retries and forwards the rest.
func (r lingoOwnedRetryer) IsErrorRetryable(err error) bool {
	if isRateLimitError(err) {
		return false
	}
	return r.Retryer.IsErrorRetryable(err)
}

// GetAttemptToken forwards to the wrapped retryer's own RetryerV2 method when
// it has one, so wrapping a retryer does not quietly downgrade it to the
// deprecated token API the SDK falls back on.
func (r lingoOwnedRetryer) GetAttemptToken(ctx context.Context) (func(error) error, error) {
	if v2, ok := r.Retryer.(aws.RetryerV2); ok {
		return v2.GetAttemptToken(ctx)
	}
	return r.Retryer.GetInitialToken(), nil
}
