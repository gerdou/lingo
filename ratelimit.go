package lingo

import (
	"context"
	"math/rand"
	"net/http"
	"strconv"
	"strings"
	"time"
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

	for attempt := 0; attempt <= r.config.MaxRetries; attempt++ {
		// Check if context is cancelled before attempting
		select {
		case <-ctx.Done():
			return ctx.Err()
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
		if attempt >= r.config.MaxRetries {
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
			Int("max_retries", r.config.MaxRetries).
			Str("wait_duration", waitDuration.String()).
			Msg("Rate limited, waiting before retry")

		// Wait with context cancellation support
		select {
		case <-ctx.Done():
			return ctx.Err()
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

// isRateLimitError checks if an error is a rate limit error
func isRateLimitError(err error) bool {
	if err == nil {
		return false
	}

	errStr := strings.ToLower(err.Error())

	// Check for common rate limit indicators
	rateLimitIndicators := []string{
		"rate limit",
		"rate_limit",
		"ratelimit",
		"too many requests",
		"429",
		"quota exceeded",
		"quota_exceeded",
		"overloaded",
		"capacity",
		"throttl",
	}

	for _, indicator := range rateLimitIndicators {
		if strings.Contains(errStr, indicator) {
			return true
		}
	}

	return false
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
