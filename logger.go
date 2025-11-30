package lingo

import (
	"github.com/rs/zerolog"
)

// ZerologAdapter adapts zerolog.Logger to our Logger interface
type ZerologAdapter struct {
	logger zerolog.Logger
}

// NewZerologAdapter creates a new adapter for zerolog
func NewZerologAdapter(logger zerolog.Logger) *ZerologAdapter {
	return &ZerologAdapter{logger: logger}
}

func (z *ZerologAdapter) Debug() LogEvent {
	return &zerologEvent{event: z.logger.Debug()}
}

func (z *ZerologAdapter) Info() LogEvent {
	return &zerologEvent{event: z.logger.Info()}
}

func (z *ZerologAdapter) Error() LogEvent {
	return &zerologEvent{event: z.logger.Error()}
}

type zerologEvent struct {
	event *zerolog.Event
}

func (e *zerologEvent) Msg(msg string) {
	e.event.Msg(msg)
}

func (e *zerologEvent) Str(key, val string) LogEvent {
	e.event = e.event.Str(key, val)
	return e
}

func (e *zerologEvent) Int(key string, val int) LogEvent {
	e.event = e.event.Int(key, val)
	return e
}

func (e *zerologEvent) Int64(key string, val int64) LogEvent {
	e.event = e.event.Int64(key, val)
	return e
}

func (e *zerologEvent) Bool(key string, val bool) LogEvent {
	e.event = e.event.Bool(key, val)
	return e
}

func (e *zerologEvent) Err(err error) LogEvent {
	e.event = e.event.Err(err)
	return e
}

// NopLogger is a no-op logger that discards all logs
type NopLogger struct{}

func (n *NopLogger) Debug() LogEvent { return &nopEvent{} }
func (n *NopLogger) Info() LogEvent  { return &nopEvent{} }
func (n *NopLogger) Error() LogEvent { return &nopEvent{} }

type nopEvent struct{}

func (e *nopEvent) Msg(msg string)                 {}
func (e *nopEvent) Str(key, val string) LogEvent   { return e }
func (e *nopEvent) Int(key string, val int) LogEvent { return e }
func (e *nopEvent) Int64(key string, val int64) LogEvent { return e }
func (e *nopEvent) Bool(key string, val bool) LogEvent { return e }
func (e *nopEvent) Err(err error) LogEvent         { return e }

