package llm

import (
	"bytes"
	"encoding/json"
)

// MarshalToolParams encodes tool parameters for provider wire formats.
// empty is used when Params is nil/null (Anthropic wants "{}", Gemini wants
// `{"type":"object"}`).
func MarshalToolParams(params *FunctionParameters, empty string) json.RawMessage {
	if empty == "" {
		empty = "{}"
	}
	raw, err := json.Marshal(params)
	if err != nil || len(raw) == 0 || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return json.RawMessage(empty)
	}
	return raw
}

// AssistantDone builds the terminal stream event for an assistant turn.
func AssistantDone(content, reasoning string, tools []ToolCall, usage Usage) StreamEvent {
	return StreamEvent{
		Type: StreamEventTypeDone,
		Final: &Message{
			Role:             RoleAssistant,
			Content:          content,
			ReasoningContent: reasoning,
			ToolCalls:        tools,
			Usage:            usage,
		},
	}
}
