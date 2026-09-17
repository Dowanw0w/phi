package llm

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestMarshalToolParams(t *testing.T) {
	assert.Equal(t, `{}`, string(MarshalToolParams(nil, "{}")))
	assert.JSONEq(t, `{"type":"object"}`, string(MarshalToolParams(nil, `{"type":"object"}`)))
	raw := MarshalToolParams(&FunctionParameters{Type: "object", Properties: Object{}}, "{}")
	assert.Contains(t, string(raw), `"type":"object"`)
}

func TestAssistantDone(t *testing.T) {
	ev := AssistantDone("hi", "think", []ToolCall{{ID: "1", Function: Function{Name: "read"}}}, Usage{TotalTokens: 3})
	assert.Equal(t, StreamEventTypeDone, ev.Type)
	assert.NotNil(t, ev.Final)
	assert.Equal(t, "hi", ev.Final.Content)
	assert.Equal(t, "think", ev.Final.ReasoningContent)
	assert.Equal(t, 3, ev.Final.Usage.TotalTokens)
	assert.Len(t, ev.Final.ToolCalls, 1)
}
