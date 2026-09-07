package gemini

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/pulseaiclub/phi/internal/llm"
)

func TestGetStreamURL(t *testing.T) {
	testcases := []struct {
		name    string
		apiKey  string
		model   string
		baseURL string
		expect  string
	}{
		{
			name:    "with_api_key",
			model:   "gemini-2.5",
			apiKey:  "sk-xxxx123456789",
			baseURL: "https://generativelanguage.googleapis.com/v1beta",
			expect:  "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5:streamGenerateContent?alt=sse&key=sk-xxxx123456789",
		},
		{
			name:    "without_api_key",
			model:   "gemini-2.5",
			baseURL: "https://aiplatform.googleapis.com",
			expect:  "https://aiplatform.googleapis.com/models/gemini-2.5:streamGenerateContent?alt=sse",
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			url := getStreamURL(tc.model, tc.baseURL, tc.apiKey)
			assert.Equal(t, tc.expect, url)
		})
	}
}

func TestGetURLNonStreaming(t *testing.T) {
	url := getURL("gemini-2.5", "https://generativelanguage.googleapis.com/v1beta", "key123", false)
	assert.Equal(
		t,
		"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5:generateContent?key=key123",
		url,
	)
}

func TestBuildRequestSystemInstructionWireName(t *testing.T) {
	req := BuildRequest("be brief", []llm.Message{{Role: llm.RoleUser, Content: "hi"}}, nil)
	body, err := json.Marshal(req)
	require.NoError(t, err)
	// Gemini REST API expects camelCase; snake_case would be ignored or rejected.
	assert.Contains(t, string(body), `"systemInstruction"`)
	assert.NotContains(t, string(body), "system_instruction")
}

func TestBuildRequestMergesToolResults(t *testing.T) {
	messages := []llm.Message{
		{Role: llm.RoleUser, Content: "hi"},
		{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{{ID: "read", Function: llm.Function{Name: "read"}}}},
		{Role: llm.RoleTool, ToolCallID: "read", Content: "a"},
		{Role: llm.RoleTool, ToolCallID: "write", Content: "b"},
	}
	req := BuildRequest("", messages, nil)
	require.Len(t, req.Contents, 3)
	last := req.Contents[2]
	assert.Equal(t, "user", last.Role)
	require.Len(t, last.Parts, 2)
	assert.Equal(t, "a", last.Parts[0].FunctionResponse.Response.(map[string]any)["output"])
	assert.Equal(t, "b", last.Parts[1].FunctionResponse.Response.(map[string]any)["output"])
}

func TestBuildRequestUserContent(t *testing.T) {
	req := BuildRequest("", []llm.Message{{
		Role:    llm.RoleUser,
		Content: "what is this",
		Images:  []llm.Image{{MimeType: "image/png", Data: "aGVsbG8="}},
	}}, nil)
	require.Len(t, req.Contents, 1)
	parts := req.Contents[0].Parts
	require.Len(t, parts, 2)
	assert.Equal(t, "what is this", parts[0].Text)
	require.NotNil(t, parts[1].InlineData)
	assert.Equal(t, "image/png", parts[1].InlineData.MIMEType)
	assert.Equal(t, "aGVsbG8=", parts[1].InlineData.Data)
}

func TestDisableThinking(t *testing.T) {
	testcases := []struct {
		name  string
		model string
		want  string
	}{
		{name: "gemini_2x_budget_zero", model: "gemini-2.5-flash", want: `"thinkingBudget":0`},
		{name: "gemini_3_pro_lowest_level", model: "gemini-3-pro", want: `"thinkingLevel":"LOW"`},
		{name: "gemini_3_flash_minimal", model: "gemini-3-flash", want: `"thinkingLevel":"MINIMAL"`},
	}
	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			var req GeminiRequest
			req.DisableThinking(tc.model)
			require.NotNil(t, req.ThinkingConfig)
			body, err := json.Marshal(req.ThinkingConfig)
			require.NoError(t, err)
			assert.Contains(t, string(body), tc.want)
		})
	}
}

// processForTest runs processStream and returns the yielded events.
func processForTest(sse string) []llm.StreamEvent {
	var events []llm.StreamEvent
	processStream(strings.NewReader(sse), func(ev llm.StreamEvent, err error) bool {
		if err != nil {
			events = append(events, llm.StreamEvent{Type: llm.StreamEventTypeError, Err: err.Error()})
			return false
		}
		events = append(events, ev)
		return true
	})
	return events
}

func TestProcessStreamTextAndUsage(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"candidates":[{"content":{"parts":[{"text":"Hello"}]}}]}`,
		"",
		`data: {"candidates":[{"content":{"parts":[{"text":" world"}]}}]}`,
		"",
		`data: {"usageMetadata":{"promptTokenCount":12,"candidatesTokenCount":7,"cachedContentTokenCount":5,"thoughtsTokenCount":3,"totalTokenCount":22}}`,
		"",
	}, "\n")

	events := processForTest(sse)

	var text strings.Builder
	var done *llm.StreamEvent
	for _, ev := range events {
		if ev.Type == llm.StreamEventTypeError {
			t.Fatalf("stream error: %s", ev.Err)
		}
		switch ev.Type {
		case llm.StreamEventTypeDelta:
			text.WriteString(ev.Delta.Content)
		case llm.StreamEventTypeDone:
			done = &ev
		}
	}

	require.NotNil(t, done, "expected done event")
	msg := done.Partial.Choices[0].Message
	assert.Equal(t, "Hello world", text.String())
	assert.Equal(t, "Hello world", msg.Content)
	assert.Equal(t, 7, done.Partial.Usage.PromptTokens)
	assert.Equal(t, 10, done.Partial.Usage.CompletionTokens)
	assert.Equal(t, 22, done.Partial.Usage.TotalTokens)
}

func TestProcessStreamThinking(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"candidates":[{"content":{"parts":[{"thought":true,"text":"let me think"}]}}]}`,
		"",
		`data: {"candidates":[{"content":{"parts":[{"text":"answer"}]}}]}`,
		"",
	}, "\n")

	events := processForTest(sse)

	var text, reasoning strings.Builder
	var done *llm.StreamEvent
	for _, ev := range events {
		if ev.Type == llm.StreamEventTypeError {
			t.Fatalf("stream error: %s", ev.Err)
		}
		switch ev.Type {
		case llm.StreamEventTypeDelta:
			text.WriteString(ev.Delta.Content)
			reasoning.WriteString(ev.Delta.ReasoningContent)
		case llm.StreamEventTypeDone:
			done = &ev
		}
	}

	require.NotNil(t, done, "expected done event")
	msg := done.Partial.Choices[0].Message
	assert.Equal(t, "answer", text.String())
	assert.Equal(t, "let me think", reasoning.String())
	assert.Equal(t, "answer", msg.Content)
	assert.Equal(t, "let me think", msg.ReasoningContent)
}

func TestCompactUsesNonStreamingEndpoint(t *testing.T) {
	var gotPath string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"candidates":[{"content":{"parts":[{"text":"summary"}]}}]}`)
	}))
	defer srv.Close()

	out, err := Compact(
		t.Context(),
		srv.Client(),
		llm.ModelConfig{Name: "gemini-2.5-flash", BaseURL: srv.URL, APIKey: "k"},
		"summarize",
	)
	require.NoError(t, err)
	assert.Equal(t, "summary", out)
	assert.Equal(t, "/models/gemini-2.5-flash:generateContent", gotPath)
}
