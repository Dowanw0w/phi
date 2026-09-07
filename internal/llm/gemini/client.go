package gemini

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"net/http"
	"net/url"
	"strings"

	"github.com/pulseaiclub/phi/internal/llm"
	"github.com/pulseaiclub/phi/internal/util"
)

const defaultBaseURL = "https://generativelanguage.googleapis.com/v1beta"

type part struct {
	Text             string            `json:"text,omitempty"`
	Thought          bool              `json:"thought,omitempty"`
	InlineData       *inlineData       `json:"inlineData,omitempty"`
	FunctionCall     *functionCall     `json:"functionCall,omitempty"`
	FunctionResponse *functionResponse `json:"functionResponse,omitempty"`
}

type inlineData struct {
	MIMEType string `json:"mimeType"`
	Data     string `json:"data"`
}

type functionCall struct {
	Name string         `json:"name"`
	Args map[string]any `json:"args,omitempty"`
}

type functionResponse struct {
	Name     string `json:"name"`
	Response any    `json:"response"`
}

type content struct {
	Role  string `json:"role,omitempty"`
	Parts []part `json:"parts"`
}

type functionDeclaration struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

type thinkingConfig struct {
	ThinkingBudget *int   `json:"thinkingBudget,omitempty"`
	ThinkingLevel  string `json:"thinkingLevel,omitempty"`
}

type GeminiRequest struct {
	SystemInstruction *content  `json:"systemInstruction,omitempty"`
	Contents          []content `json:"contents"`
	Tools             []struct {
		FunctionDeclarations []functionDeclaration `json:"functionDeclarations"`
	} `json:"tools,omitempty"`
	// ThinkingConfig opts Gemini 2.x+ out of default thinking so reasoning text
	// stays out of the visible stream unless the model cannot disable it.
	ThinkingConfig *thinkingConfig `json:"thinkingConfig,omitempty"`
}

// DisableThinking sets thinkingConfig for models that think by default.
// Gemini 2.x accepts thinkingBudget 0; Gemini 3 Pro cannot disable thinking and
// Flash/Flash-Lite have no full off, so use the lowest level without
// includeThoughts to keep hidden reasoning invisible.
func (req *GeminiRequest) DisableThinking(model string) {
	m := strings.ToLower(model)
	switch {
	case strings.Contains(m, "gemini-3") && strings.Contains(m, "pro"):
		req.ThinkingConfig = &thinkingConfig{ThinkingLevel: "LOW"}
	case strings.Contains(m, "gemini-3"):
		req.ThinkingConfig = &thinkingConfig{ThinkingLevel: "MINIMAL"}
	default:
		// Zero disables thinking; a pointer keeps it on the wire (omitempty would drop it).
		zero := 0
		req.ThinkingConfig = &thinkingConfig{ThinkingBudget: &zero}
	}
}

func BuildRequest(
	system string,
	messages []llm.Message,
	tools []llm.ToolDefinition,
) GeminiRequest {
	var req GeminiRequest

	if strings.TrimSpace(system) != "" {
		req.SystemInstruction = &content{
			Parts: []part{
				{
					Text: system,
				},
			},
		}
	}

	for _, m := range messages {
		switch m.Role {
		case llm.RoleUser:
			parts := make([]part, 0, len(m.Images)+1)
			if m.Content != "" {
				parts = append(parts, part{Text: m.Content})
			}
			for _, img := range m.Images {
				parts = append(parts, part{InlineData: &inlineData{MIMEType: img.MimeType, Data: img.Data}})
			}
			req.Contents = append(req.Contents, content{Role: "user", Parts: parts})
		case llm.RoleTool:
			name := "tool"
			if m.ToolCallID != "" {
				name = m.ToolCallID
			}
			resp := part{
				FunctionResponse: &functionResponse{
					Name:     name,
					Response: map[string]any{"output": m.Content},
				},
			}
			// Gemini requires alternating turns; merge consecutive tool results into
			// the same user content so parallel tool calls produce a single user turn.
			if last := lastFunctionResponseContent(req.Contents); last != nil {
				last.Parts = append(last.Parts, resp)
			} else {
				req.Contents = append(req.Contents, content{Role: "user", Parts: []part{resp}})
			}
		case llm.RoleAssistant:
			c := toAssistantMessage(m)
			req.Contents = append(req.Contents, c)
		}
	}
	req.Tools = []struct {
		FunctionDeclarations []functionDeclaration `json:"functionDeclarations"`
	}{{FunctionDeclarations: toToolMessage(tools)}}
	return req
}

// lastFunctionResponseContent returns the trailing user content holding
// functionResponse parts, if any, so subsequent tool results can be merged into it.
func lastFunctionResponseContent(contents []content) *content {
	if len(contents) == 0 {
		return nil
	}
	last := &contents[len(contents)-1]
	if last.Role != "user" {
		return nil
	}
	for _, p := range last.Parts {
		if p.FunctionResponse != nil {
			return last
		}
	}
	return nil
}

func toToolMessage(tools []llm.ToolDefinition) []functionDeclaration {
	decls := make([]functionDeclaration, 0, len(tools))
	for _, t := range tools {
		params, _ := json.Marshal(t.Params)
		if len(params) == 0 || bytes.Equal(bytes.TrimSpace(params), []byte("null")) {
			params = json.RawMessage(`{"type":"object"}`)
		}
		decls = append(decls, functionDeclaration{Name: t.Name, Description: t.Description, Parameters: params})
	}
	return decls
}

func toAssistantMessage(m llm.Message) content {
	c := content{Role: "model"}
	if m.Content != "" {
		c.Parts = append(c.Parts, part{Text: m.Content})
	}
	for _, tc := range m.ToolCalls {
		var args map[string]any
		_ = json.Unmarshal([]byte(tc.Function.Arguments), &args)
		c.Parts = append(c.Parts, part{FunctionCall: &functionCall{Name: tc.Function.Name, Args: args}})
	}
	if len(c.Parts) == 0 {
		c.Parts = []part{{Text: ""}}
	}
	return c
}

func getURL(model, baseURL, apiKey string, stream bool) string {
	if strings.TrimSpace(baseURL) == "" {
		baseURL = defaultBaseURL
	}
	baseURL = strings.TrimRight(baseURL, "/")
	if model == "" {
		model = "gemini-2.5-flash"
	}
	if !strings.Contains(model, "/") {
		model = "models/" + model
	}
	action := ":generateContent"
	if stream {
		action = ":streamGenerateContent"
	}
	u, _ := url.Parse(baseURL + "/" + model + action)
	q := u.Query()
	if stream {
		q.Set("alt", "sse")
	}
	if apiKey != "" && !strings.Contains(baseURL, "aiplatform.googleapis.com") {
		q.Set("key", apiKey)
	}
	u.RawQuery = q.Encode()
	return u.String()
}

func getStreamURL(model, baseURL, apiKey string) string {
	return getURL(model, baseURL, apiKey, true)
}

func Stream(
	ctx context.Context,
	client *http.Client,
	config llm.ModelConfig,
	req *GeminiRequest,
) iter.Seq2[llm.StreamEvent, error] {
	return func(yield func(llm.StreamEvent, error) bool) {
		// Gemini 2.x+ thinks by default; opt out unless reasoning was requested.
		req.DisableThinking(config.Name)
		body, err := json.Marshal(req)
		if err != nil {
			yield(llm.StreamEvent{}, err)
			return
		}

		httpReq, err := http.NewRequestWithContext(
			ctx,
			http.MethodPost,
			getStreamURL(config.Name, config.BaseURL, config.APIKey), bytes.NewReader(body),
		)
		if err != nil {
			yield(llm.StreamEvent{}, err)
			return
		}

		httpReq.Header.Set("Content-Type", "application/json")
		if strings.Contains(strings.ToLower(config.BaseURL), "aiplatform.googleapis.com") && config.APIKey != "" {
			httpReq.Header.Set("Authorization", "Bearer "+config.APIKey)
		}

		httpResp, err := util.DoWithRetry(client, httpReq)
		if err != nil {
			yield(llm.StreamEvent{}, err)
			return
		}
		defer httpResp.Body.Close()
		if httpResp.StatusCode != http.StatusOK {
			raw, _ := io.ReadAll(httpResp.Body)
			yield(llm.StreamEvent{}, fmt.Errorf("gemini API error: (%d) %s", httpResp.StatusCode, string(raw)))
			return
		}
		processStream(httpResp.Body, yield)
	}
}

type chunk struct {
	Candidates []struct {
		Content struct {
			Parts []part `json:"parts"`
		} `json:"content"`
	} `json:"candidates"`

	UsageMetadata struct {
		PromptTokenCount        int `json:"promptTokenCount"`
		CandidatesTokenCount    int `json:"candidatesTokenCount"`
		CachedContentTokenCount int `json:"cachedContentTokenCount"`
		ThoughtsTokenCount      int `json:"thoughtsTokenCount"`
		TotalTokenCount         int `json:"totalTokenCount"`
	} `json:"usageMetadata"`
}

func processStream(body io.Reader, yield func(llm.StreamEvent, error) bool) {
	var (
		text      strings.Builder
		reasoning strings.Builder
		toolCalls []llm.ToolCall
		usage     llm.Usage
	)
	for data, parseErr := range util.ParseDataStream(body) {
		if parseErr != nil {
			yield(llm.StreamEvent{Type: llm.StreamEventTypeError, Err: parseErr.Error()}, parseErr)
			return
		}
		line := bytes.TrimSpace(data)
		if len(line) == 0 {
			continue
		}

		var ck chunk
		if err := json.Unmarshal(line, &ck); err != nil {
			continue
		}

		u := ck.UsageMetadata
		if u.TotalTokenCount > 0 || u.PromptTokenCount > 0 {
			// Cached tokens are counted in promptTokenCount; split them out so
			// input is net of cache reads.
			usage.PromptTokens = u.PromptTokenCount - u.CachedContentTokenCount
			usage.CompletionTokens = u.CandidatesTokenCount + u.ThoughtsTokenCount
			usage.TotalTokens = u.TotalTokenCount
			if u.CachedContentTokenCount > 0 {
				usage.PromptTokensDetails = &llm.PromptTokensDetails{CachedTokens: u.CachedContentTokenCount}
			}
		}

		for _, cand := range ck.Candidates {
			for _, p := range cand.Content.Parts {
				// Thinking parts (Gemini 2.x+ with thoughts enabled) must not leak
				// into visible content; surface them as reasoning instead.
				if p.Thought && p.Text != "" {
					reasoning.WriteString(p.Text)
					if !yield(llm.StreamEvent{
						Type:  llm.StreamEventTypeDelta,
						Delta: llm.StreamDelta{ReasoningContent: p.Text},
					}, nil) {
						return
					}
					continue
				}
				if p.Text != "" {
					text.WriteString(p.Text)
					if !yield(llm.StreamEvent{
						Type:  llm.StreamEventTypeDelta,
						Delta: llm.StreamDelta{Content: p.Text},
					}, nil) {
						return
					}
				}

				if p.FunctionCall != nil && p.FunctionCall.Name != "" {
					call := toLLMToolCall(p, len(toolCalls))
					toolCalls = append(toolCalls, call)
					if !yield(llm.StreamEvent{
						Type:  llm.StreamEventTypeDelta,
						Delta: llm.StreamDelta{ToolCalls: []llm.ToolCall{call}},
					}, nil) {
						return
					}
				}
			}
		}
	}
	yield(llm.StreamEvent{
		Type: llm.StreamEventTypeDone,
		Partial: llm.Response{
			Choices: []llm.Choice{{
				Message: llm.Message{
					Role:             llm.RoleAssistant,
					Content:          text.String(),
					ReasoningContent: reasoning.String(),
					ToolCalls:        toolCalls,
				},
			}},
			Usage: usage,
		},
	}, nil)
}

func toLLMToolCall(p part, index int) llm.ToolCall {
	args, _ := json.Marshal(p.FunctionCall.Args)
	toolCall := llm.ToolCall{
		Index: index,
		ID:    p.FunctionCall.Name,
		Type:  "function",
		Function: llm.Function{
			Name:      p.FunctionCall.Name,
			Arguments: string(args),
		},
	}
	return toolCall
}

func Compact(
	ctx context.Context,
	client *http.Client,
	cfg llm.ModelConfig,
	prompt string,
) (string, error) {
	req := BuildRequest("", []llm.Message{{Role: llm.RoleUser, Content: prompt}}, nil)
	req.DisableThinking(cfg.Name)
	body, err := json.Marshal(req)
	if err != nil {
		return "", nil
	}
	request, err := http.NewRequestWithContext(
		ctx,
		http.MethodPost,
		// Non-streaming endpoint: the whole body is a single JSON response.
		getURL(cfg.Name, cfg.BaseURL, cfg.APIKey, false),
		bytes.NewReader(body),
	)
	if err != nil {
		return "", nil
	}
	request.Header.Set("Content-Type", "application/json")
	if strings.Contains(strings.ToLower(cfg.BaseURL), "aiplatform.googleapis.com") && cfg.APIKey != "" {
		request.Header.Set("Authorization", "Bearer "+cfg.APIKey)
	}
	httpResp, err := util.DoWithRetry(client, request)
	if err != nil {
		return "", err
	}
	defer httpResp.Body.Close()
	raw, err := io.ReadAll(httpResp.Body)
	if err != nil {
		return "", err
	}
	if httpResp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("gemini API error: (%d) %s", httpResp.StatusCode, string(raw))
	}
	var resp chunk
	if err := json.Unmarshal(raw, &resp); err != nil {
		return "", err
	}

	var b strings.Builder
	for _, c := range resp.Candidates {
		for _, p := range c.Content.Parts {
			b.WriteString(p.Text)
		}
	}
	if b.Len() == 0 {
		return "", errors.New("gemini API error: empty response")
	}
	return b.String(), nil
}
