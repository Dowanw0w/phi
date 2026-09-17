package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"iter"
	"net/http"
	"strings"

	"github.com/pulseaiclub/phi/internal/llm"
	"github.com/pulseaiclub/phi/internal/util"
)

const (
	chatCompletionsPath = "/chat/completions"
	// finishReasonLength is the OpenAI finish reason for a completion that hit
	// the output cap: the content is a prefix, not a finished answer.
	finishReasonLength = "length"
)

type streamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

type apiTool struct {
	Type     string             `json:"type"`
	Function llm.ToolDefinition `json:"function"`
}

type apiMessage struct {
	Role             llm.Role       `json:"role"`
	Content          any            `json:"content"` // string, or []contentPart when images are attached
	ReasoningContent string         `json:"reasoning_content,omitempty"`
	ToolCalls        []llm.ToolCall `json:"tool_calls,omitempty"`
	ToolCallID       string         `json:"tool_call_id,omitempty"`
}

// Request is the OpenAI-compatible chat completions body.
type Request struct {
	Model           string         `json:"model"`
	Messages        []apiMessage   `json:"messages"`
	Tools           []apiTool      `json:"tools,omitempty"`
	Stream          bool           `json:"stream,omitempty"`
	StreamOptions   *streamOptions `json:"stream_options,omitempty"`
	ExtraBody       *ExtraBody     `json:"extra_body,omitempty"`
	ReasoningEffort string         `json:"reasoning_effort,omitempty"`
	// MaxTokens caps the completion length. Compaction sets it so a runaway
	// summary cannot outgrow the context the summary is meant to free.
	MaxTokens int `json:"max_tokens,omitempty"`
}

// ExtraBody holds vendor extensions nested under extra_body (e.g. DeepSeek).
type ExtraBody struct {
	Thinking *ThinkingConfig `json:"thinking,omitempty"`
}

// ThinkingConfig is a vendor thinking toggle inside ExtraBody. ClearThinking is
// a z.ai addition: false keeps earlier reasoning in context ("preserved
// thinking"), which interleaved tool calling needs; nil omits the field so
// providers that do not know it are unaffected.
type ThinkingConfig struct {
	Type          string `json:"type"`
	ClearThinking *bool  `json:"clear_thinking,omitempty"`
}

type streamChunk struct {
	Choices []streamChoice `json:"choices"`
	Usage   *llm.Usage     `json:"usage,omitempty"`
}

type streamChoice struct {
	Delta   llm.StreamDelta `json:"delta"`
	Message *llm.Message    `json:"message,omitempty"`
}

// toAPIMessage converts a normalized message. With attached images the
// content becomes the OpenAI content-parts array (text + image_url data
// URLs); otherwise it stays a plain string, keeping the wire shape stable.
func toAPIMessage(m llm.Message) apiMessage {
	out := apiMessage{
		Role:             m.Role,
		ReasoningContent: m.ReasoningContent,
		ToolCalls:        m.ToolCalls,
		ToolCallID:       m.ToolCallID,
	}
	if len(m.Images) == 0 {
		out.Content = m.Content
		return out
	}
	parts := make([]any, 0, len(m.Images)+1)
	if m.Content != "" {
		parts = append(parts, map[string]string{"type": "text", "text": m.Content})
	}
	for _, img := range m.Images {
		parts = append(parts, map[string]any{
			"type": "image_url",
			"image_url": map[string]string{
				"url": "data:" + img.MimeType + ";base64," + img.Data,
			},
		})
	}
	out.Content = parts
	return out
}

// BuildRequest converts the normalized messages into an OpenAI-shaped request.
// The system prompt is prepended as a system message, mirroring the previous
// in-client behavior. Vendor-specific fields (e.g. DeepSeek extra_body) belong
// on model presets via RequestInterceptor, not here.
func BuildRequest(cfg llm.ModelConfig, system string, messages []llm.Message, tools []llm.ToolDefinition) *Request {
	msgs := make([]apiMessage, 0, len(messages)+1)
	if strings.TrimSpace(system) != "" {
		msgs = append(msgs, apiMessage{Role: llm.RoleSystem, Content: system})
	}
	for _, m := range messages {
		msgs = append(msgs, toAPIMessage(m))
	}

	apiTools := make([]apiTool, len(tools))
	for i, t := range tools {
		apiTools[i] = apiTool{Type: "function", Function: t}
	}

	var reasoningEffort string
	if cfg.Think.Enabled {
		reasoningEffort = string(cfg.Think.Mode)
	}

	return &Request{
		Model:           cfg.Name,
		Messages:        msgs,
		Tools:           apiTools,
		Stream:          true,
		StreamOptions:   &streamOptions{IncludeUsage: true},
		ReasoningEffort: reasoningEffort,
	}
}

func chatCompletionsURL(baseURL string) string {
	if strings.HasSuffix(baseURL, chatCompletionsPath) {
		return baseURL
	}
	return baseURL + chatCompletionsPath
}

func newChatRequest(ctx context.Context, url, apiKey string, body []byte, stream bool) (*http.Request, error) {
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+apiKey)
	if stream {
		httpReq.Header.Set("Accept", util.ContentEventStream)
	}
	return httpReq, nil
}

// NewCompactRequest builds a minimal non-streaming chat body for Compact.
func NewCompactRequest(model, prompt string, maxTokens int) *Request {
	return &Request{
		Model:     model,
		Messages:  []apiMessage{{Role: llm.RoleUser, Content: prompt}},
		MaxTokens: maxTokens,
	}
}

// CompactRequest POSTs a non-streaming chat request body and returns assistant text.
func CompactRequest(
	ctx context.Context,
	httpClient *http.Client,
	cfg llm.ModelConfig,
	req *Request,
) (llm.CompactResult, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return llm.CompactResult{}, err
	}

	httpReq, err := newChatRequest(ctx, chatCompletionsURL(cfg.BaseURL), cfg.APIKey, body, false)
	if err != nil {
		return llm.CompactResult{}, err
	}

	httpResp, err := util.DoWithRetry(httpClient, httpReq)
	if err != nil {
		return llm.CompactResult{}, err
	}
	defer httpResp.Body.Close()

	respBody, err := io.ReadAll(httpResp.Body)
	if err != nil {
		return llm.CompactResult{}, err
	}
	if httpResp.StatusCode != http.StatusOK {
		return llm.CompactResult{}, llm.FormatAPIError("LLM", httpResp.StatusCode, respBody)
	}

	var resp struct {
		Choices []struct {
			Message      llm.Message `json:"message"`
			FinishReason string      `json:"finish_reason"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(respBody, &resp); err != nil {
		return llm.CompactResult{}, err
	}
	if len(resp.Choices) == 0 {
		return llm.CompactResult{}, errors.New("LLM API error: empty choices")
	}
	return llm.CompactResult{
		Text:      resp.Choices[0].Message.Content,
		Truncated: resp.Choices[0].FinishReason == finishReasonLength,
	}, nil
}

// StreamChatCompletion POSTs a streaming chat completion and yields normalized events.
func StreamChatCompletion(
	ctx context.Context,
	httpClient *http.Client,
	baseURL string,
	apiKey string,
	payload any,
) iter.Seq2[llm.StreamEvent, error] {
	return func(yield func(llm.StreamEvent, error) bool) {
		body, err := json.Marshal(payload)
		if err != nil {
			yield(llm.StreamEvent{}, err)
			return
		}

		httpReq, err := newChatRequest(ctx, chatCompletionsURL(baseURL), apiKey, body, true)
		if err != nil {
			yield(llm.StreamEvent{}, err)
			return
		}

		httpResp, err := util.DoWithRetry(httpClient, httpReq)
		if err != nil {
			yield(llm.StreamEvent{}, err)
			return
		}
		defer httpResp.Body.Close()

		if httpResp.StatusCode != http.StatusOK {
			respBody, _ := io.ReadAll(httpResp.Body)
			yield(llm.StreamEvent{}, llm.FormatAPIError("LLM", httpResp.StatusCode, respBody))
			return
		}

		var usage llm.Usage
		acc := newStreamAccumulator()

		for data, parseErr := range util.ParseDataStream(httpResp.Body) {
			if parseErr != nil {
				yield(llm.StreamEvent{}, parseErr)
				return
			}
			payloadLine := bytes.TrimSpace(data)
			if len(payloadLine) == 0 {
				continue
			}
			if bytes.Equal(payloadLine, []byte("[DONE]")) {
				break
			}
			decodeData := data
			if bytes.Contains(decodeData, []byte("\t")) {
				decodeData = bytes.ReplaceAll(decodeData, []byte("\t"), []byte(" "))
			}

			var chunk streamChunk
			if err := json.Unmarshal(decodeData, &chunk); err != nil {
				continue
			}
			if chunk.Usage != nil {
				usage = *chunk.Usage
			}
			if len(chunk.Choices) == 0 {
				continue
			}

			sc := chunk.Choices[0]
			delta := sc.Delta
			acc.applyDelta(delta)
			if sc.Message != nil {
				acc.applyMessage(sc.Message)
			}

			if hasStreamDelta(delta, sc.Message) {
				if !yield(llm.StreamEvent{
					Type:  llm.StreamEventTypeDelta,
					Delta: delta,
					Usage: usage,
				}, nil) {
					return
				}
			}
		}

		msg := acc.message()
		msg.Usage = usage
		yield(llm.StreamEvent{Type: llm.StreamEventTypeDone, Final: &msg}, nil)
	}
}

func hasStreamDelta(delta llm.StreamDelta, msg *llm.Message) bool {
	if delta.Content != "" || delta.ReasoningContent != "" || delta.Role != "" || len(delta.ToolCalls) > 0 {
		return true
	}
	if msg == nil {
		return false
	}
	return strings.TrimSpace(msg.Content) != "" ||
		strings.TrimSpace(msg.ReasoningContent) != "" ||
		len(msg.ToolCalls) > 0
}
