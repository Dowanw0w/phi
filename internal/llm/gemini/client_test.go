package gemini

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestGetStreamURL(t *testing.T) {
	testcases := []struct {
		name    string
		apiKey  string
		model   string
		baseUrl string
		expect  string
	}{
		{
			name:    "",
			apiKey:  "",
			model:   "",
			baseUrl: "",
			expect:  "",
		},
		{
			name:   "",
			apiKey: "",
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			url := getStreamURL(tc.model, tc.baseUrl, tc.apiKey)
			assert.Equal(t, tc.expect, url)
		})
	}
}
