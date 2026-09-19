package diffreview

import (
	"context"
	"fmt"
	"os/exec"
	"strings"

	"github.com/pulseaiclub/phi/internal/util/gitx"
)

// LabelForSpec is the overlay title suffix for a /diff argument list.
func LabelForSpec(spec []string) string {
	if len(spec) == 0 {
		return "working tree"
	}
	switch strings.ToLower(spec[0]) {
	case "staged", "--staged", "--cached":
		return "staged"
	case "head":
		return "HEAD"
	default:
		return strings.Join(spec, " ")
	}
}

// EmptyNote explains an empty diff for spec. Untracked files never appear in
// `git diff` output, which is the usual reason a reviewer sees nothing.
func EmptyNote(spec []string) string {
	if len(spec) == 0 {
		return "No unstaged changes. Untracked files are not shown — run git status."
	}
	switch strings.ToLower(spec[0]) {
	case "staged", "--staged", "--cached":
		return "No staged changes. Index matches HEAD; untracked files are not shown."
	case "head":
		return "HEAD has no changes to show."
	default:
		return fmt.Sprintf("No changes vs %s. Untracked files are not shown.", LabelForSpec(spec))
	}
}

// GitArgv is the git command that produces a unified diff for spec.
func GitArgv(spec []string) []string {
	argv := []string{"git", "-c", "color.ui=never"}
	if len(spec) == 0 {
		return append(argv, "diff")
	}
	switch strings.ToLower(spec[0]) {
	case "staged", "--staged", "--cached":
		return append(argv, append([]string{"diff", "--staged"}, spec[1:]...)...)
	case "head":
		return append(argv, "show", "--pretty=medium", "-p", "HEAD")
	case "--":
		if len(spec) == 1 {
			return append(argv, "diff")
		}
		return spec[1:]
	default:
		return append(argv, append([]string{"diff"}, spec...)...)
	}
}

// LoadGit runs git in cwd and returns unified-diff text. Errors are formatted
// via gitx.FormatError so the status bar sees one line plus a hint.
func LoadGit(ctx context.Context, cwd string, spec []string) (string, error) {
	argv := GitArgv(spec)
	cmd := exec.CommandContext(ctx, argv[0], argv[1:]...) //nolint:gosec // G204: git from GitArgv, or /diff -- argv
	if cwd != "" {
		cmd.Dir = cwd
	}
	out, err := cmd.CombinedOutput()
	text := string(out)
	if err == nil || looksLikeDiff(text) {
		return text, nil
	}
	return "", gitx.FormatError(argv, text, err)
}

func looksLikeDiff(text string) bool {
	return strings.Contains(text, "\ndiff --git ") || strings.HasPrefix(text, "diff --git ") ||
		strings.HasPrefix(text, "commit ")
}
