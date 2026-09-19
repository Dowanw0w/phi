package diffreview

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func newGitRepo(t *testing.T) string {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not installed")
	}
	dir := t.TempDir()
	runGit(t, dir, "git", "init", "--template=")
	runGit(t, dir, "git", "config", "user.email", "t@t")
	runGit(t, dir, "git", "config", "user.name", "t")
	return dir
}

func runGit(t *testing.T, dir string, args ...string) {
	t.Helper()
	cmd := exec.CommandContext(t.Context(), args[0], args[1:]...)
	cmd.Dir = dir
	cmd.Env = append(
		os.Environ(),
		"GIT_AUTHOR_NAME=t",
		"GIT_AUTHOR_EMAIL=t@t",
		"GIT_COMMITTER_NAME=t",
		"GIT_COMMITTER_EMAIL=t@t",
	)
	out, err := cmd.CombinedOutput()
	if err != nil {
		if strings.Contains(string(out), "Operation not permitted") {
			t.Skipf("sandbox blocked git: %s", out)
		}
		require.NoError(t, err, string(out))
	}
}

func TestLoadGitWorkingTree(t *testing.T) {
	dir := newGitRepo(t)
	require.NoError(t, os.WriteFile(filepath.Join(dir, "a.txt"), []byte("old\n"), 0o644))
	runGit(t, dir, "git", "add", "a.txt")
	runGit(t, dir, "git", "commit", "-m", "init")
	require.NoError(t, os.WriteFile(filepath.Join(dir, "a.txt"), []byte("new\n"), 0o644))

	text, err := LoadGit(t.Context(), dir, nil)
	require.NoError(t, err)
	assert.Contains(t, text, "diff --git")
	assert.Contains(t, text, "-old")
	assert.Contains(t, text, "+new")

	_, err = Parse(text)
	require.NoError(t, err)
}

func TestLoadGitMissingRepo(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not installed")
	}
	_, err := LoadGit(t.Context(), t.TempDir(), nil)
	require.Error(t, err)
	// git answers with a warning plus the whole `git diff` usage dump; the status
	// bar gets the warning and the next step, not 4 KB of options.
	assert.Contains(t, err.Error(), "Not a git repository")
	assert.Contains(t, err.Error(), "this directory is not a git repository")
	assert.NotContains(t, err.Error(), "usage:")
	assert.NotContains(t, err.Error(), "\n")
	assert.Less(t, len(err.Error()), 300)
}

func TestLoadGitUnknownRevision(t *testing.T) {
	dir := newGitRepo(t)
	require.NoError(t, os.WriteFile(filepath.Join(dir, "a.txt"), []byte("x\n"), 0o644))
	runGit(t, dir, "git", "add", "a.txt")
	runGit(t, dir, "git", "commit", "-m", "init")

	_, err := LoadGit(t.Context(), dir, []string{"deadbeef"})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "deadbeef")
	assert.Contains(t, err.Error(), "fetch first")
	assert.NotContains(t, err.Error(), "\n")
}
