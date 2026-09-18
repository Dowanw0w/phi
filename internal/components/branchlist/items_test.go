package branchlist

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/pulseaiclub/phi/internal/components/listpicker"
	"github.com/pulseaiclub/phi/internal/util/gitx"
)

func TestItemsAnchorsCurrentThenRecentThenLocalThenRemote(t *testing.T) {
	branches := []gitx.Branch{
		{Name: "main", Current: true},
		{Name: "fix/agent-wait", Subject: "wip"},
		{Name: "feature/old"},
		{Name: "origin/main", Remote: true},
	}

	items := Items(branches, []string{"feature/old", "main"})

	assert.Equal(t, []string{"main", "feature/old", "fix/agent-wait", "origin/main"}, primaries(items),
		"current anchors the list, then recent, then local by commit date, remotes last")
	assert.Len(t, items, 4, "no branch is listed twice")
}

func TestItemsIgnoresRecentNamesThatNoLongerExist(t *testing.T) {
	branches := []gitx.Branch{{Name: "main", Current: true}}

	items := Items(branches, []string{"deleted-branch", "main"})

	assert.Equal(t, []string{"main"}, primaries(items))
}

func TestRowsAreNameAndLastCommitOnly(t *testing.T) {
	tests := []struct {
		name    string
		branch  gitx.Branch
		leading string
		detail  string
	}{
		{
			name: "current branch",
			branch: gitx.Branch{
				Name: "main", Current: true, Upstream: "origin/main",
				Ahead: 2, Behind: 3, Committed: "3 hours ago", Subject: "chore: bump",
			},
			leading: "●",
			detail:  "chore: bump",
		},
		{
			name:   "diverged branch keeps the counts out of the row",
			branch: gitx.Branch{Name: "fix/y", Upstream: "origin/fix/y", Ahead: 2, Behind: 3, Subject: "wip"},
			detail: "wip",
		},
		{
			name:   "deleted upstream keeps the badge out of the row",
			branch: gitx.Branch{Name: "stale", Upstream: "origin/stale", Gone: true, Subject: "old"},
			detail: "old",
		},
		{
			name:   "remote branch",
			branch: gitx.Branch{Name: "origin/main", Remote: true, Subject: "chore: bump"},
			detail: "chore: bump",
		},
		{
			name:   "no commits yet",
			branch: gitx.Branch{Name: "empty"},
			detail: "",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			item := Items([]gitx.Branch{tc.branch}, nil)[0]
			assert.Equal(t, tc.branch.LocalName(), item.ID,
				"accepting a remote row checks out its local branch")
			assert.Equal(t, tc.branch.Name, item.Primary)
			assert.Equal(t, tc.leading, item.Leading)
			assert.Equal(t, tc.detail, item.Detail)
			assert.Empty(t, item.Badge, "two columns means no badge column")
			assert.Contains(t, item.Keywords, tc.branch.Name)
		})
	}
}

func TestRemoteRowAcceptsTheTrackingBranch(t *testing.T) {
	item := Items([]gitx.Branch{{Name: "origin/fix/one", Remote: true, Subject: "wip"}}, nil)[0]

	assert.Equal(t, "fix/one", item.ID)
	assert.Equal(t, "origin/fix/one", item.Primary, "the row still says where it lives")
}

func TestConfigKeepsBranchNamesReadable(t *testing.T) {
	cfg := Config()
	assert.Equal(t, "Branches", cfg.Title)
	assert.GreaterOrEqual(t, cfg.PrimaryWidth, len("fix/agent-wait-timeout"))
	assert.Contains(t, cfg.Hint, "switch")
}

// primaries is the display order; IDs collapse a remote row onto its local
// tracking branch, so they are asserted separately.
func primaries(items []listpicker.Item) []string {
	out := make([]string, 0, len(items))
	for _, item := range items {
		out = append(out, item.Primary)
	}
	return out
}
