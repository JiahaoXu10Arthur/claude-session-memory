import os
import subprocess

import docs


def write(root, rel, text, mtime=None):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    if mtime is not None:
        os.utime(p, (mtime, mtime))
    return str(p)


def test_sections_split_on_headings_and_keep_the_heading_line(tmp_path):
    p = write(tmp_path, ".claude/adr/0001-a.md", "# Title one\n\nbody one\n\n## Sub\n\nbody two\n")
    out = docs.sections(p, str(tmp_path))
    assert [o["title"] for o in out] == [".claude/adr/0001-a.md § Title one", ".claude/adr/0001-a.md § Sub"]
    assert out[0]["narrative"] == "# Title one\n\nbody one" and out[1]["narrative"] == "## Sub\n\nbody two"
    o = out[0]
    assert o["type"] == "doc" and o["session_id"] == "" and o["subtitle"] == "" and o["facts"] == []
    assert o["files"] == [".claude/adr/0001-a.md"] and o["project"] == tmp_path.name and "id" not in o
    assert o["created_at"].endswith("Z")


def test_sections_text_before_first_heading_is_a_preface_and_h4_stays_in_parent(tmp_path):
    p = write(tmp_path, "docs/x.md", "intro line\n\n# A\n\na\n\n#### deep\n\nstill a\n")
    out = docs.sections(p, str(tmp_path))
    assert [o["title"].split(" § ")[1] for o in out] == ["前言", "A"]
    assert "#### deep" in out[1]["narrative"]


def test_sections_ignore_hashes_inside_fenced_code_and_drop_empty_sections(tmp_path):
    p = write(tmp_path, "docs/x.md", "# A\n\n```bash\n# not a heading\necho hi\n```\n\n# B\n\n# C\n\nc\n")
    out = docs.sections(p, str(tmp_path))
    assert [o["title"].split(" § ")[1] for o in out] == ["A", "C"]  # B 只有标题行，丢掉
    assert "# not a heading" in out[0]["narrative"]


def test_sections_over_2000_chars_split_on_blank_lines_with_the_same_title(tmp_path):
    para = "x" * 900
    p = write(tmp_path, "docs/x.md", "# A\n\n" + "\n\n".join([para] * 4) + "\n")
    out = docs.sections(p, str(tmp_path))
    assert len(out) >= 2 and {o["title"] for o in out} == {"docs/x.md § A"}
    assert all(len(o["narrative"]) <= 2000 for o in out)


def test_files_cover_the_doc_dirs_and_skip_superseded(tmp_path):
    a = write(tmp_path, ".claude/adr/0001.md", "# a\n\nStatus: accepted\n")
    write(tmp_path, ".claude/adr/0002.md", "# b\n\nStatus: superseded by ADR-0003\n")
    s = write(tmp_path, ".claude/specs/s.md", "# s\n")
    d = write(tmp_path, ".claude/design/d.md", "# d\n")
    r = write(tmp_path, "docs/adr/0009.md", "# r\n")
    m = write(tmp_path, "docs/readme.md", "# m\n")
    write(tmp_path, "docs/superpowers/plans/p.md", "# p\n")
    write(tmp_path, ".claude/findings/f.md", "# f\n")
    write(tmp_path, "CLAUDE.md", "# c\n")
    assert docs.files(str(tmp_path)) == sorted([a, s, d, r, m])


def test_mtimes_and_stale(tmp_path):
    a = write(tmp_path, ".claude/adr/0001.md", "# a\n", mtime=1_700_000_000)
    known = docs.mtimes(str(tmp_path))
    assert known == {a: os.stat(a).st_mtime_ns} and not docs.stale(str(tmp_path), known)
    os.utime(a, (1_700_000_001, 1_700_000_001))
    assert docs.stale(str(tmp_path), known)
    write(tmp_path, ".claude/adr/0002.md", "# b\n")
    assert docs.stale(str(tmp_path), docs.mtimes(str(tmp_path)) | {a: 0})
    assert docs.stale(str(tmp_path), {**docs.mtimes(str(tmp_path)), str(tmp_path / "gone.md"): 1})


def git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def test_repo_root_gives_the_main_checkout_for_a_worktree_and_none_otherwise(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git("init", "-q", cwd=repo)
    git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "--allow-empty", "-m", "init", cwd=repo)
    write(repo, ".claude/adr/0001.md", "# a\n")
    wt = tmp_path / "wt"
    git("worktree", "add", "-q", str(wt), "-b", "x", cwd=repo)
    assert docs.repo_root(str(repo)) == str(repo)
    assert docs.repo_root(str(repo / ".claude")) == str(repo)
    assert docs.repo_root(str(wt)) == str(repo)
    assert docs.repo_root(str(tmp_path)) is None  # 不是 repo
    bare = tmp_path / "nodocs"
    bare.mkdir()
    git("init", "-q", cwd=bare)
    assert docs.repo_root(str(bare)) is None  # 是 repo，没有文档目录
