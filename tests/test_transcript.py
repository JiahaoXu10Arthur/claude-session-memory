import json
import os
import time

import transcript as T

SID = "11111111-2222-3333-4444-555555555555"


def t(hms):
    return f"2026-09-25T{hms}.000Z"


def row(type_, ts, content=None, **kw):
    r = {"type": type_, "timestamp": ts, "sessionId": SID, "cwd": "/Users/x/Desktop/proj", **kw}
    if content is not None:
        r["message"] = {"role": type_, "content": content}
    return r


def human(ts, text, **kw):
    return row("user", ts, text, origin={"kind": "human"}, **kw)


def assistant(ts, text):
    return row("assistant", ts, [{"type": "text", "text": text}])


def tool_result(ts, content):
    return row("user", ts, [{"type": "tool_result", "tool_use_id": "t1", "content": content}])


def summary(ts):
    return row("system", ts, subtype="stop_hook_summary")


def queued(ts, prompt, mode="prompt"):
    return row("attachment", ts, attachment={"type": "queued_command", "prompt": prompt, "commandMode": mode})


def write(path, rows, tail=b""):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False).encode() + b"\n")
        f.write(tail)
    return str(path)


def offsets(path):
    """每行的字节偏移"""
    out, o = [], 0
    for line in open(path, "rb"):
        out.append(o)
        o += len(line)
    return out


def test_is_start_rules():
    assert T.is_start(human(t("10:00:00"), "hi"))
    assert not T.is_start(human(t("10:00:00"), "hi", isCompactSummary=True))
    assert not T.is_start(row("user", t("10:00:00"), "hi", origin={"kind": "peer"}))
    assert not T.is_start(row("user", t("10:00:00"), "hi"))  # 没 origin 的 user 行（tool 结果之类）
    goal = "<command-name>/goal</command-name>\n<command-args>tests green</command-args>"
    assert T.is_start(row("user", t("10:00:00"), goal))
    assert not T.is_start(row("user", t("10:00:00"), goal, isMeta=True))
    assert T.is_start(queued(t("10:00:00"), "可以"))
    assert not T.is_start(queued(t("10:00:00"), "<task-notification>", mode="task-notification"))


def test_pieces_drops_meta_rows_compact_summaries_images_and_truncates_tool_output():
    assert T.pieces(row("user", t("10:00:00"), "skill text", isMeta=True)) == []
    assert T.pieces(human(t("10:00:00"), "This session is being continued...", isCompactSummary=True)) == []
    img = tool_result(t("10:00:00"), [{"type": "image", "source": {"data": "AAAA"}}, {"type": "text", "text": "seen"}])
    assert T.pieces(img) == ['[{"type": "text", "text": "seen"}]']
    assert T.pieces(tool_result(t("10:00:00"), "x" * 3000)) == ["x" * 2000]
    assert T.pieces(queued(t("10:00:00"), "可以")) == ["可以"]
    assert T.pieces(queued(t("10:00:00"), "<task-notification>", mode="task-notification")) == []
    assert T.pieces(row("attachment", t("10:00:00"), attachment={"type": "hook_success", "content": "memory：..."})) == []


def test_rows_skips_half_line_and_other_sessions(tmp_path):
    other = dict(human(t("10:00:00"), "from parent"), sessionId="other")
    path = write(tmp_path / f"{SID}.jsonl", [other, human(t("10:00:01"), "mine")], tail=b'{"type": "user", "sess')
    rs = T.rows(path, SID)
    assert [r["message"]["content"] for _, _, r in rs] == ["mine"]
    o = offsets(path)
    assert rs[0][0] == o[1] and rs[0][1] == o[2]  # (这行的偏移, 下一行的偏移)


def test_next_turn_cuts_at_last_summary_and_reports_end(tmp_path):
    rows = [row("attachment", t("09:59:59"), attachment={"type": "environment"}),  # 起点之前的行不压
            human(t("10:00:00"), "first prompt"), assistant(t("10:00:05"), "answer one"), summary(t("10:00:06")),
            human(t("10:01:00"), "second prompt"), assistant(t("10:01:05"), "working...")]  # 还没 Stop
    path = write(tmp_path / f"{SID}.jsonl", rows)
    o = offsets(path)
    turn = T.next_turn(path, 0)
    assert (turn.session_id, turn.start, turn.created_at, turn.project) == (SID, o[1], t("10:00:00"), "proj")
    assert (turn.prompt, turn.continuation) == ("first prompt", False)
    assert turn.text == "first prompt\nanswer one"
    assert turn.end == o[4]  # summary 行之后、下一个起点的偏移
    assert T.next_turn(path, turn.end) == o[4]  # 第二个 turn 还没 Stop：没得压，游标停在 summary 之后
    assert T.next_turn(path, 0) == turn  # 游标不动就再切出同一个
    assert T.next_turn(path, os.path.getsize(path)) == os.path.getsize(path)  # 游标已经过了 summary：原样返回
    assert T.started_at(path) == t("09:59:59")
    fork = write(tmp_path / "fork.jsonl", rows[:2] + [{"type": "custom-title", "sessionId": "fork"}])  # 带着父 session 的行
    assert T.started_at(fork) is None


def test_next_turn_continuation_keeps_key_and_is_flagged(tmp_path):
    rows = [human(t("10:00:00"), "P" * 400), assistant(t("10:00:05"), "part one"), summary(t("10:00:06")),
            row("user", t("10:02:00"), "<task-notification>done</task-notification>", origin={"kind": "task-notification"}),
            assistant(t("10:02:05"), "part two"), summary(t("10:02:06"))]
    path = write(tmp_path / f"{SID}.jsonl", rows[:3])  # Stop 当时只写到 summary
    first = T.next_turn(path, 0)
    assert first.text == "P" * 400 + "\npart one" and first.end == os.path.getsize(path)
    write(tmp_path / f"{SID}.jsonl", rows)  # task-notification 叫醒模型接着干，又一次 Stop
    second = T.next_turn(path, first.end)
    assert (second.start, second.created_at, second.prompt, second.continuation) == (first.start, first.created_at, "P" * 400, True)
    assert second.text == "<task-notification>done</task-notification>\npart two"
    assert second.end == os.path.getsize(path)
    assert T.next_turn(path, second.end) == second.end


def test_next_turn_skips_empty_segments_and_needs_a_summary(tmp_path):
    rows = [human(t("10:00:00"), "only prompt"), summary(t("10:00:01")),
            row("attachment", t("10:00:02"), attachment={"type": "hook_success"}),  # summary 后的空段
            human(t("10:01:00"), "again"), assistant(t("10:01:05"), "ok"), summary(t("10:01:06")),
            row("custom-title", t("10:01:07"))]  # summary 之后的元数据行，永远压不到
    path = write(tmp_path / f"{SID}.jsonl", rows[:3])  # 第一次 Stop 时读到这里
    o = offsets(path)
    first = T.next_turn(path, 0)
    assert first.text == "only prompt" and first.end == o[2]
    write(tmp_path / f"{SID}.jsonl", rows)
    o = offsets(path)
    second = T.next_turn(path, first.end)  # 游标停在 summary 后：hook_success 那段是空的，跳过去
    assert second.text == "again\nok" and second.start == o[3]
    assert T.next_turn(path, second.end) == o[6]
    assert T.next_turn(write(tmp_path / f"{SID}.jsonl", rows[:1]), 7) == 7  # 一行 summary 都没有：游标不动
    sdk = write(tmp_path / f"{SID}.jsonl", [row("user", t("10:00:00"), "claude -p prompt"), summary(t("10:00:01"))])
    assert T.next_turn(sdk, 0) == os.path.getsize(sdk)  # 有 summary 没起点（claude -p）：不压，游标到底


def spawn(ts, tool_use_id):
    return row("assistant", ts, [{"type": "tool_use", "id": tool_use_id, "name": "Agent", "input": {}}])


def test_next_subagent_waits_for_parent_summary_and_idle_file(tmp_path, monkeypatch):
    parent_rows = [human(t("10:00:00"), "spawn one"), spawn(t("10:00:01"), "toolu_1"),
                   queued(t("10:00:02"), "和上一句一起"),  # tool_use 之后、subagent 第一行之前插进来的排队 prompt
                   summary(t("10:05:00")),
                   human(t("10:10:00"), "spawn two"), spawn(t("10:10:01"), "toolu_2")]  # 第二轮还没 Stop
    parent = write(tmp_path / f"{SID}.jsonl", parent_rows)
    o = offsets(parent)

    def sub(name, ts, tool_use_id=None):
        path = write(tmp_path / SID / "subagents" / f"{name}.jsonl",
                     [row("user", ts, "task brief", isSidechain=True), assistant(ts, "report")])
        if tool_use_id:
            (tmp_path / SID / "subagents" / f"{name}.meta.json").write_text(json.dumps({"toolUseId": tool_use_id}))
        return path

    a = sub("agent-a", t("10:00:03"), "toolu_1")
    b = sub("agent-b", t("10:10:30"), "toolu_2")
    c = sub("agent-c", t("10:00:03"))  # 没有 meta：按时间归，落到排队那句
    e = sub("agent-e", t("10:00:03"), "toolu_nested")  # 嵌套的 subagent：meta 里的 tool_use 不在父文件里，也按时间归
    wf = write(tmp_path / SID / "subagents" / "workflows" / "wf_1" / "agent-d.jsonl",
               [row("user", t("10:00:40"), "wf brief", isSidechain=True), assistant(t("10:00:40"), "wf report")])
    monkeypatch.setattr(T, "SUBAGENT_IDLE", 0)
    turn = T.next_subagent(a, 0)
    assert (turn.session_id, turn.start, turn.prompt, turn.continuation) == (SID, o[0], "spawn one", True)
    assert turn.text == "task brief\nreport" and turn.end == os.path.getsize(a)
    assert T.next_subagent(a, offsets(a)[1]).text == "report"  # 游标之前的行不再压
    assert T.next_subagent(c, 0).start == o[2]
    assert T.next_subagent(e, 0).start == o[2]
    assert T.next_subagent(wf, 0).start == o[2]
    assert T.next_subagent(b, 7) == 7  # 父的最后一个 summary 早于它的最后一行：没写完，游标不动
    empty = write(tmp_path / SID / "subagents" / "agent-f.jsonl", [row("user", t("10:00:03"), "meta", isMeta=True)])
    assert T.next_subagent(empty, 0) == os.path.getsize(empty)  # 写完了但没正文：游标到底
    monkeypatch.setattr(T, "SUBAGENT_IDLE", 3600)
    assert T.next_subagent(a, 7) == 7  # 刚改过
    assert T.parent_of(wf) == parent


def test_next_subagent_of_a_parent_without_starts_is_skipped(tmp_path, monkeypatch):
    parent = write(tmp_path / f"{SID}.jsonl", [row("user", t("10:00:00"), "claude -p"), summary(t("10:05:00"))])
    a = write(tmp_path / SID / "subagents" / "agent-a.jsonl", [assistant(t("10:00:30"), "report")])
    monkeypatch.setattr(T, "SUBAGENT_IDLE", 0)
    assert T.next_subagent(a, 0) == os.path.getsize(a)


def test_windows_prefixes_prompt_head_from_second_window(monkeypatch):
    monkeypatch.setattr(T, "WINDOW", 10)
    monkeypatch.setattr(T, "MAX_WINDOWS", 3)
    turn = T.Turn(SID, 0, t("10:00:00"), "proj", "P" * 400, "a" * 10 + "b" * 10 + "c" * 10 + "d" * 10, 1, False)
    assert T.windows(turn) == ["a" * 10, "P" * 300 + "\n" + "b" * 10, "P" * 300 + "\n" + "c" * 10]  # 第 4 段丢掉
    turn.continuation = True
    assert T.windows(turn)[0] == "P" * 300 + "\n" + "a" * 10


def test_project_is_the_session_first_cwd_with_worktrees_named_after_repo():
    rs = lambda *cwds: [(0, 1, {"cwd": c}) for c in cwds]
    assert T.project([(0, 1, {"type": "custom-title"}), *rs("/Users/x/Desktop/proj", "/Users/x/Desktop/proj/scratch")]) == "proj"
    assert T.project(rs("/Users/x/Desktop/claimscope/.claude/worktrees/agitated-hypatia-7daa8b")) == "claimscope/agitated-hypatia-7daa8b"
    assert T.project([]) == ""


def test_sweep_lists_recent_files_newest_first_without_excluded_dirs(tmp_path):
    root = tmp_path / "projects"
    old = write(root / "-p" / "old.jsonl", [human(t("10:00:00"), "x")])
    main = write(root / "-p" / f"{SID}.jsonl", [human(t("10:00:00"), "x")])
    sub = write(root / "-p" / SID / "subagents" / "agent-a.jsonl", [human(t("10:00:00"), "x")])
    wf = write(root / "-p" / SID / "subagents" / "workflows" / "wf_1" / "agent-b.jsonl", [human(t("10:00:00"), "x")])
    write(root / "-Users-x--claude-mem-observer-sessions" / "o.jsonl", [human(t("10:00:00"), "x")])
    now = time.time()
    os.utime(old, (now - 3 * 86400, now - 3 * 86400))
    os.utime(sub, (now - 30, now - 30))
    os.utime(main, (now - 20, now - 20))
    os.utime(wf, (now - 10, now - 10))
    assert T.sweep(str(root), 2) == [wf, main, sub]
