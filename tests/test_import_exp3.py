import json
import sqlite3
import tomllib

import numpy as np
import pytest

import core
import embed
import import_exp3
from import_exp3 import (
    find_starts,
    pieces,
    read_turns,
    session_turns,
    user_text,
    window_observation_ids,
)

BEFORE = "2026-08-26T18:00:00.000Z"  # T0 之前
AFTER = "2026-09-24T16:00:00.000Z"  # CUT 之后


def t(hms):
    """窗口里 9/1 这一天的时间"""
    return f"2026-09-01T{hms}.000Z"


def user(ts, content, **kw):
    return {"type": "user", "timestamp": ts, "message": {"role": "user", "content": content}, **kw}


def tool_result(ts, content):
    return user(ts, [{"type": "tool_result", "tool_use_id": "t1", "content": content}])


def assistant(ts, *blocks):
    return {"type": "assistant", "timestamp": ts,
            "message": {"role": "assistant", "content": list(blocks)}}


def say(text):
    return {"type": "text", "text": text}


def attachment(ts, **a):
    return {"type": "attachment", "timestamp": ts, "attachment": a}


def world(tmp_path, prompts, transcripts, sessions=(), observations=()):
    """造一个小的 claude-mem 库和 transcript 目录。prompts 是 (session, prompt_number, prompt_text, created_at)；
    transcripts 是 {相对路径: 各行}"""
    src = str(tmp_path / "claude-mem.db")
    conn = sqlite3.connect(src)
    conn.execute("CREATE TABLE user_prompts (content_session_id, prompt_number, prompt_text, created_at)")
    conn.execute("CREATE TABLE sdk_sessions (content_session_id, memory_session_id)")
    conn.execute("CREATE TABLE observations (id, memory_session_id, created_at)")
    conn.executemany("INSERT INTO user_prompts VALUES (?, ?, ?, ?)", prompts)
    conn.executemany("INSERT INTO sdk_sessions VALUES (?, ?)", sessions)
    conn.executemany("INSERT INTO observations VALUES (?, ?, ?)", observations)
    conn.commit()
    conn.close()
    root = tmp_path / "transcripts"
    root.mkdir()
    for rel, rows in transcripts.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return src, str(root)


def test_pieces_keep_answer_thinking_tool_input_and_tool_output():
    row = assistant(t("10:00:01"), {"type": "thinking", "thinking": "想一下"}, say("先看日志"),
                    {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls 日志"}})
    assert pieces(row) == ["想一下", "先看日志", 'Bash {"command": "ls 日志"}']
    assert pieces(tool_result(t("10:00:02"), "a.log")) == ["a.log"]
    assert pieces(user(t("10:00:00"), "看一下日志")) == ["看一下日志"]
    assert pieces(user(t("10:00:00"), [say("看"), {"type": "image", "source": {}}])) == ["看"]


def test_pieces_cut_each_tool_output_to_2000_chars_but_not_tool_input():
    assert pieces(tool_result(t("10:00:02"), "x" * 2500)) == ["x" * 2000]
    blocks = [say("y" * 3000)]
    assert pieces(tool_result(t("10:00:02"), blocks)) == [json.dumps(blocks)[:2000]]
    row = assistant(t("10:00:01"), {"type": "tool_use", "name": "Write", "input": {"content": "z" * 3000}})
    assert pieces(row) == ['Write {"content": "' + "z" * 3000 + '"}']


def test_pieces_of_attachments_are_only_queued_prompts():
    assert pieces(attachment(t("10:05:00"), type="queued_command", prompt="再跑一次")) == ["再跑一次"]
    assert pieces(attachment(t("10:05:00"), type="queued_command",
                             prompt=[say("再"), say("跑")])) == ["再跑"]
    assert pieces(attachment(t("10:05:00"), type="hook_success", content="hook 输出")) == []
    assert pieces({"type": "queue-operation", "operation": "enqueue", "timestamp": t("10:04:00"),
                   "content": "再跑一次"}) == []
    assert pieces({"type": "system", "timestamp": t("10:04:00"), "content": "token 提醒"}) == []


def test_user_text_counts_task_notifications_but_not_tool_results_meta_or_sidechain():
    note = "<task-notification>done</task-notification>"
    assert user_text(user(t("10:00:00"), note)) == note
    assert user_text(user(t("10:00:00"), [say("a"), say("b")])) == "ab"
    assert user_text(tool_result(t("10:00:00"), "out")) is None
    assert user_text(user(t("10:00:00"), "skill 正文", isMeta=True)) is None
    assert user_text(user(t("10:00:00"), "subagent 的话", isSidechain=True)) is None
    assert user_text(assistant(t("10:00:00"), say("a"))) is None


def test_find_starts_at_the_same_words_nearest_in_time():
    rows = [user(t("10:00:00"), "continue"), assistant(t("10:00:05"), say("a")),
            user(t("11:00:00"), "continue"), assistant(t("11:00:05"), say("b"))]
    assert find_starts(rows, [(7, "continue", t("11:00:00"))]) == [(2, 7, t("11:00:00"), "continue")]
    spaced = [user(t("10:00:00"), "fix the\n bug  now")]
    assert find_starts(spaced, [(1, "fix the bug now", t("10:00:00"))]) == [
        (0, 1, t("10:00:00"), "fix the bug now")]
    assert find_starts(rows[:2], [(8, "continue", "2026-09-03T10:00:00.000Z")]) == []


def test_find_starts_at_the_queued_command_row():
    rows = [user(t("10:00:00"), "第一句"), assistant(t("10:00:05"), say("a")),
            attachment(t("10:00:03"), type="queued_command", prompt="第二句"),
            assistant(t("10:00:09"), say("b"))]
    prompts = [(1, "第一句", t("10:00:00")), (2, "第二句", t("10:00:03"))]
    assert find_starts(rows, prompts) == [(0, 1, t("10:00:00"), "第一句"),
                                          (2, 2, t("10:00:03"), "第二句")]


def test_find_starts_by_time_when_the_words_differ():
    rows = [user(t("10:00:00"), "[Image #1] 这张图"), assistant(t("10:00:05"), say("a"))]
    assert find_starts(rows, [(1, "这张图", t("10:01:30"))]) == [(0, 1, t("10:01:30"), "这张图")]
    assert find_starts(rows, [(1, "这张图", t("10:02:01"))]) == []


def test_turn_runs_until_the_next_prompt():
    rows = [user(t("10:00:00"), "第一句"), assistant(t("10:00:05"), say("a"), say("  ")),
            user(t("10:10:00"), "第二句"), assistant(t("10:10:05"), say("b"))]
    prompts = [(1, "第一句", t("10:00:00")), (2, "第二句", t("10:10:00"))]
    assert session_turns(rows, prompts, []) == [(t("10:00:00"), 1, "第一句", "第一句\na"),
                                                (t("10:10:00"), 2, "第二句", "第二句\nb")]


def test_a_prompt_not_found_stays_in_the_previous_turn():
    note = "<task-notification>完成</task-notification>"
    rows = [user(t("10:00:00"), "第一句"), assistant(t("10:00:05"), say("a")),
            user(t("10:30:00"), note), assistant(t("10:30:05"), say("b"))]
    prompts = [(1, "第一句", t("10:00:00")), (2, "找不到的一句", t("12:00:00"))]
    assert session_turns(rows, prompts, []) == [
        (t("10:00:00"), 1, "第一句", f"第一句\na\n{note}\nb")]


def test_turn_keeps_only_what_falls_inside_the_window():
    last = "2026-09-24T15:50:00.000Z"
    rows = [user(BEFORE, "窗口前开头"), assistant("2026-08-26T18:10:00.000Z", say("窗口前的回答")),
            assistant("2026-08-26T18:20:00.000Z", say("窗口里的回答")),
            user(last, "最后一句"), assistant("2026-09-24T15:55:00.000Z", say("赶上了")),
            assistant(AFTER, say("窗口后的回答"))]
    prompts = [(1, "窗口前开头", BEFORE), (2, "最后一句", last)]
    assert session_turns(rows, prompts, []) == [(BEFORE, 1, "窗口前开头", "窗口里的回答"),
                                                (last, 2, "最后一句", "最后一句\n赶上了")]
    outside = [user(BEFORE, "全在窗口前"), assistant(BEFORE, say("x")),
               user(AFTER, "全在窗口后"), assistant(AFTER, say("y"))]
    assert session_turns(outside, [(1, "全在窗口前", BEFORE), (2, "全在窗口后", AFTER)], []) == []


def test_a_subagent_goes_whole_into_the_turn_of_its_first_line():
    rows = [user(t("10:00:00"), "第一句"), assistant(t("10:00:05"), say("派 agent")),
            user(t("10:10:00"), "第二句"), assistant(t("10:10:05"), say("b"))]
    prompts = [(1, "第一句", t("10:00:00")), (2, "第二句", t("10:10:00"))]
    agent = [user(t("10:00:06"), "agent 的任务"), assistant(t("10:20:00"), say("agent 干完了")),
             assistant(AFTER, say("窗口后"))]
    late = [user(t("10:10:30"), "第二个 agent")]
    assert session_turns(rows, prompts, [(t("10:00:06"), agent), (t("10:10:30"), late)]) == [
        (t("10:00:00"), 1, "第一句", "第一句\n派 agent\nagent 的任务\nagent 干完了"),
        (t("10:10:00"), 2, "第二句", "第二句\nb\n第二个 agent")]


def test_read_turns_numbers_turns_by_prompt_time_across_sessions(tmp_path):
    long = "b 的 第一句 " + "长" * 70
    src, root = world(
        tmp_path,
        prompts=[("s-a", 1, "a 的第一句", t("10:00:00")), ("s-a", 2, "a 的第二句", t("10:20:00")),
                 ("s-b", 1, "b 的  第一句\n" + "长" * 70, t("10:10:00"))],
        transcripts={
            "-Users-x-p/s-a.jsonl": [user(t("10:00:00"), "a 的第一句"), assistant(t("10:00:05"), say("a1")),
                                     user(t("10:20:00"), "a 的第二句"), assistant(t("10:20:05"), say("a2"))],
            "-Users-x-q/s-b.jsonl": [user(t("10:10:00"), long), assistant(t("10:10:05"), say("b1"))],
        })
    turns = read_turns(src, root)
    assert [(x["id"], x["session_id"], x["narrative"]) for x in turns] == [
        (1000001, "s-a", "a 的第一句\na1"), (1000002, "s-b", f"{long}\nb1"), (1000003, "s-a", "a 的第二句\na2")]
    assert turns[1] == {"id": 1000002, "session_id": "s-b", "project": "", "type": "turn",
                        "title": "b 的 第一句 " + "长" * 52, "subtitle": "", "narrative": f"{long}\nb1",
                        "facts": [], "files": [], "created_at": t("10:10:00")}


def test_read_turns_skips_sessions_without_transcript_or_outside_the_window(tmp_path):
    src, root = world(
        tmp_path,
        prompts=[("s-a", 1, "有 transcript", t("10:00:00")), ("s-gone", 1, "transcript 被删了", t("10:05:00")),
                 ("s-old", 1, "窗口前的 session", BEFORE), ("s-late", 1, "CUT 之后才有的 prompt", AFTER)],
        transcripts={
            "-Users-x-p/s-a.jsonl": [user(t("10:00:00"), "有 transcript"), assistant(t("10:00:05"), say("a1"))],
            "-Users-x-p/s-old.jsonl": [user(BEFORE, "窗口前的 session"), assistant(BEFORE, say("old"))],
            "-Users-x-p/s-late.jsonl": [user(AFTER, "CUT 之后才有的 prompt"), assistant(AFTER, say("late"))],
        })
    assert [x["session_id"] for x in read_turns(src, root)] == ["s-a"]


def test_read_turns_reads_subagents_and_workflow_agents_but_not_journals(tmp_path):
    src, root = world(
        tmp_path,
        prompts=[("s-a", 1, "派两个 agent", t("10:00:00"))],
        transcripts={
            "-Users-x-p/s-a.jsonl": [user(t("10:00:00"), "派两个 agent"), assistant(t("10:00:05"), say("派了"))],
            "-Users-x-p/s-a/subagents/agent-b.jsonl": [user(t("10:00:08"), "subagent 后开始")],
            "-Users-x-p/s-a/subagents/workflows/wf_1/agent-a.jsonl": [user(t("10:00:07"), "workflow agent 先开始")],
            "-Users-x-p/s-a/subagents/workflows/wf_1/journal.jsonl": [user(t("10:00:06"), "journal 不放")],
            "-Users-x-p/s-a/tool-results/r.jsonl": [user(t("10:00:06"), "tool-results 不放")],
        })
    assert read_turns(src, root)[0]["narrative"] == "派两个 agent\n派了\nworkflow agent 先开始\nsubagent 后开始"


def test_claude_mem_is_opened_read_only(tmp_path, monkeypatch):
    src, root = world(tmp_path, prompts=[], transcripts={})
    seen = []
    real_connect = sqlite3.connect

    def spy(*args, **kwargs):
        seen.append((args, kwargs))
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", spy)
    read_turns(src, root)
    window_observation_ids(src, root)
    assert [("mode=ro" in args[0], kwargs.get("uri")) for args, kwargs in seen] == [(True, True)] * 2


def test_window_observation_ids_drop_old_ones_and_sessions_without_transcript(tmp_path):
    src, root = world(
        tmp_path,
        prompts=[("s-a", 1, "有 transcript", t("10:00:00")), ("s-gone", 1, "transcript 被删了", t("10:05:00"))],
        transcripts={"-Users-x-p/s-a.jsonl": [user(t("10:00:00"), "有 transcript")]},
        sessions=[("s-a", "m-a"), ("s-gone", "m-gone")],
        observations=[(1, "m-a", BEFORE), (2, "m-a", t("10:00:30")), (3, "m-gone", t("10:05:30")),
                      (4, "m-other", t("10:06:00")), (37234, "m-a", t("11:00:00"))])
    assert window_observation_ids(src, root) == {2, 4}


class FakeEmbedder:
    dim = 2

    def __init__(self, model_name):
        pass

    def embed(self, texts):
        return np.full((len(texts), 2), 2 ** -0.5, dtype=np.float32)

    def split_text(self, text, size, overlap):
        return [text]


def config(tmp_path, unit):
    path = tmp_path / f"{unit}.toml"
    path.write_text(f'db = "{tmp_path / unit}.db"\nunit = "{unit}"\n\n[embed]\nmodel = "fake"\n\n'
                    '[search]\nmode = "hybrid"\n')
    return str(path)


@pytest.mark.parametrize("done", [0, 1])
def test_main_imports_turns_and_resumes_after_the_last_one(tmp_path, monkeypatch, capsys, done):
    src, root = world(
        tmp_path,
        prompts=[("s-a", n, f"第 {n} 句", t(f"10:0{n}:00")) for n in (1, 2, 3)],
        transcripts={"-Users-x-p/s-a.jsonl": [row for n in (1, 2, 3) for row in (
            user(t(f"10:0{n}:00"), f"第 {n} 句"), assistant(t(f"10:0{n}:05"), say(f"答 {n}")))]})
    monkeypatch.setattr(import_exp3, "SRC", src)
    monkeypatch.setattr(import_exp3, "TRANSCRIPTS", root)
    monkeypatch.setattr(import_exp3, "BATCH", 1)
    monkeypatch.setattr(embed, "Embedder", FakeEmbedder)
    path = config(tmp_path, "turn")
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    core.Memory(cfg).add(read_turns(src, root)[:done])  # 上次断之前灌进去的
    import_exp3.main(path)
    assert capsys.readouterr().out.splitlines() == [f"已灌 {n}/3" for n in range(done + 1, 4)] + [
        "(3, 1000003)"]
    assert [o["title"] for o in core.Memory(cfg).store.get([1000001, 1000002, 1000003])] == [
        "第 1 句", "第 2 句", "第 3 句"]


def test_main_refuses_to_copy_into_an_existing_db(tmp_path):
    path = config(tmp_path, "observation")
    (tmp_path / "observation.db").write_bytes(b"")
    with pytest.raises(SystemExit, match="已经存在"):
        import_exp3.main(path)
