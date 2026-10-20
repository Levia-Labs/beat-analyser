"""The tools that keep agent sessions small: lanes, token accounting, gate, outline."""

import importlib.util
import json
import sys
from pathlib import Path

TASKS = Path(__file__).resolve().parents[1] / "tasks"


def load(relative: str):
    path = TASKS / relative
    spec = importlib.util.spec_from_file_location(path.stem + "_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their module by name
    spec.loader.exec_module(module)
    return module


sys.path.insert(0, str(TASKS / "claude"))  # hooks.py imports lanes as a sibling
lanes = load("claude/lanes.py")
hooks = load("claude/hooks.py")
tokens = load("claude/tokens.py")
allow = load("claude/allow_tasks.py")
gate = load("dev/gate.py")
outline = load("dev/outline.py")


def test_a_catch_all_agent_is_refused():
    state = lanes.empty()
    for kind in ("general-purpose", "fork", "Explore", "Plan", ""):
        assert lanes.refusal({"subagent_type": kind}, state)
    assert lanes.refusal({}, state)
    assert lanes.refusal({"subagent_type": "tool-smith"}, state) is None


def test_a_seventh_lane_is_refused_and_a_finished_one_frees_a_slot():
    state = lanes.empty()
    for n in range(6):
        lanes.reserve(state, f"call-{n}", 0.0)
    assert lanes.refusal({"subagent_type": "tool-smith"}, state)
    lanes.started(state, "agent-a", 1.0)
    assert lanes.in_use(state) == 6
    lanes.stopped(state, "agent-a")
    assert lanes.refusal({"subagent_type": "tool-smith"}, state) is None


def test_stale_lanes_expire():
    state = {"running": {"old": 0.0}, "pending": {"call": 0.0}}
    assert lanes.in_use(lanes.prune(state, lanes.STALE_SECONDS + 1)) == 0
    assert lanes.in_use(lanes.prune(state, 1.0)) == 2


def test_the_hook_denies_a_catch_all_and_counts_a_named_agent(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(hooks, "budget_refusal", lambda: None)
    monkeypatch.setattr(hooks.lanes, "STATE", tmp_path / "lanes.json")
    monkeypatch.setattr(hooks.lanes, "LOCK", tmp_path / "lanes.lock")

    hooks.pre_tool_use({"tool_name": "Agent", "tool_input": {"subagent_type": "Explore"}})
    decision = json.loads(capsys.readouterr().out)["hookSpecificOutput"]
    assert decision["permissionDecision"] == "deny"

    event = {"tool_name": "Agent", "tool_use_id": "t1", "tool_input": {"subagent_type": "x"}}
    hooks.pre_tool_use(event)
    assert capsys.readouterr().out == ""
    with hooks.lanes.locked_state() as state:
        assert lanes.in_use(state) == 1


def test_tokens_count_one_response_once_and_size_tool_results(tmp_path):
    usage = {"input_tokens": 10, "cache_read_input_tokens": 100, "output_tokens": 5}
    read = {"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "a.py"}}
    entries = [
        {"message": {"id": "m1", "usage": usage, "content": [{"type": "text", "text": "x"}]}},
        {"message": {"id": "m1", "usage": usage, "content": [read]}},
        {"message": {"content": [{"type": "tool_result", "tool_use_id": "t1", "content": "abc"}]}},
    ]
    transcript = tmp_path / "s.jsonl"
    transcript.write_text("\n".join(json.dumps(e) for e in entries), encoding="utf-8")
    row = tokens.tally(transcript, "s", "session")
    assert (row.calls, row.total) == (1, 115)
    assert row.tool_chars["Read"] == 3
    assert row.reads["a.py"] == 1


def test_settings_findings_and_rewrite():
    settings = {
        "permissions": {
            "allow": ["Bash(task test)", "PowerShell(task test)", "Bash(git *)"],
            "deny": ["Bash(task uploads:clean)"],
        },
        "hooks": {},
    }
    assert len(allow.findings(settings)) == 6
    fixed = allow.rewrite(settings)
    assert fixed["permissions"]["allow"] == ["Bash(task *)", "Bash(git *)"]
    assert "Bash(rtk task uploads:clean)" in fixed["permissions"]["deny"]
    assert allow.findings(fixed) == []


def test_gate_excerpt_keeps_the_reason_and_drops_build_noise():
    output = "   Compiling foo v1\nrunning 3 tests\nthread 'x' panicked at src/a.rs:3\nok\n"
    lines = gate.excerpt(output, 30)
    assert "thread 'x' panicked at src/a.rs:3" in lines
    assert not any("Compiling" in line for line in lines)


def test_gate_log_names_are_file_safe():
    assert gate.log_name("test K=notes") == "test_K_notes.log"
    assert gate.log_name("rust:test") == "rust_test.log"


def test_outline_python_and_taskfile(tmp_path):
    source = tmp_path / "m.py"
    source.write_text("LIMIT = 3\n\nclass A:\n    def go(self):\n        pass\n", encoding="utf-8")
    assert outline.outline(source) == [(1, "const LIMIT"), (3, "class A  -5"), (4, "def A.go  -5")]
    taskfile = tmp_path / "Taskfile.yml"
    taskfile.write_text("vars:\n  PY: x\ntasks:\n  test:\n    cmds: []\n", encoding="utf-8")
    assert outline.outline(taskfile) == [(1, "vars"), (3, "tasks"), (4, "task test")]
