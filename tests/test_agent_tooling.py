"""Agent-session tools: gate and outline (lanes, hooks and tokens live in Crepidoma)."""

import importlib.util
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


gate = load("dev/gate.py")
outline = load("dev/outline.py")


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
