"""The gate runner's output survives consoles that cannot encode a failing tool's characters."""

import importlib.util
import io
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "tasks" / "dev" / "gate.py"
spec = importlib.util.spec_from_file_location("dev_gate", SCRIPT)
assert spec and spec.loader
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def cp1250_stream():
    raw = io.BytesIO()
    return raw, io.TextIOWrapper(raw, encoding="cp1250", errors="strict")


def test_failing_gate_prints_on_cp1250_console(monkeypatch, tmp_path):
    raw, stream = cp1250_stream()
    monkeypatch.setattr(sys, "stdout", stream)
    monkeypatch.setattr(sys, "stderr", stream)
    monkeypatch.setattr(gate, "LOGS", tmp_path)
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    done = subprocess.CompletedProcess([], 1, stdout="✖ test failed\n", stderr="")
    monkeypatch.setattr(gate.subprocess, "run", lambda *a, **k: done)
    gate.configure_output()
    assert gate.run("x", 30) is False
    stream.flush()
    text = raw.getvalue().decode("cp1250")
    assert text.startswith("FAIL  x")
    assert "test failed" in text


def test_passing_gate_format_unchanged(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(gate, "LOGS", tmp_path)
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    done = subprocess.CompletedProcess([], 0, stdout="fine", stderr="")
    monkeypatch.setattr(gate.subprocess, "run", lambda *a, **k: done)
    assert gate.run("x", 30) is True
    assert capsys.readouterr().out.startswith("ok    x  ")
