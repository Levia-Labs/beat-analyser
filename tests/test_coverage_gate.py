"""The coverage ratchet: a file may hold or gain coverage, never lose it."""

import importlib.util
import json
import os
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "tasks" / "quality" / "coverage.py"
spec = importlib.util.spec_from_file_location("coverage_gate", SCRIPT)
assert spec and spec.loader
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def test_holding_or_gaining_passes():
    assert gate.compare({"TOTAL": 80.0, "app/a.py": 90.0}, {"TOTAL": 80.0, "app/a.py": 85.0}) == []


def test_a_drop_fails():
    found = gate.compare({"TOTAL": 80.0, "app/a.py": 84.0}, {"TOTAL": 80.0, "app/a.py": 85.0})
    assert found == ["app/a.py: 84.00% < baseline 85.00%"]


def test_a_drop_of_a_hundredth_fails():
    assert gate.compare({"app/a.py": 84.98}, {"app/a.py": 85.0}) != []


def test_a_new_file_under_the_floor_fails():
    found = gate.compare({"TOTAL": 80.0, "app/new.py": 10.0}, {"TOTAL": 80.0})
    assert found == [f"app/new.py: 10.00% < {gate.NEW_FILE_FLOOR:.2f}% floor for a new file"]


def test_float_noise_passes():
    assert gate.compare({"TOTAL": 79.997}, {"TOTAL": 80.0}) == []


def test_a_file_whose_tests_need_a_missing_tool_is_not_compared():
    current = {"TOTAL": 60.0, "app/notes.py": 40.0, "app/storage.py": 93.0}
    baseline = {"TOTAL": 74.0, "app/notes.py": 94.0, "app/storage.py": 93.0}
    assert gate.compare(current, baseline, missing={"onnxruntime"}) == []
    assert gate.compare(current, baseline) != []


def test_missing_report_says_what_to_run(tmp_path):
    with pytest.raises(SystemExit, match="task test:cov"):
        gate.load_current(tmp_path / "absent.json")


def test_a_report_older_than_the_source_is_stale(tmp_path):
    report = tmp_path / "coverage.json"
    report.write_text(json.dumps({"files": {}, "totals": {"percent_covered": 0}}), encoding="utf-8")
    source = tmp_path / "mod.py"
    source.write_text("x = 1\n", encoding="utf-8")
    os.utime(report, (1_000_000, 1_000_000))
    with pytest.raises(SystemExit, match="older than"):
        gate.check_fresh(report, [source])


def test_a_missing_baseline_says_what_to_run(tmp_path, monkeypatch, capsys):
    report = tmp_path / "coverage.json"
    report.write_text(
        json.dumps({"files": {}, "totals": {"percent_covered": 50}}), encoding="utf-8"
    )
    monkeypatch.setattr(gate, "REPORT", report)
    monkeypatch.setattr(gate, "BASELINE", tmp_path / "coverage.baseline")
    monkeypatch.setattr(gate, "sources", list)
    monkeypatch.setattr("sys.argv", ["coverage.py", "--check"])
    assert gate.main() == 1
    assert "task quality:coverage:baseline" in capsys.readouterr().err
