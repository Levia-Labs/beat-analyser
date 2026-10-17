"""The coverage ratchet: a file may hold or gain coverage, never lose it."""

import importlib.util
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
    assert found == ["app/a.py: 84.0% < baseline 85.0%"]


def test_a_new_file_under_the_floor_fails():
    found = gate.compare({"TOTAL": 80.0, "app/new.py": 10.0}, {"TOTAL": 80.0})
    assert found == [f"app/new.py: 10.0% < {gate.NEW_FILE_FLOOR:.1f}% floor for a new file"]


def test_rounding_noise_passes():
    assert gate.compare({"TOTAL": 79.96}, {"TOTAL": 80.0}) == []


def test_missing_report_says_what_to_run(tmp_path):
    with pytest.raises(SystemExit, match="task test:cov"):
        gate.load_current(tmp_path / "absent.json")
