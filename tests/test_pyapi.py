"""The pyapi inspector: attribute description, module listing, refusals."""

import importlib.util
from pathlib import Path

from hypothesis import given
from hypothesis import strategies as st

SCRIPT = Path(__file__).resolve().parents[1] / "tasks" / "dev" / "pyapi.py"
spec = importlib.util.spec_from_file_location("pyapi", SCRIPT)
assert spec and spec.loader
pyapi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pyapi)


def test_attribute_prints_signature_and_doc(capsys):
    assert pyapi.main(["json:dumps"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("json:dumps(obj")
    assert out[1] == "Serialize ``obj`` to a JSON formatted ``str``."


def test_dotted_attribute(capsys):
    assert pyapi.main(["json:JSONDecoder.decode"]) == 0
    assert capsys.readouterr().out.startswith("json:JSONDecoder.decode(self, s")


def test_module_listing(capsys):
    assert pyapi.main(["json"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert any(line.startswith("dumps(") for line in out)
    assert "JSONDecodeError(msg, doc, pos)" in out


def test_listing_is_capped(capsys):
    assert pyapi.main(["os", "--top", "5"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 6
    assert out[-1].startswith("... ")


def test_app_module_resolves_by_bare_name(capsys):
    assert pyapi.main(["storage:is_valid_name"]) == 0
    assert capsys.readouterr().out.startswith("storage:is_valid_name(")


def test_missing_attribute(capsys):
    assert pyapi.main(["json:nope"]) == 1
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 1
    assert out[0].startswith("missing attribute: json:nope")


def test_missing_module(capsys):
    assert pyapi.main(["no_such_module_xyz:a"]) == 1
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 1
    assert out[0].startswith("import failed: no_such_module_xyz:a")


@given(st.text(alphabet="abcdefghij._", max_size=12))
def test_split_target_roundtrip(text):
    module, attr = pyapi.split_target(f"{text}:x.y")
    assert module == text
    assert attr == "x.y"
    assert pyapi.split_target(text) == (text, "")
