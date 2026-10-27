"""Inspect a Python API without reading site-packages.

Commands:
    pyapi.py <module>[:<attr.path>] [...] [--top N]

With an attribute, prints `module:qualname(signature)` and the first docstring line.
With a module only, lists its public names, one line each, capped at --top (default 60).
Exit 1 with one line on ImportError or a missing attribute, 2 on usage errors.

Usage: task dev:pyapi -- mcp.server.mcpserver.server:MCPServer.streamable_http_app
"""

import argparse
import importlib
import inspect
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[2] / "app"


def signature(obj: object) -> str:
    """The call signature as text, empty for non-callables and builtins without one."""
    if not callable(obj):
        return ""
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return "(...)"


def first_line(obj: object) -> str:
    doc = inspect.getdoc(obj) or ""
    return doc.splitlines()[0] if doc else ""


def split_target(target: str) -> tuple[str, str]:
    module, _, attr = target.partition(":")
    return module, attr


def resolve(module: str, attr: str) -> tuple[object, object]:
    """Return (module object, target object). Raises ImportError or AttributeError."""
    mod = importlib.import_module(module)
    obj: object = mod
    for part in filter(None, attr.split(".")):
        obj = getattr(obj, part)
    return mod, obj


def describe(module: str, attr: str, obj: object) -> list[str]:
    name = getattr(obj, "__qualname__", attr)
    lines = [f"{module}:{name}{signature(obj)}"]
    doc = first_line(obj)
    if doc:
        lines.append(doc)
    return lines


def public_names(mod: object) -> list[str]:
    declared = getattr(mod, "__all__", None)
    if declared is not None:
        return sorted(str(n) for n in declared)
    return sorted(n for n in vars(mod) if not n.startswith("_"))


def listing(mod: object, top: int) -> list[str]:
    lines = []
    for name in public_names(mod):
        member = getattr(mod, name, None)
        lines.append(f"{name}{signature(member)}" if callable(member) else name)
    if len(lines) > top:
        extra = len(lines) - top
        lines = [*lines[:top], f"... {extra} more"]
    return lines


def run(target: str, top: int) -> list[str]:
    module, attr = split_target(target)
    mod, obj = resolve(module, attr)
    if attr:
        return describe(module, attr, obj)
    return listing(mod, top)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("targets", nargs="+", help="module or module:attr.path")
    parser.add_argument("--top", type=int, default=60, help="max lines per module listing")
    args = parser.parse_args(argv)
    # App modules import by bare name; pytest.ini puts app/ on the path, so do the same here.
    sys.path.insert(0, str(APP_DIR))
    status = 0
    for target in args.targets:
        try:
            lines = run(target, args.top)
        except ImportError as exc:
            print(f"import failed: {target}: {exc}")
            status = 1
            continue
        except AttributeError as exc:
            print(f"missing attribute: {target}: {exc}")
            status = 1
            continue
        for line in lines:
            print(line)
    return status


if __name__ == "__main__":
    sys.exit(main())
