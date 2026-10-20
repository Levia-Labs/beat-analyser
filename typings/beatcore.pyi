"""Type stub for beatcore, the Rust extension in beatcore/src/lib.rs (pyo3 module)."""

from typing import Any

__version__: str

def analyse(path: str) -> dict[str, Any]: ...
def analyse_samples(samples: list[float], sample_rate: int) -> dict[str, Any]: ...
