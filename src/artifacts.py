from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import tomli_w


def atomic_write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", dir=destination.parent, delete=False, encoding="utf-8"
    ) as file:
        file.write(text)
        temporary = Path(file.name)
    os.replace(temporary, destination)


def atomic_write_toml(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb", dir=destination.parent, delete=False
    ) as file:
        tomli_w.dump(payload, file)
        temporary = Path(file.name)
    os.replace(temporary, destination)


def write_config_safely(path: Path, payload: dict[str, Any]) -> str:
    if path.exists():
        try:
            import tomllib
        except ModuleNotFoundError:  # Python 3.10
            import tomli as tomllib

        with path.open("rb") as file:
            current = tomllib.load(file)
        if current != payload:
            raise RuntimeError(f"Refusing to overwrite a different config: {path}")
        return "exists"
    atomic_write_toml(path, payload)
    return "written"


def save_predictions(path: Path, actions: np.ndarray, outputs: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb", dir=path.parent, suffix=".npz", delete=False
    ) as file:
        temporary = Path(file.name)
    np.savez_compressed(
        temporary,
        actions=np.asarray(actions, dtype=np.float32),
        pnl=np.asarray(outputs["pnl"], dtype=np.float32),
        fees=np.asarray(outputs["fees"], dtype=np.float32),
        turnover=np.asarray(outputs["turnover"], dtype=np.float32),
    )
    os.replace(temporary, path)


def touch_terminal(directory: Path, marker: str) -> None:
    for other in ("RUNNING", "DONE", "FAILED"):
        path = directory / other
        if path.exists() and other != marker:
            path.unlink()
    (directory / marker).touch()
