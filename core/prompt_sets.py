"""Seed-prompt sets: the shipped prompts (P1) and frozen alternative sets (P2).

``P1`` is ``configs/prompts/<topology>/<dataset>/<role>.txt``, the set every runner
loads. ``P2`` is generated once from the same meta prompt and role, tool and domain
catalogs by ``configs/generate_role_prompts.py --prompt-set P2`` into
``configs/prompt_sets/p2/`` and frozen there by ``manifest.json``: the generation
settings, and for every prompt file the SHA-256 of its generator input and of its
text. :func:`load` checks every file against the manifest, so a set edited after
freezing is rejected. The run protocol starts a job from a set with ``--prompt-set``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from core.paths import PROMPT_SETS_DIR, PROMPTS_DIR

PROMPT_SETS = ("P1", "P2")
DEFAULT_PROMPT_SET = "P1"
MANIFEST = "manifest.json"
MANIFEST_SCHEMA = "mas-promptbench-prompt-set"


def check(name: str) -> str:
    """``name`` if it is a prompt set, else ValueError."""
    if name not in PROMPT_SETS:
        raise ValueError(f"prompt set must be one of {PROMPT_SETS}, got {name!r}")
    return name


def directory(name: str) -> Path:
    """Root of a set's ``<topology>/<dataset>/<role>.txt`` files."""
    return PROMPTS_DIR if check(name) == DEFAULT_PROMPT_SET else PROMPT_SETS_DIR / name.lower()


def text_sha256(text: str) -> str:
    """SHA-256 of ``text`` encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PromptSet:
    """A frozen prompt set: its manifest, verified against the files on disk."""

    name: str
    manifest: Mapping[str, Any]
    digest: str  # SHA-256 of manifest.json

    def prompt(self, topology: str, dataset: str, role: str) -> str:
        """The prompt of ``role``, stripped."""
        key = f"{topology}/{dataset}/{role}.txt"
        if key not in self.manifest["files"]:
            raise KeyError(f"prompt set {self.name} has no {key}")
        return (directory(self.name) / key).read_text().strip()


@cache
def load(name: str) -> PromptSet:
    """The frozen set ``name`` (P2); every prompt file is checked against its manifest."""
    if check(name) == DEFAULT_PROMPT_SET:
        raise ValueError("P1 is configs/prompts/ itself and has no manifest; read it with seed_prompt()")
    root = directory(name)
    path = root / MANIFEST
    if not path.exists():
        raise FileNotFoundError(
            f"prompt set {name} is not frozen (no {MANIFEST}); "
            f"generate it with configs/generate_role_prompts.py --prompt-set {name}"
        )
    text = path.read_text()
    manifest = json.loads(text)
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("name") != name:
        raise ValueError(f"{name}/{MANIFEST} is not a {MANIFEST_SCHEMA} manifest of {name}")
    files = {item.relative_to(root).as_posix() for item in root.rglob("*.txt")}
    if files != set(manifest["files"]):
        raise ValueError(f"prompt set {name}: the prompt files differ from {MANIFEST}")
    for key, entry in manifest["files"].items():
        if text_sha256((root / key).read_text()) != entry["sha256"]:
            raise ValueError(f"prompt set {name}: {key} was changed after freezing")
    return PromptSet(name, manifest, text_sha256(text))


def seed_prompt(topology: str, dataset: str, role: str, prompt_set: str = DEFAULT_PROMPT_SET) -> str:
    """The seed prompt of ``role`` in ``prompt_set``, stripped."""
    if check(prompt_set) == DEFAULT_PROMPT_SET:
        return (PROMPTS_DIR / topology / dataset / f"{role}.txt").read_text().strip()
    return load(prompt_set).prompt(topology, dataset, role)


def freeze(name: str, *, generator: Mapping[str, Any], inputs: Mapping[str, str]) -> Path:
    """Write ``manifest.json`` for every prompt file of ``name``; a frozen set is never refrozen.

    ``generator`` holds the generation settings; ``inputs`` maps each file key
    (``<topology>/<dataset>/<role>.txt``) to the rendered generator input.
    """
    if check(name) == DEFAULT_PROMPT_SET:
        raise ValueError("P1 is the shipped set and is not frozen by a manifest")
    root = directory(name)
    path = root / MANIFEST
    if path.exists():
        raise FileExistsError(f"prompt set {name} is already frozen")
    keys = sorted(item.relative_to(root).as_posix() for item in root.rglob("*.txt"))
    missing = [key for key in keys if key not in inputs]
    if not keys or missing:
        raise ValueError(f"prompt set {name}: no generator input recorded for {missing[:5] or 'any file'}")
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "name": name,
        "generator": dict(generator),
        "files": {
            key: {"input_sha256": text_sha256(inputs[key]), "sha256": text_sha256((root / key).read_text())}
            for key in keys
        },
    }
    path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return path
