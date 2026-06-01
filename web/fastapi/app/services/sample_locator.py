# -*- coding: utf-8 -*-
from __future__ import annotations

import configparser
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.ini"
SHA256_RE = re.compile(r"\A[0-9a-fA-F]{64}\Z")
SAMPLE_CLASSES = ("malicious", "benign")
FILE_KINDS = ("elf", "pe", "others")


@dataclass(frozen=True)
class SampleLocation:
    sample_dir_path: str
    sample_file_path: str
    sample_class: str
    file_kind: str
    is_upload: bool = False


def is_valid_sha256(value: str) -> bool:
    return bool(SHA256_RE.fullmatch(value or ""))


def safe_resolve_path(root: Path, *parts: str) -> Path:
    root_resolved = root.resolve()
    candidate = root_resolved.joinpath(*parts).resolve()
    candidate.relative_to(root_resolved)
    return candidate


def load_path_roots() -> tuple[Path, Path]:
    config = configparser.ConfigParser()
    config.read(CONFIG_PATH, encoding="utf-8")

    base_dir = CONFIG_PATH.parent
    samples_dir = config.get("paths", "sample_root", fallback="../../data/samples")
    web_upload_dir = config.get("paths", "web_upload_dir", fallback="../../data/web_upload_file")

    samples_root = (base_dir / samples_dir).resolve()
    web_upload_root = (base_dir / web_upload_dir).resolve()
    return samples_root, web_upload_root


def infer_file_kind(db_name: str | None, fallback: str = "") -> str:
    lowered = (db_name or "").lower()
    for kind in FILE_KINDS:
        if lowered.endswith(f"_{kind}") or f"_{kind}_" in lowered:
            return kind
    return fallback if fallback in FILE_KINDS else ""


def infer_sample_class(db_name: str | None, fallback: str = "") -> str:
    lowered = (db_name or "").lower()
    if lowered.startswith("malicious"):
        return "malicious"
    if lowered.startswith("benign"):
        return "benign"
    return fallback if fallback in SAMPLE_CLASSES else ""


def record_sample_class(record: dict[str, Any] | None) -> str:
    if not record:
        return ""
    value = record.get("__sample_class") or record.get("_sample_class")
    return value if value in SAMPLE_CLASSES else infer_sample_class(record.get("__source_db"))


def record_file_kind(record: dict[str, Any] | None) -> str:
    if not record:
        return ""
    value = record.get("__file_kind") or record.get("_file_kind")
    return value if value in FILE_KINDS else infer_file_kind(record.get("__source_db"))


def candidate_sample_paths(sha256: str, record: dict[str, Any] | None = None) -> list[SampleLocation]:
    sha256 = (sha256 or "").strip().lower()
    if not is_valid_sha256(sha256):
        return []

    samples_root, web_upload_root = load_path_roots()
    prefix_parts = list(sha256[:5])

    sample_class = record_sample_class(record)
    file_kind = record_file_kind(record)

    candidates: list[tuple[str, str]] = []
    if sample_class and file_kind:
        candidates.append((sample_class, file_kind))

    for cls in SAMPLE_CLASSES:
        for kind in FILE_KINDS:
            candidate = (cls, kind)
            if candidate not in candidates:
                candidates.append(candidate)

    locations: list[SampleLocation] = []
    for cls, kind in candidates:
        sample_dir = safe_resolve_path(samples_root, cls, kind, *prefix_parts)
        sample_path = safe_resolve_path(samples_root, cls, kind, *prefix_parts, sha256)
        locations.append(SampleLocation(str(sample_dir), str(sample_path), cls, kind, False))

    upload_path = safe_resolve_path(web_upload_root, sha256)
    locations.append(SampleLocation(str(web_upload_root), str(upload_path), "upload", "upload", True))
    return locations


def locate_sample(sha256: str, record: dict[str, Any] | None = None) -> SampleLocation | None:
    for location in candidate_sample_paths(sha256, record):
        if Path(location.sample_file_path).exists():
            return location
    return None
