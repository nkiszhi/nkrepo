# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any


def get_record_value(record: Any, field_name: str, index: int | None = None, default: Any = "") -> Any:
    try:
        if isinstance(record, dict):
            value = record.get(field_name, default)
            return default if value is None else value
        if index is not None and isinstance(record, (list, tuple)) and len(record) > index:
            value = record[index]
            return default if value is None else value
    except Exception:
        return default
    return default


def sample_class_label(record: dict[str, Any]) -> str:
    sample_class = record.get("__sample_class")
    if sample_class == "malicious":
        return "恶意样本"
    if sample_class == "benign":
        return "白样本"
    if record.get("__is_upload"):
        return "上传样本"
    return sample_class or ""


def build_sample_detail(record: Any, sha_label: str = "SHA256") -> dict[str, Any]:
    if not isinstance(record, dict):
        return {
            "MD5": get_record_value(record, "md5", 3, ""),
            sha_label: get_record_value(record, "sha256", 2, ""),
            "SSDEEP": get_record_value(record, "ssdeep", 4, ""),
            "vhash": get_record_value(record, "vhash", 5, ""),
            "Authentihash": get_record_value(record, "authentihash", 6, ""),
            "Imphash": get_record_value(record, "imphash", 7, ""),
            "Rich header hash": get_record_value(record, "rich_header_hash", 8, ""),
            "类型": get_record_value(record, "category", 11, ""),
            "平台": get_record_value(record, "platform", 12, ""),
            "家族": get_record_value(record, "family", 13, "")
        }

    sample_class = record.get("__sample_class")
    detail = {
        "MD5": record.get("md5", ""),
        sha_label: record.get("sha256", ""),
        "SSDEEP": record.get("ssdeep", ""),
        "vhash": record.get("vhash", ""),
        "Authentihash": record.get("authentihash", ""),
        "Imphash": record.get("imphash", ""),
        "Rich header hash": record.get("rich_header_hash", ""),
        "样本性质": sample_class_label(record),
        "文件类别": record.get("__file_kind", ""),
        "来源库": record.get("__source_db", "")
    }

    if sample_class == "benign":
        detail.update({
            "软件名称": record.get("software_name", ""),
            "软件类型": record.get("software_type", ""),
            "原始文件名": record.get("file_name", ""),
            "文件大小": f"{record.get('length', 0) or 0} bytes",
            "文件类型": record.get("filetype", ""),
            "平台": record.get("platform", ""),
            "加壳": record.get("packer", ""),
            "适用系统": record.get("os_versions", ""),
            "文件路径": record.get("file_path", "")
        })
    else:
        detail.update({
            "类型": record.get("category", ""),
            "平台": record.get("platform", ""),
            "家族": record.get("family", ""),
            "文件大小": f"{record.get('length', 0) or 0} bytes",
            "文件类型": record.get("filetype", ""),
            "来源": record.get("source", ""),
            "卡巴结果": record.get("kav_result", ""),
            "Defender结果": record.get("defender_result", ""),
            "加壳": record.get("packer", "")
        })

    return detail


def get_sample_display_name(record: Any, fallback: str) -> str:
    if isinstance(record, dict):
        return (
            record.get("name")
            or record.get("file_name")
            or record.get("software_name")
            or fallback
        )
    return fallback
