#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create and refresh sample_stats rollup tables.

Default behavior is a full refresh. Incremental mode uses per-source-table
COUNT(*) and MAX(updated_at) to skip unchanged sample_xx tables.
"""

from __future__ import annotations

import argparse
import configparser
import logging
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import mysql.connector
from mysql.connector import Error


LOGGER = logging.getLogger("sample_stats_refresh")
IDENTIFIER_RE = re.compile(r"\A[A-Za-z0-9_]{1,64}\Z")
HEX_SUFFIXES = [f"{i:02x}" for i in range(256)]
MALICIOUS_KINDS = ("elf", "pe", "others")
BENIGN_KINDS = ("elf", "pe", "others")


@dataclass(frozen=True)
class SourceDb:
    sample_class: str
    file_kind: str
    db_name: str
    storage_subdir: str


def quote_identifier(identifier: str) -> str:
    if not IDENTIFIER_RE.fullmatch(identifier or ""):
        raise ValueError(f"Unsafe MySQL identifier: {identifier!r}")
    return f"`{identifier}`"


def split_config_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def infer_kind(db_name: str, fallback_index: int) -> str:
    lowered = db_name.lower()
    for kind in ("elf", "pe", "others"):
        if lowered.endswith(f"_{kind}") or f"_{kind}_" in lowered:
            return kind
    return ("elf", "pe", "others")[fallback_index % 3]


def load_config(config_path: Path) -> tuple[dict[str, Any], str, list[SourceDb]]:
    parser = configparser.ConfigParser()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    parser.read(config_path, encoding="utf-8")

    mysql_cfg = {
        "host": parser.get("mysql", "host"),
        "user": parser.get("mysql", "user"),
        "password": parser.get("mysql", "passwd"),
        "port": parser.getint("mysql", "port", fallback=3306),
        "charset": parser.get("mysql", "charset", fallback="utf8mb4"),
        "autocommit": False,
    }
    stats_db = parser.get("mysql", "db_stats", fallback="sample_stats")

    malicious_dbs = split_config_list(parser.get("mysql", "malicious_dbs", fallback=""))
    if not malicious_dbs:
        malicious_dbs = [parser.get("mysql", "db", fallback="malicious")]

    benign_dbs = split_config_list(parser.get("mysql", "benign_dbs", fallback=""))
    if not benign_dbs:
        benign_dbs = [parser.get("mysql", "db_benign", fallback="benign")]

    sources: list[SourceDb] = []
    for idx, db_name in enumerate(malicious_dbs):
        kind = infer_kind(db_name, idx)
        sources.append(SourceDb("malicious", kind, db_name, f"malicious/{kind}"))
    for idx, db_name in enumerate(benign_dbs):
        kind = infer_kind(db_name, idx)
        sources.append(SourceDb("benign", kind, db_name, f"benign/{kind}"))

    return mysql_cfg, stats_db, sources


def connect(mysql_cfg: dict[str, Any], database: str | None = None):
    cfg = dict(mysql_cfg)
    if database:
        cfg["database"] = database
    return mysql.connector.connect(**cfg)


def execute_many(cursor, sql: str, rows: Iterable[tuple[Any, ...]]) -> None:
    rows = list(rows)
    if rows:
        cursor.executemany(sql, rows)


def create_database_and_tables(mysql_cfg: dict[str, Any], stats_db: str) -> None:
    with connect(mysql_cfg) as conn:
        cursor = conn.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {quote_identifier(stats_db)} DEFAULT CHARSET utf8mb4")
        conn.commit()

    with connect(mysql_cfg, stats_db) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS sample_db_registry (
              id INT AUTO_INCREMENT PRIMARY KEY,
              sample_class VARCHAR(16) NOT NULL,
              file_kind VARCHAR(16) NOT NULL,
              db_name VARCHAR(64) NOT NULL,
              storage_subdir VARCHAR(64) NOT NULL,
              enabled TINYINT(1) NOT NULL DEFAULT 1,
              updated_at TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
              UNIQUE KEY uk_sample_db (sample_class, file_kind)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS sample_source_table_state (
              sample_class VARCHAR(16) NOT NULL,
              file_kind VARCHAR(16) NOT NULL,
              source_db VARCHAR(64) NOT NULL,
              source_table VARCHAR(64) NOT NULL,
              row_count BIGINT NOT NULL DEFAULT 0,
              last_updated_at DATETIME NULL,
              last_refreshed_at TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
              PRIMARY KEY (sample_class, file_kind, source_db, source_table)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS sample_table_total_stats (
              sample_class VARCHAR(16) NOT NULL,
              file_kind VARCHAR(16) NOT NULL,
              source_db VARCHAR(64) NOT NULL,
              source_table VARCHAR(64) NOT NULL,
              total_samples BIGINT NOT NULL DEFAULT 0,
              has_vt_count BIGINT NOT NULL DEFAULT 0,
              has_vt_summary_count BIGINT NOT NULL DEFAULT 0,
              has_vt_mitre_count BIGINT NOT NULL DEFAULT 0,
              updated_at TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
              PRIMARY KEY (sample_class, file_kind, source_db, source_table)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS sample_table_monthly_stats (
              sample_class VARCHAR(16) NOT NULL,
              file_kind VARCHAR(16) NOT NULL,
              source_db VARCHAR(64) NOT NULL,
              source_table VARCHAR(64) NOT NULL,
              year INT NOT NULL,
              month INT NOT NULL,
              total_samples BIGINT NOT NULL DEFAULT 0,
              updated_at TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
              PRIMARY KEY (sample_class, file_kind, source_db, source_table, year, month)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """
        )
        for table_name, key_name, key_type in (
            ("sample_table_yearly_stats", "year", "INT"),
            ("sample_table_category_stats", "category", "VARCHAR(255)"),
            ("sample_table_platform_stats", "platform", "VARCHAR(255)"),
            ("sample_table_family_stats", "family", "VARCHAR(255)"),
            ("sample_table_filetype_stats", "filetype", "VARCHAR(255)"),
        ):
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {quote_identifier(table_name)} (
                  sample_class VARCHAR(16) NOT NULL,
                  file_kind VARCHAR(16) NOT NULL,
                  source_db VARCHAR(64) NOT NULL,
                  source_table VARCHAR(64) NOT NULL,
                  {key_name} {key_type} NOT NULL,
                  total_samples BIGINT NOT NULL DEFAULT 0,
                  updated_at TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                  PRIMARY KEY (sample_class, file_kind, source_db, source_table, {key_name})
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS sample_total_stats (
              sample_class VARCHAR(16) NOT NULL,
              file_kind VARCHAR(16) NOT NULL,
              total_samples BIGINT NOT NULL DEFAULT 0,
              has_vt_count BIGINT NOT NULL DEFAULT 0,
              has_vt_summary_count BIGINT NOT NULL DEFAULT 0,
              has_vt_mitre_count BIGINT NOT NULL DEFAULT 0,
              updated_at TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
              PRIMARY KEY (sample_class, file_kind)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS sample_behavior_stats (
              sample_class VARCHAR(16) NOT NULL,
              file_kind VARCHAR(16) NOT NULL,
              total_samples BIGINT NOT NULL DEFAULT 0,
              has_vt_1 BIGINT NOT NULL DEFAULT 0,
              has_vt_summary_1 BIGINT NOT NULL DEFAULT 0,
              has_vt_mitre_1 BIGINT NOT NULL DEFAULT 0,
              updated_at TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
              PRIMARY KEY (sample_class, file_kind)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS sample_monthly_stats (
              sample_class VARCHAR(16) NOT NULL,
              file_kind VARCHAR(16) NOT NULL,
              year INT NOT NULL,
              month INT NOT NULL,
              total_samples BIGINT NOT NULL DEFAULT 0,
              updated_at TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
              PRIMARY KEY (sample_class, file_kind, year, month)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """
        )
        for table_name, key_name, key_type in (
            ("sample_yearly_stats", "year", "INT"),
            ("sample_category_stats", "category", "VARCHAR(255)"),
            ("sample_platform_stats", "platform", "VARCHAR(255)"),
            ("sample_family_stats", "family", "VARCHAR(255)"),
            ("sample_filetype_stats", "filetype", "VARCHAR(255)"),
        ):
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {quote_identifier(table_name)} (
                  sample_class VARCHAR(16) NOT NULL,
                  file_kind VARCHAR(16) NOT NULL,
                  {key_name} {key_type} NOT NULL,
                  total_samples BIGINT NOT NULL DEFAULT 0,
                  updated_at TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                  PRIMARY KEY (sample_class, file_kind, {key_name})
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS sample_stats_refresh_log (
              id BIGINT AUTO_INCREMENT PRIMARY KEY,
              mode VARCHAR(16) NOT NULL,
              started_at TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP,
              finished_at TIMESTAMP NULL,
              status VARCHAR(16) NOT NULL,
              refreshed_tables INT NOT NULL DEFAULT 0,
              skipped_tables INT NOT NULL DEFAULT 0,
              message TEXT NULL
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """
        )
        conn.commit()


def register_sources(mysql_cfg: dict[str, Any], stats_db: str, sources: list[SourceDb]) -> None:
    sql = """
        INSERT INTO sample_db_registry
          (sample_class, file_kind, db_name, storage_subdir, enabled)
        VALUES (%s, %s, %s, %s, 1)
        ON DUPLICATE KEY UPDATE
          db_name = VALUES(db_name),
          storage_subdir = VALUES(storage_subdir),
          enabled = 1
    """
    rows = [(s.sample_class, s.file_kind, s.db_name, s.storage_subdir) for s in sources]
    with connect(mysql_cfg, stats_db) as conn:
        cursor = conn.cursor()
        execute_many(cursor, sql, rows)
        conn.commit()


def cleanup_stale_source_stats(mysql_cfg: dict[str, Any], stats_db: str, sources: list[SourceDb]) -> int:
    """Delete cached table stats whose source DB is no longer in config.ini."""
    valid_sources = {(s.sample_class, s.file_kind, s.db_name) for s in sources}
    stale_sources: set[tuple[str, str, str]] = set()
    source_tables = (
        "sample_source_table_state",
        "sample_table_total_stats",
        "sample_table_yearly_stats",
        "sample_table_monthly_stats",
        "sample_table_category_stats",
        "sample_table_platform_stats",
        "sample_table_family_stats",
        "sample_table_filetype_stats",
    )

    with connect(mysql_cfg, stats_db) as conn:
        cursor = conn.cursor()
        for table in source_tables:
            cursor.execute(
                f"""
                SELECT DISTINCT sample_class, file_kind, source_db
                FROM {quote_identifier(table)}
                """
            )
            for sample_class, file_kind, source_db in cursor.fetchall():
                key = (sample_class, file_kind, source_db)
                if key not in valid_sources:
                    stale_sources.add(key)

        deleted = 0
        for sample_class, file_kind, source_db in stale_sources:
            for table in source_tables:
                cursor.execute(
                    f"""
                    DELETE FROM {quote_identifier(table)}
                    WHERE sample_class = %s
                      AND file_kind = %s
                      AND source_db = %s
                    """,
                    (sample_class, file_kind, source_db),
                )
                deleted += cursor.rowcount
        conn.commit()

    if stale_sources:
        LOGGER.warning("deleted stale source stats: %s", sorted(stale_sources))
    return deleted


def table_exists(conn, db_name: str, table_name: str) -> bool:
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT 1
        FROM information_schema.TABLES
        WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
        LIMIT 1
        """,
        (db_name, table_name),
    )
    return cursor.fetchone() is not None


def get_columns(conn, db_name: str, table_name: str) -> set[str]:
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT COLUMN_NAME
        FROM information_schema.COLUMNS
        WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
        """,
        (db_name, table_name),
    )
    return {row[0] for row in cursor.fetchall()}


def get_source_state(stats_conn, source: SourceDb, table_name: str) -> tuple[int, Any] | None:
    cursor = stats_conn.cursor()
    cursor.execute(
        """
        SELECT row_count, last_updated_at
        FROM sample_source_table_state
        WHERE sample_class = %s
          AND file_kind = %s
          AND source_db = %s
          AND source_table = %s
        """,
        (source.sample_class, source.file_kind, source.db_name, table_name),
    )
    row = cursor.fetchone()
    if not row:
        return None
    return int(row[0]), row[1]


def get_table_signature(source_conn, db_name: str, table_name: str, columns: set[str]) -> tuple[int, Any]:
    cursor = source_conn.cursor()
    if "updated_at" in columns:
        cursor.execute(f"SELECT COUNT(*), MAX(updated_at) FROM {quote_identifier(table_name)}")
    else:
        cursor.execute(f"SELECT COUNT(*), NULL FROM {quote_identifier(table_name)}")
    row = cursor.fetchone()
    return int(row[0] or 0), row[1]


def fetch_one_counter(source_conn, sql: str) -> Counter:
    cursor = source_conn.cursor()
    cursor.execute(sql)
    counter = Counter()
    for key, count in cursor.fetchall():
        if key is None or key == "":
            continue
        counter[key] += int(count or 0)
    return counter


def fetch_table_stats(source_conn, table_name: str, source: SourceDb, columns: set[str]) -> dict[str, Any]:
    q_table = quote_identifier(table_name)
    has_vt_expr = "SUM(CASE WHEN has_vt = 1 THEN 1 ELSE 0 END)" if "has_vt" in columns else "0"
    has_vt_summary_expr = "SUM(CASE WHEN has_vt_summary = 1 THEN 1 ELSE 0 END)" if "has_vt_summary" in columns else "0"
    has_vt_mitre_expr = "SUM(CASE WHEN has_vt_mitre = 1 THEN 1 ELSE 0 END)" if "has_vt_mitre" in columns else "0"

    cursor = source_conn.cursor()
    cursor.execute(
        f"""
        SELECT COUNT(*), {has_vt_expr}, {has_vt_summary_expr}, {has_vt_mitre_expr}
        FROM {q_table}
        """
    )
    total, has_vt, has_vt_summary, has_vt_mitre = cursor.fetchone()

    stats: dict[str, Any] = {
        "total": int(total or 0),
        "has_vt": int(has_vt or 0),
        "has_vt_summary": int(has_vt_summary or 0),
        "has_vt_mitre": int(has_vt_mitre or 0),
        "yearly": Counter(),
        "monthly": Counter(),
        "category": Counter(),
        "platform": Counter(),
        "family": Counter(),
        "filetype": Counter(),
    }

    if source.sample_class == "malicious":
        if "date" in columns:
            date_expr = "COALESCE(`date`, DATE(created_at))" if "created_at" in columns else "`date`"
        elif "created_at" in columns:
            date_expr = "DATE(created_at)"
        else:
            date_expr = None

        if date_expr:
            stats["yearly"] = fetch_one_counter(
                source_conn,
                f"""
                SELECT YEAR({date_expr}) AS stat_year, COUNT(*)
                FROM {q_table}
                WHERE {date_expr} IS NOT NULL
                GROUP BY stat_year
                """,
            )
            stats["monthly"] = fetch_one_counter(
                source_conn,
                f"""
                SELECT DATE_FORMAT({date_expr}, '%Y-%m') AS stat_month, COUNT(*)
                FROM {q_table}
                WHERE {date_expr} IS NOT NULL
                GROUP BY stat_month
                """,
            )

        for field in ("category", "platform", "family", "filetype"):
            if field in columns:
                stats[field] = fetch_one_counter(
                    source_conn,
                    f"""
                    SELECT TRIM({quote_identifier(field)}) AS stat_key, COUNT(*)
                    FROM {q_table}
                    WHERE {quote_identifier(field)} IS NOT NULL
                      AND TRIM({quote_identifier(field)}) <> ''
                    GROUP BY stat_key
                    """,
                )

    return stats


def delete_table_stats(stats_conn, source: SourceDb, table_name: str) -> None:
    cursor = stats_conn.cursor()
    for stats_table in (
        "sample_table_total_stats",
        "sample_table_yearly_stats",
        "sample_table_monthly_stats",
        "sample_table_category_stats",
        "sample_table_platform_stats",
        "sample_table_family_stats",
        "sample_table_filetype_stats",
    ):
        cursor.execute(
            f"""
            DELETE FROM {quote_identifier(stats_table)}
            WHERE sample_class = %s
              AND file_kind = %s
              AND source_db = %s
              AND source_table = %s
            """,
            (source.sample_class, source.file_kind, source.db_name, table_name),
        )


def upsert_table_stats(stats_conn, source: SourceDb, table_name: str, stats: dict[str, Any]) -> None:
    cursor = stats_conn.cursor()
    cursor.execute(
        """
        INSERT INTO sample_table_total_stats
          (sample_class, file_kind, source_db, source_table, total_samples,
           has_vt_count, has_vt_summary_count, has_vt_mitre_count)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
          total_samples = VALUES(total_samples),
          has_vt_count = VALUES(has_vt_count),
          has_vt_summary_count = VALUES(has_vt_summary_count),
          has_vt_mitre_count = VALUES(has_vt_mitre_count)
        """,
        (
            source.sample_class,
            source.file_kind,
            source.db_name,
            table_name,
            stats["total"],
            stats["has_vt"],
            stats["has_vt_summary"],
            stats["has_vt_mitre"],
        ),
    )

    table_map = (
        ("sample_table_yearly_stats", "year", "yearly"),
        ("sample_table_category_stats", "category", "category"),
        ("sample_table_platform_stats", "platform", "platform"),
        ("sample_table_family_stats", "family", "family"),
        ("sample_table_filetype_stats", "filetype", "filetype"),
    )
    for table, key_column, stats_key in table_map:
        rows = [
            (source.sample_class, source.file_kind, source.db_name, table_name, key, count)
            for key, count in stats[stats_key].items()
        ]
        execute_many(
            cursor,
            f"""
            INSERT INTO {quote_identifier(table)}
              (sample_class, file_kind, source_db, source_table, {key_column}, total_samples)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE total_samples = VALUES(total_samples)
            """,
            rows,
        )

    monthly_rows = []
    for year_month, count in stats["monthly"].items():
        year_str, month_str = str(year_month).split("-", 1)
        monthly_rows.append(
            (
                source.sample_class,
                source.file_kind,
                source.db_name,
                table_name,
                int(year_str),
                int(month_str),
                count,
            )
        )
    execute_many(
        cursor,
        """
        INSERT INTO sample_table_monthly_stats
          (sample_class, file_kind, source_db, source_table, year, month, total_samples)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE total_samples = VALUES(total_samples)
        """,
        monthly_rows,
    )


def update_source_state(
    stats_conn,
    source: SourceDb,
    table_name: str,
    row_count: int,
    last_updated_at: Any,
) -> None:
    cursor = stats_conn.cursor()
    cursor.execute(
        """
        INSERT INTO sample_source_table_state
          (sample_class, file_kind, source_db, source_table, row_count, last_updated_at)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
          row_count = VALUES(row_count),
          last_updated_at = VALUES(last_updated_at)
        """,
        (source.sample_class, source.file_kind, source.db_name, table_name, row_count, last_updated_at),
    )


def refresh_source_tables(
    mysql_cfg: dict[str, Any],
    stats_db: str,
    source: SourceDb,
    mode: str,
    only_tables: set[str] | None = None,
    force: bool = False,
) -> tuple[int, int]:
    refreshed = 0
    skipped = 0

    with connect(mysql_cfg, source.db_name) as source_conn, connect(mysql_cfg, stats_db) as stats_conn:
        stats_conn.start_transaction()
        table_names = sorted(only_tables) if only_tables else [f"sample_{suffix}" for suffix in HEX_SUFFIXES]
        for table_name in table_names:
            if not table_exists(source_conn, source.db_name, table_name):
                skipped += 1
                continue

            columns = get_columns(source_conn, source.db_name, table_name)
            row_count, last_updated_at = get_table_signature(source_conn, source.db_name, table_name, columns)
            previous_state = get_source_state(stats_conn, source, table_name)

            if not force and mode == "incremental" and previous_state == (row_count, last_updated_at):
                skipped += 1
                continue

            delete_table_stats(stats_conn, source, table_name)
            table_stats = fetch_table_stats(source_conn, table_name, source, columns)
            upsert_table_stats(stats_conn, source, table_name, table_stats)
            update_source_state(stats_conn, source, table_name, row_count, last_updated_at)
            refreshed += 1

        stats_conn.commit()

    return refreshed, skipped


def truncate_table_level_stats(mysql_cfg: dict[str, Any], stats_db: str) -> None:
    tables = (
        "sample_source_table_state",
        "sample_table_total_stats",
        "sample_table_yearly_stats",
        "sample_table_monthly_stats",
        "sample_table_category_stats",
        "sample_table_platform_stats",
        "sample_table_family_stats",
        "sample_table_filetype_stats",
    )
    with connect(mysql_cfg, stats_db) as conn:
        cursor = conn.cursor()
        for table in tables:
            cursor.execute(f"TRUNCATE TABLE {quote_identifier(table)}")
        conn.commit()


def rebuild_rollups(mysql_cfg: dict[str, Any], stats_db: str) -> None:
    with connect(mysql_cfg, stats_db) as conn:
        cursor = conn.cursor()
        for table in (
            "sample_total_stats",
            "sample_behavior_stats",
            "sample_yearly_stats",
            "sample_monthly_stats",
            "sample_category_stats",
            "sample_platform_stats",
            "sample_family_stats",
            "sample_filetype_stats",
        ):
            cursor.execute(f"TRUNCATE TABLE {quote_identifier(table)}")

        cursor.execute(
            """
            INSERT INTO sample_total_stats
              (sample_class, file_kind, total_samples, has_vt_count,
               has_vt_summary_count, has_vt_mitre_count)
            SELECT sample_class, file_kind,
                   SUM(total_samples),
                   SUM(has_vt_count),
                   SUM(has_vt_summary_count),
                   SUM(has_vt_mitre_count)
            FROM sample_table_total_stats
            GROUP BY sample_class, file_kind
            """
        )
        cursor.execute(
            """
            INSERT INTO sample_behavior_stats
              (sample_class, file_kind, total_samples, has_vt_1,
               has_vt_summary_1, has_vt_mitre_1)
            SELECT sample_class, file_kind,
                   SUM(total_samples),
                   SUM(has_vt_count),
                   SUM(has_vt_summary_count),
                   SUM(has_vt_mitre_count)
            FROM sample_table_total_stats
            WHERE sample_class = 'malicious'
            GROUP BY sample_class, file_kind
            """
        )

        cursor.execute(
            """
            INSERT INTO sample_monthly_stats
              (sample_class, file_kind, year, month, total_samples)
            SELECT sample_class, file_kind, year, month, SUM(total_samples)
            FROM sample_table_monthly_stats
            GROUP BY sample_class, file_kind, year, month
            """
        )

        rollup_map = (
            ("sample_yearly_stats", "sample_table_yearly_stats", "year"),
            ("sample_category_stats", "sample_table_category_stats", "category"),
            ("sample_platform_stats", "sample_table_platform_stats", "platform"),
            ("sample_family_stats", "sample_table_family_stats", "family"),
            ("sample_filetype_stats", "sample_table_filetype_stats", "filetype"),
        )
        for target, source, key_column in rollup_map:
            cursor.execute(
                f"""
                INSERT INTO {quote_identifier(target)}
                  (sample_class, file_kind, {key_column}, total_samples)
                SELECT sample_class, file_kind, {key_column}, SUM(total_samples)
                FROM {quote_identifier(source)}
                GROUP BY sample_class, file_kind, {key_column}
                """
            )
        conn.commit()


def insert_refresh_log(
    mysql_cfg: dict[str, Any],
    stats_db: str,
    mode: str,
    status: str,
    refreshed: int,
    skipped: int,
    started_at: datetime,
    message: str | None = None,
) -> None:
    with connect(mysql_cfg, stats_db) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO sample_stats_refresh_log
              (mode, started_at, finished_at, status, refreshed_tables, skipped_tables, message)
            VALUES (%s, %s, NOW(), %s, %s, %s, %s)
            """,
            (mode, started_at, status, refreshed, skipped, message),
        )
        conn.commit()


def parse_changed_tables(values: list[str] | None) -> dict[str, set[str]]:
    changed: dict[str, set[str]] = defaultdict(set)
    for value in values or []:
        if "." not in value:
            raise ValueError(f"Invalid --changed-table value: {value}. Expected db.sample_xx")
        db_name, table_name = value.split(".", 1)
        quote_identifier(db_name)
        quote_identifier(table_name)
        if not table_name.startswith("sample_"):
            raise ValueError(f"Invalid sample table: {table_name}")
        changed[db_name].add(table_name)
    return changed


def refresh_stats(config_path: Path, mode: str, changed_table_values: list[str] | None = None) -> bool:
    started_at = datetime.now()
    mysql_cfg, stats_db, sources = load_config(config_path)
    changed_tables = parse_changed_tables(changed_table_values)
    create_database_and_tables(mysql_cfg, stats_db)
    register_sources(mysql_cfg, stats_db, sources)
    stale_deleted = cleanup_stale_source_stats(mysql_cfg, stats_db, sources)
    if stale_deleted:
        LOGGER.warning("removed %s stale cached stat rows before refresh", stale_deleted)

    if changed_tables and mode == "full":
        raise ValueError("--changed-table cannot be used with --mode full")

    if mode == "full":
        truncate_table_level_stats(mysql_cfg, stats_db)

    refreshed_total = 0
    skipped_total = 0
    try:
        source_by_db = {source.db_name: source for source in sources}
        unknown_dbs = sorted(set(changed_tables) - set(source_by_db))
        if unknown_dbs:
            raise ValueError(f"--changed-table contains DBs not configured in malicious_dbs/benign_dbs: {unknown_dbs}")

        for source in sources:
            only_tables = changed_tables.get(source.db_name) if changed_tables else None
            if changed_tables and not only_tables:
                continue

            refreshed, skipped = refresh_source_tables(
                mysql_cfg,
                stats_db,
                source,
                mode,
                only_tables=only_tables,
                force=bool(changed_tables),
            )
            refreshed_total += refreshed
            skipped_total += skipped
            LOGGER.info(
                "source=%s/%s db=%s refreshed=%s skipped=%s",
                source.sample_class,
                source.file_kind,
                source.db_name,
                refreshed,
                skipped,
            )

        rebuild_rollups(mysql_cfg, stats_db)
        insert_refresh_log(
            mysql_cfg,
            stats_db,
            mode,
            "success",
            refreshed_total,
            skipped_total,
            started_at,
        )
        LOGGER.info("sample_stats refresh success: refreshed=%s skipped=%s", refreshed_total, skipped_total)
        return True
    except Exception as exc:
        LOGGER.exception("sample_stats refresh failed")
        try:
            insert_refresh_log(
                mysql_cfg,
                stats_db,
                mode,
                "failed",
                refreshed_total,
                skipped_total,
                started_at,
                str(exc),
            )
        except Exception:
            LOGGER.exception("failed to write refresh log")
        return False


def parse_args(argv: list[str]) -> argparse.Namespace:
    default_config = Path(__file__).resolve().parents[3] / "config.ini"
    parser = argparse.ArgumentParser(description="Refresh sample_stats rollup tables.")
    parser.add_argument("--config", default=str(default_config), help="Path to web/fastapi/config.ini")
    parser.add_argument(
        "--mode",
        choices=("full", "incremental"),
        default="full",
        help="full refresh or incremental refresh by table updated_at signature",
    )
    parser.add_argument(
        "--changed-table",
        action="append",
        default=[],
        help="Refresh only a known changed table, format: database.sample_xx. Can be passed multiple times.",
    )
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    ok = refresh_stats(Path(args.config).resolve(), args.mode, args.changed_table)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
