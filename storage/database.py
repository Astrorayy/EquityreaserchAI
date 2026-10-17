"""SQLite persistence.

Design notes
------------
* A fresh connection is opened per operation.  Streamlit re-runs the script in
  worker threads, and SQLite connections are thread-bound; opening per call is
  the simplest correct pattern at this scale.
* The schema stores *structured records*, not blobs of Python objects: company
  profiles, sources, observations, valuation runs, generated reports and an
  HTTP cache.
* Monetary values are stored in millions of the reporting currency, exactly as
  in the in-memory schema, plus the original reported figure and its scale.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS companies (
    company_id     TEXT PRIMARY KEY,
    name           TEXT NOT NULL,
    short_name     TEXT,
    ticker         TEXT,
    exchange       TEXT,
    country        TEXT,
    currency       TEXT,
    sector         TEXT,
    industry       TEXT,
    fiscal_year_end TEXT,
    sec_cik        TEXT,
    profile_json   TEXT NOT NULL,
    updated_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sources (
    source_id        TEXT PRIMARY KEY,
    company_id       TEXT,
    name             TEXT NOT NULL,
    url              TEXT,
    publisher        TEXT,
    publication_date TEXT,
    document_type    TEXT,
    tier             INTEGER,
    currency         TEXT,
    scale            TEXT,
    retrieved_at     TEXT NOT NULL,
    notes            TEXT
);

CREATE TABLE IF NOT EXISTS observations (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id         TEXT NOT NULL,
    line_item          TEXT NOT NULL,
    period_label       TEXT NOT NULL,
    fiscal_year        INTEGER NOT NULL,
    period_type        TEXT NOT NULL,
    value_millions     REAL,
    currency           TEXT NOT NULL,
    reported_value     REAL,
    reported_scale     TEXT,
    reported_currency  TEXT,
    status             TEXT NOT NULL,
    source_id          TEXT,
    extraction_method  TEXT,
    notes              TEXT,
    created_at         TEXT NOT NULL,
    -- NULL source_id would be "distinct" in a plain UNIQUE constraint, which
    -- would let the same figure be inserted twice without a source.  The
    -- generated key normalises NULL to '' so duplicates are actually blocked.
    source_key         TEXT GENERATED ALWAYS AS (IFNULL(source_id, '')) VIRTUAL,
    UNIQUE (company_id, line_item, period_label, status, source_key)
);

CREATE INDEX IF NOT EXISTS idx_obs_lookup
    ON observations (company_id, line_item, period_label);

CREATE TABLE IF NOT EXISTS valuation_runs (
    run_id       TEXT PRIMARY KEY,
    company_id   TEXT NOT NULL,
    method       TEXT NOT NULL,
    scenario     TEXT,
    currency     TEXT,
    created_at   TEXT NOT NULL,
    assumptions  TEXT NOT NULL,
    result       TEXT NOT NULL,
    value_per_share REAL,
    simulated    INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS reports (
    report_id    TEXT PRIMARY KEY,
    company_id   TEXT NOT NULL,
    title        TEXT,
    path         TEXT,
    created_at   TEXT NOT NULL,
    meta         TEXT
);

CREATE TABLE IF NOT EXISTS peers (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id TEXT NOT NULL,
    name       TEXT NOT NULL,
    payload    TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (company_id, name)
);

CREATE TABLE IF NOT EXISTS http_cache (
    url          TEXT PRIMARY KEY,
    fetched_at   TEXT NOT NULL,
    status_code  INTEGER,
    content_type TEXT,
    body         BLOB,
    headers      TEXT
);

CREATE TABLE IF NOT EXISTS manual_overrides (
    company_id TEXT NOT NULL,
    field      TEXT NOT NULL,
    value      REAL,
    note       TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (company_id, field)
);
"""


@dataclass(frozen=True)
class DatabaseInfo:
    path: Path
    size_bytes: int
    tables: int
    schema_version: int


class Database:
    """Thin SQLite wrapper: connect, migrate, inspect."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # -- connection --------------------------------------------------------

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def initialise(self) -> None:
        with self.connect() as conn:
            conn.executescript(_SCHEMA)
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )

    # -- introspection -----------------------------------------------------

    def info(self) -> DatabaseInfo:
        tables = 0
        version = 0
        if self.path.exists():
            with self.connect() as conn:
                tables = conn.execute(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
                ).fetchone()[0]
                try:
                    row = conn.execute(
                        "SELECT value FROM meta WHERE key='schema_version'"
                    ).fetchone()
                    version = int(row[0]) if row else 0
                except sqlite3.Error:
                    version = 0
        size = self.path.stat().st_size if self.path.exists() else 0
        return DatabaseInfo(self.path, size, tables, version)

    def table_counts(self) -> dict[str, int]:
        tables = (
            "companies", "sources", "observations", "valuation_runs",
            "reports", "peers", "http_cache", "manual_overrides",
        )
        out: dict[str, int] = {}
        with self.connect() as conn:
            for t in tables:
                try:
                    out[t] = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                except sqlite3.Error:
                    out[t] = -1
        return out

    def reset(self, tables: Optional[list[str]] = None) -> None:
        """Delete stored rows. Used by the Settings page and the tests."""
        targets = tables or [
            "companies", "sources", "observations", "valuation_runs",
            "reports", "peers", "http_cache", "manual_overrides",
        ]
        with self.connect() as conn:
            for t in targets:
                conn.execute(f"DELETE FROM {t}")


def dumps(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


def loads(raw: Optional[str]) -> object:
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return None
