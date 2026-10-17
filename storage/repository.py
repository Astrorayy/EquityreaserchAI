"""Repository: translate between the pydantic schema and SQLite rows."""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timezone
from typing import Dict, Iterable, List, Optional

from financial_engine.schemas import (
    CompanyProfile,
    ComparableCompany,
    Currency,
    ExtractionStatus,
    FiscalPeriod,
    FinancialDataSet,
    LineItem,
    Observation,
    PeriodType,
    Scale,
    SourceRecord,
    SourceTier,
)
from storage.database import Database, dumps, loads


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------
# Companies
# --------------------------------------------------------------------------


def upsert_company(db: Database, profile: CompanyProfile) -> None:
    with db.connect() as conn:
        conn.execute(
            """
            INSERT INTO companies (
                company_id, name, short_name, ticker, exchange, country, currency,
                sector, industry, fiscal_year_end, sec_cik, profile_json, updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(company_id) DO UPDATE SET
                name=excluded.name, short_name=excluded.short_name, ticker=excluded.ticker,
                exchange=excluded.exchange, country=excluded.country,
                currency=excluded.currency, sector=excluded.sector,
                industry=excluded.industry, fiscal_year_end=excluded.fiscal_year_end,
                sec_cik=excluded.sec_cik, profile_json=excluded.profile_json,
                updated_at=excluded.updated_at
            """,
            (
                profile.company_id, profile.name, profile.short_name, profile.ticker,
                profile.exchange, profile.country, profile.currency.value,
                profile.sector, profile.industry, profile.fiscal_year_end,
                profile.sec_cik, profile.model_dump_json(), _now(),
            ),
        )


def list_companies(db: Database) -> List[CompanyProfile]:
    out: List[CompanyProfile] = []
    with db.connect() as conn:
        for row in conn.execute("SELECT profile_json FROM companies ORDER BY name"):
            raw = loads(row["profile_json"])
            if raw:
                out.append(CompanyProfile.model_validate(raw))
    return out


def get_company(db: Database, company_id: str) -> Optional[CompanyProfile]:
    with db.connect() as conn:
        row = conn.execute(
            "SELECT profile_json FROM companies WHERE company_id = ?", (company_id,)
        ).fetchone()
    if row is None:
        return None
    raw = loads(row["profile_json"])
    return CompanyProfile.model_validate(raw) if raw else None


# --------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------


def upsert_source(db: Database, source: SourceRecord, company_id: Optional[str] = None) -> None:
    with db.connect() as conn:
        conn.execute(
            """
            INSERT INTO sources (
                source_id, company_id, name, url, publisher, publication_date,
                document_type, tier, currency, scale, retrieved_at, notes
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(source_id) DO UPDATE SET
                name=excluded.name, url=excluded.url, publisher=excluded.publisher,
                publication_date=excluded.publication_date,
                document_type=excluded.document_type, tier=excluded.tier,
                currency=excluded.currency, scale=excluded.scale,
                retrieved_at=excluded.retrieved_at, notes=excluded.notes
            """,
            (
                source.source_id, company_id, source.name, source.url, source.publisher,
                source.publication_date.isoformat() if source.publication_date else None,
                source.document_type, int(source.tier),
                source.currency.value if source.currency else None,
                source.scale.value if source.scale else None,
                source.retrieved_at.isoformat(), source.notes,
            ),
        )


def list_sources(db: Database, company_id: Optional[str] = None) -> List[SourceRecord]:
    query = "SELECT * FROM sources"
    params: tuple = ()
    if company_id:
        query += " WHERE company_id = ? OR company_id IS NULL"
        params = (company_id,)
    query += " ORDER BY COALESCE(publication_date, '') DESC, name"
    out: List[SourceRecord] = []
    with db.connect() as conn:
        for row in conn.execute(query, params):
            out.append(
                SourceRecord(
                    source_id=row["source_id"],
                    name=row["name"],
                    url=row["url"],
                    publisher=row["publisher"],
                    publication_date=_parse_date(row["publication_date"]),
                    document_type=row["document_type"],
                    tier=SourceTier(row["tier"]) if row["tier"] else SourceTier.OFFICIAL,
                    currency=Currency(row["currency"]) if row["currency"] else None,
                    scale=Scale(row["scale"]) if row["scale"] else None,
                    retrieved_at=_parse_dt(row["retrieved_at"]) or datetime.now(timezone.utc),
                    notes=row["notes"],
                )
            )
    return out


# --------------------------------------------------------------------------
# Observations
# --------------------------------------------------------------------------


def upsert_observations(db: Database, observations: Iterable[Observation]) -> int:
    rows = []
    for o in observations:
        rows.append(
            (
                o.company_id, o.line_item.value, o.period.label, o.period.fiscal_year,
                o.period.period_type.value, o.value, o.currency.value,
                o.reported_value, o.reported_scale.value if o.reported_scale else None,
                o.reported_currency.value if o.reported_currency else None,
                o.status.value, o.source_id, o.extraction_method, o.notes, _now(),
            )
        )
    if not rows:
        return 0
    with db.connect() as conn:
        conn.executemany(
            """
            INSERT INTO observations (
                company_id, line_item, period_label, fiscal_year, period_type,
                value_millions, currency, reported_value, reported_scale,
                reported_currency, status, source_id, extraction_method, notes, created_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(company_id, line_item, period_label, status, source_key)
            DO UPDATE SET
                value_millions=excluded.value_millions, currency=excluded.currency,
                reported_value=excluded.reported_value,
                reported_scale=excluded.reported_scale,
                reported_currency=excluded.reported_currency,
                extraction_method=excluded.extraction_method, notes=excluded.notes,
                created_at=excluded.created_at
            """,
            rows,
        )
    return len(rows)


def load_dataset(db: Database, company_id: str) -> FinancialDataSet:
    """Rebuild a :class:`FinancialDataSet` from storage."""
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT * FROM observations WHERE company_id = ? ORDER BY fiscal_year, period_type",
            (company_id,),
        ).fetchall()
    observations: List[Observation] = []
    currency = Currency.USD
    for row in rows:
        currency = Currency(row["currency"])
        observations.append(
            Observation(
                company_id=row["company_id"],
                line_item=LineItem(row["line_item"]),
                period=FiscalPeriod(
                    fiscal_year=row["fiscal_year"],
                    period_type=PeriodType(row["period_type"]),
                    period_number=_period_number(row["period_label"]),
                ),
                value=row["value_millions"],
                currency=currency,
                reported_value=row["reported_value"],
                reported_scale=Scale(row["reported_scale"]) if row["reported_scale"] else None,
                reported_currency=(
                    Currency(row["reported_currency"]) if row["reported_currency"] else None
                ),
                status=ExtractionStatus(row["status"]),
                source_id=row["source_id"],
                extraction_method=row["extraction_method"],
                notes=row["notes"],
            )
        )
    return FinancialDataSet(company_id=company_id, currency=currency, observations=observations)


def delete_observations(db: Database, company_id: str, status: Optional[ExtractionStatus] = None) -> int:
    """Remove observations, optionally only those with a given status."""
    with db.connect() as conn:
        if status is None:
            cur = conn.execute("DELETE FROM observations WHERE company_id = ?", (company_id,))
        else:
            cur = conn.execute(
                "DELETE FROM observations WHERE company_id = ? AND status = ?",
                (company_id, status.value),
            )
        return cur.rowcount


# --------------------------------------------------------------------------
# Valuation runs, peers, reports
# --------------------------------------------------------------------------


def save_valuation_run(
    db: Database,
    run_id: str,
    company_id: str,
    method: str,
    scenario: Optional[str],
    currency: Optional[str],
    assumptions: Dict,
    result: Dict,
    value_per_share: Optional[float] = None,
    simulated: bool = False,
) -> None:
    with db.connect() as conn:
        conn.execute(
            """
            INSERT INTO valuation_runs (
                run_id, company_id, method, scenario, currency, created_at,
                assumptions, result, value_per_share, simulated
            ) VALUES (?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(run_id) DO UPDATE SET
                assumptions=excluded.assumptions, result=excluded.result,
                value_per_share=excluded.value_per_share, created_at=excluded.created_at
            """,
            (
                run_id, company_id, method, scenario, currency, _now(),
                dumps(assumptions), dumps(result), value_per_share, int(simulated),
            ),
        )


def list_valuation_runs(db: Database, company_id: Optional[str] = None, limit: int = 50) -> List[Dict]:
    query = "SELECT * FROM valuation_runs"
    params: tuple = ()
    if company_id:
        query += " WHERE company_id = ?"
        params = (company_id,)
    query += " ORDER BY created_at DESC LIMIT ?"
    params = params + (limit,)
    out: List[Dict] = []
    with db.connect() as conn:
        for row in conn.execute(query, params):
            item = dict(row)
            item["assumptions"] = loads(item["assumptions"])
            item["result"] = loads(item["result"])
            out.append(item)
    return out


def upsert_peers(db: Database, company_id: str, peers: Iterable[ComparableCompany]) -> int:
    rows = [
        (company_id, p.name, p.model_dump_json(), _now())
        for p in peers
    ]
    if not rows:
        return 0
    with db.connect() as conn:
        conn.executemany(
            """
            INSERT INTO peers (company_id, name, payload, updated_at)
            VALUES (?,?,?,?)
            ON CONFLICT(company_id, name) DO UPDATE SET
                payload=excluded.payload, updated_at=excluded.updated_at
            """,
            rows,
        )
    return len(rows)


def list_peers(db: Database, company_id: str) -> List[ComparableCompany]:
    out: List[ComparableCompany] = []
    with db.connect() as conn:
        for row in conn.execute(
            "SELECT payload FROM peers WHERE company_id = ? ORDER BY name", (company_id,)
        ):
            raw = loads(row["payload"])
            if raw:
                out.append(ComparableCompany.model_validate(raw))
    return out


def delete_peer(db: Database, company_id: str, name: str) -> int:
    with db.connect() as conn:
        return conn.execute(
            "DELETE FROM peers WHERE company_id = ? AND name = ?", (company_id, name)
        ).rowcount


def save_report(db: Database, report_id: str, company_id: str, title: str, path: str, meta: Optional[Dict] = None) -> None:
    with db.connect() as conn:
        conn.execute(
            """
            INSERT INTO reports (report_id, company_id, title, path, created_at, meta)
            VALUES (?,?,?,?,?,?)
            ON CONFLICT(report_id) DO UPDATE SET
                title=excluded.title, path=excluded.path, meta=excluded.meta
            """,
            (report_id, company_id, title, path, _now(), dumps(meta or {})),
        )


def list_reports(db: Database, company_id: Optional[str] = None) -> List[Dict]:
    query = "SELECT * FROM reports"
    params: tuple = ()
    if company_id:
        query += " WHERE company_id = ?"
        params = (company_id,)
    query += " ORDER BY created_at DESC"
    out: List[Dict] = []
    with db.connect() as conn:
        for row in conn.execute(query, params):
            item = dict(row)
            item["meta"] = loads(item["meta"])
            out.append(item)
    return out


# --------------------------------------------------------------------------
# Manual overrides
# --------------------------------------------------------------------------


def set_manual_override(db: Database, company_id: str, field: str, value: Optional[float], note: str = "") -> None:
    with db.connect() as conn:
        conn.execute(
            """
            INSERT INTO manual_overrides (company_id, field, value, note, updated_at)
            VALUES (?,?,?,?,?)
            ON CONFLICT(company_id, field) DO UPDATE SET
                value=excluded.value, note=excluded.note, updated_at=excluded.updated_at
            """,
            (company_id, field, value, note, _now()),
        )


def get_manual_overrides(db: Database, company_id: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    with db.connect() as conn:
        for row in conn.execute(
            "SELECT field, value FROM manual_overrides WHERE company_id = ? AND value IS NOT NULL",
            (company_id,),
        ):
            out[row["field"]] = row["value"]
    return out


# --------------------------------------------------------------------------
# HTTP cache
# --------------------------------------------------------------------------


def cache_get(db: Database, url: str, max_age_seconds: Optional[int] = None) -> Optional[Dict]:
    with db.connect() as conn:
        row = conn.execute("SELECT * FROM http_cache WHERE url = ?", (url,)).fetchone()
    if row is None:
        return None
    fetched = _parse_dt(row["fetched_at"])
    if max_age_seconds is not None and fetched is not None:
        age = (datetime.now(timezone.utc) - fetched).total_seconds()
        if age > max_age_seconds:
            return None
    return {
        "url": row["url"],
        "fetched_at": fetched,
        "status_code": row["status_code"],
        "content_type": row["content_type"],
        "body": row["body"],
        "headers": loads(row["headers"]) or {},
    }


def cache_put(db: Database, url: str, status_code: int, content_type: str, body: bytes, headers: Optional[Dict] = None) -> None:
    with db.connect() as conn:
        conn.execute(
            """
            INSERT INTO http_cache (url, fetched_at, status_code, content_type, body, headers)
            VALUES (?,?,?,?,?,?)
            ON CONFLICT(url) DO UPDATE SET
                fetched_at=excluded.fetched_at, status_code=excluded.status_code,
                content_type=excluded.content_type, body=excluded.body,
                headers=excluded.headers
            """,
            (url, _now(), status_code, content_type, body, dumps(headers or {})),
        )


def cache_stats(db: Database) -> Dict[str, object]:
    with db.connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n, COALESCE(SUM(LENGTH(body)), 0) AS size FROM http_cache"
        ).fetchone()
    return {"entries": row["n"], "bytes": row["size"]}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _parse_date(raw: Optional[str]) -> Optional[date]:
    if not raw:
        return None
    try:
        return date.fromisoformat(raw[:10])
    except ValueError:
        return None


def _parse_dt(raw: Optional[str]) -> Optional[datetime]:
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _period_number(label: str) -> Optional[int]:
    import re

    m = re.match(r"^[TS](\d) ", label)
    return int(m.group(1)) if m else None
