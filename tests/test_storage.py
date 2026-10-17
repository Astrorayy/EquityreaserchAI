"""Storage-layer tests: SQLite round-trips, upsert semantics, duplicate blocking.

These tests use a temporary database file and never touch the application
database in ``./data``.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from financial_engine.schemas import (
    ComparableCompany,
    Currency,
    ExtractionStatus,
    FiscalPeriod,
    LineItem,
    Observation,
    PeriodType,
    Scale,
    SourceRecord,
    SourceTier,
)
from storage import repository as repo
from storage.database import Database

COMPANY = "TESTCO"


@pytest.fixture()
def db(tmp_path):
    database = Database(tmp_path / "test.db")
    database.initialise()
    return database


def obs(
    item: LineItem,
    value: float,
    *,
    period: FiscalPeriod | None = None,
    status: ExtractionStatus = ExtractionStatus.REPORTED,
    source_id: str | None = "doc-1",
    currency: Currency = Currency.MAD,
    reported_value: float | None = None,
    reported_scale: Scale | None = None,
) -> Observation:
    period = period or FiscalPeriod(fiscal_year=2024, period_type=PeriodType.ANNUAL)
    return Observation(
        company_id=COMPANY,
        line_item=item,
        period=period,
        value=value,
        currency=currency,
        status=status,
        source_id=source_id,
        reported_value=reported_value,
        reported_scale=reported_scale,
    )


def test_initialise_creates_expected_tables(db):
    counts = db.table_counts()
    for table in ("companies", "sources", "observations", "valuation_runs", "reports", "peers"):
        assert table in counts
        assert counts[table] == 0


def test_observations_round_trip_through_load_dataset(db):
    period = FiscalPeriod(fiscal_year=2024, period_type=PeriodType.ANNUAL)
    dataset = [
        obs(LineItem.REVENUE, 1250.5, reported_value=1_250_500.0, reported_scale=Scale.THOUSANDS),
        obs(LineItem.NET_INCOME, 90.0, period=period),
        obs(
            LineItem.REVENUE,
            320.0,
            period=FiscalPeriod(
                fiscal_year=2024, period_type=PeriodType.QUARTERLY, period_number=1
            ),
        ),
    ]
    assert repo.upsert_observations(db, dataset) == 3

    loaded = repo.load_dataset(db, COMPANY)
    assert loaded.company_id == COMPANY
    assert loaded.currency == Currency.MAD
    assert loaded.value(LineItem.REVENUE, "FY2024") == pytest.approx(1250.5)
    assert loaded.value(LineItem.NET_INCOME, "FY2024") == pytest.approx(90.0)
    assert loaded.value(LineItem.REVENUE, "T1 2024") == pytest.approx(320.0)

    revenue = next(
        o for o in loaded.observations
        if o.line_item is LineItem.REVENUE and o.period.period_type is PeriodType.ANNUAL
    )
    assert revenue.reported_value == pytest.approx(1_250_500.0)
    assert revenue.reported_scale is Scale.THOUSANDS
    assert revenue.status is ExtractionStatus.REPORTED
    assert revenue.source_id == "doc-1"


def test_same_observation_twice_updates_instead_of_duplicating(db):
    repo.upsert_observations(db, [obs(LineItem.REVENUE, 1000.0)])
    repo.upsert_observations(db, [obs(LineItem.REVENUE, 1234.0)])

    assert db.table_counts()["observations"] == 1
    loaded = repo.load_dataset(db, COMPANY)
    assert loaded.value(LineItem.REVENUE, "FY2024") == pytest.approx(1234.0)


def test_missing_source_id_still_blocks_duplicates(db):
    # source_id NULL becomes the generated key '' -> the UNIQUE constraint must
    # still catch the second insert and turn it into an update.
    repo.upsert_observations(db, [obs(LineItem.REVENUE, 700.0, source_id=None)])
    repo.upsert_observations(db, [obs(LineItem.REVENUE, 800.0, source_id=None)])

    assert db.table_counts()["observations"] == 1
    assert repo.load_dataset(db, COMPANY).value(LineItem.REVENUE, "FY2024") == pytest.approx(800.0)


def test_reported_and_derived_values_can_coexist(db):
    repo.upsert_observations(
        db,
        [
            obs(LineItem.EBITDA, 200.0, status=ExtractionStatus.REPORTED),
            obs(LineItem.EBITDA, 205.0, status=ExtractionStatus.DERIVED),
        ],
    )
    assert db.table_counts()["observations"] == 2


def test_delete_observations_can_target_one_status(db):
    repo.upsert_observations(
        db,
        [
            obs(LineItem.REVENUE, 1000.0, status=ExtractionStatus.REPORTED),
            obs(LineItem.GROSS_PROFIT, 400.0, status=ExtractionStatus.DERIVED),
        ],
    )
    removed = repo.delete_observations(db, COMPANY, status=ExtractionStatus.DERIVED)

    assert removed == 1
    loaded = repo.load_dataset(db, COMPANY)
    assert loaded.value(LineItem.REVENUE, "FY2024") == pytest.approx(1000.0)
    assert loaded.value(LineItem.GROSS_PROFIT, "FY2024") is None


def test_peers_upsert_list_and_delete(db):
    peer = ComparableCompany(
        name="Pair A", ticker="PA", currency=Currency.MAD,
        market_cap=10_000.0, ev_ebitda=8.5, pe=12.0, pb=1.8,
    )
    assert repo.upsert_peers(db, COMPANY, [peer]) == 1

    updated = ComparableCompany(
        name="Pair A", ticker="PA", currency=Currency.MAD,
        market_cap=11_000.0, ev_ebitda=9.1, pe=13.0, pb=2.0,
    )
    repo.upsert_peers(db, COMPANY, [updated])

    peers = repo.list_peers(db, COMPANY)
    assert len(peers) == 1
    assert peers[0].market_cap == pytest.approx(11_000.0)
    assert peers[0].ev_ebitda == pytest.approx(9.1)

    assert repo.delete_peer(db, COMPANY, "Pair A") == 1
    assert repo.list_peers(db, COMPANY) == []


def test_sources_round_trip_with_metadata(db):
    source = SourceRecord(
        source_id="src-1",
        name="Rapport annuel 2024",
        url="https://example.test/annual-report.pdf",
        publisher="Société test",
        publication_date=date(2025, 3, 31),
        document_type="annual report",
        tier=SourceTier.OFFICIAL,
        currency=Currency.MAD,
        scale=Scale.MILLIONS,
        retrieved_at=datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
        notes="Document simulé pour les tests.",
    )
    repo.upsert_source(db, source, COMPANY)

    stored = repo.list_sources(db, COMPANY)
    assert len(stored) == 1
    got = stored[0]
    assert got.source_id == "src-1"
    assert got.name == "Rapport annuel 2024"
    assert got.publication_date == date(2025, 3, 31)
    assert got.tier is SourceTier.OFFICIAL
    assert got.currency is Currency.MAD
    assert got.scale is Scale.MILLIONS


def test_source_upsert_updates_instead_of_duplicating(db):
    repo.upsert_source(
        db, SourceRecord(source_id="src-1", name="Version 1"), COMPANY
    )
    repo.upsert_source(
        db, SourceRecord(source_id="src-1", name="Version 2"), COMPANY
    )

    sources = repo.list_sources(db, COMPANY)
    assert len(sources) == 1
    assert sources[0].name == "Version 2"


def test_manual_overrides_set_and_clear(db):
    repo.set_manual_override(db, COMPANY, "reference_price", 412.5, note="cours saisi")
    assert repo.get_manual_overrides(db, COMPANY) == {"reference_price": pytest.approx(412.5)}

    repo.set_manual_override(db, COMPANY, "reference_price", None)
    assert repo.get_manual_overrides(db, COMPANY) == {}


def test_cache_put_and_get(db):
    repo.cache_put(db, "https://example.test/data", 200, "text/plain", b"payload")
    hit = repo.cache_get(db, "https://example.test/data")
    assert hit is not None
    assert hit["status_code"] == 200
    assert hit["body"] == b"payload"
    assert repo.cache_get(db, "https://example.test/absent") is None
