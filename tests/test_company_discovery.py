"""Company discovery: search any listed company without network access in tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from data_sources import company_discovery as cd
from financial_engine.schemas import Currency

SEC_TABLE = {
    "0": {"cik_str": 1045810, "ticker": "NVDA", "title": "NVIDIA CORP"},
    "1": {"cik_str": 723125, "ticker": "MU", "title": "MICRON TECHNOLOGY INC"},
    "2": {"cik_str": 1652044, "ticker": "GOOGL", "title": "Alphabet Inc."},
    "3": {"cik_str": 1652044, "ticker": "GOOG", "title": "Alphabet Inc."},
}

YAHOO_PAYLOAD = {
    "quotes": [
        {"symbol": "AIR.PA", "longname": "Airbus SE", "exchDisp": "Paris", "quoteType": "EQUITY",
         "sectorDisp": "Produits industriels", "industryDisp": "Aérospatiale et défense"},
        {"symbol": "AIR=F", "shortname": "Airbus future", "exchDisp": "CME", "quoteType": "FUTURE"},
        {"symbol": "NVDA", "longname": "NVIDIA Corporation", "exchDisp": "NASDAQ", "quoteType": "EQUITY",
         "sectorDisp": "Technologie"},
        {"symbol": "BARC.L", "longname": "Barclays PLC", "exchDisp": "LSE", "quoteType": "EQUITY"},
    ]
}


class FakeHttp:
    def __init__(self, sec=SEC_TABLE, yahoo=YAHOO_PAYLOAD):
        self.sec, self.yahoo = sec, yahoo

    def get_json(self, url, **kwargs):
        if "sec.gov" in url:
            return self.sec, SimpleNamespace(error=None if self.sec else "blocked")
        return self.yahoo, SimpleNamespace(error=None if self.yahoo else "blocked")


SETTINGS = SimpleNamespace(sec_user_agent="tests (test@example.com)")


def test_make_company_id_is_storage_safe():
    assert cd.make_company_id("air.pa") == "AIR_PA"
    assert cd.make_company_id(" BRK-B ") == "BRK_B"
    assert cd.make_company_id("") == ""


@pytest.mark.parametrize(
    "ticker, exchange, expected",
    [
        ("AIR.PA", "Paris", ("France", Currency.EUR)),
        ("BARC.L", "LSE", ("Royaume-Uni", Currency.GBP)),
        ("NVDA", "NASDAQ", ("États-Unis", Currency.USD)),
        ("AIRE.XD", "DXE", ("", Currency.OTHER)),
        ("XYZ", "Bourse inconnue", ("", Currency.OTHER)),
    ],
)
def test_market_from_ticker(ticker, exchange, expected):
    assert cd.market_from_ticker(ticker, exchange) == expected


def test_sec_search_ranks_exact_ticker_and_dedupes_share_classes():
    hits, error = cd.search_sec("goog", http=FakeHttp(), settings=SETTINGS)
    assert error is None
    assert [h.ticker for h in hits] == ["GOOG"]  # one registrant, one entry
    assert hits[0].sec_cik == "0001652044"
    assert hits[0].source == cd.SOURCE_SEC


def test_sec_search_matches_names():
    hits, _ = cd.search_sec("micron", http=FakeHttp(), settings=SETTINGS)
    assert hits and hits[0].ticker == "MU"


def test_sec_unavailable_reports_error():
    hits, error = cd.search_sec("nvda", http=FakeHttp(sec=None), settings=SETTINGS)
    assert hits == [] and "SEC" in error


def test_yahoo_search_keeps_equities_and_flags_verification():
    hits, error = cd.search_yahoo("airbus", http=FakeHttp())
    assert error is None
    tickers = [h.ticker for h in hits]
    assert "AIR=F" not in tickers
    airbus = hits[0]
    assert (airbus.ticker, airbus.country, airbus.currency) == ("AIR.PA", "France", Currency.EUR)
    assert any("non officielle" in note for note in airbus.notes)
    barclays = next(h for h in hits if h.ticker == "BARC.L")
    assert any("pence" in note for note in barclays.notes)


def test_combined_search_prefers_official_sec_listing():
    result = cd.search_companies("nvidia", http=FakeHttp(), settings=SETTINGS)
    nvda = [c for c in result.candidates if c.ticker == "NVDA"]
    assert len(nvda) == 1
    assert nvda[0].source == cd.SOURCE_SEC
    assert nvda[0].sector == "Technologie"  # borrowed from Yahoo
    assert any(c.ticker == "AIR.PA" for c in result.candidates)


def test_combined_search_rejects_too_short_query():
    result = cd.search_companies("a", http=FakeHttp(), settings=SETTINGS)
    assert result.candidates == [] and result.errors


def test_us_only_search_skips_yahoo():
    result = cd.search_companies("airbus", include_international=False, http=FakeHttp(), settings=SETTINGS)
    assert all(c.source == cd.SOURCE_SEC for c in result.candidates)


def test_manual_profile_is_flagged_and_normalised():
    candidate = cd.manual_candidate(
        name="Attijariwafa Bank", ticker="atw", exchange="Bourse de Casablanca",
        country="Maroc", currency=Currency.MAD, sector="Banques", fiscal_year_end="1231",
    )
    profile = cd.candidate_to_profile(candidate)
    assert profile.company_id == "ATW"
    assert profile.currency == Currency.MAD
    assert profile.fiscal_year_end == "12-31"
    assert profile.sec_cik is None
    assert "manuellement" in profile.identity_notes


def test_unknown_currency_and_missing_fields_are_explicit():
    candidate = cd.manual_candidate(name="Société X", ticker="SX")
    profile = cd.candidate_to_profile(candidate)
    assert profile.exchange == "Non renseignée"
    assert profile.country == "Non renseigné"
    assert profile.fiscal_year_end is None
    assert "Devise non reconnue" in profile.identity_notes


def test_invalid_fiscal_year_end_is_dropped():
    candidate = cd.manual_candidate(name="Y", ticker="Y", fiscal_year_end="31/12")
    assert cd.candidate_to_profile(candidate).fiscal_year_end is None


def test_sector_key_uses_word_hints():
    from workflows import _sector_key

    assert _sector_key("Semiconductors & Related Devices") == "semiconductors"
    assert _sector_key("Mines et metaux") == "mining"
    assert _sector_key("Gold and Silver Ores") == "mining"
    assert _sector_key("Administration publique") == "generic"


def test_delete_company_keeps_data(tmp_path):
    from storage.database import Database
    from storage import repository as repo

    db = Database(tmp_path / "t.db")
    db.initialise()
    profile = cd.candidate_to_profile(cd.manual_candidate(name="Z Corp", ticker="ZZ"))
    repo.upsert_company(db, profile)
    assert repo.get_company(db, "ZZ") is not None
    assert repo.delete_company(db, "ZZ") is True
    assert repo.get_company(db, "ZZ") is None
    assert repo.delete_company(db, "ZZ") is False
