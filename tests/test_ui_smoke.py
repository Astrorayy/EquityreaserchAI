"""Opt-in UI smoke test: render every page with Streamlit's AppTest harness.

Skipped by default because it starts a full script run for each page and takes
tens of seconds.  To run it, point the application at a scratch database and
set the opt-in flag (PowerShell)::

    $env:ERAI_RUN_UI_SMOKE = '1'
    $env:ERAI_DATA_DIR = './_smoke_data'
    .\\.venv\\Scripts\\python.exe -m pytest tests/test_ui_smoke.py -q

The test seeds the scratch database with the SIMULATED dataset for SanDisk, so
no real company data is touched and no external request is made.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("ERAI_RUN_UI_SMOKE") != "1",
    reason=(
        "UI smoke test disabled; set ERAI_RUN_UI_SMOKE=1 and ERAI_DATA_DIR to a "
        "scratch directory to run it."
    ),
)

os.environ.setdefault("ERAI_LANG", "fr")

ROOT = Path(__file__).resolve().parents[1]
PAGES = (
    "dashboard",
    "research",
    "statements",
    "valuation",
    "comparables",
    "assistant",
    "sources",
    "settings",
)


def seed_demo_data() -> str:
    """Attach the simulated dataset and simulated peers to SanDisk."""
    from data_sources.company_registry import get_profile, list_profiles
    from data_sources.sample import synthetic_dataset
    from storage.database import Database
    from storage import repository as repo
    import workflows
    from config import get_settings

    settings = get_settings()
    settings.ensure_dirs()
    db = Database(settings.db_path)
    db.initialise()
    for profile in list_profiles():
        repo.upsert_company(db, profile)

    profile = get_profile("sandisk")
    dataset = workflows.relabel_dataset(
        synthetic_dataset(), profile.company_id, currency=profile.currency
    )
    repo.upsert_observations(db, dataset.observations)
    repo.upsert_source(
        db,
        repo.SourceRecord(
            source_id="simulated:demo",
            name="Jeu de démonstration SIMULÉ",
            publisher="Equity Research AI",
            document_type="données synthétiques",
            currency=profile.currency,
        ),
        profile.company_id,
    )
    for peer in repo.list_peers(db, profile.company_id):
        repo.delete_peer(db, profile.company_id, peer.name)
    repo.upsert_peers(
        db,
        profile.company_id,
        [
            repo.ComparableCompany(
                name="Pair simulé A", ticker="SIM-A", currency=profile.currency,
                market_cap=18000, ev_ebitda=9.0, pe=12.0, pb=1.9, sector="Mémoire",
            ),
            repo.ComparableCompany(
                name="Pair simulé B", ticker="SIM-B", currency=profile.currency,
                market_cap=26000, ev_ebitda=11.0, pe=15.0, pb=4.0, sector="Mémoire",
            ),
            repo.ComparableCompany(
                name="Pair simulé C", ticker="SIM-C", currency=profile.currency,
                market_cap=21000, ev_ebitda=10.0, pe=14.0, pb=2.4, sector="Mémoire",
            ),
        ],
    )
    return profile.company_id


def _interact(at, page: str) -> str:
    """Click the page's main action, if it has one, and report what happened."""
    if page == "valuation":
        for button in at.button:
            if (button.label or "").strip() == "Lancer la valorisation":
                button.click()
                at.run()
                return "clicked: Lancer la valorisation"
        return "submit button not found"
    if page == "comparables":
        try:
            at.button(key="run_multiples").click()
        except KeyError:
            return "run_multiples button not found"
        at.run()
        return "clicked: Lancer la valorisation par comparables"
    if page == "assistant":
        if not at.chat_input:
            return "chat input not found (expected without an API key)"
        at.chat_input[0].set_value("Quel est le chiffre d'affaires de la dernière période ?")
        at.run()
        return "asked the assistant a question"
    return "no action"


def test_all_pages_render_without_exception(tmp_path):
    if not os.environ.get("ERAI_DATA_DIR"):
        pytest.skip("ERAI_DATA_DIR must point to a scratch directory for the UI smoke test.")

    from streamlit.testing.v1 import AppTest

    company_id = seed_demo_data()
    failures: list[str] = []

    for page in PAGES:
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90)
        try:
            at.session_state["nav_page"] = page
            at.session_state["company_id"] = company_id
            at.run()
            action = _interact(at, page)
            if at.exception:
                details = "; ".join(
                    str(exc.value)[:300] if hasattr(exc, "value") else str(exc)
                    for exc in at.exception
                )
                failures.append(f"{page}: {details}")
                continue
            if at.error:
                messages = "; ".join(str(m.value)[:200] for m in at.error)
                failures.append(f"{page}: UI error messages: {messages}")
        except Exception as exc:  # noqa: BLE001 - the harness itself failed
            failures.append(f"{page}: harness error {type(exc).__name__}: {exc}")

    assert not failures, "UI smoke failures:\n" + "\n".join(failures)
