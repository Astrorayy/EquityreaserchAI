"""Streamlit Cloud secrets must reach the app through ``os.environ``.

Streamlit Community Cloud exposes configuration via ``st.secrets`` and does not
create environment variables; ``config.apply_streamlit_secrets`` bridges the two.
"""

from __future__ import annotations

import os
import sys
import types

import config


def test_streamlit_secrets_are_exported(monkeypatch):
    fake = types.ModuleType("streamlit")
    fake.secrets = {
        "ERAI_SEARCH_PROVIDER": "duckduckgo",
        "ERAI_SEARCH_MAX_RESULTS": 5,
        "NOT_OURS": "ignored",
        "ERAI_NESTED": {"nested": "ignored"},
    }
    monkeypatch.setitem(sys.modules, "streamlit", fake)
    for key in (
        "ERAI_SEARCH_PROVIDER",
        "ERAI_SEARCH_MAX_RESULTS",
        "NOT_OURS",
        "ERAI_NESTED",
    ):
        monkeypatch.delenv(key, raising=False)

    added = config.apply_streamlit_secrets()

    assert added == 2
    assert os.environ["ERAI_SEARCH_PROVIDER"] == "duckduckgo"
    assert os.environ["ERAI_SEARCH_MAX_RESULTS"] == "5"
    assert "NOT_OURS" not in os.environ
    assert "ERAI_NESTED" not in os.environ


def test_environment_values_win_over_secrets(monkeypatch):
    fake = types.ModuleType("streamlit")
    fake.secrets = {"ERAI_SEARCH_PROVIDER": "duckduckgo"}
    monkeypatch.setitem(sys.modules, "streamlit", fake)
    monkeypatch.setenv("ERAI_SEARCH_PROVIDER", "none")

    added = config.apply_streamlit_secrets()

    assert added == 0
    assert os.environ["ERAI_SEARCH_PROVIDER"] == "none"


def test_load_settings_reads_streamlit_secrets(monkeypatch):
    fake = types.ModuleType("streamlit")
    fake.secrets = {"ERAI_SEARCH_PROVIDER": "duckduckgo"}
    monkeypatch.setitem(sys.modules, "streamlit", fake)
    original = os.environ.get("ERAI_SEARCH_PROVIDER")
    monkeypatch.delenv("ERAI_SEARCH_PROVIDER", raising=False)

    try:
        settings = config.load_settings()
        assert settings.search_provider == "duckduckgo"
        assert settings.search_enabled
    finally:
        if original is None:
            os.environ.pop("ERAI_SEARCH_PROVIDER", None)
        else:
            os.environ["ERAI_SEARCH_PROVIDER"] = original
