"""Application-wide configuration.

Loads ``.env`` (if present) and exposes typed settings.  Nothing here requires
a paid subscription: every external integration is optional and the core
application runs with an empty environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent

# Load .env once, at import time. override=False keeps real environment
# variables (e.g. set in CI) authoritative over the file.
load_dotenv(ROOT_DIR / ".env", override=False)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of the runtime configuration."""

    lang: str = "fr"

    http_timeout: float = 20.0
    http_min_interval: float = 1.0
    user_agent: str = "equity-research-ai/0.1 (research prototype)"

    data_dir: Path = field(default_factory=lambda: ROOT_DIR / "data")
    cache_dir: Path = field(default_factory=lambda: ROOT_DIR / "data" / "cache")
    reports_dir: Path = field(default_factory=lambda: ROOT_DIR / "data" / "reports")

    sec_user_agent: str = "equity-research-ai/0.1 (contact@example.com)"
    sec_ticker_map_url: str = "https://www.sec.gov/files/company_tickers.json"

    search_provider: str = "none"
    search_api_key: str = ""
    search_max_results: int = 8

    llm_provider: str = "custom"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 1500
    llm_timeout: float = 90.0

    enable_yfinance: bool = True

    @property
    def db_path(self) -> Path:
        return self.data_dir / "equity_research.db"

    @property
    def ai_enabled(self) -> bool:
        """True when an LLM endpoint and key are configured."""
        return bool(self.llm_api_key) and bool(self.llm_base_url)

    @property
    def search_enabled(self) -> bool:
        """True when a Tier-2 search provider is actually usable."""
        if self.search_provider == "none":
            return False
        if self.search_provider == "duckduckgo":
            return True
        return bool(self.search_api_key)

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.cache_dir, self.reports_dir):
            path.mkdir(parents=True, exist_ok=True)


def load_settings() -> Settings:
    data_dir_raw = os.getenv("ERAI_DATA_DIR", "./data").strip() or "./data"
    data_dir = Path(data_dir_raw)
    if not data_dir.is_absolute():
        data_dir = (ROOT_DIR / data_dir).resolve()

    return Settings(
        lang=os.getenv("ERAI_LANG", "fr").strip().lower() or "fr",
        http_timeout=_env_float("ERAI_HTTP_TIMEOUT", 20.0),
        http_min_interval=_env_float("ERAI_HTTP_MIN_INTERVAL", 1.0),
        user_agent=os.getenv("ERAI_WEB_USER_AGENT", "").strip()
        or "equity-research-ai/0.1 (research prototype)",
        data_dir=data_dir,
        cache_dir=data_dir / "cache",
        reports_dir=data_dir / "reports",
        sec_user_agent=os.getenv("ERAI_SEC_USER_AGENT", "").strip()
        or "equity-research-ai/0.1 (contact@example.com)",
        sec_ticker_map_url=os.getenv(
            "ERAI_SEC_TICKER_MAP", "https://www.sec.gov/files/company_tickers.json"
        ).strip(),
        search_provider=os.getenv("ERAI_SEARCH_PROVIDER", "none").strip().lower() or "none",
        search_api_key=os.getenv("ERAI_SEARCH_API_KEY", "").strip(),
        search_max_results=_env_int("ERAI_SEARCH_MAX_RESULTS", 8),
        llm_provider=os.getenv("ERAI_LLM_PROVIDER", "custom").strip().lower() or "custom",
        llm_base_url=os.getenv("ERAI_LLM_BASE_URL", "https://api.openai.com/v1").strip(),
        llm_api_key=os.getenv("ERAI_LLM_API_KEY", "").strip(),
        llm_model=os.getenv("ERAI_LLM_MODEL", "gpt-4o-mini").strip(),
        llm_temperature=_env_float("ERAI_LLM_TEMPERATURE", 0.2),
        llm_max_tokens=_env_int("ERAI_LLM_MAX_TOKENS", 1500),
        llm_timeout=_env_float("ERAI_LLM_TIMEOUT", 90.0),
        enable_yfinance=_env_bool("ERAI_ENABLE_YFINANCE", True),
    )


# Module-level singleton, refreshed by get_settings() so that tests (and the
# Settings page) can reload after the environment changes.
_settings: Settings | None = None


def get_settings(reload: bool = False) -> Settings:
    global _settings
    if _settings is None or reload:
        _settings = load_settings()
        _settings.ensure_dirs()
    return _settings
