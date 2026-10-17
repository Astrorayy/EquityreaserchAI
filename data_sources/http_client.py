"""Polite HTTP client: timeouts, rate limiting, caching, robots.txt.

Rules implemented here
----------------------
* A descriptive ``User-Agent`` on every request (the SEC requires one).
* A minimum delay between two requests to the same host.
* Responses are cached in SQLite with a TTL, so re-running the dashboard does
  not re-download anything.
* ``robots.txt`` is respected by default for HTML pages.
* TLS verification is **never** disabled silently.  A site whose certificate
  chain is not trusted locally (the Casablanca Stock Exchange currently is not)
  fails loudly, with a documented opt-in switch rather than a silent bypass.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests

from config import Settings, get_settings


@dataclass
class HttpResponse:
    url: str
    status_code: int
    content: bytes
    content_type: str
    from_cache: bool = False
    error: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None and 200 <= self.status_code < 300

    @property
    def text(self) -> str:
        """Decoded body, guessing the charset as the browser would."""
        if not self.content:
            return ""
        charset = None
        ct = self.content_type or ""
        if "charset=" in ct:
            charset = ct.split("charset=", 1)[1].split(";")[0].strip()
        for candidate in filter(None, [charset, "utf-8", "cp1252", "latin-1"]):
            try:
                return self.content.decode(candidate)
            except (UnicodeDecodeError, LookupError):
                continue
        return self.content.decode("utf-8", errors="replace")


class RateLimiter:
    """Enforce a minimum interval between calls to the same host."""

    def __init__(self, min_interval: float):
        self.min_interval = max(0.0, min_interval)
        self._last: Dict[str, float] = {}

    def wait(self, host: str) -> None:
        if self.min_interval <= 0:
            return
        now = time.monotonic()
        last = self._last.get(host)
        if last is not None:
            elapsed = now - last
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
        self._last[host] = time.monotonic()


class HttpClient:
    """Small requests wrapper used by every connector."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        db=None,
        cache_ttl: int = 6 * 3600,
        respect_robots: bool = True,
        insecure: bool = False,
    ):
        self.settings = settings or get_settings()
        self.db = db
        self.cache_ttl = cache_ttl
        self.respect_robots = respect_robots
        self.insecure = insecure
        self.limiter = RateLimiter(self.settings.http_min_interval)
        self._robots: Dict[str, Optional[RobotFileParser]] = {}
        self._session = requests.Session()
        self.last_error: Optional[str] = None

    # -- internals ---------------------------------------------------------

    def _headers(self, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        headers = {
            "User-Agent": self.settings.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml,application/json,*/*",
            "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
        }
        if extra:
            headers.update(extra)
        return headers

    def _robots_allows(self, url: str) -> tuple[bool, Optional[str]]:
        if not self.respect_robots:
            return True, None
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        if origin not in self._robots:
            robots_url = f"{origin}/robots.txt"
            parser: Optional[RobotFileParser] = None
            cached = self._cache_read(robots_url) if self.db is not None else None
            if cached is not None and cached.ok:
                parser = RobotFileParser()
                parser.parse(cached.text.splitlines())
            else:
                try:
                    self.limiter.wait(parsed.netloc)
                    resp = self._session.get(
                        robots_url, headers=self._headers(), timeout=self.settings.http_timeout
                    )
                    if resp.status_code == 200:
                        parser = RobotFileParser()
                        parser.parse(resp.text.splitlines())
                        if self.db is not None:
                            self._cache_write(robots_url, resp)
                except requests.RequestException:
                    # No robots.txt reachable: treat as "not restricted".
                    parser = None
            self._robots[origin] = parser

        parser = self._robots[origin]
        if parser is None:
            return True, None
        allowed = parser.can_fetch(self.settings.user_agent, url)
        return allowed, None if allowed else f"robots.txt de {parsed.netloc} interdit cette URL"

    def _cache_read(self, url: str) -> Optional[HttpResponse]:
        if self.db is None:
            return None
        from storage.repository import cache_get

        row = cache_get(self.db, url, max_age_seconds=self.cache_ttl)
        if row is None or row["status_code"] is None:
            return None
        return HttpResponse(
            url=url,
            status_code=row["status_code"],
            content=row["body"] or b"",
            content_type=row["content_type"] or "",
            from_cache=True,
            headers=row["headers"] or {},
        )

    def _cache_write(self, url: str, response: requests.Response) -> None:
        if self.db is None:
            return
        from storage.repository import cache_put

        try:
            cache_put(
                self.db,
                url,
                response.status_code,
                response.headers.get("Content-Type", ""),
                response.content,
                dict(response.headers),
            )
        except Exception:
            # Caching is an optimisation; never let it break a download.
            pass

    # -- public API --------------------------------------------------------

    def get(
        self,
        url: str,
        use_cache: bool = True,
        extra_headers: Optional[Dict[str, str]] = None,
        allow_robots_override: bool = False,
    ) -> HttpResponse:
        url = url.strip()
        if not url.lower().startswith(("http://", "https://")):
            return HttpResponse(url, 0, b"", "", error="L'URL doit commencer par http:// ou https://")

        if use_cache:
            cached = self._cache_read(url)
            if cached is not None:
                return cached

        if not allow_robots_override:
            allowed, reason = self._robots_allows(url)
            if not allowed:
                return HttpResponse(url, 0, b"", "", error=reason)

        host = urlparse(url).netloc
        try:
            self.limiter.wait(host)
            response = self._session.get(
                url,
                headers=self._headers(extra_headers),
                timeout=self.settings.http_timeout,
                allow_redirects=True,
                verify=not self.insecure,
            )
        except requests.exceptions.SSLError as exc:
            self.last_error = str(exc)
            return HttpResponse(
                url, 0, b"", "",
                error=(
                    "Le certificat TLS n'a pas pu être vérifié : "
                    f"{exc}. Ce site peut nécessiter une chaîne de certificats non "
                    "installée sur cette machine. Téléchargez le document manuellement "
                    "et importez-le, plutôt que de désactiver les contrôles de certificat."
                ),
            )
        except requests.exceptions.Timeout:
            self.last_error = "timeout"
            return HttpResponse(url, 0, b"", "", error=f"Requête expirée après {self.settings.http_timeout:.0f}s")
        except requests.RequestException as exc:
            self.last_error = str(exc)
            return HttpResponse(url, 0, b"", "", error=f"{type(exc).__name__}: {exc}")

        if response.status_code == 200:
            self._cache_write(url, response)

        return HttpResponse(
            url=url,
            status_code=response.status_code,
            content=response.content,
            content_type=response.headers.get("Content-Type", ""),
            headers=dict(response.headers),
        )

    def get_json(self, url: str, **kwargs) -> tuple[Optional[object], Optional[HttpResponse]]:
        response = self.get(url, **kwargs)
        if not response.ok:
            return None, response
        try:
            import json

            return json.loads(response.content.decode("utf-8")), response
        except (ValueError, UnicodeDecodeError) as exc:
            return None, HttpResponse(
                url, response.status_code, response.content, response.content_type,
                error=f"invalid JSON: {exc}",
            )

    def download_to_file(self, url: str, destination) -> tuple[bool, Optional[str]]:
        """Fetch a binary document (PDF, XLSX) and write it to disk."""
        from pathlib import Path

        response = self.get(url)
        if not response.ok:
            return False, response.error or f"HTTP {response.status_code}"
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
        return True, None


# --------------------------------------------------------------------------
# Convenience singletons used by the UI
# --------------------------------------------------------------------------


_client: Optional[HttpClient] = None


def get_client(db=None, **kwargs) -> HttpClient:
    """Shared client so the rate limiter and cache are process-wide."""
    global _client
    if _client is None or kwargs:
        _client = HttpClient(db=db or getattr(_client, "db", None), **kwargs)
    elif db is not None and _client.db is None:
        _client.db = db
    return _client


def reset_client() -> None:
    global _client
    _client = None
