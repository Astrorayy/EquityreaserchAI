"""Interchangeable Tier-2 web search.

Design rules
------------
* A search provider is **optional**.  The default is ``none``: the application
  is fully functional without any search key.
* Nothing here treats a search snippet as a financial fact.  Every result is
  returned with ``is_authoritative = False`` and a reminder that the snippet is
  a lead, not a number.
* Providers are interchangeable behind :class:`SearchProvider`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional
from urllib.parse import urlencode

from config import Settings, get_settings
from data_sources.http_client import HttpClient, get_client

PROVIDERS = ("none", "duckduckgo", "tavily", "brave", "serper")


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str
    provider: str
    rank: int = 0
    published: Optional[str] = None
    retrieved_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    #: A snippet is a pointer to a document, never a verified financial fact.
    is_authoritative: bool = False

    def as_dict(self) -> Dict[str, object]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "provider": self.provider,
            "rank": self.rank,
            "published": self.published,
            "retrieved_at": self.retrieved_at.isoformat(),
            "is_authoritative": self.is_authoritative,
        }


@dataclass
class SearchResponse:
    query: str
    provider: str
    results: List[SearchResult] = field(default_factory=list)
    error: Optional[str] = None
    note: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def as_dict(self) -> Dict[str, object]:
        return {
            "query": self.query,
            "provider": self.provider,
            "error": self.error,
            "note": self.note,
            "results": [r.as_dict() for r in self.results],
        }


class SearchProvider:
    """Base class: a provider only has to implement :meth:`search`."""

    name = "base"

    def __init__(self, http: Optional[HttpClient] = None, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.http = http or get_client()

    @property
    def available(self) -> bool:
        return False

    def search(self, query: str, max_results: Optional[int] = None) -> SearchResponse:
        raise NotImplementedError


# --------------------------------------------------------------------------
# No-op provider (the default)
# --------------------------------------------------------------------------


class NullSearchProvider(SearchProvider):
    """Explicitly reports that no search backend is configured."""

    name = "none"

    @property
    def available(self) -> bool:
        return False

    def search(self, query: str, max_results: Optional[int] = None) -> SearchResponse:
        return SearchResponse(
            query=query,
            provider=self.name,
            error=(
                "Aucun fournisseur de recherche n'est configuré. La recherche web est "
                "une fonction optionnelle : définissez ERAI_SEARCH_PROVIDER (et "
                "ERAI_SEARCH_API_KEY si nécessaire) dans le fichier .env pour l'activer. "
                "Toutes les données financières officielles restent accessibles sans cela."
            ),
        )


# --------------------------------------------------------------------------
# DuckDuckGo HTML endpoint (no API key; best effort)
# --------------------------------------------------------------------------


class DuckDuckGoProvider(SearchProvider):
    """Scrapes the public ``html.duckduckgo.com`` result page.

    No API key and no paid tier, but the endpoint is not contractual: it can
    change or rate limit at any time, hence the best-effort error handling.
    """

    name = "duckduckgo"
    ENDPOINT = "https://html.duckduckgo.com/html/"

    @property
    def available(self) -> bool:
        return True

    def search(self, query: str, max_results: Optional[int] = None) -> SearchResponse:
        limit = max_results or self.settings.search_max_results
        url = f"{self.ENDPOINT}?{urlencode({'q': query})}"
        response = self.http.get(url, extra_headers={"Accept": "text/html"})
        if not response.ok:
            return SearchResponse(
                query=query, provider=self.name,
                error=f"Recherche DuckDuckGo indisponible : {response.error or response.status_code}",
            )
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            return SearchResponse(query=query, provider=self.name,
                                  error="beautifulsoup4 est requis pour ce fournisseur")

        soup = BeautifulSoup(response.text, "html.parser")
        results: List[SearchResult] = []
        for rank, block in enumerate(soup.select(".result, .web-result"), start=1):
            anchor = block.select_one("a.result__a")
            if anchor is None:
                continue
            href = anchor.get("href", "")
            from urllib.parse import parse_qs, unquote, urlparse

            if href.startswith("//duckduckgo.com/l/") or "duckduckgo.com/l/" in href:
                qs = parse_qs(urlparse("https:" + href if href.startswith("//") else href).query)
                href = unquote(qs.get("uddg", [href])[0])
            snippet_el = block.select_one(".result__snippet")
            results.append(
                SearchResult(
                    title=anchor.get_text(" ", strip=True),
                    url=href,
                    snippet=snippet_el.get_text(" ", strip=True) if snippet_el else "",
                    provider=self.name,
                    rank=rank,
                )
            )
            if len(results) >= limit:
                break

        note = None
        if not results:
            note = (
                "Aucun résultat analysé. DuckDuckGo a peut-être renvoyé une page de "
                "vérification anti-robot, ou modifie son balisage HTML."
            )
        return SearchResponse(query=query, provider=self.name, results=results, note=note)


# --------------------------------------------------------------------------
# API-key providers
# --------------------------------------------------------------------------


class TavilyProvider(SearchProvider):
    name = "tavily"
    ENDPOINT = "https://api.tavily.com/search"

    @property
    def available(self) -> bool:
        return bool(self.settings.search_api_key)

    def search(self, query: str, max_results: Optional[int] = None) -> SearchResponse:
        if not self.available:
            return SearchResponse(query=query, provider=self.name,
                                  error="ERAI_SEARCH_API_KEY manquante pour Tavily")
        payload = {
            "api_key": self.settings.search_api_key,
            "query": query,
            "max_results": max_results or self.settings.search_max_results,
            "search_depth": "basic",
        }
        response = self.http._session.post(  # noqa: SLF001 - deliberate, keeps one session
            self.ENDPOINT, json=payload, timeout=self.settings.http_timeout
        )
        if response.status_code != 200:
            return SearchResponse(query=query, provider=self.name,
                                  error=f"Tavily a repondu {response.status_code}")
        data = response.json()
        results = [
            SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("content", "")[:500],
                provider=self.name,
                rank=i,
            )
            for i, item in enumerate(data.get("results", [])[: max_results or self.settings.search_max_results])
        ]
        return SearchResponse(query=query, provider=self.name, results=results)


class BraveProvider(SearchProvider):
    name = "brave"
    ENDPOINT = "https://api.search.brave.com/res/v1/web/search"

    @property
    def available(self) -> bool:
        return bool(self.settings.search_api_key)

    def search(self, query: str, max_results: Optional[int] = None) -> SearchResponse:
        if not self.available:
            return SearchResponse(query=query, provider=self.name,
                                  error="ERAI_SEARCH_API_KEY manquante pour Brave Search")
        limit = max_results or self.settings.search_max_results
        url = f"{self.ENDPOINT}?{urlencode({'q': query, 'count': min(limit, 20)})}"
        response = self.http.get(
            url,
            use_cache=False,
            extra_headers={
                "Accept": "application/json",
                "X-Subscription-Token": self.settings.search_api_key,
            },
        )
        if not response.ok:
            return SearchResponse(query=query, provider=self.name,
                                  error=f"Brave Search a repondu {response.status_code}: {response.error}")
        try:
            data = json.loads(response.content.decode("utf-8"))
        except ValueError as exc:
            return SearchResponse(query=query, provider=self.name, error=f"JSON invalide : {exc}")
        results = [
            SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("description", "")[:500],
                provider=self.name,
                rank=i,
                published=item.get("age"),
            )
            for i, item in enumerate((data.get("web", {}) or {}).get("results", [])[:limit])
        ]
        return SearchResponse(query=query, provider=self.name, results=results)


class SerperProvider(SearchProvider):
    name = "serper"
    ENDPOINT = "https://google.serper.dev/search"

    @property
    def available(self) -> bool:
        return bool(self.settings.search_api_key)

    def search(self, query: str, max_results: Optional[int] = None) -> SearchResponse:
        if not self.available:
            return SearchResponse(query=query, provider=self.name,
                                  error="ERAI_SEARCH_API_KEY manquante pour Serper")
        limit = max_results or self.settings.search_max_results
        response = self.http._session.post(  # noqa: SLF001
            self.ENDPOINT,
            json={"q": query, "num": min(limit, 20)},
            headers={"X-API-KEY": self.settings.search_api_key, "Content-Type": "application/json"},
            timeout=self.settings.http_timeout,
        )
        if response.status_code != 200:
            return SearchResponse(query=query, provider=self.name,
                                  error=f"Serper a repondu {response.status_code}")
        data = response.json()
        results = [
            SearchResult(
                title=item.get("title", ""),
                url=item.get("link", ""),
                snippet=item.get("snippet", "")[:500],
                provider=self.name,
                rank=i,
                published=item.get("date"),
            )
            for i, item in enumerate((data.get("organic") or [])[:limit])
        ]
        return SearchResponse(query=query, provider=self.name, results=results)


_PROVIDER_REGISTRY = {
    "none": NullSearchProvider,
    "duckduckgo": DuckDuckGoProvider,
    "tavily": TavilyProvider,
    "brave": BraveProvider,
    "serper": SerperProvider,
}


def get_search_provider(settings: Optional[Settings] = None, http: Optional[HttpClient] = None) -> SearchProvider:
    """Instantiate the configured provider, falling back to the no-op one."""
    settings = settings or get_settings()
    cls = _PROVIDER_REGISTRY.get(settings.search_provider, NullSearchProvider)
    provider = cls(http=http, settings=settings)
    if not provider.available:
        return NullSearchProvider(http=http, settings=settings)
    return provider


def describe_providers(settings: Optional[Settings] = None) -> List[Dict[str, object]]:
    """Table for the Settings page: what each provider needs, and its status."""
    settings = settings or get_settings()
    notes = {
        "none": "Recherche desactivee (valeur par defaut). Aucune cle requise.",
        "duckduckgo": "Sans cle. API non contractuelle : peut cesser de fonctionner sans preavis.",
        "tavily": "Cle API requise. Offre gratuite limitee disponible.",
        "brave": "Cle API requise. Offre gratuite limitee disponible.",
        "serper": "Cle API requise. Quota gratuit puis payant.",
    }
    rows: List[Dict[str, object]] = []
    for name in PROVIDERS:
        cls = _PROVIDER_REGISTRY[name]
        instance = cls(settings=settings)
        rows.append(
            {
                "fournisseur": name,
                "actif": name == settings.search_provider,
                "utilisable": instance.available,
                "cle_requise": name not in ("none", "duckduckgo"),
                "remarque": notes[name],
            }
        )
    return rows


def research_brief(company_name: str, sector: str, settings: Optional[Settings] = None, http: Optional[HttpClient] = None) -> List[SearchResponse]:
    """Run the standard Tier-2 query set for a company.

    The queries are generic research questions - the results are displayed as
    leads with their URLs, never as authoritative figures.
    """
    provider = get_search_provider(settings, http)
    queries = [
        f"{company_name} résultats financiers {datetime.now().year}",
        f"{company_name} guidance perspectives direction",
        f"{sector} prix et tendances du marché {datetime.now().year}",
        f"{company_name} risques facteurs de risque",
    ]
    return [provider.search(q) for q in queries]
