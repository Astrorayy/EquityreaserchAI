"""Provider-independent LLM access (OpenAI-compatible chat completions).

Only the standard library plus ``requests`` is used, so no vendor SDK is needed
and the user can point the assistant at any OpenAI-compatible endpoint (OpenAI,
Groq, Gemini's compat endpoint, OpenRouter, Mistral, a local Ollama, …).

Safety rules enforced here
--------------------------
* The API key is read from settings and sent in the ``Authorization`` header
  only.  It is never logged, never echoed in an error message, and never stored
  in the database.
* Responses are parsed defensively: a malformed payload raises :class:`LLMError`
  with the raw body truncated, rather than half-parsed into the conversation.
* Tool-call arguments arrive as a JSON *string* and are parsed with a fallback:
  unparsable arguments are surfaced to the caller so the model can be told to
  retry, instead of being silently dropped.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import requests

from config import Settings, get_settings

logger = logging.getLogger(__name__)

#: Truncation applied to any upstream body quoted in an error message.
MAX_ERROR_BODY = 400

ROLE_SYSTEM = "system"
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_TOOL = "tool"


class LLMError(RuntimeError):
    """Raised for any failure that prevents a usable model response."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    raw_arguments: str = ""
    parse_error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.parse_error is None


@dataclass
class ToolSpec:
    """A function exposed to the model (OpenAI "tools" payload)."""

    name: str
    description: str
    parameters: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ChatMessage:
    role: str
    content: str = ""
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"role": self.role}
        # ``content`` must be present but may be null when tool calls are made.
        payload["content"] = self.content if self.content != "" else (None if self.tool_calls else "")
        if self.name:
            payload["name"] = self.name
        if self.tool_call_id:
            payload["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.raw_arguments or "{}"},
                }
                for call in self.tool_calls
            ]
        return payload


@dataclass
class LLMResponse:
    content: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    finish_reason: Optional[str] = None
    model: Optional[str] = None
    usage: Dict[str, Any] = field(default_factory=dict)
    raw: Optional[Dict[str, Any]] = None

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


class LLMProvider:
    """Base class. ``available`` gates every call in the UI."""

    name = "disabled"
    model = ""

    @property
    def available(self) -> bool:  # pragma: no cover - trivial
        return False

    def describe(self) -> Dict[str, Any]:
        return {"provider": self.name, "model": self.model, "available": self.available}

    def chat(  # pragma: no cover - overridden
        self,
        messages: Sequence[ChatMessage],
        tools: Optional[Sequence[ToolSpec]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        raise LLMError("Aucun fournisseur de modèle n'est configuré.")


class DisabledLLMProvider(LLMProvider):
    """Used when no API key is configured: the rest of the app keeps working."""

    name = "disabled"

    def __init__(self, reason: str = "") -> None:
        self.reason = reason or (
            "Assistant IA désactivé : aucune clé d'API n'est configurée "
            "(variable ERAI_LLM_API_KEY)."
        )

    def chat(self, messages, tools=None, temperature=None, max_tokens=None) -> LLMResponse:
        raise LLMError(self.reason)


class OpenAICompatibleProvider(LLMProvider):
    """POST ``/chat/completions`` against any OpenAI-compatible base URL."""

    name = "openai-compatible"

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        timeout: float = 90.0,
        max_tokens: int = 1500,
        temperature: float = 0.2,
        session: Optional[requests.Session] = None,
    ) -> None:
        if not base_url:
            raise LLMError("URL de base du modèle manquante (ERAI_LLM_BASE_URL).")
        if not api_key:
            raise LLMError("Clé d'API du modèle manquante (ERAI_LLM_API_KEY).")
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self.model = model
        self.timeout = timeout
        self.default_max_tokens = max_tokens
        self.default_temperature = temperature
        self._session = session or requests.Session()

    @property
    def available(self) -> bool:
        return True

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    def describe(self) -> Dict[str, Any]:
        info = super().describe()
        info["endpoint"] = self.endpoint
        return info

    # -- internals ------------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _payload(
        self,
        messages: Sequence[ChatMessage],
        tools: Optional[Sequence[ToolSpec]],
        temperature: Optional[float],
        max_tokens: Optional[int],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [m.as_dict() for m in messages],
            "temperature": self.default_temperature if temperature is None else temperature,
            "max_tokens": self.default_max_tokens if max_tokens is None else max_tokens,
        }
        if tools:
            payload["tools"] = [t.as_dict() for t in tools]
            payload["tool_choice"] = "auto"
        return payload

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            response = self._session.post(
                self.endpoint,
                headers=self._headers(),
                json=payload,
                timeout=self.timeout,
            )
        except requests.Timeout as exc:
            raise LLMError(
                f"Le fournisseur de modèle n'a pas répondu en {self.timeout:.0f} s. "
                "Réessayez ou réduisez la longueur de la question."
            ) from exc
        except requests.RequestException as exc:
            raise LLMError(f"Connexion au fournisseur de modèle impossible : {exc}") from exc

        body_text = response.text or ""
        if response.status_code >= 400:
            hint = ""
            if response.status_code in (401, 403):
                hint = " (clé d'API absente, invalide ou sans droits sur ce modèle)"
            elif response.status_code == 404:
                hint = " (URL de base ou nom de modèle incorrect)"
            elif response.status_code == 429:
                hint = " (quota ou limite de débit atteinte — un palier gratuit suffit rarement pour un long échange)"
            raise LLMError(
                f"Le fournisseur a répondu HTTP {response.status_code}{hint}. "
                f"Détail : {body_text[:MAX_ERROR_BODY]}"
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise LLMError(
                "Réponse illisible du fournisseur (JSON invalide) : "
                f"{body_text[:MAX_ERROR_BODY]}"
            ) from exc
        if not isinstance(data, dict):
            raise LLMError("Réponse inattendue du fournisseur (objet JSON attendu).")
        if "error" in data and "choices" not in data:
            message = data["error"]
            if isinstance(message, dict):
                message = message.get("message") or json.dumps(message)[:MAX_ERROR_BODY]
            raise LLMError(f"Erreur renvoyée par le fournisseur : {message}")
        return data

    # -- public API -----------------------------------------------------

    def chat(
        self,
        messages: Sequence[ChatMessage],
        tools: Optional[Sequence[ToolSpec]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        data = self._post(self._payload(messages, tools, temperature, max_tokens))
        choices = data.get("choices") or []
        if not choices:
            raise LLMError("Le fournisseur n'a renvoyé aucune réponse (liste « choices » vide).")
        message = choices[0].get("message") or {}
        content = message.get("content") or ""
        if isinstance(content, list):
            # Some endpoints return content as a list of parts.
            content = "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        calls: List[ToolCall] = []
        for raw_call in message.get("tool_calls") or []:
            function = (raw_call or {}).get("function") or {}
            raw_arguments = function.get("arguments") or "{}"
            arguments: Dict[str, Any] = {}
            parse_error: Optional[str] = None
            try:
                parsed = json.loads(raw_arguments)
                if isinstance(parsed, dict):
                    arguments = parsed
                else:
                    parse_error = "les arguments doivent être un objet JSON"
            except (ValueError, TypeError) as exc:
                parse_error = f"arguments JSON illisibles : {exc}"
            calls.append(
                ToolCall(
                    id=str(raw_call.get("id") or f"call_{len(calls)}"),
                    name=str(function.get("name") or ""),
                    arguments=arguments,
                    raw_arguments=raw_arguments,
                    parse_error=parse_error,
                )
            )
        return LLMResponse(
            content=content,
            tool_calls=calls,
            finish_reason=choices[0].get("finish_reason"),
            model=data.get("model") or self.model,
            usage=data.get("usage") or {},
            raw=data,
        )


def get_provider(settings: Optional[Settings] = None, **kwargs) -> LLMProvider:
    """Return a usable provider, or a :class:`DisabledLLMProvider` with a reason."""
    settings = settings or get_settings()
    if not settings.ai_enabled:
        missing = []
        if not settings.llm_api_key:
            missing.append("ERAI_LLM_API_KEY")
        if not settings.llm_base_url:
            missing.append("ERAI_LLM_BASE_URL")
        return DisabledLLMProvider(
            "Assistant IA désactivé : renseignez "
            + " et ".join(missing or ["ERAI_LLM_API_KEY"])
            + " dans le fichier .env puis redémarrez l'application."
        )
    try:
        return OpenAICompatibleProvider(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            timeout=settings.llm_timeout,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
            **kwargs,
        )
    except LLMError as exc:
        return DisabledLLMProvider(str(exc))


def check_connection(provider: LLMProvider) -> Dict[str, Any]:
    """Tiny round-trip used by the Settings page to validate the configuration."""
    if not provider.available:
        return {"ok": False, "error": getattr(provider, "reason", "assistant désactivé")}
    try:
        response = provider.chat(
            [
                ChatMessage(ROLE_SYSTEM, "Tu réponds en un mot."),
                ChatMessage(ROLE_USER, "Réponds exactement : OK"),
            ],
            temperature=0.0,
            max_tokens=8,
        )
        return {"ok": True, "model": response.model, "reply": response.content.strip()[:60]}
    except LLMError as exc:
        return {"ok": False, "error": str(exc)}
