"""The assistant loop: question → tool calls → grounded answer.

The loop is deliberately small and boring:

1. Build the message list (system prompt + session context + conversation).
2. Ask the model. If it requests tools, execute them **with the deterministic
   Python functions in** :mod:`agent.tools` and feed the JSON results back.
3. Repeat until the model answers without asking for tools, or until
   ``max_iterations`` is reached — in which case the transcript is returned with
   an explicit note rather than a fabricated answer.
4. Return the answer *plus* the full tool log, so the UI can show exactly which
   functions ran and what they returned. A suspicious answer is therefore
   auditable without re-running the session.

Nothing returned by a tool is executed; nothing the model says is executed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from agent.llm_provider import (
    ChatMessage,
    LLMError,
    LLMProvider,
    ROLE_ASSISTANT,
    ROLE_SYSTEM,
    ROLE_TOOL,
    ROLE_USER,
    ToolCall,
    get_provider,
)
from agent.prompts import (
    company_context_block,
    first_turn_instruction,
    system_prompt,
    tool_error_message,
    tool_result_block,
)
from agent.tools import TOOLS_BY_NAME, ToolContext, dispatch, tool_specs

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 6
MAX_QUESTION_CHARS = 4000


@dataclass
class ToolInvocation:
    """One executed tool call, kept for the UI audit trail."""

    name: str
    arguments: Dict[str, Any]
    ok: bool
    summary: str
    payload: Dict[str, Any]
    parse_error: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "tool": self.name,
            "arguments": self.arguments,
            "ok": self.ok,
            "summary": self.summary,
            "payload": self.payload,
            "parse_error": self.parse_error,
        }


@dataclass
class AgentRun:
    answer: str
    invocations: List[ToolInvocation] = field(default_factory=list)
    iterations: int = 0
    stopped_early: bool = False
    error: Optional[str] = None
    model: Optional[str] = None
    usage: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.answer.strip())

    def tools_used(self) -> List[str]:
        seen: List[str] = []
        for invocation in self.invocations:
            if invocation.name not in seen:
                seen.append(invocation.name)
        return seen


class FinancialAgent:
    """Thin orchestrator over an :class:`LLMProvider` and the tool registry."""

    def __init__(
        self,
        ctx: ToolContext,
        provider: Optional[LLMProvider] = None,
        *,
        lang: str = "fr",
        max_iterations: int = MAX_ITERATIONS,
    ) -> None:
        self.ctx = ctx
        self.provider = provider or get_provider(ctx.settings)
        self.lang = lang
        self.max_iterations = max(1, int(max_iterations))

    # -- context --------------------------------------------------------

    @property
    def available(self) -> bool:
        return bool(getattr(self.provider, "available", False))

    def unavailable_reason(self) -> str:
        return getattr(self.provider, "reason", "Assistant IA indisponible.")

    def _session_context(self) -> str:
        profile = (
            self.ctx.resolve_company(self.ctx.default_company_id)
            if self.ctx.default_company_id
            else None
        )
        summary = None
        quality = None
        valuation = None
        if profile is not None:
            from workflows import dataset_summary

            dataset = self.ctx.dataset_for(profile.company_id)
            if dataset.observations:
                summary = dataset_summary(dataset)
                report = None
                try:
                    from workflows import assess

                    report = assess(dataset)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug("Quality assessment failed: %s", exc)
                if report is not None:
                    quality = {
                        "errors": len(report.errors),
                        "warnings": len(report.warnings),
                    }
        cached = self.ctx.cache.get("valuation")
        base = cached.base if cached is not None else None
        if base is not None:
            valuation = {
                "scenario": base.scenario,
                "wacc": base.wacc,
                "terminal_growth": base.terminal_growth,
                "value_per_share": base.value_per_share,
                "currency": base.currency,
            }
        return company_context_block(profile, summary, quality, valuation)

    def _system_messages(self) -> List[ChatMessage]:
        return [
            ChatMessage(ROLE_SYSTEM, system_prompt(self.lang)),
            ChatMessage(ROLE_SYSTEM, self._session_context()),
        ]

    # -- run ------------------------------------------------------------

    def run(
        self,
        question: str,
        history: Optional[Sequence[ChatMessage]] = None,
        *,
        require_initial_lookup: bool = True,
    ) -> AgentRun:
        question = (question or "").strip()
        if not question:
            return AgentRun(answer="", error="Question vide.")
        if len(question) > MAX_QUESTION_CHARS:
            question = question[:MAX_QUESTION_CHARS]
        if not self.available:
            return AgentRun(answer="", error=self.unavailable_reason())

        messages: List[ChatMessage] = self._system_messages()
        if history:
            messages.extend(history)
        user_text = question
        if require_initial_lookup and not history:
            user_text = f"{question}\n\n({first_turn_instruction(self._profile_name())})"
        messages.append(ChatMessage(ROLE_USER, user_text))

        specs = tool_specs()
        run = AgentRun(answer="")

        for iteration in range(1, self.max_iterations + 1):
            run.iterations = iteration
            try:
                response = self.provider.chat(messages, tools=specs)
            except LLMError as exc:
                run.error = str(exc)
                run.answer = (
                    "L'appel au fournisseur de modèle a échoué. Détail technique : "
                    f"{exc}"
                )
                return run

            run.model = response.model or run.model
            if response.usage:
                run.usage = response.usage

            if not response.has_tool_calls:
                run.answer = (response.content or "").strip()
                if not run.answer:
                    run.error = (
                        "Le modèle a renvoyé une réponse vide "
                        f"(finish_reason = {response.finish_reason})."
                    )
                    run.answer = (
                        "Le modèle n'a produit aucune réponse exploitable. Reformulez la "
                        "question ou vérifiez la configuration du fournisseur dans les "
                        "paramètres."
                    )
                return run

            messages.append(
                ChatMessage(
                    ROLE_ASSISTANT,
                    content=response.content or "",
                    tool_calls=response.tool_calls,
                )
            )

            for call in response.tool_calls:
                invocation = self._execute(call)
                run.invocations.append(invocation)
                messages.append(
                    ChatMessage(
                        ROLE_TOOL,
                        content=self._tool_message(invocation),
                        name=call.name,
                        tool_call_id=call.id,
                    )
                )

        run.stopped_early = True
        run.notes.append(
            f"Limite de {self.max_iterations} tours d'outils atteinte : la réponse n'est pas "
            "une conclusion, seulement l'état des données consultées."
        )
        last = run.invocations[-1] if run.invocations else None
        run.answer = (
            "Je n'ai pas pu conclure dans la limite d'appels autorisée. Voici l'état factuel "
            "des données consultées :\n\n"
            + (last.summary if last else "aucun résultat exploitable")
        )
        return run

    # -- internals ------------------------------------------------------

    def _profile_name(self) -> Optional[str]:
        if not self.ctx.default_company_id:
            return None
        profile = self.ctx.resolve_company(self.ctx.default_company_id)
        return profile.name if profile else None

    def _execute(self, call: ToolCall) -> ToolInvocation:
        if not call.name:
            return ToolInvocation(
                name="(sans nom)",
                arguments={},
                ok=False,
                summary="Appel de fonction sans nom : ignoré.",
                payload={"ok": False, "error": "appel sans nom"},
            )
        if call.name not in TOOLS_BY_NAME:
            result = dispatch(self.ctx, call.name, {})
            return ToolInvocation(
                name=call.name,
                arguments={},
                ok=False,
                summary=result.summary,
                payload=result.to_dict(),
            )
        if not call.ok:
            return ToolInvocation(
                name=call.name,
                arguments={},
                ok=False,
                summary=f"Arguments illisibles pour {call.name}.",
                payload={"ok": False, "error": call.parse_error},
                parse_error=call.parse_error,
            )
        arguments = dict(call.arguments or {})
        if (
            self.ctx.default_company_id
            and "company_id" in TOOLS_BY_NAME[call.name].parameters.get("properties", {})
            and not arguments.get("company_id")
        ):
            # Only default it when the model gave nothing: an explicit other
            # company must never be overridden.
            arguments["company_id"] = self.ctx.default_company_id
        result = dispatch(self.ctx, call.name, arguments)
        return ToolInvocation(
            name=call.name,
            arguments=arguments,
            ok=result.ok,
            summary=result.summary,
            payload=result.to_dict(),
        )

    def _tool_message(self, invocation: ToolInvocation) -> str:
        if invocation.ok:
            return tool_result_block(invocation.name, invocation.payload)
        reason = (
            invocation.parse_error
            or invocation.payload.get("error")
            or invocation.summary
        )
        return tool_error_message(invocation.name, str(reason))


def quick_answer(
    ctx: ToolContext,
    question: str,
    provider: Optional[LLMProvider] = None,
    lang: str = "fr",
) -> AgentRun:
    """Convenience wrapper used by the UI and by tests."""
    agent = FinancialAgent(ctx, provider, lang=lang)
    return agent.run(question)
