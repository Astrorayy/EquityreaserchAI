"""AI research assistant: provider-agnostic LLM plus deterministic tools.

Layers (each importable on its own):

``agent.llm_provider``   any OpenAI-compatible chat endpoint; *disabled* without a key
``agent.tools``          the deterministic functions the model may call
``agent.prompts``        the behavioural contract given to the model
``agent.financial_agent``the tool-calling loop, with a full audit trail
"""

from agent import financial_agent, llm_provider, prompts, tools

__all__ = ["financial_agent", "llm_provider", "prompts", "tools"]
