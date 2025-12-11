# agents/router_agent.py

import logging
import os
from typing import Dict, Any

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

OPENAI_MODEL_ROUTER = os.getenv("OPENAI_MODEL_ROUTER", "gpt-4o-mini")

router_llm = ChatOpenAI(
    model=OPENAI_MODEL_ROUTER,
    temperature=0,
)

ROUTER_SYSTEM_PROMPT = """
You are the Router Agent in a multi-agent customer service system.

You work with:
- Customer Data Agent: uses MCP tools to read/update customer records and tickets.
- Support Agent: handles customer support conversations, explanations, and empathy.

Given the latest user query, you MUST:
1. Briefly explain what the user wants.
2. Decide the next step in the workflow.
3. Output a line starting with: ROUTE: <route>

Valid routes:
- data              -> send to Customer Data Agent only
- support           -> send to Support Agent only
- data_then_support -> first Customer Data Agent, then Support Agent
- end               -> no further action needed

Examples:

User: "Get customer information for ID 5"
ROUTE: data

User: "I'm customer 12345 and need help upgrading my account"
ROUTE: data_then_support

User: "I've been charged twice, please refund immediately!"
ROUTE: support

User: "Thanks, that answers my question"
ROUTE: end
""".strip()


async def router_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Router node for LangGraph."""
    messages = state["messages"]
    # Take the latest user-facing message content
    last = messages[-1]
    last_content = getattr(last, "content", "")
    if isinstance(last_content, list):
        # If it's a list of content blocks, join text parts
        last_content = " ".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in last_content
        )

    logger.info("[Router] Received query: %s", last_content)

    response = await router_llm.ainvoke(
        [
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=last_content),
        ]
    )

    logger.info("[Router] LLM routing decision: %s", response.content)

    return {"messages": [response]}


def _extract_route(text: str) -> str:
    text_lower = text.lower()
    if "route:" in text_lower:
        after = text_lower.split("route:", 1)[1].strip()
        token = after.split()[0]
        if token in {"data", "support", "data_then_support", "end"}:
            return token
    # Fallback heuristic based on keywords
    if any(k in text_lower for k in ["id ", "customer", "ticket", "history", "active customers"]):
        if any(k in text_lower for k in ["help", "upgrade", "cancel", "refund", "issue", "support"]):
            return "data_then_support"
        return "data"
    if any(k in text_lower for k in ["help", "upgrade", "cancel", "refund", "issue", "support"]):
        return "support"
    return "end"


def route_from_router(state: Dict[str, Any]) -> str:
    """Decide next node based on the Router Agent's message."""
    messages = state["messages"]
    last = messages[-1]
    text = getattr(last, "content", "")
    if isinstance(text, list):
        text = " ".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in text
        )

    route = _extract_route(text)
    logger.info("[Router] Parsed route: %s", route)

    # Map to graph edges
    if route == "data":
        return "customer_data"
    if route == "support":
        return "support"
    if route == "data_then_support":
        return "customer_data"
    return "end"
