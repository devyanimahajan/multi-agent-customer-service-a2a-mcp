# agents/support_agent.py

import logging
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    BaseMessage,
)
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient

logger = logging.getLogger(__name__)

OPENAI_MODEL_SUPPORT = os.getenv("OPENAI_MODEL_SUPPORT", "gpt-4o-mini")

support_llm = ChatOpenAI(
    model=OPENAI_MODEL_SUPPORT,
    temperature=0.4,
)

SUPPORT_SYSTEM_PROMPT = """
You are the Support Agent in a multi-agent customer service system.

Your responsibilities:
- Handle general customer support queries (account issues, upgrades, cancellations, billing questions, etc.).
- When ticket or history information is provided (for example, from a database lookup),
  use it to answer questions about ticket status, especially high-priority or open tickets.
- Escalate urgent issues appropriately (for example, billing disputes, double charges, or repeated failures).
- Be clear, concise, and empathetic. Where appropriate, summarise the ticket situation for the user.

If you are given a special context message that contains ticket information
(e.g., "TICKET_CONTEXT: ..."), read it carefully and incorporate it into your answer.
""".strip()

MCP_SERVER_PATH = Path(__file__).resolve().parents[1] / "mcp_server" / "server.py"


def _extract_customer_ids_from_messages(messages: List[BaseMessage]) -> List[int]:
    """
    Extract possible customer_id values from the conversation by scanning
    message contents for integers. We de-duplicate and sort them.
    This is a heuristic but works well for:
      - "customer ID 12345"
      - "customers 1, 2, and 3"
      - Data Agent summaries that mention IDs.
    """
    ids: List[int] = []
    for m in messages:
        content = getattr(m, "content", "")
        if isinstance(content, list):
            text = " ".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            )
        else:
            text = str(content)

        for match in re.finditer(r"\b\d+\b", text):
            try:
                value = int(match.group(0))
                if value not in ids:
                    ids.append(value)
            except ValueError:
                continue

    return ids


def _needs_ticket_lookup(user_text: str) -> bool:
    """
    Decide whether this query should trigger MCP ticket lookups.
    We look for signals like 'ticket', 'history', 'status', 'open tickets',
    'high-priority', etc.
    """
    text = user_text.lower()
    keywords = [
        "ticket",
        "ticket history",
        "history of my tickets",
        "status of",
        "open tickets",
        "high-priority",
        "high priority",
        "escalate",
    ]
    return any(k in text for k in keywords)


async def _lookup_ticket_context_via_mcp(customer_ids: List[int]) -> Optional[str]:
    """
    Use MCP to call get_customer_history(customer_id) for each provided ID.
    Return a human-readable summary string, or None if nothing could be retrieved.
    """
    if not customer_ids:
        return None

    logger.info("[Support] Connecting to MCP server at %s", MCP_SERVER_PATH)
    client = MultiServerMCPClient(
        {
            "customer_support_db": {
                "command": "python",
                "args": [str(MCP_SERVER_PATH)],
                "transport": "stdio",
            }
        }
    )

    tools = await client.get_tools()
    logger.info("[Support] Loaded %d MCP tools", len(tools))

    tools_by_name = {t.name: t for t in tools}

    def find_tool(name: str):
        if name in tools_by_name:
            return tools_by_name[name]
        for n, t in tools_by_name.items():
            if n.endswith(name):
                return t
        return None

    history_tool = find_tool("get_customer_history")
    if history_tool is None:
        logger.warning(
            "[Support] Could not find 'get_customer_history' tool. Available: %s",
            list(tools_by_name.keys()),
        )
        return None

    lines: List[str] = []
    for cid in customer_ids:
        try:
            logger.info("[Support] Invoking get_customer_history for customer_id=%s", cid)
            result = await history_tool.ainvoke({"customer_id": cid})
            # We make minimal assumptions about the result shape:
            # try to extract something useful if keys exist.
            if isinstance(result, dict):
                ticket_count = result.get("ticket_count")
                customer_id = result.get("customer_id", cid)
                tickets = result.get("tickets", [])
                lines.append(
                    f"Customer {customer_id}: {ticket_count} tickets in history."
                )
                # If ticket details exist, highlight high-priority/open ones.
                if isinstance(tickets, list):
                    high_priority_open = [
                        t
                        for t in tickets
                        if str(t.get("priority", "")).lower() == "high"
                        and str(t.get("status", "")).lower() in ("open", "in_progress")
                    ]
                    if high_priority_open:
                        lines.append(
                            f"  High-priority open/in-progress tickets: {len(high_priority_open)}"
                        )
            else:
                lines.append(f"Customer {cid}: raw history result = {result!r}")
        except Exception as e:
            logger.exception(
                "[Support] Error while calling get_customer_history for %s: %s",
                cid,
                e,
            )
            lines.append(
                f"Customer {cid}: error while retrieving ticket history: {e}"
            )

    if not lines:
        return None

    return "\n".join(lines)


async def support_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    messages: List[BaseMessage] = state["messages"]

    # Find the latest human message to judge intent.
    user_text = ""
    for m in reversed(messages):
        if isinstance(m, HumanMessage) or getattr(m, "type", "") == "human":
            content = getattr(m, "content", "")
            if isinstance(content, list):
                user_text = " ".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in content
                )
            else:
                user_text = str(content)
            break

    logger.info("[Support] Generating support response for %d messages", len(messages))

    # Decide whether to call MCP for ticket context.
    ticket_context_text = None
    if user_text and _needs_ticket_lookup(user_text):
        customer_ids = _extract_customer_ids_from_messages(messages)
        logger.info("[Support] Detected ticket-related query. Candidate IDs: %s", customer_ids)
        ticket_context_text = await _lookup_ticket_context_via_mcp(customer_ids)

    prompt_messages: List[BaseMessage] = [SystemMessage(content=SUPPORT_SYSTEM_PROMPT)]
    prompt_messages.extend(messages)

    if ticket_context_text:
        # Add a synthetic human-style message so the LLM treats this as context to use.
        context_msg = HumanMessage(
            content=f"TICKET_CONTEXT:\n{ticket_context_text}\n\nUse this ticket information to answer the user's question."
        )
        prompt_messages.append(context_msg)
        logger.info("[Support] Added ticket context to LLM prompt:\n%s", ticket_context_text)

    response = await support_llm.ainvoke(prompt_messages)
    logger.info("[Support] Response: %s", getattr(response, "content", ""))

    return {"messages": [response]}
