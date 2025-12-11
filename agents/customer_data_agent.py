# agents/customer_data_agent.py

import logging
import os
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

from langchain_core.messages import (
    SystemMessage,
    AIMessage,
    HumanMessage,
    BaseMessage,
)
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient

logger = logging.getLogger(__name__)

OPENAI_MODEL_DATA = os.getenv("OPENAI_MODEL_DATA", "gpt-4o-mini")

data_llm = ChatOpenAI(
    model=OPENAI_MODEL_DATA,
    temperature=0,
)

DATA_PLANNER_PROMPT = """
You are the Customer Data Agent in a multi-agent customer service system.

You have access to MCP tools from the `customer_support_db` server, including:
- get_customer(customer_id)
- list_customers(status, limit)
- update_customer(customer_id, data)
- create_ticket(customer_id, issue, priority)
- get_customer_history(customer_id)

Rules:
- customer_id must be an INTEGER, not a string like "unknown" or "provided_customer_id".
- If the user provides a numeric customer ID in their message, use that exact number.
- For list_customers, valid status values are only "active" or "disabled".
  If the user says "premium customers", treat that as "active".

Given the latest USER REQUEST (not internal summaries), decide:

1. Which ONE operation to perform.
2. What arguments to pass.

Respond ONLY as valid JSON with the structure:

{
  "operation": "get_customer" | "list_customers" | "update_customer" | "create_ticket" | "get_customer_history",
  "args": { ... appropriate arguments ... }
}
""".strip()

MCP_SERVER_PATH = Path(__file__).resolve().parents[1] / "mcp_server" / "server.py"


def _safe_json_from_llm(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("[CustomerData] Failed to parse JSON from LLM: %s", text)
        return {"operation": None, "args": {}}


def _extract_customer_id_from_text(text: str) -> Optional[int]:
    """
    Extract the first integer appearing in the user text.
    E.g. "customer ID 12345" -> 12345.
    """
    match = re.search(r"\d+", text)
    if match:
        try:
            return int(match.group(0))
        except ValueError:
            return None
    return None


def _normalize_customer_id(args: Dict[str, Any], user_text: str) -> Optional[int]:
    """
    Ensure args['customer_id'] is a proper int. If the planner gave a string like
    'provided_customer_id' or 'unknown', fall back to parsing from user_text.
    """
    cid = args.get("customer_id")

    # If planner already gave an int, accept it.
    if isinstance(cid, int):
        return cid

    # If it's a string, try to parse as int.
    if isinstance(cid, str):
        try:
            return int(cid)
        except ValueError:
            pass

    # Fallback: extract from user text.
    parsed = _extract_customer_id_from_text(user_text)
    if parsed is not None:
        args["customer_id"] = parsed
        return parsed

    return None


def _normalize_status(args: Dict[str, Any]) -> None:
    """
    For list_customers, ensure status is 'active' or 'disabled'.
    Treat 'premium' as 'active'. If unknown, default to 'active'.
    """
    status = str(args.get("status", "active")).lower()
    if status in ("active", "disabled"):
        args["status"] = status
        return
    if "premium" in status:
        args["status"] = "active"
        return
    # Default fallback
    args["status"] = "active"


def _summarise_result(operation: str, args: Dict[str, Any], result: Any) -> str:
    base = f"Customer Data Agent executed '{operation}' with args {args}.\n"

    if isinstance(result, dict):
        if operation == "get_customer":
            if result.get("success") and result.get("customer"):
                return base + f"Found customer record: {result['customer']}"
            return base + f"Operation failed: {result.get('message')}"

        if operation == "list_customers":
            customers = result.get("customers")
            if customers:
                return base + f"Found {result.get('count', 0)} customers: {customers}"
            return base + f"Found {result.get('count', 0)} customers."

        if operation == "update_customer":
            if result.get("success"):
                return base + f"Update succeeded. New record: {result.get('customer')}"
            return base + f"Update failed: {result.get('message')}"

        if operation == "create_ticket":
            if result.get("success"):
                return base + (
                    f"Created ticket with ID {result.get('ticket_id')} "
                    f"for customer {result.get('customer_id')}."
                )
            return base + f"Ticket creation failed: {result.get('message')}"

        if operation == "get_customer_history":
            return base + (
                f"Found {result.get('ticket_count', 0)} tickets "
                f"for customer {result.get('customer_id')}."
            )

    return base + f"Raw result: {result}"


async def customer_data_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    messages: List[BaseMessage] = state["messages"]

    # Use the LAST HUMAN MESSAGE as the true user request (not router summaries).
    user_text = None
    for m in reversed(messages):
        if isinstance(m, HumanMessage) or getattr(m, "type", "") == "human":
            user_text = getattr(m, "content", "")
            break

    if user_text is None:
        # Fallback: use last message content if no human found.
        last = messages[-1]
        user_text = getattr(last, "content", "")

    if isinstance(user_text, list):
        user_text = " ".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in user_text
        )

    logger.info("[CustomerData] Planning operation for user text: %s", user_text)

    # 1) Planner step: decide operation + args
    planner_response = await data_llm.ainvoke(
        [
            SystemMessage(content=DATA_PLANNER_PROMPT),
            HumanMessage(content=user_text),
        ]
    )

    logger.info("[CustomerData] Planner raw response: %s", planner_response.content)

    plan = _safe_json_from_llm(planner_response.content)
    operation = plan.get("operation")
    args = plan.get("args", {})

    if not operation:
        summary = (
            "Customer Data Agent could not determine a valid MCP operation "
            "from the user's request. Please clarify what data action is needed."
        )
        return {"messages": [AIMessage(content=summary)]}

    # 2) Normalize arguments before calling MCP tools
    if operation in {"get_customer", "update_customer", "create_ticket", "get_customer_history"}:
        cid = _normalize_customer_id(args, user_text)
        if cid is None:
            summary = (
                "Customer Data Agent could not determine a valid numeric customer ID "
                "from your request. Please provide a specific customer ID (e.g., 12345)."
            )
            return {"messages": [AIMessage(content=summary)]}

    if operation == "list_customers":
        _normalize_status(args)
        # Make sure limit is an int
        try:
            args["limit"] = int(args.get("limit", 50))
        except ValueError:
            args["limit"] = 50

    # 3) Connect to MCP server and load tools
    logger.info("[CustomerData] Connecting to MCP server at %s", MCP_SERVER_PATH)

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
    logger.info("[CustomerData] Loaded %d MCP tools", len(tools))

    tools_by_name = {t.name: t for t in tools}

    def find_tool(op_name: str):
        if op_name in tools_by_name:
            return tools_by_name[op_name]
        for name, tool in tools_by_name.items():
            if name.endswith(op_name):
                return tool
        return None

    tool = find_tool(operation)
    if tool is None:
        summary = (
            f"Customer Data Agent planned to call '{operation}' but could not "
            f"find a matching MCP tool. Available tools: {list(tools_by_name.keys())}"
        )
        return {"messages": [AIMessage(content=summary)]}

    # 4) Invoke MCP tool
    logger.info("[CustomerData] Invoking MCP tool '%s' with args %s", tool.name, args)
    try:
        result = await tool.ainvoke(args)
    except Exception as e:
        logger.exception("[CustomerData] Error invoking MCP tool: %s", e)
        summary = (
            f"Customer Data Agent tried to call '{operation}' but encountered an error: {e}"
        )
        return {"messages": [AIMessage(content=summary)]}

    logger.info("[CustomerData] Raw MCP result: %s", result)

    # 5) Summarise into plain AIMessage (NO tool_calls)
    summary = _summarise_result(operation, args, result)
    logger.info("[CustomerData] Summary: %s", summary)

    return {"messages": [AIMessage(content=summary)]}


def route_from_customer_data(state: Dict[str, Any]) -> str:
    texts: List[str] = []
    for m in state["messages"]:
        content = getattr(m, "content", "")
        if isinstance(content, list):
            content = " ".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            )
        texts.append(str(content))

    all_text = " ".join(texts).lower()
    keywords = ["help", "upgrade", "cancel", "refund", "issue", "support", "explain"]

    route = "support" if any(k in all_text for k in keywords) else "end"

    logger.info("[CustomerData] Routing after data step: %s", route)
    return route
