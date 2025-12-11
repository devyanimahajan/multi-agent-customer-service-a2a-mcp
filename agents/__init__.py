# agents/__init__.py

from typing import List, TypedDict, Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, MessagesState, START, END, add_messages

from .router_agent import router_agent, route_from_router
from .customer_data_agent import customer_data_agent, route_from_customer_data
from .support_agent import support_agent


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


def build_customer_service_graph():
    """
    Build the LangGraph StateGraph for the multi-agent customer service system.

    Nodes:
    - router: orchestrator / intent detection
    - customer_data: data specialist using MCP tools
    - support: support specialist

    Edges:
    - START -> router
    - router -> customer_data / support / END based on route_from_router
    - customer_data -> support / END based on route_from_customer_data
    - support -> END
    """
    graph = StateGraph(AgentState)

    graph.add_node("router", router_agent)
    graph.add_node("customer_data", customer_data_agent)
    graph.add_node("support", support_agent)

    # Start at router
    graph.add_edge(START, "router")

    # Router branching
    graph.add_conditional_edges(
        "router",
        route_from_router,
        {
            "customer_data": "customer_data",
            "support": "support",
            "end": END,
        },
    )

    # After customer data, either go to support or end
    graph.add_conditional_edges(
        "customer_data",
        route_from_customer_data,
        {
            "support": "support",
            "end": END,
        },
    )

    # Support always ends
    graph.add_edge("support", END)

    return graph.compile()
