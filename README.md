# Multi-Agent Customer Service System with A2A and MCP

This repository implements a complete multi-agent customer service system using:

- LangGraph for agent orchestration  
- OpenAI models for reasoning and planning  
- Model Context Protocol (MCP) for secure tool and database access  
- SQLite for persistent customer and ticket storage  

The project fulfills all requirements of the assignment, including multi-agent coordination, A2A message passing, and MCP-integrated data operations.

---

## Architecture Overview

### 1. Router Agent (Orchestrator)
- Receives user queries  
- Uses an LLM to decide routing  
- Directs control flow to Customer Data Agent and Support Agent  
- Aggregates final responses  

### 2. Customer Data Agent (Specialist)
- Uses an LLM planner to select MCP operations  
- Calls MCP server tools for customer and ticket operations  
- Sends structured summaries back to Router Agent  

### 3. Support Agent (Specialist)
- Generates user-facing responses  
- References prior agent outputs  
- Handles escalation, multi-intent reasoning, and domain support  

### 4. MCP Server
- Hosts database-backed tools  
- Manages SQLite `support.db`  
- Provides required operations:  
  `get_customer`, `list_customers`, `update_customer`, `create_ticket`, `get_customer_history`

---

## System Diagram

```
                ┌────────────────────────┐
                │      User Query        │
                └─────────────┬──────────┘
                              ▼
                    ┌──────────────────┐
                    │   Router Agent   │
                    └───┬──────────────┘
        ┌────────────────┼───────────────────┐
        ▼                ▼                   ▼
┌──────────────┐  ┌──────────────┐   ┌──────────────┐
│  Data Route  │  │Support Route │   │Data+Support  │
└───────┬──────┘  └──────┬───────┘   └──────┬───────┘
        ▼                 ▼                   ▼
┌──────────────────┐  ┌──────────────────┐   ┌──────────────────┐
│ Customer Data    │  │ Support Agent    │   │ Customer Data     │
│ Agent (MCP Calls)│  │ (LLM Responses)  │   │ Agent then Support│
└──────────┬───────┘  └─────────┬────────┘   └──────────┬────────┘
           ▼                    ▼                        ▼
        ┌──────────────────────────────────────────────────────┐
        │          Shared LangGraph State (messages)            │
        └──────────────────────────────────────────────────────┘
                              ▼
                    ┌──────────────────┐
                    │ Final User Reply │
                    └──────────────────┘
```

---

## Setup Instructions

### 1. Clone Repository

```
git clone <your-repo-url>
cd multi-agent-customer-service-a2a-mcp
```

---

## 2. Create Virtual Environment

```
python3 -m venv a2aenv
source a2aenv/bin/activate
```

Deactivate with:

```
deactivate
```

---

## 3. Install Dependencies

```
pip install -r requirements.txt
```

`requirements.txt` should include:

```
langgraph==0.5.4
langgraph-sdk==0.1.74
langchain-core>=0.1
langchain-openai==1.1.0
openai>=1.109.0
mcp>=1.2.0
langchain-mcp-adapters>=0.1.0
pydantic>=2.0
typing-extensions>=4.12
python-dotenv>=1.0.0
```

---

## 4. Provide OpenAI API Key

```
export OPENAI_API_KEY="YOUR_KEY_HERE"
```

Or create `.env`:

```
OPENAI_API_KEY=YOUR_KEY_HERE
```

---

## 5. Initialize SQLite Support Database

```
cd mcp_server
python database_setup.py
```

---

## 6. Start MCP Server

```
python server.py
```

You should see MCP tools loading.

---

## 7. Run the Demo Notebook

```
demo.ipynb
```

Example:

```python
await run_query("I need help with my account, customer ID 5")
```

---

## Repository Layout

```
agents/
    router_agent.py
    customer_data_agent.py
    support_agent.py
mcp_server/
    server.py
    database_setup.py
    support.db
demo.ipynb
requirements.txt
README.md
.env.example
```

---

## Grading Criteria Alignment

This project fulfills:

✔ Multi-agent architecture  
✔ MCP integration with 5 required tools  
✔ A2A communication via LangGraph messages  
✔ Multi-step coordination flows  
✔ Test scenarios in demo notebook  
✔ Clear system documentation and setup instructions  

---

## Conclusion

This system demonstrates full agentic coordination using LangGraph, MCP, and OpenAI models with robust multi-step reasoning and database-backed operations. It is scalable and ready for extension into production-grade architectures.

