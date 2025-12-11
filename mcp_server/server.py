# mcp_server/server.py

import sqlite3
from datetime import datetime
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# Database file created by database_setup.py
DB_PATH = Path(__file__).with_name("support.db")

mcp = FastMCP("customer_support_db")


def get_conn():
    return sqlite3.connect(DB_PATH)


@mcp.tool()
def get_customer(customer_id: int) -> dict:
    """
    Get a single customer record by ID.
    """
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, name, email, phone, status, created_at, updated_at
        FROM customers
        WHERE id = ?
        """,
        (customer_id,),
    )
    row = cur.fetchone()
    conn.close()

    if not row:
        return {"success": False, "message": "Customer not found"}

    return {
        "success": True,
        "customer": {
            "id": row[0],
            "name": row[1],
            "email": row[2],
            "phone": row[3],
            "status": row[4],
            "created_at": row[5],
            "updated_at": row[6],
        },
    }


@mcp.tool()
def list_customers(status: str = None, limit: int = 50) -> dict:
    """
    List customers, optionally filtered by status, up to a given limit.
    """
    conn = get_conn()
    cur = conn.cursor()

    if status:
        cur.execute(
            """
            SELECT id, name, email, phone, status
            FROM customers
            WHERE status = ?
            LIMIT ?
            """,
            (status, limit),
        )
    else:
        cur.execute(
            """
            SELECT id, name, email, phone, status
            FROM customers
            LIMIT ?
            """,
            (limit,),
        )

    rows = cur.fetchall()
    conn.close()

    customers = [
        {
            "id": r[0],
            "name": r[1],
            "email": r[2],
            "phone": r[3],
            "status": r[4],
        }
        for r in rows
    ]

    return {
        "success": True,
        "count": len(customers),
        "customers": customers,
    }


@mcp.tool()
def update_customer(customer_id: int, data: dict) -> dict:
    """
    Update a customer's fields using the keys provided in `data`.
    """
    if not data:
        return {"success": False, "message": "No fields provided to update"}

    allowed_fields = {"name", "email", "phone", "status"}
    updates = []
    values = []

    for key, value in data.items():
        if key in allowed_fields:
            updates.append(f"{key} = ?")
            values.append(value)

    if not updates:
        return {"success": False, "message": "No valid fields to update"}

    # Add updated_at and id
    values.append(datetime.utcnow().isoformat())
    values.append(customer_id)

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        f"""
        UPDATE customers
        SET {', '.join(updates)}, updated_at = ?
        WHERE id = ?
        """,
        tuple(values),
    )
    conn.commit()

    # Fetch updated row
    cur.execute(
        """
        SELECT id, name, email, phone, status, created_at, updated_at
        FROM customers
        WHERE id = ?
        """,
        (customer_id,),
    )
    row = cur.fetchone()
    conn.close()

    if not row:
        return {"success": False, "message": "Customer not found after update"}

    return {
        "success": True,
        "customer": {
            "id": row[0],
            "name": row[1],
            "email": row[2],
            "phone": row[3],
            "status": row[4],
            "created_at": row[5],
            "updated_at": row[6],
        },
    }


@mcp.tool()
def create_ticket(customer_id: int, issue: str, priority: str = "medium") -> dict:
    """
    Create a support ticket for a given customer.
    """
    if priority not in {"low", "medium", "high"}:
        priority = "medium"

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO tickets (customer_id, issue, status, priority, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (customer_id, issue, "open", priority, datetime.utcnow().isoformat()),
    )
    ticket_id = cur.lastrowid
    conn.commit()
    conn.close()

    return {
        "success": True,
        "ticket_id": ticket_id,
        "customer_id": customer_id,
        "issue": issue,
        "status": "open",
        "priority": priority,
    }


@mcp.tool()
def get_customer_history(customer_id: int) -> dict:
    """
    Get all tickets for a given customer.
    """
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, issue, status, priority, created_at
        FROM tickets
        WHERE customer_id = ?
        ORDER BY created_at DESC
        """,
        (customer_id,),
    )
    rows = cur.fetchall()
    conn.close()

    tickets = [
        {
            "id": r[0],
            "issue": r[1],
            "status": r[2],
            "priority": r[3],
            "created_at": r[4],
        }
        for r in rows
    ]

    return {
        "success": True,
        "customer_id": customer_id,
        "ticket_count": len(tickets),
        "tickets": tickets,
    }


if __name__ == "__main__":
    # Run over stdio so MultiServerMCPClient can talk to it
    mcp.run(transport="stdio")
