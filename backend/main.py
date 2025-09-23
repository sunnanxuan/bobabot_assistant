# -*- coding: utf-8 -*-
"""
backend/main.py
FastAPI 后端：接收订单 -> 生成取件码 -> 写入 SQLite
运行：
  uvicorn backend.main:app --reload --port 8000
环境变量（.env）：
  BACKEND_TOKEN=change-this-to-a-long-random-secret
  ORDERS_DB=./data/orders.db
"""

import os
import json
import uuid
import random
import sqlite3
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# 加载 .env
load_dotenv()

DB_PATH = os.getenv("ORDERS_DB", "orders.db")
BACKEND_TOKEN = os.getenv("BACKEND_TOKEN", "devtoken")

app = FastAPI(title="BobaBot Orders API")

def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS orders (
      id TEXT PRIMARY KEY,
      client_order_id TEXT UNIQUE,
      pickup_code TEXT UNIQUE,
      total INTEGER NOT NULL,
      channel TEXT,
      customer_name TEXT,
      customer_phone TEXT,
      created_at TEXT NOT NULL,
      status TEXT NOT NULL
    )
    """)
    c.execute("""
    CREATE TABLE IF NOT EXISTS order_items (
      id TEXT PRIMARY KEY,
      order_id TEXT NOT NULL,
      drink TEXT NOT NULL,
      size TEXT NOT NULL,
      hot INTEGER,
      sugar TEXT,
      ice TEXT,
      extras TEXT,
      base_price INTEGER,
      extras_cost INTEGER,
      line_total INTEGER,
      FOREIGN KEY(order_id) REFERENCES orders(id)
    )
    """)
    conn.commit()
    conn.close()

def gen_pickup_code(conn: sqlite3.Connection) -> str:
    while True:
        code = f"{random.randint(0, 999999):06d}"
        cur = conn.execute("SELECT 1 FROM orders WHERE pickup_code=?", (code,))
        if not cur.fetchone():
            return code

class ItemIn(BaseModel):
    drink: str
    size: str
    hot: bool = False
    sugar: Optional[str] = None
    ice: Optional[str] = None
    extras: List[str] = []
    base_price: int
    extras_cost: int
    line_total: int

class OrderIn(BaseModel):
    items: List[ItemIn]
    total: int
    channel: str = "chat"
    customer_name: Optional[str] = None
    customer_phone: Optional[str] = None
    client_order_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

class OrderOut(BaseModel):
    order_id: str
    pickup_code: str
    total: int
    created_at: str
    status: str = "pending"

@app.on_event("startup")
def _startup() -> None:
    init_db()

@app.post("/orders", response_model=OrderOut)
def create_order(order: OrderIn, authorization: str = Header(None)) -> OrderOut:
    if authorization != f"Bearer {BACKEND_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    # 幂等：client_order_id 已存在则直接返回
    cur = c.execute("SELECT * FROM orders WHERE client_order_id=?", (order.client_order_id,))
    row = cur.fetchone()
    if row:
        conn.close()
        return OrderOut(
            order_id=row["id"],
            pickup_code=row["pickup_code"],
            total=row["total"],
            created_at=row["created_at"],
            status=row["status"],
        )

    order_id = str(uuid.uuid4())
    pickup_code = gen_pickup_code(conn)
    created_at = datetime.utcnow().isoformat()

    c.execute(
        """
        INSERT INTO orders (id, client_order_id, pickup_code, total, channel,
                            customer_name, customer_phone, created_at, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            order_id,
            order.client_order_id,
            pickup_code,
            order.total,
            order.channel,
            order.customer_name,
            order.customer_phone,
            created_at,
            "pending",
        ),
    )

    for it in order.items:
        c.execute(
            """
            INSERT INTO order_items (id, order_id, drink, size, hot, sugar, ice, extras,
                                     base_price, extras_cost, line_total)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                order_id,
                it.drink,
                it.size,
                1 if it.hot else 0,
                it.sugar,
                it.ice,
                json.dumps(it.extras, ensure_ascii=False),
                it.base_price,
                it.extras_cost,
                it.line_total,
            ),
        )

    conn.commit()
    conn.close()

    return OrderOut(
        order_id=order_id,
        pickup_code=pickup_code,
        total=order.total,
        created_at=created_at,
        status="pending",
    )

@app.get("/health")
def health():
    return {"ok": True, "db": DB_PATH}
