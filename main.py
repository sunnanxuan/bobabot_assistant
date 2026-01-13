# -*- coding: utf-8 -*-
"""
backend/main.py
FastAPI 后端：接收订单 -> 生成取件码 -> 写入 SQLite（服务端权威定价）
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
import requests
from fastapi import FastAPI, Header, HTTPException, UploadFile, File, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

# ==== 价格与品类（与前端/工具共用同一份配置） ====
# 请确保 menu_config.py 与 tools.py 使用的是同一份菜单配置
from menu_config import PRICES, CATEGORY, EXTRAS, hot_allowed

# 加载 .env
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

DB_PATH = os.getenv("ORDERS_DB", "orders.db")
BACKEND_TOKEN = os.getenv("BACKEND_TOKEN", "devtoken")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
STT_MODEL = os.getenv("STT_MODEL", "gpt-4o-mini-transcribe")
TTS_MODEL = os.getenv("TTS_MODEL", "gpt-4o-mini-tts")
TTS_VOICE = os.getenv("TTS_VOICE", "alloy")
TTS_INSTRUCTIONS = os.getenv("TTS_INSTRUCTIONS", "Speak clearly and naturally.")

# 确保数据库目录存在
_db_dir = os.path.dirname(DB_PATH)
if _db_dir:
    os.makedirs(_db_dir, exist_ok=True)

app = FastAPI(title="BobaBot Orders API", version="1.1.0")

# 可选：本地联调前端需要跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 需要更严格可改白名单
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- DB ----------
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

# ---------- 定价（服务端权威） ----------
ALLOWED_SIZES = {"小杯", "中杯", "大杯"}
ALLOWED_ICES = {"正常冰", "少冰", "去冰", "热饮"}
ALLOWED_SUGARS = {"全糖", "半糖", "少糖", "无糖"}

def server_quote_item(
    drink: str,
    size: str,
    extras: List[str],
    sugar: Optional[str],
    ice: Optional[str],
    hot_flag: Optional[bool] = None,
) -> dict:
    """
    返回：{ ok, error?, base_price, extras_cost, line_total, hot, ice }
    - 统一在后端校验饮品、规格、小料、冷热限制（果茶仅冷饮）。
    - 以 “热饮” 或 hot_flag=True 视为热饮；若该饮品不可热则报错。
    """
    # 基本校验
    if drink not in PRICES:
        return {"ok": False, "error": f"未找到饮品：{drink}"}
    if size not in ALLOWED_SIZES or size not in PRICES[drink]:
        return {"ok": False, "error": f"规格错误：{size}"}
    if sugar and sugar not in ALLOWED_SUGARS:
        return {"ok": False, "error": f"甜度非法：{sugar}"}
    if ice and ice not in ALLOWED_ICES:
        return {"ok": False, "error": f"冰量非法：{ice}"}

    # 统一冷热逻辑
    is_hot = bool(hot_flag) or (ice == "热饮")
    ice_final = "热饮" if is_hot else (ice or "正常冰")

    # 果茶/不可热拦截
    if is_hot and not hot_allowed(drink):
        return {"ok": False, "error": f"{drink} 为果茶类，只能做冷饮"}

    # 小料校验与加价
    extras_cost = 0
    for e in extras or []:
        if e not in EXTRAS:
            return {"ok": False, "error": f"未知小料：{e}"}
        extras_cost += EXTRAS[e]

    base = PRICES[drink][size]
    line_total = base + extras_cost

    return {
        "ok": True,
        "base_price": base,
        "extras_cost": extras_cost,
        "line_total": line_total,
        "hot": is_hot,
        "ice": ice_final,
    }

# ---------- Schemas ----------
class ItemIn(BaseModel):
    drink: str
    size: str
    hot: Optional[bool] = None
    sugar: Optional[str] = None
    ice: Optional[str] = None
    extras: List[str] = []

    # 允许前端传金额，但**后端会完全忽略并重算**：
    base_price: Optional[int] = None
    extras_cost: Optional[int] = None
    line_total: Optional[int] = None

class OrderIn(BaseModel):
    items: List[ItemIn]
    # 允许前端传 total，但**后端会完全忽略并重算**：
    total: Optional[int] = None

    channel: str = "chat"
    customer_name: Optional[str] = None
    customer_phone: Optional[str] = None
    client_order_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    @validator("items")
    def _not_empty(cls, v):
        if not v:
            raise ValueError("订单不得为空")
        return v

class OrderOut(BaseModel):
    order_id: str
    pickup_code: str
    total: int
    created_at: str
    status: str = "pending"

# ---------- App ----------
@app.on_event("startup")
def _startup() -> None:
    init_db()

@app.get("/health")
def health():
    return {"ok": True, "db": DB_PATH}

@app.post("/orders", response_model=OrderOut)
def create_order(order: OrderIn, authorization: str = Header(None)) -> OrderOut:
    # 鉴权
    if authorization != f"Bearer {BACKEND_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    # 服务端权威定价（逐项重算）
    computed_items = []
    computed_total = 0
    for it in order.items:
        quoted = server_quote_item(
            drink=it.drink,
            size=it.size,
            extras=it.extras or [],
            sugar=it.sugar,
            ice=it.ice,
            hot_flag=it.hot,
        )
        if not quoted["ok"]:
            raise HTTPException(status_code=400, detail=quoted["error"])
        computed_items.append({
            "drink": it.drink,
            "size": it.size,
            "hot": quoted["hot"],
            "sugar": it.sugar,
            "ice": quoted["ice"],
            "extras": it.extras or [],
            "base_price": quoted["base_price"],
            "extras_cost": quoted["extras_cost"],
            "line_total": quoted["line_total"],
        })
        computed_total += quoted["line_total"]

    # 幂等：client_order_id 已存在则直接返回
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute("SELECT * FROM orders WHERE client_order_id=?", (order.client_order_id,))
        row = cur.fetchone()
        if row:
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

        # 写入主表
        conn.execute(
            """
            INSERT INTO orders (id, client_order_id, pickup_code, total, channel,
                                customer_name, customer_phone, created_at, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                order_id,
                order.client_order_id,
                pickup_code,
                computed_total,  # 以服务端重算为准
                order.channel,
                order.customer_name,
                order.customer_phone,
                created_at,
                "pending",
            ),
        )

        # 写入明细
        for ci in computed_items:
            conn.execute(
                """
                INSERT INTO order_items (id, order_id, drink, size, hot, sugar, ice, extras,
                                         base_price, extras_cost, line_total)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    order_id,
                    ci["drink"],
                    ci["size"],
                    1 if ci["hot"] else 0,
                    ci["sugar"],
                    ci["ice"],
                    json.dumps(ci["extras"], ensure_ascii=False),
                    ci["base_price"],
                    ci["extras_cost"],
                    ci["line_total"],
                ),
            )

        conn.commit()
        return OrderOut(
            order_id=order_id,
            pickup_code=pickup_code,
            total=computed_total,
            created_at=created_at,
            status="pending",
        )
    except sqlite3.IntegrityError as e:
        conn.rollback()
        # 可能是 UNIQUE 约束冲突（极端并发生成了同码等）
        raise HTTPException(status_code=409, detail=f"Conflict: {e}")
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")
    finally:
        conn.close()


# === 新增：查询/支付两个接口 ===
from fastapi import Path

def _fetch_order(conn, order_id: str):
    cur = conn.execute("SELECT * FROM orders WHERE id=?", (order_id,))
    return cur.fetchone()

@app.get("/orders/{order_id}", response_model=OrderOut)
def get_order(order_id: str, authorization: str = Header(None)) -> OrderOut:
    if authorization != f"Bearer {BACKEND_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        row = _fetch_order(conn, order_id)
        if not row:
            raise HTTPException(status_code=404, detail="Order not found")
        return OrderOut(
            order_id=row["id"],
            pickup_code=row["pickup_code"],
            total=row["total"],
            created_at=row["created_at"],
            status=row["status"],
        )
    finally:
        conn.close()

@app.post("/orders/{order_id}/pay", response_model=OrderOut)
def mark_paid(order_id: str = Path(...), authorization: str = Header(None)) -> OrderOut:
    if authorization != f"Bearer {BACKEND_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        row = _fetch_order(conn, order_id)
        if not row:
            raise HTTPException(status_code=404, detail="Order not found")
        if row["status"] == "paid":
            return OrderOut(
                order_id=row["id"],
                pickup_code=row["pickup_code"],
                total=row["total"],
                created_at=row["created_at"],
                status=row["status"],
            )
        conn.execute("UPDATE orders SET status=? WHERE id=?", ("paid", order_id))
        conn.commit()
        row = _fetch_order(conn, order_id)
        return OrderOut(
            order_id=row["id"],
            pickup_code=row["pickup_code"],
            total=row["total"],
            created_at=row["created_at"],
            status=row["status"],
        )
    finally:
        conn.close()




# =========================
# Voice AI: STT / TTS APIs
# =========================

MAX_AUDIO_BYTES = 25 * 1024 * 1024  # OpenAI docs: uploads currently limited to 25MB

def _require_auth(authorization: str):
    print("AUTH HEADER =", authorization)
    print("EXPECTED   =", f"Bearer {BACKEND_TOKEN}")
    if authorization != f"Bearer {BACKEND_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")

@app.post("/voice/stt")
async def voice_stt(
    file: UploadFile = File(...),
    authorization: str = Header(None)
):
    """
    Speech-to-Text: upload audio -> transcript text
    """
    _require_auth(authorization)
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set on backend")

    audio_bytes = await file.read()
    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Audio file too large (max 25MB)")

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    data = {
        "model": STT_MODEL,
        "response_format": "json",
        "language": "zh",  # ✅ 强制中文（普通话）
        "prompt": "请将音频内容转写为简体中文，使用中文标点；只输出中文，不要输出任何其他语言。",
    }
    files = {
        "file": (
            file.filename or "audio.wav",
            audio_bytes,
            file.content_type or "application/octet-stream",
        )
    }

    r = requests.post(
        "https://api.openai.com/v1/audio/transcriptions",
        headers=headers,
        data=data,
        files=files,
        timeout=60,
    )

    if r.status_code >= 400:
        raise HTTPException(status_code=r.status_code, detail=r.text)

    j = r.json()
    return {"text": (j.get("text") or "").strip()}

@app.post("/voice/tts")
def voice_tts(
    payload: dict,
    authorization: str = Header(None)
):
    """
    Text-to-Speech: text -> mp3 bytes
    payload: { "text": "...", "voice": "alloy"(optional), "instructions": "..."(optional) }
    """
    _require_auth(authorization)
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set on backend")

    text = (payload.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Missing text")
    # OpenAI docs: input text max length 4096 chars
    text = text[:4096]

    voice = payload.get("voice") or TTS_VOICE
    instructions = payload.get("instructions") or TTS_INSTRUCTIONS

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": TTS_MODEL,
        "voice": voice,
        "input": text,
        "instructions": instructions,
        "response_format": "mp3",
    }

    r = requests.post(
        "https://api.openai.com/v1/audio/speech",
        headers=headers,
        json=body,
        timeout=60,
    )

    if r.status_code >= 400:
        raise HTTPException(status_code=r.status_code, detail=r.text)

    return Response(content=r.content, media_type="audio/mpeg")
