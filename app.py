# -*- coding: utf-8 -*-
"""
app.py — Streamlit 前端（对话点单 + 确认提交）
依赖：
  pip install streamlit langchain langchain-openai python-dotenv requests tiktoken

环境变量（.env）：
  OPENAI_API_KEY=sk-xxxx
  FT_MODEL=ft:gpt-4o-mini-2024-07-18:personal::YOUR_ID
  MODEL_TEMP=0.3
  BOT_NAME=BobaBot
  BACKEND_URL=http://localhost:8000
  BACKEND_TOKEN=change-this-to-a-long-random-secret
"""

import os
import json
import uuid
import requests
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from prompts import get_system_prompt
from tools import TOOLS, quote_price, list_menu, can_make_hot

# ========== 读取 .env ==========
load_dotenv()
FT_MODEL = os.getenv("FT_MODEL") or os.getenv("BASE_FALLBACK_MODEL", "gpt-4o-mini-2024-07-18")
MODEL_TEMP = float(os.getenv("MODEL_TEMP", "0.3"))
BOT_NAME = os.getenv("BOT_NAME", "BobaBot")
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
BACKEND_TOKEN = os.getenv("BACKEND_TOKEN", "devtoken")

# ========== 模型与提示词 ==========
SYSTEM_PROMPT = get_system_prompt(BOT_NAME)
llm = ChatOpenAI(model=FT_MODEL, temperature=MODEL_TEMP)
llm_with_tools = llm.bind_tools(TOOLS)

# ========== Streamlit UI ==========
st.set_page_config(page_title=f"{BOT_NAME} · 奶茶店员", layout="centered")
st.title(f"🧋 {BOT_NAME} · 奶茶店员")

# 对话与购物车状态
if "msgs" not in st.session_state:
    st.session_state.msgs = [SystemMessage(content=SYSTEM_PROMPT)]
if "cart" not in st.session_state:
    st.session_state.cart = []              # 已确认加入订单的杯子
if "pending_item" not in st.session_state:
    st.session_state.pending_item = None    # 最近一次报价（待确认）

# 展示历史消息
for m in st.session_state.msgs:
    if isinstance(m, HumanMessage):
        st.chat_message("user").write(m.content)
    elif isinstance(m, AIMessage):
        st.chat_message("assistant").write(m.content)

# 侧栏购物车
with st.sidebar:
    st.subheader("🛒 当前订单")
    if st.session_state.pending_item:
        pi = st.session_state.pending_item
        st.write(f"待确认：{pi['drink']} {pi['size']} - ¥{pi['total']}")
    if st.session_state.cart:
        for i, it in enumerate(st.session_state.cart, 1):
            st.write(f"{i}. {it['drink']} {it['size']} - ¥{it['total']}")
    st.write("---")
    st.write("合计：¥", sum(x["total"] for x in st.session_state.cart))

# ===== 提交订单到 FastAPI 后端 =====
def submit_order_to_backend(items: list):
    total = sum(x["total"] for x in items)
    payload = {
        "items": [
            {
                "drink": x["drink"],
                "size": x["size"],
                "hot": bool(x.get("hot")),
                "sugar": x.get("sugar"),
                "ice": x.get("ice"),
                "extras": [e["name"] for e in x.get("extras", [])],
                "base_price": x.get("base_price", 0),
                "extras_cost": sum(e["price"] for e in x.get("extras", [])),
                "line_total": x["total"],
            }
            for x in items
        ],
        "total": total,
        "channel": "streamlit",
        "client_order_id": str(uuid.uuid4()),  # 幂等键
    }
    r = requests.post(
        f"{BACKEND_URL}/orders",
        headers={"Authorization": f"Bearer {BACKEND_TOKEN}"},
        json=payload,
        timeout=15,
    )
    r.raise_for_status()
    return r.json()  # {order_id, pickup_code, total, created_at, status}

# ===== 处理工具调用 =====
def handle_tool_call(tc):
    name, args = tc["name"], (tc.get("args") or {})
    if name == "quote_price":
        result = quote_price.invoke(args)
        # 报价成功：把上一杯（如果有）先加入购物车；当前作为待确认项
        if result.get("ok"):
            if st.session_state.pending_item:
                st.session_state.cart.append(st.session_state.pending_item)
            st.session_state.pending_item = result
        return result
    elif name == "list_menu":
        return list_menu.invoke(args)
    elif name == "can_make_hot":
        return can_make_hot.invoke(args)
    else:
        return {"ok": False, "error": f"未知工具：{name}"}

# ===== 单轮执行（已修复“ai=null/不显示文本”） =====
def run_turn(user_text: str):
    # 触发提交订单
    if user_text.strip() in {"确认点单", "确认下单", "确认点餐"}:
        items = list(st.session_state.cart)
        if st.session_state.pending_item:
            items.append(st.session_state.pending_item)
            st.session_state.pending_item = None
        if not items:
            st.chat_message("assistant").write("当前没有已确认的饮品，请先点单哦。")
            return
        try:
            resp = submit_order_to_backend(items)
            st.session_state.cart.clear()
            msg = f"✅ 下单成功！取件码：**{resp['pickup_code']}**，合计：¥{resp['total']}。需要我再帮你加单或修改吗？"
            st.session_state.msgs.append(AIMessage(content=msg))
            st.chat_message("assistant").write(msg)
        except Exception as e:
            st.session_state.msgs.append(AIMessage(content=f"下单失败：{e}"))
            st.chat_message("assistant").write(f"下单失败：{e}")
        return

    # 普通对话（模型 + 工具）
    st.session_state.msgs.append(HumanMessage(user_text))
    st.chat_message("user").write(user_text)   # 立刻渲染当前用户消息

    try:
        ai: AIMessage = llm_with_tools.invoke(st.session_state.msgs)
    except Exception as e:
        st.error(f"调用模型失败：{e}")
        return

    st.session_state.msgs.append(ai)

    # 兼容不同版本的 tool_calls 取法
    tool_calls = getattr(ai, "tool_calls", None)
    if not tool_calls and hasattr(ai, "additional_kwargs"):
        tool_calls = ai.additional_kwargs.get("tool_calls")

    if tool_calls:
        # 1) 执行工具并用 ToolMessage 回传
        for tc in tool_calls:
            result = handle_tool_call(tc)
            st.session_state.msgs.append(
                ToolMessage(content=json.dumps(result, ensure_ascii=False), tool_call_id=tc["id"])
            )

        # 2) 再取最终文本；若仍有工具调用，再补 2 轮
        final: AIMessage = llm_with_tools.invoke(st.session_state.msgs)
        for _ in range(2):
            more_calls = getattr(final, "tool_calls", None)
            if not more_calls and hasattr(final, "additional_kwargs"):
                more_calls = final.additional_kwargs.get("tool_calls")
            if not more_calls:
                break
            for tc in more_calls:
                result = handle_tool_call(tc)
                st.session_state.msgs.append(
                    ToolMessage(content=json.dumps(result, ensure_ascii=False), tool_call_id=tc["id"])
                )
            final = llm_with_tools.invoke(st.session_state.msgs)

        st.session_state.msgs.append(final)
        st.chat_message("assistant").write(final.content or "（已完成工具调用）")
    else:
        # 没有工具调用：直接渲染文本（避免 None 显示为 null）
        st.chat_message("assistant").write(ai.content or "（已收到）")

# ===== 输入框 =====
user = st.chat_input("点单/询价/推荐/介绍……（下单时可直接输入：确认点单）")
if user:
    run_turn(user)
