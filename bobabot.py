# -*- coding: utf-8 -*-
"""
bobabot.py — 命令行版（购物车合单 + 两段式确认）
依赖：
  pip install langchain langchain-openai python-dotenv requests tiktoken

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
import re
import requests
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from system_prompt import get_system_prompt
from tools import TOOLS, quote_price, list_menu, can_make_hot

# ====== 环境变量 ======
load_dotenv()
FT_MODEL = os.getenv("FT_MODEL") or os.getenv("BASE_FALLBACK_MODEL", "gpt-4o-mini-2024-07-18")
MODEL_TEMP = float(os.getenv("MODEL_TEMP", "0.3"))
BOT_NAME = os.getenv("BOT_NAME", "BobaBot")
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
BACKEND_TOKEN = os.getenv("BACKEND_TOKEN", "devtoken")

# ====== 模型与系统提示 ======
SYSTEM_PROMPT = get_system_prompt(BOT_NAME)
llm = ChatOpenAI(model=FT_MODEL, temperature=MODEL_TEMP)
llm_with_tools = llm.bind_tools(TOOLS)

# ====== 状态 ======
msgs = [SystemMessage(content=SYSTEM_PROMPT)]
cart = []            # 多杯
pending_item = None  # 待加入
stage = "BROWSING"   # BROWSING / AWAIT_CONFIRM_ORDER / AWAIT_CONFIRM_SUBMIT

def submit_order_to_backend(items: list) -> dict:
    payload = {
        "items": [{
            "drink": x["drink"],
            "size": x["size"],
            "hot": bool(x.get("hot")),
            "sugar": x.get("sugar"),
            "ice": x.get("ice"),
            "extras": [e["name"] for e in x.get("extras", [])],
        } for x in items],
        "channel": "cli",
        "client_order_id": str(uuid.uuid4()),
    }
    r = requests.post(
        f"{BACKEND_URL}/orders",
        headers={"Authorization": f"Bearer {BACKEND_TOKEN}"},
        json=payload,
        timeout=20,
    )
    r.raise_for_status()
    return r.json()

def cart_total():
    return sum(x.get("total", 0) for x in cart)

def cart_summary():
    if not cart:
        return "购物车为空。"
    lines = []
    for i, it in enumerate(cart, 1):
        ex = it.get("extras") or []
        ex_txt = "、".join(e["name"] for e in ex) if ex else "无"
        lines.append(f"{i}) {it['drink']} {it['size']} {it.get('sugar') or ''} {it.get('ice') or ''} 小料：{ex_txt} 小计：¥{it.get('total',0)}")
    return "订单摘要：\n" + "\n".join(lines) + f"\n合计：¥{cart_total()}"

def print_state():
    print("\n🧾 当前：")
    if pending_item:
        ex = pending_item.get("extras") or []
        ex_txt = "、".join(e["name"] for e in ex) if ex else "无"
        print(f"  待加入：{pending_item['drink']} {pending_item['size']} 小料：{ex_txt} 小计（前端展示）：¥{pending_item.get('total')}")
    else:
        print("  无待加入项")
    print("  购物车共", len(cart), "杯，合计：¥", cart_total())
    print("  阶段：", stage)
    print("")

def handle_tool_call(tc):
    global pending_item, stage
    name, args = tc["name"], (tc.get("args") or {})
    if name == "quote_price":
        result = quote_price.invoke(args)
        if result.get("ok"):
            pending_item = result
            stage = "BROWSING"
        return result
    elif name == "list_menu":
        return list_menu.invoke(args)
    elif name == "can_make_hot":
        return can_make_hot.invoke(args)
    else:
        return {"ok": False, "error": f"未知工具：{name}"}

def help_text():
    print("""
命令：
  /help            帮助
  /state           查看状态
  /add             将“待加入”杯加入购物车（等同“加入购物车”）
  /del N           删除第N杯（如 /del 2）
  /clear           清空购物车
  /checkout        去结算（生成摘要并进入确认流程）
  /confirm         确认订单（第一段确认）
  /submit          确认下单（第二段确认 → 提交后端）
  /exit            退出

说明：
  - 正常输入自然语言就能推荐/咨询/点单；每次报价成功后，该杯会成为“待加入”。
  - 多杯下单：反复 /add 或输入“加入购物车”把杯子加入，再 /checkout → /confirm → /submit。
""")

def main():
    global pending_item, cart, stage, msgs

    print(f"{BOT_NAME}（模型：{FT_MODEL}）已启动。输入 /help 查看命令，/exit 退出。\n")

    while True:
        try:
            user_text = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if not user_text:
            continue

        # —— 命令 ——
        low = user_text.lower()
        if low in {"/exit", "exit", "quit"}:
            print("再见！")
            break

        if low == "/help":
            help_text(); continue

        if low == "/state":
            print_state(); continue

        if low == "/add" or user_text in {"加入购物车", "加到购物车", "确认本杯", "添加本杯"}:
            if not pending_item:
                print("BobaBot：当前没有待加入的饮品，请先报价。")
                continue
            cart.append(pending_item)
            pending_item = None
            stage = "BROWSING"
            print("BobaBot：已加入购物车。还要再加一杯吗？若结算请输入 /checkout 或 “去结算”。")
            print_state()
            continue

        if low.startswith("/del"):
            m = re.search(r"/del\s+(\d+)", low)
            if not m:
                print("用法：/del 2  删除第2杯")
                continue
            idx = int(m.group(1)) - 1
            if 0 <= idx < len(cart):
                cart.pop(idx)
                print("BobaBot：已删除。"); print_state()
            else:
                print("BobaBot：未找到该序号。")
            continue

        if low == "/clear" or user_text in {"清空", "清空购物车"}:
            cart.clear()
            print("BobaBot：已清空购物车。"); print_state()
            continue

        if low == "/checkout" or user_text in {"去结算", "结算"}:
            if pending_item:
                cart.append(pending_item)
                pending_item = None
            if not cart:
                print("BobaBot：购物车为空，请先加入至少一杯。")
                continue
            stage = "AWAIT_CONFIRM_ORDER"
            print("BobaBot：", cart_summary())
            print("BobaBot：请回复 /confirm 或 “确认订单”；如需调整可用 /del N。")
            continue

        if low == "/confirm" or user_text == "确认订单":
            if stage != "AWAIT_CONFIRM_ORDER":
                print("BobaBot：尚未进入结算，请先 /checkout 或输入“去结算”。")
                continue
            stage = "AWAIT_CONFIRM_SUBMIT"
            print("BobaBot：已确认订单。请回复 /submit 或 “确认下单”以提交并生成取件码。")
            continue

        if low == "/submit" or user_text in {"确认下单", "确认点单", "确认点餐"}:
            if stage != "AWAIT_CONFIRM_SUBMIT":
                print("BobaBot：尚未确认订单，请先 /confirm 或输入“确认订单”。")
                continue
            try:
                resp = submit_order_to_backend(cart)
                cart.clear()
                stage = "BROWSING"
                print(f"BobaBot：✅ 下单成功！取件码：{resp['pickup_code']}，合计：¥{resp['total']}。还需要再来一单吗？")
            except Exception as e:
                print(f"BobaBot：下单失败：{e}")
            continue

        # —— 正常对话（模型 + 工具）——
        msgs.append(HumanMessage(user_text))
        try:
            ai: AIMessage = llm_with_tools.invoke(msgs)
        except Exception as e:
            print(f"BobaBot：调用模型失败：{e}")
            continue
        msgs.append(ai)

        tool_calls = getattr(ai, "tool_calls", None) or getattr(getattr(ai, "additional_kwargs", {}), "get", lambda k: None)("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                result = handle_tool_call(tc)
                msgs.append(ToolMessage(content=json.dumps(result, ensure_ascii=False), tool_call_id=tc["id"]))
            final: AIMessage = llm_with_tools.invoke(msgs)
            for _ in range(2):
                more = getattr(final, "tool_calls", None) or getattr(getattr(final, "additional_kwargs", {}), "get", lambda k: None)("tool_calls")
                if not more:
                    break
                for tc in more:
                    result = handle_tool_call(tc)
                    msgs.append(ToolMessage(content=json.dumps(result, ensure_ascii=False), tool_call_id=tc["id"]))
                final = llm_with_tools.invoke(msgs)
            msgs.append(final)
            print("BobaBot：", final.content or "（已完成工具调用）")
        else:
            print("BobaBot：", ai.content or "（已收到）")

        print_state()

if __name__ == "__main__":
    main()
