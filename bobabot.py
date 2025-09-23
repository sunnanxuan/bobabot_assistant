# -*- coding: utf-8 -*-
"""
bobabot.py — 命令行版“智能奶茶店员”
依赖：
  pip install langchain langchain-openai python-dotenv requests tiktoken

环境变量（.env）：
  OPENAI_API_KEY=sk-xxxx
  FT_MODEL=ft:gpt-4o-mini-2024-07-18:personal::YOUR_ID
  MODEL_TEMP=0.3
  BOT_NAME=BobaBot
  # 可选：启用“确认点单”提交到后端
  BACKEND_URL=http://localhost:8000
  BACKEND_TOKEN=change-this-to-a-long-random-secret

用法：
  python bobabot.py
  输入 /exit 退出；输入 /help 查看命令。
"""

import os
import json
import uuid
import requests
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from prompts import get_system_prompt
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

# ====== 订单/购物车状态（命令行也支持） ======
msgs = [SystemMessage(content=SYSTEM_PROMPT)]
cart = []             # 已确认加入订单的杯子
pending_item = None   # 最近一次报价（待确认未入 cart）

def submit_order_to_backend(items: list) -> dict:
    """将购物车提交到 FastAPI 后端，返回 {pickup_code,...}"""
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
        "channel": "cli",
        "client_order_id": str(uuid.uuid4()),  # 幂等键
    }
    r = requests.post(
        f"{BACKEND_URL}/orders",
        headers={"Authorization": f"Bearer {BACKEND_TOKEN}"},
        json=payload,
        timeout=15,
    )
    r.raise_for_status()
    return r.json()

def print_cart():
    print("\n🛒 当前订单：")
    if pending_item:
        print(f"  待确认：{pending_item['drink']} {pending_item['size']} - ¥{pending_item['total']}")
    if cart:
        total = 0
        for i, it in enumerate(cart, 1):
            print(f"  {i}. {it['drink']} {it['size']} - ¥{it['total']}")
            total += it["total"]
        print(f"  合计：¥{total}")
    elif not pending_item:
        print("  （空）")
    print("")

def handle_tool_call(tc):
    """执行工具调用并根据结果更新购物车状态"""
    global pending_item, cart
    name, args = tc["name"], (tc.get("args") or {})

    if name == "quote_price":
        result = quote_price.invoke(args)
        if result.get("ok"):
            # 把上一杯挂起项先放入购物车，当前报价成为新的待确认
            if pending_item:
                cart.append(pending_item)
            pending_item = result
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
              /help        显示帮助
              /cart        查看当前订单
              /clear       清空购物车（不影响待确认项）
              /drop        丢弃当前待确认项
              /confirm     等同“确认点单”，将订单提交后端
              /exit        退出
            说明：
              - 正常输入自然语言即可点单/询价/推荐。
              - 模型每次报价后会把那杯作为“待确认”；再次点单或确认时会自动入购物车。
              - “/confirm” 会把购物车 + 待确认项一并提交到后端生成取件码。
            """)

def main():
    global pending_item, cart
    print(f"{BOT_NAME}（模型：{FT_MODEL}）已启动。输入 /help 查看命令，/exit 退出。\n")

    while True:
        try:
            user_text = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_text:
            continue

        # ======= 命令处理 =======
        if user_text.lower() in {"/exit", "exit", "quit"}:
            print("再见！")
            break

        if user_text.lower() == "/help":
            help_text()
            continue

        if user_text.lower() == "/cart":
            print_cart()
            continue

        if user_text.lower() == "/clear":
            cart.clear()
            print("已清空购物车。")
            print_cart()
            continue

        if user_text.lower() == "/drop":
            pending_item = None
            print("已丢弃当前待确认项。")
            print_cart()
            continue

        if user_text.lower() in {"/confirm"} or user_text in {"确认点单", "确认下单", "确认点餐"}:
            items = list(cart)
            if pending_item:
                items.append(pending_item)
                pending_item = None
            if not items:
                print("BobaBot：当前没有已确认的饮品，请先点单哦。")
                continue
            try:
                resp = submit_order_to_backend(items)
                cart.clear()
                print(f"BobaBot：✅ 下单成功！取件码：{resp['pickup_code']}，合计：¥{resp['total']}。")
            except Exception as e:
                print(f"BobaBot：下单失败：{e}")
            continue

        # ======= 正常对话（模型 + 工具） =======
        msgs.append(HumanMessage(user_text))
        ai: AIMessage = llm_with_tools.invoke(msgs)
        msgs.append(ai)

        # 尝试获取工具调用
        tool_calls = getattr(ai, "tool_calls", None)
        if not tool_calls and hasattr(ai, "additional_kwargs"):
            tool_calls = ai.additional_kwargs.get("tool_calls")

        if tool_calls:
            for tc in tool_calls:
                result = handle_tool_call(tc)
                msgs.append(ToolMessage(content=json.dumps(result, ensure_ascii=False), tool_call_id=tc["id"]))
            final: AIMessage = llm_with_tools.invoke(msgs)
            msgs.append(final)
            print("BobaBot：", final.content)
        else:
            print("BobaBot：", ai.content)

        # 每轮结束显示简易购物车摘要（可注释掉）
        print_cart()

if __name__ == "__main__":
    main()
