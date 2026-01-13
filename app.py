import os
import re
import json
import uuid
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from prompts import *
from tools import TOOLS, quote_price, list_menu, can_make_hot
from menu_config import PRICES, CATEGORY, EXTRAS

from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import io
import wave

# ========== 读取 .env ==========
from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

FT_MODEL = os.getenv("FT_MODEL")
MODEL_TEMP = float(os.getenv("TEMPERATURE"))
BOT_NAME = os.getenv("BOT_NAME", "BobaBot")
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
BACKEND_TOKEN = os.getenv("BACKEND_TOKEN", "devtoken")

# ========== 模型与提示词 ==========
SYSTEM_PROMPT = get_system_prompt(BOT_NAME)
llm = ChatOpenAI(model=FT_MODEL, temperature=MODEL_TEMP)
llm_with_tools = llm.bind_tools(TOOLS)

# ========== Streamlit UI ==========
st.set_page_config(page_title=f"{BOT_NAME} · 奶茶店员", layout="centered")
st.title(f"🧋 {BOT_NAME} ")

# ========== 会话状态 ==========
if "msgs" not in st.session_state:
    st.session_state.msgs = [SystemMessage(content=SYSTEM_PROMPT)]
if "cart" not in st.session_state:
    st.session_state.cart = []               # 多杯购物车
if "pending_item" not in st.session_state:
    st.session_state.pending_item = None     # 最近一次报价（待加入购物车）
if "stage" not in st.session_state:
    st.session_state.stage = "BROWSING"      # BROWSING / AWAIT_CONFIRM_ORDER / AWAIT_CONFIRM_SUBMIT
# 支付相关
if "awaiting_payment" not in st.session_state:
    st.session_state.awaiting_payment = False
if "last_order" not in st.session_state:
    st.session_state.last_order = None       # {order_id, pickup_code, total, created_at, status}

# 输入模式：text / voice（默认 text）
if "input_mode" not in st.session_state:
    st.session_state.input_mode = "text"

# ========== 历史消息渲染 ==========
for m in st.session_state.msgs:
    if isinstance(m, HumanMessage):
        st.chat_message("user").write(m.content)
    elif isinstance(m, AIMessage):
        st.chat_message("assistant").write(m.content)

# ========== 工具函数 ==========
def cart_total():
    return sum(x.get("total", 0) for x in st.session_state.cart)

def cart_summary_text():
    if not st.session_state.cart:
        return "购物车为空。"
    lines = []
    for i, it in enumerate(st.session_state.cart, 1):
        ex = it.get("extras") or []
        ex_txt = "、".join(e["name"] for e in ex) if ex else "无"
        lines.append(
            f"{i}) {it['drink']} {it['size']} {it.get('sugar') or ''} {it.get('ice') or ''} "
            f"小料：{ex_txt}  小计：¥{it.get('total',0)}"
        )
    return "订单摘要：\n" + "\n".join(lines) + f"\n合计：¥{cart_total()}"

def remove_item_by_text(text: str) -> bool:
    m = re.search(r"(删除|去掉)第(\d+)杯", text)
    if not m:
        return False
    idx = int(m.group(2)) - 1
    if 0 <= idx < len(st.session_state.cart):
        st.session_state.cart.pop(idx)
        return True
    return False


class AudioRecorder:
    """Collects microphone audio frames into PCM bytes."""
    def __init__(self):
        self.frames = []
        self.sample_rate = 48000
        self.channels = 1

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        pcm = frame.to_ndarray()
        if pcm.ndim == 2:
            pcm = pcm[0]
        if pcm.dtype != "int16":
            pcm = pcm.astype("int16")
        if frame.sample_rate:
            self.sample_rate = int(frame.sample_rate)
        self.frames.append(pcm.tobytes())
        return frame

    def to_wav_bytes(self) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # int16 => 2 bytes
            wf.setframerate(self.sample_rate)
            wf.writeframes(b"".join(self.frames))
        return buf.getvalue()

    def clear(self):
        self.frames = []


# ===== 提交订单到 FastAPI 后端（一个订单一个取件码，金额后端重算） =====
def submit_order_to_backend(items: list):
    payload = {
        "items": [
            {
                "drink": x["drink"],
                "size": x["size"],
                "hot": bool(x.get("hot")),
                "sugar": x.get("sugar"),
                "ice": x.get("ice"),
                "extras": [e["name"] for e in x.get("extras", [])],
            }
            for x in items
        ],
        "channel": "streamlit",
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

def pay_order(order_id: str):
    r = requests.post(
        f"{BACKEND_URL}/orders/{order_id}/pay",
        headers={"Authorization": f"Bearer {BACKEND_TOKEN}"},
        timeout=15,
    )
    r.raise_for_status()
    return r.json()

# ====== 侧栏：购物车 + 规则 + 菜单 + 小料 ======
def _fmt_sizes(sizes: dict) -> str:
    sm = sizes.get("小杯"); md = sizes.get("中杯"); lg = sizes.get("大杯")
    return f"小¥{sm}/中¥{md}/大¥{lg}"

with st.sidebar:
    st.subheader("🛒 购物车")
    if st.session_state.stage == "BROWSING" and st.session_state.pending_item:
        pi = st.session_state.pending_item
        st.write(f"待加入：{pi['drink']} {pi['size']}  小计：¥{pi['total']}")
        ex = pi.get("extras") or []
        if ex:
            st.caption("小料：" + "、".join(e["name"] for e in ex))

    if st.session_state.cart:
        for i, it in enumerate(st.session_state.cart, 1):
            st.write(f"{i}. {it['drink']} {it['size']} - ¥{it.get('total', 0)}")

    st.write("---")
    st.write("合计：¥", cart_total())
    st.caption("指令：加入购物车 / 删除第N杯 / 去结算 / 确认订单 / 确认下单 / 已支付 / 清空")

    with st.expander("📋 点单规则（必读）", expanded=False):
        st.markdown(
            """
1. 逐杯确认：饮品 → 小料 → 杯型/甜度/冰量（果茶仅冷饮）。
2. 核对订单摘要，先回复【确认订单】。
3. 再回复【确认下单】生成待支付订单；支付后回复【已支付】获取取件码。
小贴士：可用“删除第N杯”“清空购物车”快速修改。
"""
        )

    with st.expander("🍵 全菜单（按类别）", expanded=False):
        cats = ["奶茶类", "纯茶类", "茶拿铁类", "果茶类"]
        for cat in cats:
            names = [name for name, c in CATEGORY.items() if c == cat]
            if not names:
                continue
            st.markdown(f"**{cat}**")
            lines = []
            for name in names:
                sizes = PRICES.get(name, {})
                lines.append(f"- {name}：{_fmt_sizes(sizes)}")
            st.markdown("\n".join(lines))
        st.caption("说明：果茶类仅提供冷饮；其余类别可冷热。")

    with st.expander("➕ 小料与加价", expanded=False):
        extras_lines = [f"- {k}：+¥{v}" for k, v in sorted(EXTRAS.items(), key=lambda x: (x[1], x[0]))]
        st.markdown("\n".join(extras_lines))
        st.caption("温馨提示：部分饮品自带配料已在菜单中体现，额外加料按上表加价。")

    st.write("---")
    st.subheader("🔊 语音播报")

    st.session_state.voice_output = st.toggle(
        "启用语音播报（TTS）",
        value=st.session_state.get("voice_output", True),
    )

    st.session_state.tts_voice = st.selectbox(
        "TTS voice",
        ["alloy", "coral", "nova", "onyx", "sage", "shimmer", "verse",
         "ash", "fable", "echo", "ballad", "marin", "cedar"],
        index=0,
    )
    st.caption("提示：语音为 AI 生成（TTS）。")

# ===== 处理工具调用 =====
def handle_tool_call(tc):
    name, args = tc["name"], (tc.get("args") or {})
    if name == "quote_price":
        result = quote_price.invoke(args)
        if result.get("ok"):
            if st.session_state.pending_item:
                st.session_state.cart.append(st.session_state.pending_item)
            st.session_state.pending_item = result
            st.session_state.stage = "BROWSING"
        return result
    elif name == "list_menu":
        return list_menu.invoke(args)
    elif name == "can_make_hot":
        return can_make_hot.invoke(args)
    else:
        return {"ok": False, "error": f"未知工具：{name}"}


def stt_transcribe_wav_bytes(wav_bytes: bytes, filename: str = "recording.wav") -> str:
    files = {"file": (filename, wav_bytes, "audio/wav")}
    url = f"{BACKEND_URL.rstrip('/')}/voice/stt"
    r = requests.post(
        url,
        headers={"Authorization": f"Bearer {BACKEND_TOKEN}"},
        files=files,
        timeout=60,
    )
    r.raise_for_status()
    return (r.json().get("text") or "").strip()


def tts_speak(text: str):
    if not st.session_state.get("voice_output"):
        return
    if not text:
        return
    if len(text) > 600:
        text = text[:600]

    r = requests.post(
        f"{BACKEND_URL}/voice/tts",
        headers={"Authorization": f"Bearer {BACKEND_TOKEN}"},
        json={"text": text, "voice": st.session_state.get("tts_voice", "alloy")},
        timeout=60,
    )
    r.raise_for_status()
    st.audio(r.content, format="audio/mp3")


# ===== 单轮执行 =====
def run_turn(user_text: str):
    text = user_text.strip()

    # —— 命令优先 —— #
    if text in {"加入购物车", "加到购物车", "确认本杯", "添加本杯"}:
        if not st.session_state.pending_item:
            st.chat_message("assistant").write("当前没有待加入的饮品，请先选择并报价。")
            return
        st.session_state.cart.append(st.session_state.pending_item)
        st.session_state.pending_item = None
        if st.session_state.stage in {"AWAIT_CONFIRM_ORDER", "AWAIT_CONFIRM_SUBMIT"}:
            st.session_state.stage = "AWAIT_CONFIRM_ORDER"
            st.chat_message("assistant").write(
                cart_summary_text() + " 如需提交请回复“确认订单”，或继续“删除第N杯/加入购物车”。"
            )
        else:
            st.chat_message("assistant").write("已加入购物车。还要再加一杯吗？若结算请输入“去结算”。")
        return

    if text in {"清空", "清空购物车"}:
        st.session_state.cart.clear()
        st.chat_message("assistant").write("已清空购物车。")
        return

    if remove_item_by_text(text):
        if st.session_state.stage in {"AWAIT_CONFIRM_ORDER", "AWAIT_CONFIRM_SUBMIT"}:
            st.session_state.stage = "AWAIT_CONFIRM_ORDER"
            st.chat_message("assistant").write(
                cart_summary_text() + " 如需提交请回复“确认订单”，或继续“删除第N杯/加入购物车”。"
            )
        else:
            st.chat_message("assistant").write("已删除指定杯。")
        return

    if text in {"去结算", "结算"}:
        if st.session_state.pending_item:
            st.session_state.cart.append(st.session_state.pending_item)
            st.session_state.pending_item = None
        if not st.session_state.cart:
            st.chat_message("assistant").write("购物车为空，请先加入至少一杯。")
            return
        st.session_state.stage = "AWAIT_CONFIRM_ORDER"
        st.chat_message("assistant").write(
            cart_summary_text() + " 请确认信息，回复【确认订单】或输入“删除第N杯”。"
        )
        return

    if text == "确认订单":
        if st.session_state.pending_item:
            st.session_state.cart.append(st.session_state.pending_item)
            st.session_state.pending_item = None

        if not st.session_state.cart:
            msg = "购物车为空，请先选择饮品并报价。"
            st.session_state.msgs.append(AIMessage(content=msg))
            st.chat_message("assistant").write(msg)
            return

        st.session_state.stage = "AWAIT_CONFIRM_SUBMIT"
        msg = (
            cart_summary_text()
            + " 已确认订单。请回复【确认下单】以生成待支付订单，或输入“删除第N杯”继续修改。"
        )
        st.session_state.msgs.append(AIMessage(content=msg))
        st.chat_message("assistant").write(msg)
        st.rerun()
        return

    if text in {"确认下单", "确认点单", "确认点餐"}:
        if st.session_state.stage != "AWAIT_CONFIRM_SUBMIT":
            st.chat_message("assistant").write("尚未确认订单，请先回复“确认订单”。")
            return
        try:
            if st.session_state.pending_item:
                st.session_state.cart.append(st.session_state.pending_item)
                st.session_state.pending_item = None
            if not st.session_state.cart:
                st.chat_message("assistant").write("购物车为空，请先加入至少一杯。")
                return

            resp = submit_order_to_backend(st.session_state.cart)
            st.session_state.cart.clear()
            st.session_state.stage = "BROWSING"

            st.session_state.last_order = resp
            st.session_state.awaiting_payment = True

            st.chat_message("assistant").write(
                f"🧾 订单已生成（金额：¥{resp['total']}，状态：待支付）。"
                "请完成支付后回复“已支付”，我再为您显示取件码。"
            )
        except Exception as e:
            st.chat_message("assistant").write(f"下单失败：{e}")
        return

    if text in {"已支付", "完成支付", "支付完成"}:
        if not st.session_state.awaiting_payment or not st.session_state.last_order:
            st.chat_message("assistant").write("当前没有待支付的订单。若已下单，请先“去结算→确认订单→确认下单”。")
            return
        try:
            paid = pay_order(st.session_state.last_order["order_id"])
            st.session_state.awaiting_payment = False
            st.session_state.last_order = None
            st.chat_message("assistant").write(f"✅ 支付成功！取件码：{paid['pickup_code']}。祝您用餐愉快～")
        except Exception as e:
            st.chat_message("assistant").write(f"支付状态更新失败：{e}")
        return

    # —— 普通对话（模型 + 工具）——
    st.session_state.msgs.append(HumanMessage(text))
    st.chat_message("user").write(text)

    try:
        ai: AIMessage = llm_with_tools.invoke(st.session_state.msgs)
    except Exception as e:
        st.error(f"调用模型失败：{e}")
        return
    st.session_state.msgs.append(ai)

    tool_calls = getattr(ai, "tool_calls", None)
    if not tool_calls and hasattr(ai, "additional_kwargs"):
        tool_calls = ai.additional_kwargs.get("tool_calls")

    if tool_calls:
        for tc in tool_calls:
            result = handle_tool_call(tc)
            st.session_state.msgs.append(
                ToolMessage(content=json.dumps(result, ensure_ascii=False), tool_call_id=tc["id"])
            )

        final: AIMessage = llm_with_tools.invoke(st.session_state.msgs)
        for _ in range(2):
            more = getattr(final, "tool_calls", None)
            if not more and hasattr(final, "additional_kwargs"):
                more = final.additional_kwargs.get("tool_calls")
            if not more:
                break
            for tc in more:
                result = handle_tool_call(tc)
                st.session_state.msgs.append(
                    ToolMessage(content=json.dumps(result, ensure_ascii=False), tool_call_id=tc["id"])
                )
            final = llm_with_tools.invoke(st.session_state.msgs)

        st.session_state.msgs.append(final)
        reply = final.content or "（已完成工具调用）"
        st.chat_message("assistant").write(reply)
        tts_speak(reply)
    else:
        reply = ai.content or "（已收到）"
        st.chat_message("assistant").write(reply)
        tts_speak(reply)


# =========================
# 输入区：默认文字 + 旁边麦克风按钮；切换后显示语音 + 旁边键盘按钮
# =========================

def _send_text_from_box():
    text = (st.session_state.get("text_box") or "").strip()
    if text:
        st.session_state.text_box = ""  # 清空
        run_turn(text)

# 用 columns 模拟“输入框旁边一个按钮”的布局
col_main, col_btn = st.columns([0.86, 0.14], vertical_alignment="bottom")

with col_btn:
    if st.session_state.input_mode == "text":
        if st.button("🎙", key="switch_to_voice", help="切换到语音输入"):
            st.session_state.input_mode = "voice"
            st.rerun()
    else:
        if st.button("⌨️", key="switch_to_text", help="切换到文字输入"):
            st.session_state.input_mode = "text"
            st.rerun()

with col_main:
    if st.session_state.input_mode == "text":
        # 说明：st.chat_input 无法真正放进 columns 旁边放按钮（Streamlit 限制），
        # 所以这里用 text_input + 回车发送（on_change）来实现“旁边一个按钮”的 UI。
        st.text_input(
            " ",
            placeholder="加入购物车 → 确认订单 → 确认下单（生成待支付订单）→ 回复“已支付”获取取件码",
            key="text_box",
            label_visibility="collapsed",
            on_change=_send_text_from_box,
        )
        # 备用：也可加一个显式发送按钮（可删）
        # if st.button("发送", key="send_text_btn"):
        #     _send_text_from_box()

    else:
        st.caption("🎙 点击 Start 开始说话；说完点击 Stop；然后点 Send 发送。")

        ctx = webrtc_streamer(
            key="mic-recorder",
            mode=WebRtcMode.SENDONLY,
            audio_receiver_size=1024,
            media_stream_constraints={"audio": True, "video": False},
            async_processing=True,
            audio_processor_factory=AudioRecorder,
        )
        if ctx and ctx.audio_processor:
            st.session_state.recorder = ctx.audio_processor

        c1, c2 = st.columns(2)
        with c1:
            if st.button("🧹 清空", key="voice_clear"):
                if ctx and ctx.audio_processor:
                    ctx.audio_processor.clear()
                st.toast("已清空录音", icon="🧹")

        with c2:
            send = st.button("📨 Send", key="voice_send_recording")

        if send:
            rec = st.session_state.get("recorder")
            if not rec or not getattr(rec, "frames", None):
                st.warning("还没有录到声音：先点 Start 说话，再点 Stop。")
            else:
                try:
                    wav_bytes = rec.to_wav_bytes()
                    rec.clear()
                    transcript = stt_transcribe_wav_bytes(wav_bytes)
                    if transcript:
                        run_turn(transcript)
                    else:
                        st.warning("没有识别到有效文本，请再试一次。")
                except Exception as e:
                    st.error(f"语音识别失败：{e}")

