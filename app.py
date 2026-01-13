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

import time
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

st.set_page_config(
    page_title=f"{BOT_NAME} · 奶茶店员",
    layout="centered"
)

# 👇 就放在这里
st.markdown(
    """ <style> ... </style> """,
    unsafe_allow_html=True,
)

st.title(f"🧋 {BOT_NAME}")



# ========== 会话状态 ==========
if "msgs" not in st.session_state:
    st.session_state.msgs = [SystemMessage(content=SYSTEM_PROMPT)]
if "cart" not in st.session_state:
    st.session_state.cart = []
if "pending_item" not in st.session_state:
    st.session_state.pending_item = None
if "stage" not in st.session_state:
    st.session_state.stage = "BROWSING"
if "awaiting_payment" not in st.session_state:
    st.session_state.awaiting_payment = False
if "last_order" not in st.session_state:
    st.session_state.last_order = None

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
    """Collects microphone audio frames into PCM bytes (mono, 16-bit)."""
    def __init__(self):
        self.frames = []
        self.sample_rate = 48000
        self.channels = 1

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        # ✅ 强制转换成 16-bit PCM
        try:
            pcm = frame.to_ndarray(format="s16")
        except TypeError:
            # 老版本 av 可能不支持 format 参数
            pcm = frame.to_ndarray()
            if pcm.dtype != "int16":
                pcm = pcm.astype("int16")

        # pcm 可能是:
        # 1) (channels, samples)  或 2) (samples, channels) 或 3) (samples,)
        if pcm.ndim == 2:
            ch = getattr(frame.layout, "channels", None)
            ch = len(ch) if ch else None

            # 判断哪一维是 channel
            if ch and pcm.shape[0] == ch:
                # (channels, samples)
                mono = pcm[0]
            elif ch and pcm.shape[1] == ch:
                # (samples, channels)
                mono = pcm[:, 0]
            else:
                # 猜测：取更小的那一维当 channel
                if pcm.shape[0] <= pcm.shape[1]:
                    mono = pcm[0]
                else:
                    mono = pcm[:, 0]
        else:
            mono = pcm

        # 更新采样率
        if frame.sample_rate:
            self.sample_rate = int(frame.sample_rate)

        # ✅ 单声道 int16 bytes
        self.channels = 1
        self.frames.append(mono.tobytes())
        return frame

    def to_wav_bytes(self) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # int16
            wf.setframerate(self.sample_rate)
            wf.writeframes(b"".join(self.frames))
        return buf.getvalue()

    def clear(self):
        self.frames = []



# ===== 提交订单到 FastAPI 后端 =====
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
        st.session_state.text_box = ""
        run_turn(text)

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
        st.text_input(
            " ",
            placeholder="加入购物车 → 确认订单 → 确认下单（生成待支付订单）→ 回复“已支付”获取取件码",
            key="text_box",
            label_visibility="collapsed",
            on_change=_send_text_from_box,
        )
    else:
        # ===== 美化语音输入：主按钮三步（开始→结束→发送），最长 60s =====
        MAX_REC_SECONDS = 60

        if "voice_step" not in st.session_state:
            st.session_state.voice_step = "idle"  # idle / recording / stopped
        if "voice_started_at" not in st.session_state:
            st.session_state.voice_started_at = None
        if "recorder" not in st.session_state:
            st.session_state.recorder = AudioRecorder()
        if "voice_last_wav" not in st.session_state:
            st.session_state.voice_last_wav = None  # 用于回听/发送

        desired_playing = (st.session_state.voice_step == "recording")

        # webrtc 在后台采集（控件已被 CSS 隐藏）
        ctx = webrtc_streamer(
            key="mic-recorder",
            mode=WebRtcMode.SENDONLY,
            audio_receiver_size=1024,
            media_stream_constraints={"audio": True, "video": False},
            async_processing=True,
            audio_processor_factory=AudioRecorder,
            desired_playing_state=desired_playing,  # ✅ 由我们控制开始/停止
        )

        if ctx and ctx.audio_processor:
            st.session_state.recorder = ctx.audio_processor

        # 自动停止（达到 60s 后：recording -> stopped，不自动发送）
        if st.session_state.voice_step == "recording" and st.session_state.voice_started_at:
            elapsed_sec = time.time() - st.session_state.voice_started_at
            if elapsed_sec >= MAX_REC_SECONDS:
                rec = st.session_state.get("recorder")
                if rec and getattr(rec, "frames", None):
                    st.session_state.voice_last_wav = rec.to_wav_bytes()
                st.session_state.voice_step = "stopped"
                st.toast("已达到 60s 上限，自动结束录音。点击发送。", icon="⏹")
                st.rerun()

        # 顶部：状态 + 计时
        row1, row2 = st.columns([0.68, 0.32], vertical_alignment="center")
        with row1:
            if st.session_state.voice_step == "idle":
                st.caption("🎙 点一次开始；说完点一次结束；第三次点发送。（最长 60s）")
            elif st.session_state.voice_step == "recording":
                st.caption("🔴 录音中… 最长 60s，超时自动结束。")
            else:
                st.caption("✅ 已结束：可回听/删除，点发送提交转写。")

        with row2:
            if st.session_state.voice_step == "recording" and st.session_state.voice_started_at:
                elapsed = int(time.time() - st.session_state.voice_started_at)
                if elapsed > MAX_REC_SECONDS:
                    elapsed = MAX_REC_SECONDS
                st.markdown(f"**⏱ {elapsed}s / {MAX_REC_SECONDS}s**")
            elif st.session_state.voice_step == "stopped":
                st.markdown("**⏱ 已结束**")
            else:
                st.markdown("")

        # 按钮行：主按钮 + 删除 + 回听
        b1, b2, b3 = st.columns([0.56, 0.22, 0.22], vertical_alignment="center")

        with b1:
            if st.session_state.voice_step == "idle":
                main_label = "🎙 开始"
            elif st.session_state.voice_step == "recording":
                main_label = "⏹ 结束"
            else:
                main_label = "📨 发送"
            main_click = st.button(main_label, use_container_width=True, key="voice_main_btn")

        with b2:
            del_click = st.button("🗑 删除", use_container_width=True, key="voice_delete_btn")

        with b3:
            play_click = st.button(
                "▶️ 回听",
                use_container_width=True,
                key="voice_play_btn",
                disabled=not bool(st.session_state.voice_last_wav),
            )

        # 删除：清空录音缓冲 + 清空回听
        if del_click:
            rec = st.session_state.get("recorder")
            if rec:
                rec.clear()
            st.session_state.voice_last_wav = None
            st.session_state.voice_step = "idle"
            st.session_state.voice_started_at = None
            st.toast("已删除录音", icon="🗑")
            st.rerun()

        # 回听：只在点击时显示播放器（不展示任何文件/上传）
        if play_click and st.session_state.voice_last_wav:
            st.audio(st.session_state.voice_last_wav, format="audio/wav")

        # 主按钮：三步状态机
        if main_click:
            rec = st.session_state.get("recorder")

            # 1) idle -> recording
            if st.session_state.voice_step == "idle":
                if rec:
                    rec.clear()
                st.session_state.voice_last_wav = None
                st.session_state.voice_step = "recording"
                st.session_state.voice_started_at = time.time()
                st.toast("开始录音…", icon="🎙")
                st.rerun()

            # 2) recording -> stopped（封 wav，等待发送）
            elif st.session_state.voice_step == "recording":
                if not rec or not getattr(rec, "frames", None):
                    st.warning("没有录到声音：请检查浏览器麦克风权限/输入设备后重试。")
                else:
                    wav_bytes = rec.to_wav_bytes()
                    st.session_state.voice_last_wav = wav_bytes
                    st.session_state.voice_step = "stopped"
                    st.toast("已结束录音，点击发送", icon="⏹")
                    st.rerun()

            # 3) stopped -> send（转写并 run_turn）
            else:
                wav_bytes = st.session_state.voice_last_wav
                if not wav_bytes:
                    st.warning("没有可发送的录音，请重新录制。")
                else:
                    try:
                        transcript = stt_transcribe_wav_bytes(wav_bytes, filename="recording.wav")

                        # 发送后清空
                        if rec:
                            rec.clear()
                        st.session_state.voice_last_wav = None
                        st.session_state.voice_step = "idle"
                        st.session_state.voice_started_at = None

                        if transcript:
                            run_turn(transcript)
                        else:
                            st.warning("没有识别到有效文本，请再试一次。")
                    except Exception as e:
                        st.error(f"语音识别失败：{e}")
                        # 出错时保留 voice_last_wav，方便回听/重试
