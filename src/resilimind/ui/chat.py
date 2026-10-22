import logging
import streamlit as st
from typing import Any, Dict, List, Optional
from resilimind.services.resilience_service import process_user_message
from resilimind.core.agents.common import resilience_graph

# Initialize module logger
logger = logging.getLogger(__name__)


def get_node_display_name(node_id: str) -> str:
    """Returns the bilingual Persian and English title for a graph node if available."""
    if resilience_graph and resilience_graph.has_node(node_id):
        data = resilience_graph.nodes[node_id]
        name_fa = data.get("name_fa")
        if name_fa:
            return f"{name_fa} ({node_id})"
    return node_id


def render_crisis_banner(is_crisis: bool = False) -> None:
    """Renders the permanent research disclaimer and emergency hotline banner with dynamic crisis state."""
    if is_crisis:
        st.markdown("""
        <div class="crisis-banner-alert">
            <div style="font-weight:700; font-size:1.05rem; margin-bottom:6px;">
                🚨 هشدار فوری ایمنی و بحران روانی
            </div>
            <div style="line-height:1.7;">
                اگر شما یا یکی از اطرافیانتان در شرایط بحران شدید، افکار آسیب‌رسان یا خطر فوری قرار دارید، لطفاً فوراً با 
                <strong>اورژانس اجتماعی (۱۲۳)</strong> یا <strong>اورژانس پزشکی (۱۱۵)</strong> تماس بگیرید. 
                مشاوران تخصصی <strong>خط ملی مشاور بهزیستی (۱۴۸۰)</strong> نیز به صورت رایگان، شبانه‌روزی و محرمانه آماده پاسخگویی هستند.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="crisis-banner-safe">
            <div>
                🛡️ <strong>سامانه پژوهشی ResiliMind</strong> (ابزار کمکی و تجربی — فاقد تشخیص بالینی)
            </div>
            <div>
                📞 خطوط مشاوره و اورژانس رایگان کشور: <strong>اورژانس اجتماعی ۱۲۳</strong> | <strong>صدای مشاور بهزیستی ۱۴۸۰</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_prompt_chips() -> None:
    """Renders interactive quick conversation starter cards when conversation is fresh."""
    chips = [
        {"icon": "💼", "text": "امروز احساس فرسودگی شغلی و استرس کاری زیادی داشتم"},
        {"icon": "📉", "text": "در مواجهه با اخبار منفی و نوسانات اقتصادی استرس می‌گیرم"},
        {"icon": "🤝", "text": "می‌خواهم وضعیت روابط و مهارت‌های ارتباطی‌ام را ارزیابی کنم"},
        {"icon": "🌱", "text": "احساس می‌کنم انگیزه و پشتکارم برای اهداف روزمره افت کرده است"}
    ]

    st.markdown("""
    <div class="prompt-chips-wrapper">
        <div class="prompt-chips-title">💡 پیشنهاد موضوع برای شروع گفتگو:</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    for i, chip in enumerate(chips):
        target_col = col1 if i % 2 == 0 else col2
        with target_col:
            button_label = f"{chip['icon']} {chip['text']}"
            if st.button(button_label, key=f"quick_chip_{i}", use_container_width=True):
                st.session_state["queued_user_input"] = chip["text"]
                st.rerun()


def render_evidence_expander(signals: List[Dict[str, Any]], assessments: List[Dict[str, Any]]) -> None:
    """Renders an interactive expandable card showing detected signals and rubric breakdown."""
    if not signals and not assessments:
        return

    with st.expander("🔍 مفاهیم و شواهد شناسایی‌شده در پیام شما", expanded=False):
        if signals:
            st.markdown("##### 📡 سیگنال‌های مفهومی استخراج‌شده:")
            polarity_labels = {
                "positive": "🟢 مثبت (نقطه قوت / منبع تاب‌آوری)",
                "negative": "🔴 منفی (عامل چالش / فشار روانی)",
                "mixed": "🟡 ترکیبی (هم‌زمانی چالش و توان مقابله)"
            }

            for sig in signals:
                node_id = sig.get("node_id", "")
                node_title = get_node_display_name(node_id)
                raw_pol = sig.get("detected_signal", "mixed").lower()
                polarity_text = polarity_labels.get(raw_pol, raw_pol.upper())
                evidence = sig.get("evidence", "")
                pol_ev = sig.get("polarity_evidence") or {}

                with st.container():
                    st.markdown(f"🏷️ **مفهوم تاب‌آوری:** `{node_title}`")
                    st.markdown(f"🧭 **جهت‌گیری شناسایی‌شده:** {polarity_text}")
                    if evidence:
                        st.markdown(f'<div class="evidence-quote">💬 <strong>شاهد متنی:</strong> «{evidence}»</div>', unsafe_allow_html=True)
                    if pol_ev.get("positive"):
                        st.caption(f"✨ **شاهد مثبت:** «{pol_ev['positive']}»")
                    if pol_ev.get("negative"):
                        st.caption(f"⚠️ **شاهد منفی:** «{pol_ev['negative']}»")
                    st.markdown("<hr style='margin:8px 0; border:none; border-top:1px dashed rgba(255,255,255,0.08);'>", unsafe_allow_html=True)

        if assessments:
            st.markdown("##### 🎯 ارزیابی ابعاد چهارگانه تاب‌آوری:")
            status_labels = {
                "GREEN": "🟢 پایدار (وضعیت مطلوب)",
                "YELLOW": "🟡 هشدار (نیازمند پایش و تمرین)",
                "RED": "🔴 بحرانی (نیازمند توجه فوری و مراقبت)"
            }

            for item in assessments:
                node_id = item.get("node_id", "")
                node_title = get_node_display_name(node_id)
                raw_status = item.get("status", "YELLOW").upper()
                status_text = status_labels.get(raw_status, raw_status)
                score = item.get("score")
                confidence = int(float(item.get("confidence", 0.0)) * 100)
                reasoning = item.get("reasoning", "")
                scores = item.get("scores") or {}

                rubric_parts = []
                if "severity" in scores:
                    rubric_parts.append(f"شدت پریشانی: {scores['severity']}")
                if "frequency" in scores:
                    rubric_parts.append(f"تکرار: {scores['frequency']}")
                if "functional" in scores:
                    rubric_parts.append(f"کارکرد روزمره: {scores['functional']}")
                if "coping" in scores:
                    rubric_parts.append(f"توان مقابله: {scores['coping']}")
                rubric_summary = " | ".join(rubric_parts) if rubric_parts else ""

                with st.container():
                    st.markdown(f"🏷️ **نود:** `{node_title}`")
                    st.markdown(f"🚦 **ارزیابی تاب‌آوری:** {status_text} | **امتیاز کل:** `{score or 'N/A'}/100`")
                    if rubric_summary:
                        st.caption(f"📐 **اجزای روبیک:** {rubric_summary}")
                    st.caption(f"📊 **میزان اطمینان تحلیل هوش مصنوعی:** `{confidence}%`")
                    if reasoning:
                        st.markdown(f'<div class="evidence-quote">🧠 <strong>استدلال ارزیاب:</strong> {reasoning}</div>', unsafe_allow_html=True)


class StreamlitStreamHandler:
    """Handles real-time streaming tokens from the LLM into a designated placeholder."""

    def __init__(self, container: Any) -> None:
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str) -> None:
        self.text += token
        self.container.markdown(self.text.replace("\n", "  \n") + " ▌")


def render_chat_interface(app: Any, config: Dict[str, Any]) -> None:
    """Renders the main chat interface and handles user state updates via decoupled service results."""
    logger.debug("[UI-Chat] Rendering main chat interface...")

    # Header
    st.markdown("""
    <div class="main-header">
        <h2 style="margin:0; font-weight:700;">🌱 سامانه هوشمند مشاوره و تاب‌آوری ResiliMind</h2>
        <p style="margin:6px 0 0 0; opacity:0.85; font-size:0.95rem;">
            تحلیل چندبعدی وضعیت بر اساس شبکه گراف دانش و هوش مصنوعی
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 1. Determine crisis state for Safety & Crisis Banner
    last_state = st.session_state.get("last_state") or {}
    is_crisis = (
        last_state.get("safety_status") == "HIGH_RISK" or
        any(m.get("safety_status") == "HIGH_RISK" for m in st.session_state.get("messages", []))
    )
    render_crisis_banner(is_crisis=is_crisis)

    # 2. Render message history with evidence expanders
    messages = st.session_state.get("messages", [])
    user_message_count = sum(1 for m in messages if m.get("role") == "user")

    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"].replace("\n", "  \n"))
            if message.get("role") == "assistant":
                render_evidence_expander(
                    signals=message.get("signals", []),
                    assessments=message.get("assessments", [])
                )

    # 3. Quick Conversation Starter Chips (when no user message has been sent yet)
    if user_message_count == 0:
        render_prompt_chips()

    # 4. Input handling: check chat input or queued prompt chip
    chat_box_input = st.chat_input("پیام خود را اینجا بنویسید...")
    queued_input = st.session_state.pop("queued_user_input", None)
    user_input = queued_input or chat_box_input

    if user_input:
        logger.info("[UI-Chat] Received user input: '%s...' (length: %s)", user_input[:30], len(user_input))
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            # 1. Live Reasoning Trace using st.status
            status_box = st.status("🧠 در حال پردازش پیام و تحلیل تاب‌آوری...", expanded=True)
            status_box.write("⏳ در حال راه‌اندازی فرآیند ارزیابی هوشمند...")

            def stage_callback(node_name: str, updates: Dict[str, Any]) -> None:
                """Callback triggered as each workflow node completes execution."""
                if node_name == "safety_classifier":
                    status = updates.get("safety_status", "SAFE")
                    if status == "HIGH_RISK":
                        status_box.write("🚨 **بررسی ایمنی:** پیام دارای نشانه‌های بحران ارزیابی شد؛ ارجاع به پروتکل اورژانس.")
                        status_box.update(label="🚨 هشدار ایمنی بحرانی", state="error", expanded=True)
                    elif status == "SAFE":
                        status_box.write("✅ **بررسی ایمنی و خطرات فوری:** پیام فاقد مخاطره ارزیابی شد (امن).")
                    else:
                        status_box.write("⚠️ **بررسی ایمنی:** وضعیت ایمنی در دسترس نیست.")
                elif node_name == "extractor":
                    nodes = updates.get("active_nodes", [])
                    if nodes:
                        status_box.write(f"🔍 **استخراج سیگنال‌ها و شواهد متنی:** {len(nodes)} بعد مفهومی تاب‌آوری شناسایی و دسته‌بندی شد.")
                    else:
                        status_box.write("🔍 **استخراج سیگنال‌ها:** سیگنال مستقیمی در پیام اولیه یافت نشد.")
                elif node_name == "retriever":
                    status_box.write("🕸️ **بازیابی الگوها از گراف دانش:** نودها، محدوده‌ها و روابط علی از گراف استخراج شدند.")
                elif node_name == "assessor":
                    assessments = updates.get("assessments", [])
                    status_box.write(f"📊 **ارزیابی ابعاد چهارگانه تاب‌آوری:** روبیک‌های ۴ گانه (شدت، تکرار، کارکرد، مقابله) محاسبه شدند.")
                elif node_name == "advisor":
                    status_box.write("💬 **تدوین پاسخ مشاور:** توصیه‌های روان‌شناختی متناسب با گراف دانش و تاریخچه فرمول‌بندی شدند.")
                elif node_name == "questioner":
                    status_box.write("💬 **طرح پرسش شفاف‌ساز:** جهت شفافیت بیشتر وضعیت، پرسشی همدلانه تنظیم شد.")
                elif node_name == "emergency_response":
                    status_box.write("🚨 **پاسخ اورژانس:** راهنمای تماس فوری و مراکز بحران آماده‌سازی شد.")
                elif node_name == "service_unavailable":
                    status_box.write("⚠️ **وضعیت سرویس:** سیستم به صورت موقت امکان ارائه پاسخ کامل ندارد.")

            # 2. Response placeholder OUTSIDE the status box (in main chat flow)
            message_placeholder = st.empty()
            stream_handler = StreamlitStreamHandler(message_placeholder)

            if "configurable" not in config:
                config["configurable"] = {}
            config["configurable"]["stream_handler"] = stream_handler

            logger.debug("[UI-Chat] Invoking process_user_message service layer with live reasoning trace...")

            result = process_user_message(
                app=app,
                config=config,
                user_input=user_input,
                user_id=st.session_state.user_id,
                stage_callback=stage_callback
            )
            st.session_state.last_state = result.state

            response_text = result.final_response
            if not response_text:
                response_text = result.state.get("final_response", "")

            if not response_text:
                response_text = "⚠️ **خطا در سیستم:** فرآیند تحلیل انجام شد اما پاسخی تولید نگردید."
                logger.error("[UI-Chat] Empty response text. Graph State Dump: %s", result.state.keys())

            # Finalize status box display
            is_result_high_risk = result.state.get("safety_status") == "HIGH_RISK"
            if is_result_high_risk:
                status_box.update(label="🚨 پروتکل ایمنی و ارجاع بحران فعال شد", state="error", expanded=False)
            else:
                status_box.update(label="✨ تحلیل چندبعدی و تدوین پاسخ کامل شد", state="complete", expanded=False)

            # Render final formatted text in chat flow
            formatted_response = response_text.replace("\n", "  \n")
            message_placeholder.markdown(formatted_response)

            active_signals = result.state.get("active_signals", [])
            assessments = result.new_assessments or result.state.get("assessments", [])

            # 3. Render evidence expander directly under the assistant response (OUTSIDE status box)
            render_evidence_expander(signals=active_signals, assessments=assessments)

            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "signals": active_signals,
                "assessments": assessments,
                "safety_status": result.state.get("safety_status", "SAFE")
            })
            logger.info("[UI-Chat] Assistant streaming and evidence logging completed.")

            if is_result_high_risk:
                st.rerun()
