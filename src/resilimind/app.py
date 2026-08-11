from typing import Dict, Any, List
import streamlit as st

# Import the workflow factory function from the core layer
from resilimind.core.workflow import build_workflow

# -----------------------------------------------------------------------------
# 1. Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ResiliMind | AI Resilience Platform",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. Custom CSS Injection (RTL Support, Typography, & Modern UI Components)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Load Google Vazirmatn Font */
    @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@300;400;600;700&display=swap');

    /* Global RTL and Typography Settings */
    html, body, [class*="css"], .stChatMessage, .stTextInput, .stMarkdown {
        font-family: 'Vazirmatn', sans-serif !important;
        direction: rtl;
        text-align: right;
    }

    /* Modern Glassmorphic Dashboard Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }

    /* Node Chip Badge */
    .node-chip {
        display: inline-block;
        background-color: #1e293b;
        color: #38bdf8;
        border: 1px solid #0284c7;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.82rem;
        font-weight: 600;
        margin: 3px;
        direction: ltr;
    }

    /* Resilience Status Badges */
    .badge-green {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid #10b981;
        padding: 2px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .badge-yellow {
        background-color: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid #f59e0b;
        padding: 2px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .badge-red {
        background-color: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid #ef4444;
        padding: 2px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
    }

    /* Chat Input Fixes for RTL */
    .stChatInput textarea {
        direction: rtl;
        text-align: right;
    }

    /* Header Accent Banner */
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        border-radius: 16px;
        padding: 24px;
        color: #ffffff;
        margin-bottom: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 3. Application State & Workflow Initialization
# -----------------------------------------------------------------------------
@st.cache_resource
def load_graph_application():
    """Builds and caches the compiled LangGraph execution graph instance."""
    return build_workflow()

app = load_graph_application()

# Initialize chat history state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "سلام 👋 من **ResiliMind** هستم؛ دستیار هوشمند ارزیابی و تقویت تاب‌آوری روانی.\n\nامروز چه حسی داری یا دوست داری در مورد چه موضوعی با هم گفتگو کنیم؟"
        }
    ]

# Initialize last execution state for real-time sidebar diagnostics
if "last_state" not in st.session_state:
    st.session_state.last_state = None


# -----------------------------------------------------------------------------
# 4. Sidebar Diagnostics Dashboard
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🌱 ResiliMind Dashboard")
    st.caption("سامانه تحلیل تاب‌آوری روان‌شناختی مبتنی بر LangGraph & RAG")
    st.divider()

    st.markdown("### 📊 تحلیل زنده گراف دانش")

    if st.session_state.last_state:
        state = st.session_state.last_state

        # Display Extracted Active Graph Nodes
        active_nodes: List[str] = state.get("active_nodes", [])
        st.markdown("**نودهای فعال استخراج‌شده:**")
        if active_nodes:
            chips_html = "".join([f"<span class='node-chip'>{node}</span>" for node in active_nodes])
            st.markdown(f"<div style='margin-bottom: 12px;'>{chips_html}</div>", unsafe_allow_html=True)
        else:
            st.caption("سیگنال مستقیمی در آخرین پیام شناسایی نشد.")

        st.markdown("---")

        # Display Evaluated Node Statuses
        assessments: List[Dict[str, Any]] = state.get("assessments", [])
        st.markdown("**وضعیت تاب‌آوری ارزیابی‌شده:**")
        if assessments:
            for item in assessments:
                node_id = item.get("node_id", "N/A")
                status = item.get("status", "YELLOW").upper()
                confidence = int(item.get("confidence", 0.0) * 100)
                reasoning = item.get("reasoning", "")

                badge_class = f"badge-{status.lower()}"
                
                st.markdown(f"""
                <div class="metric-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span style="font-weight: 700; font-family: monospace; direction: ltr;">{node_id}</span>
                        <span class="{badge_class}">{status}</span>
                    </div>
                    <div style="font-size: 0.8rem; color: #94a3b8;">درصد اطمینان: {confidence}%</div>
                    <div style="font-size: 0.78rem; color: #cbd5e1; margin-top: 4px;">{reasoning}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.caption("ارزیابی جدیدی ثبت نشده است.")

        st.markdown("---")

        # Display Dynamic Agent Routing Status
        requires_disambiguation = state.get("requires_disambiguation", False)
        st.markdown("**مسیر اجرای ایجنت‌ها (Router):**")
        if requires_disambiguation:
            st.warning("⚠️ **دستور اجرا:** هدایت به ایجنت Questioner (نیازمند شفاف‌سازی)")
        else:
            st.success("✅ **دستور اجرا:** هدایت به ایجنت Advisor (ارائه راهکار درمانی)")

    else:
        st.info("با ارسال اولین پیام، لایه‌های پردازشی گراف و وضعیت ارزیابی در این پنل قرار می‌گیرند.")

    st.divider()

    # Reset Session Action
    if st.button("🔄 شروع گفتگو جدید", use_container_width=True):
        st.session_state.messages = [st.session_state.messages[0]]
        st.session_state.last_state = None
        st.rerun()


# -----------------------------------------------------------------------------
# 5. Main Interactive Chat Interface
# -----------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h2 style="margin:0; font-weight:700;">🌱 سامانه هوشمند مشاوره و تاب‌آوری ResiliMind</h2>
    <p style="margin:6px 0 0 0; opacity:0.85; font-size:0.95rem;">
        تحلیل چندبعدی وضعیت روانی بر اساس شبکه گراف دانش و هوش مصنوعی
    </p>
</div>
""", unsafe_allow_html=True)

# Render Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Chat Input Prompt
if user_input := st.chat_input("پیام خود را اینجا بنویسید..."):

    # 1. Display User Message immediately
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Process message through LangGraph state machine
    with st.chat_message("assistant"):
        with st.spinner("در حال تحلیل پیام، واکشی از گراف دانش و ارزیابی روانی..."):

            # Construct execution initial state payload
            initial_state: Dict[str, Any] = {
                "user_message": user_input,
                "active_nodes": [],
                "subgraph_context": "",
                "assessments": [],
                "requires_disambiguation": False,
                "final_response": ""
            }

            # Invoke state machine pipeline
            final_state = app.invoke(initial_state)

            # Save state context for diagnostics sidebar rendering
            st.session_state.last_state = final_state

            # Extract response string generated by the terminal agent
            response_text = final_state.get("final_response", "پاسخی از سمت سیستم دریافت نشد.")

            # Render assistant message
            st.markdown(response_text)

            # Append to session chat history
            st.session_state.messages.append({"role": "assistant", "content": response_text})

            # Refresh Streamlit layout to instantly reflect sidebar changes
            st.rerun()
