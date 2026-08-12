from typing import Dict, Any, List, Optional
import streamlit as st
from langchain_core.messages import HumanMessage

# Import core workflow and database management modules
from resilimind.core.workflow import build_workflow
from resilimind.core.database import (
    init_db, 
    register_user, 
    authenticate_user, 
    save_resilience_log, 
    get_user_resilience_history
)

# -----------------------------------------------------------------------------
# 1. Streamlit Page Configuration & Initialization
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ResiliMind | AI Resilience Platform",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize the SQLite database tables on application startup
init_db()

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
    
    /* Authentication Container Styling */
    .auth-box {
        max-width: 400px;
        margin: 0 auto;
        padding: 30px;
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        background: rgba(255,255,255,0.02);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 3. Session State & Authentication UI
# -----------------------------------------------------------------------------
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "username" not in st.session_state:
    st.session_state.username = None

# Render login/registration interface if unauthenticated
if st.session_state.user_id is None:
    st.markdown("<h2 style='text-align: center; margin-top: 50px;'>ورود به سامانه ResiliMind</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<div class='auth-box'>", unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["ورود", "ثبت‌نام"])
        
        # Login Form
        with tab1:
            login_user: str = st.text_input("نام کاربری", key="log_user")
            login_pass: str = st.text_input("رمز عبور", type="password", key="log_pass")
            
            if st.button("ورود به حساب", use_container_width=True):
                if login_user and login_pass:
                    user_id: Optional[int] = authenticate_user(login_user, login_pass)
                    if user_id:
                        st.session_state.user_id = user_id
                        st.session_state.username = login_user
                        st.success("ورود موفقیت‌آمیز بود!")
                        st.rerun()
                    else:
                        st.error("نام کاربری یا رمز عبور اشتباه است.")
                else:
                    st.warning("لطفاً همه فیلدها را پر کنید.")
                    
        # Registration Form
        with tab2:
            reg_user: str = st.text_input("نام کاربری جدید", key="reg_user")
            reg_pass: str = st.text_input("رمز عبور", type="password", key="reg_pass")
            
            if st.button("ایجاد حساب کاربری", use_container_width=True):
                if reg_user and reg_pass:
                    if register_user(reg_user, reg_pass):
                        st.success("حساب کاربری ایجاد شد. اکنون می‌توانید وارد شوید.")
                    else:
                        st.error("این نام کاربری قبلاً ثبت شده است.")
                else:
                    st.warning("لطفاً همه فیلدها را پر کنید.")
                    
        st.markdown("</div>", unsafe_allow_html=True)
        
    st.stop()


# =============================================================================
# 4. Main Application Logic & Persistent State Synchronization
# =============================================================================

@st.cache_resource
def load_graph_application() -> Any:
    """
    Builds and caches the compiled LangGraph execution graph instance.

    Returns:
        Any (CompiledStateGraph): The runnable state machine.
    """
    return build_workflow()

app: Any = load_graph_application()

# Configure thread_id specific to the authenticated user ID
config: Dict[str, Any] = {"configurable": {"thread_id": str(st.session_state.user_id)}}

# Sync and fetch existing conversation checkpoint state from SQLite database
current_graph_state: Any = app.get_state(config)

# Populate or restore session chat history from SQLite checkpoint if available
if "messages" not in st.session_state or not st.session_state.messages:
    stored_messages: List[Any] = current_graph_state.values.get("messages", [])
    if stored_messages:
        # Convert LangGraph Message objects to Streamlit chat format
        st.session_state.messages = []
        for msg in stored_messages:
            role = "user" if msg.type == "human" else "assistant"
            st.session_state.messages.append({"role": role, "content": msg.content})
    else:
        # Default welcome message for first-time users
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": f"سلام {st.session_state.username} عزیز 👋 من **ResiliMind** هستم؛ دستیار هوشمند ارزیابی و تقویت تاب‌آوری روانی.\n\nامروز چه حسی داری یا دوست داری در مورد چه موضوعی با هم گفتگو کنیم؟"
            }
        ]

# Restore last graph execution state for dashboard metrics rendering
st.session_state.last_state = current_graph_state.values if current_graph_state.values else None


# -----------------------------------------------------------------------------
# 5. Sidebar Diagnostics & Profile Dashboard
# -----------------------------------------------------------------------------
with st.sidebar:
    # User Profile Section
    st.markdown("### 👤 پروفایل کاربری")
    st.markdown(f"کاربر فعلی: **{st.session_state.username}**")
    
    # Logout action handler
    if st.button("🚪 خروج از حساب", use_container_width=True):
        st.session_state.user_id = None
        st.session_state.username = None
        st.session_state.messages = []
        st.session_state.last_state = None
        st.rerun()
        
    st.divider()

    # Sidebar Tabs for Real-Time Analysis vs. Historical Logs
    sidebar_tab1, sidebar_tab2 = st.tabs(["📊 تحلیل نشست جاری", "📜 سوابق تاب‌آوری"])

    # Tab 1: Current Session State
    with sidebar_tab1:
        if st.session_state.last_state:
            state: Dict[str, Any] = st.session_state.last_state

            # Display Extracted Active Graph Nodes
            active_nodes: List[str] = state.get("active_nodes", [])
            st.markdown("**نودهای فعال:**")
            if active_nodes:
                chips_html: str = "".join([f"<span class='node-chip'>{node}</span>" for node in active_nodes])
                st.markdown(f"<div style='margin-bottom: 12px;'>{chips_html}</div>", unsafe_allow_html=True)
            else:
                st.caption("سیگنال مستقیمی در آخرین پیام شناسایی نشد.")

            st.markdown("---")

            # Display Current Assessment Statuses
            assessments: List[Dict[str, Any]] = state.get("assessments", [])
            st.markdown("**ارزیابی نشست:**")
            if assessments:
                for item in assessments:
                    node_id: str = item.get("node_id", "N/A")
                    status: str = item.get("status", "YELLOW").upper()
                    confidence: int = int(item.get("confidence", 0.0) * 100)
                    reasoning: str = item.get("reasoning", "")

                    badge_class: str = f"badge-{status.lower()}"
                    
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
        else:
            st.info("با ارسال پیام، جزئیات نشست در این بخش قرار می‌گیرد.")

    # Tab 2: Historical Resilience Log Profile from Database
    with sidebar_tab2:
        st.markdown("**تاریخچه ارزیابی‌های اخیر:**")
        history_logs: List[Dict[str, Any]] = get_user_resilience_history(st.session_state.user_id, limit=15)
        
        if history_logs:
            for log in history_logs:
                node_id: str = log.get("node_id", "N/A")
                status: str = log.get("status", "YELLOW").upper()
                created_at: str = str(log.get("created_at", ""))[:16]
                badge_class: str = f"badge-{status.lower()}"
                
                st.markdown(f"""
                <div class="metric-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: 700; font-family: monospace; direction: ltr;">{node_id}</span>
                        <span class="{badge_class}">{status}</span>
                    </div>
                    <div style="font-size: 0.75rem; color: #64748b; margin-top: 4px; direction: ltr; text-align: left;">{created_at}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.caption("هنوز سابقه ارزیابی برای حساب شما ثبت نشده است.")


# -----------------------------------------------------------------------------
# 6. Main Interactive Chat Interface
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

    # 2. Process message through LangGraph state machine with persistent thread config
    with st.chat_message("assistant"):
        with st.spinner("در حال تحلیل پیام، واکشی از گراف دانش و ارزیابی روانی..."):

            # Construct execution initial state payload
            initial_state: Dict[str, Any] = {
                "user_id": st.session_state.user_id,
                "user_message": user_input,
                "active_nodes": [],
                "subgraph_context": "",
                "assessments": [],
                "requires_disambiguation": False,
                "final_response": "",
                "messages": [HumanMessage(content=user_input)]
            }

            # Invoke state machine pipeline with user-specific thread configuration
            final_state: Dict[str, Any] = app.invoke(initial_state, config=config)

            # Automatically persist new assessments into SQLite resilience_logs table
            new_assessments: List[Dict[str, Any]] = final_state.get("assessments", [])
            for assessment in new_assessments:
                save_resilience_log(
                    user_id=st.session_state.user_id,
                    node_id=assessment.get("node_id", ""),
                    status=assessment.get("status", "YELLOW"),
                    confidence=float(assessment.get("confidence", 0.0)),
                    reasoning=assessment.get("reasoning", "")
                )

            # Save state context for diagnostics sidebar rendering
            st.session_state.last_state = final_state

            # Extract response string generated by the terminal agent
            response_text: str = final_state.get("final_response", "پاسخی از سمت سیستم دریافت نشد.")

            # Render assistant message
            st.markdown(response_text)

            # Append to session chat history
            st.session_state.messages.append({"role": "assistant", "content": response_text})

            # Refresh Streamlit layout to instantly reflect sidebar changes
            st.rerun()
