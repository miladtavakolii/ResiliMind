from typing import Dict, Any, List, Optional
import streamlit as st
from langchain_core.messages import HumanMessage
import pandas as pd
import plotly.express as px

# Import core workflow and database management modules
from resilimind.core.workflow import build_workflow
from resilimind.core.database import (
    init_db, 
    register_user, 
    authenticate_user, 
    save_resilience_log, 
    get_user_resilience_history,
    get_user_latest_node_statuses
)

def render_domain_resilience_chart(user_id: int) -> None:
    """
    Renders an interactive donut chart showing resilience score distribution across the 6 core domains.
    Translates English backend domains to Persian UI labels.
    """
    logs = get_user_latest_node_statuses(user_id)
    if not logs:
        st.caption("هنوز ارزیابی کافی برای محاسبه نمودار ثبت نشده است.")
        return

    df = pd.DataFrame(logs)
    
    if 'category' not in df.columns or 'score' not in df.columns:
        st.caption("لطفاً پیام جدیدی ارسال کنید تا داده‌های ساختاریافته در دیتابیس قرار گیرند.")
        return

    domain_translation = {
        "Personal_Resilience": "فردی",
        "Political_Resilience": "سیاسی",
        "Economic_Resilience": "اقتصادی",
        "Physical_Resilience": "جسمانی",
        "Social_Resilience": "اجتماعی",
        "Spiritual_Cultural_Resilience": "معنوی"
    }

    df['category_fa'] = df['category'].map(domain_translation)

    all_categories_fa = ["فردی", "سیاسی", "اقتصادی", "جسمانی", "اجتماعی", "معنوی"]
    
    category_avg = df.groupby('category_fa')['score'].mean().reindex(all_categories_fa, fill_value=0).reset_index()
    category_avg.columns = ['دسته', 'امتیاز']

    custom_colors = {
        "فردی": "#38bdf8",     
        "سیاسی": "#a855f7",    
        "اقتصادی": "#f59e0b",  
        "جسمانی": "#10b981",   
        "اجتماعی": "#ec4899",  
        "معنوی": "#6366f1"     
    }

    fig = px.pie(
        category_avg,
        values='امتیاز',
        names='دسته',
        hole=0.55,
        color='دسته',
        color_discrete_map=custom_colors
    )

    fig.update_layout(
        showlegend=True,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=10, b=10, l=10, r=10),
        height=280,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5,
            font=dict(color="#cbd5e1", size=12)
        )
    )

    fig.update_traces(
        textposition='inside',
        textinfo='label+percent',
        hoverinfo='label+value'
    )

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

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
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 16px 20px;
        margin-bottom: 14px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        transition: all 0.2s ease-in-out;
    }
    .metric-card:hover {
        background: rgba(255, 255, 255, 0.05);
        border-color: rgba(255, 255, 255, 0.15);
    }

    /* Node Chip Badge */
    .node-chip {
        display: inline-block;
        background-color: rgba(14, 165, 233, 0.1);
        color: #38bdf8;
        border: 1px solid rgba(14, 165, 233, 0.3);
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 4px;
        direction: ltr;
    }

    /* Resilience Status Badges */
    .badge-green {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 4px 12px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .badge-yellow {
        background-color: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.3);
        padding: 4px 12px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .badge-red {
        background-color: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
        padding: 4px 12px;
        border-radius: 8px;
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
        border-radius: 18px;
        padding: 28px;
        color: #ffffff;
        margin-bottom: 30px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
    }

       
    /* Styled Input Fields */
    .stTextInput > div > div > input {
        border-radius: 10px !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        background-color: rgba(255,255,255,0.03) !important;
        padding: 12px 16px !important;
        transition: all 0.3s ease !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 0 1px #38bdf8 !important;
        background-color: rgba(255,255,255,0.06) !important;
    }

    /* Styled Buttons */
    .stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 8px 16px !important;
        background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%) !important;
        border: none !important;
        color: white !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(2, 132, 199, 0.4) !important;
    }

    /* Styled Tabs (for Login and Sidebar) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(255, 255, 255, 0.02);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px !important;
        padding: 10px 20px !important;
        background-color: transparent;
        border: none !important;
        font-weight: 600 !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(255, 255, 255, 0.08) !important;
        color: #38bdf8 !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
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
    # Add top margin for vertical centering
    st.markdown("<div style='margin-top: 10vh;'></div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        # Beautiful Header Section
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2.5rem;">
            <div style="font-size: 3.5rem; margin-bottom: 0.5rem; animation: float 3s ease-in-out infinite;">🌱</div>
            <h1 style="font-weight: 700; color: #f8fafc; font-size: 2.2rem; margin-bottom: 0.5rem;">ResiliMind</h1>
            <p style="color: #94a3b8; font-size: 1.1rem; margin-top: 0;">پلتفرم هوشمند مشاوره و ارزیابی تاب‌آوری</p>
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["🔑 ورود به حساب", "📝 ثبت‌نام جدید"])
        
        # Login Form
        with tab1:
            st.markdown("<br>", unsafe_allow_html=True)
            login_user: str = st.text_input("👤 نام کاربری", key="log_user")
            login_pass: str = st.text_input("🔒 رمز عبور", type="password", key="log_pass")
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button("ورود به سامانه", use_container_width=True):
                if login_user and login_pass:
                    user_id: Optional[int] = authenticate_user(login_user, login_pass)
                    if user_id:
                        st.session_state.user_id = user_id
                        st.session_state.username = login_user
                        st.success("ورود موفقیت‌آمیز بود! در حال انتقال...")
                        st.rerun()
                    else:
                        st.error("نام کاربری یا رمز عبور اشتباه است.")
                else:
                    st.warning("لطفاً همه فیلدها را پر کنید.")
                    
        # Registration Form
        with tab2:
            st.markdown("<br>", unsafe_allow_html=True)
            reg_user: str = st.text_input("👤 نام کاربری دلخواه", key="reg_user")
            reg_pass: str = st.text_input("🔒 رمز عبور", type="password", key="reg_pass")
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button("ایجاد حساب کاربری", use_container_width=True):
                if reg_user and reg_pass:
                    if register_user(reg_user, reg_pass):
                        st.success("حساب کاربری با موفقیت ایجاد شد! اکنون می‌توانید از تب ورود استفاده کنید.")
                    else:
                        st.error("این نام کاربری قبلاً ثبت شده است. لطفاً نام دیگری انتخاب کنید.")
                else:
                    st.warning("لطفاً همه فیلدها را پر کنید.")
        
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
                "content": f"سلام {st.session_state.username} عزیز 👋 من **ResiliMind** هستم؛ دستیار هوشمند ارزیابی و تقویت تاب‌آوری.\n\nامروز چه حسی داری یا دوست داری در مورد چه موضوعی با هم گفتگو کنیم؟"
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
    sidebar_tab1, sidebar_tab2 = st.tabs(["📊 نشست جاری", "📜 سوابق قبلی"])

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
        st.markdown("**📊 خلاصه وضعیت تاب‌آوری:**")
        
        render_domain_resilience_chart(st.session_state.user_id)
        
        st.divider()

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
        تحلیل چندبعدی وضعیت بر اساس شبکه گراف دانش و هوش مصنوعی
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
        with st.spinner("در حال تحلیل پیام، واکشی از گراف دانش و ارزیابی..."):

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
                    category=assessment.get("category", "Personal_Resilience"), # مقدار انگلیسی خام
                    status=assessment.get("status", "YELLOW"),
                    score=int(assessment.get("score", 50)),
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
