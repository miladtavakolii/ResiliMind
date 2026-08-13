import logging
import streamlit as st
import pandas as pd
import plotly.express as px
from typing import List, Dict, Any
from resilimind.core.database import get_user_latest_node_statuses, get_user_resilience_history

# Initialize module logger
logger = logging.getLogger(__name__)


def render_domain_resilience_chart(user_id: int) -> None:
    """Renders an interactive RADAR chart showing resilience score distribution."""
    logger.debug(f"[UI-Dashboard] Rendering domain resilience chart for user_id={user_id}...")
    logs = get_user_latest_node_statuses(user_id)
    if not logs:
        logger.debug("[UI-Dashboard] No logs available for radar chart rendering.")
        st.caption("هنوز ارزیابی کافی برای محاسبه نمودار ثبت نشده است.")
        return

    df = pd.DataFrame(logs)
    if 'category' not in df.columns or 'score' not in df.columns:
        logger.warning("[UI-Dashboard] Missing required columns ('category' or 'score') in log data frame.")
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

    fig = px.line_polar(
        category_avg, r='امتیاز', theta='دسته', line_close=True, range_r=[0, 100], markers=True
    )
    fig.update_traces(fill='toself', line_color='#38bdf8', marker=dict(size=8, color='#0ea5e9'))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], color="#64748b", gridcolor="rgba(255, 255, 255, 0.1)", tickfont=dict(size=10)),
            angularaxis=dict(color="#cbd5e1", gridcolor="rgba(255, 255, 255, 0.1)", tickfont=dict(size=13, family="Vazirmatn")),
            bgcolor="rgba(0,0,0,0)"
        ),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=40, b=40, l=40, r=40), height=320,
    )
    logger.debug("[UI-Dashboard] Radar chart successfully compiled. Rendering in Streamlit...")
    st.plotly_chart(fig, width="stretch", config={'displayModeBar': False})


def render_sidebar_dashboard() -> None:
    """Renders the complete sidebar profile and dashboard securely without raw HTML injection risks."""
    logger.debug("[UI-Dashboard] Rendering sidebar profile and dashboard securely...")
    with st.sidebar:
        st.markdown("### 👤 پروفایل کاربری")
        st.markdown(f"کاربر فعلی: **{st.session_state.username}**")
        
        if st.button("🚪 خروج از حساب", width="stretch"):
            logger.info(f"[UI-Dashboard] User '{st.session_state.username}' logged out.")
            for key in ["user_id", "username", "messages", "last_state"]:
                st.session_state[key] = None if key in ["user_id", "username", "last_state"] else []
            st.rerun()
            
        st.divider()

        sidebar_tab1, sidebar_tab2 = st.tabs(["📊 نشست جاری", "📜 سوابق قبلی"])

        with sidebar_tab1:
            if st.session_state.get("last_state"):
                state: Dict[str, Any] = st.session_state.last_state
                active_signals: List[Dict[str, Any]] = state.get("active_signals", [])
                
                st.markdown("**سیگنال‌ها و شواهد شناسایی‌شده:**")
                if active_signals:
                    logger.debug(f"[UI-Dashboard] Rendering {len(active_signals)} active signals in sidebar.")
                    for sig in active_signals:
                        node_id = sig.get("node_id", "")
                        polarity = sig.get("detected_signal", "mixed")
                        evidence = sig.get("evidence", "")
                        
                        with st.container(border=True):
                            col1, col2 = st.columns([2, 1])
                            with col1:
                                st.code(node_id, language=None)
                            with col2:
                                st.markdown(f"**{polarity.upper()}**")
                            if evidence:
                                st.caption(f'شاهد: "{evidence}"')
                else:
                    st.caption("سیگنال مستقیمی در آخرین پیام شناسایی نشد.")

                assessments: List[Dict[str, Any]] = state.get("assessments", [])
                st.markdown("**ارزیابی نشست:**")
                if assessments:
                    logger.debug(f"[UI-Dashboard] Rendering {len(assessments)} assessments in sidebar.")
                    for item in assessments:
                        node_id = item.get("node_id", "N/A")
                        status = item.get("status", "YELLOW").upper()
                        confidence = int(item.get("confidence", 0.0) * 100)
                        reasoning = item.get("reasoning", "")
                        
                        with st.container(border=True):
                            col1, col2 = st.columns([2, 1])
                            with col1:
                                st.code(node_id, language=None)
                            with col2:
                                st.markdown(f"**{status}**")
                            st.markdown(f"اطمینان: {confidence}%")
                            if reasoning:
                                st.markdown(f"تحلیل: {reasoning}")
                else:
                    st.caption("ارزیابی جدیدی ثبت نشده است.")
            else:
                st.info("با ارسال پیام، جزئیات نشست در این بخش قرار می‌گیرد.")

        with sidebar_tab2:
            st.markdown("**📊 خلاصه وضعیت تاب‌آوری:**")
            render_domain_resilience_chart(st.session_state.user_id)
            st.divider()

            st.markdown("**تاریخچه ارزیابی‌های اخیر:**")
            history_logs = get_user_resilience_history(st.session_state.user_id, limit=15)
            
            if history_logs:
                logger.debug(f"[UI-Dashboard] Rendering {len(history_logs)} historical logs in sidebar.")
                for log in history_logs:
                    node_id = log.get("node_id", "N/A")
                    status = log.get("status", "YELLOW").upper()
                    created_at = str(log.get("created_at", ""))[:16]
                    
                    with st.container(border=True):
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.code(node_id, language=None)
                        with col2:
                            st.markdown(f"**{status}**")
                        st.caption(created_at)
            else:
                st.caption("هنوز سابقه ارزیابی برای حساب شما ثبت نشده است.")
