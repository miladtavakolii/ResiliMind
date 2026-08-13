import streamlit as st
import pandas as pd
import plotly.express as px
from typing import List, Dict, Any
from resilimind.core.database import get_user_latest_node_statuses, get_user_resilience_history

def render_domain_resilience_chart(user_id: int) -> None:
    """Renders an interactive RADAR chart showing resilience score distribution."""
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
    st.plotly_chart(fig, width="stretch", config={'displayModeBar': False})

def render_sidebar_dashboard() -> None:
    """Renders the complete sidebar profile and dashboard."""
    with st.sidebar:
        st.markdown("### 👤 پروفایل کاربری")
        st.markdown(f"کاربر فعلی: **{st.session_state.username}**")
        
        if st.button("🚪 خروج از حساب", width="stretch"):
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
                    for sig in active_signals:
                        node_id = sig.get("node_id", "")
                        polarity = sig.get("detected_signal", "mixed")
                        evidence = sig.get("evidence", "")
                        
                        color_map = {"positive": "#34d399", "negative": "#f87171", "mixed": "#fbbf24"}
                        sig_color = color_map.get(polarity, "#38bdf8")
                        
                        st.markdown(f"""
                        <div class="metric-card" style="border-right: 4px solid {sig_color};">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span style="font-weight: 700; font-family: monospace;">{node_id}</span>
                                <span style="font-size: 0.75rem; color: {sig_color}; font-weight: bold;">{polarity.upper()}</span>
                            </div>
                            <div style="font-size: 0.8rem; color: #cbd5e1; margin-top: 6px; font-style: italic;">"{evidence}"</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.caption("سیگنال مستقیمی در آخرین پیام شناسایی نشد.")

                assessments: List[Dict[str, Any]] = state.get("assessments", [])
                st.markdown("**ارزیابی نشست:**")
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
            else:
                st.info("با ارسال پیام، جزئیات نشست در این بخش قرار می‌گیرد.")

        with sidebar_tab2:
            st.markdown("**📊 خلاصه وضعیت تاب‌آوری:**")
            render_domain_resilience_chart(st.session_state.user_id)
            st.divider()

            st.markdown("**تاریخچه ارزیابی‌های اخیر:**")
            history_logs = get_user_resilience_history(st.session_state.user_id, limit=15)
            
            if history_logs:
                for log in history_logs:
                    node_id = log.get("node_id", "N/A")
                    status = log.get("status", "YELLOW").upper()
                    created_at = str(log.get("created_at", ""))[:16]
                    badge_class = f"badge-{status.lower()}"
                    
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
