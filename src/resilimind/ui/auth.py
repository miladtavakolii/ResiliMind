import logging
import streamlit as st
from typing import Optional
from resilimind.core.database import authenticate_user, register_user

# Initialize module logger
logger = logging.getLogger(__name__)


def init_session_state() -> None:
    """Initializes authentication variables in session state."""
    logger.debug("[UI-Auth] Initializing session state variables...")
    if "user_id" not in st.session_state:
        st.session_state.user_id = None
        logger.debug("[UI-Auth] Initialized 'user_id' to None.")
    if "username" not in st.session_state:
        st.session_state.username = None
        logger.debug("[UI-Auth] Initialized 'username' to None.")


def render_auth_page() -> None:
    """Renders the login and registration interface."""
    logger.info("[UI-Auth] Rendering authentication page...")
    st.markdown("<div style='margin-top: 10vh;'></div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2.5rem;">
            <div style="font-size: 3.5rem; margin-bottom: 0.5rem; animation: float 3s ease-in-out infinite;">🌱</div>
            <h1 style="font-weight: 700; color: #f8fafc; font-size: 2.2rem; margin-bottom: 0.5rem;">ResiliMind</h1>
            <p style="color: #94a3b8; font-size: 1.1rem; margin-top: 0;">پلتفرم هوشمند مشاوره و ارزیابی تاب‌آوری</p>
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["🔑 ورود به حساب", "📝 ثبت‌نام جدید"])
        
        with tab1:
            st.markdown("<br>", unsafe_allow_html=True)
            login_user: str = st.text_input("👤 نام کاربری", key="log_user")
            login_pass: str = st.text_input("🔒 رمز عبور", type="password", key="log_pass")
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button("ورود به سامانه", width="stretch"):
                logger.debug(f"[UI-Auth] Login attempt initiated for user: {login_user}")
                if login_user and login_pass:
                    user_id: Optional[int] = authenticate_user(login_user, login_pass)
                    if user_id:
                        logger.info(f"[UI-Auth] User '{login_user}' authenticated successfully (id={user_id}).")
                        st.session_state.user_id = user_id
                        st.session_state.username = login_user
                        st.success("ورود موفقیت‌آمیز بود! در حال انتقال...")
                        st.rerun()
                    else:
                        logger.warning(f"[UI-Auth] Failed login attempt for user: {login_user}")
                        st.error("نام کاربری یا رمز عبور اشتباه است.")
                else:
                    logger.warning("[UI-Auth] Login attempted with empty fields.")
                    st.warning("لطفاً همه فیلدها را پر کنید.")
                    
        with tab2:
            st.markdown("<br>", unsafe_allow_html=True)
            reg_user: str = st.text_input("👤 نام کاربری دلخواه", key="reg_user")
            reg_pass: str = st.text_input("🔒 رمز عبور", type="password", key="reg_pass")
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button("ایجاد حساب کاربری", width="stretch"):
                logger.debug(f"[UI-Auth] Registration attempt initiated for user: {reg_user}")
                if reg_user and reg_pass:
                    if register_user(reg_user, reg_pass):
                        logger.info(f"[UI-Auth] User '{reg_user}' registered successfully.")
                        st.success("حساب کاربری با موفقیت ایجاد شد! اکنون می‌توانید از تب ورود استفاده کنید.")
                    else:
                        logger.warning(f"[UI-Auth] Registration failed: Username '{reg_user}' already exists.")
                        st.error("این نام کاربری قبلاً ثبت شده است. لطفاً نام دیگری انتخاب کنید.")
                else:
                    logger.warning("[UI-Auth] Registration attempted with empty fields.")
                    st.warning("لطفاً همه فیلدها را پر کنید.")
