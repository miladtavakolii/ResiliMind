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
    """Renders a modern, glassmorphic login and registration interface."""
    logger.info("[UI-Auth] Rendering modern authentication portal...")
    
    # Hide empty sidebar and center layout on authentication screen
    st.markdown("""
    <style>
        /* Hide empty sidebar on login/auth page */
        [data-testid="stSidebar"],
        [data-testid="collapsedControl"] {
            display: none !important;
        }
        /* Clean vertical spacing */
        .block-container {
            max-width: 960px !important;
            padding-top: 5vh !important;
            padding-bottom: 3rem !important;
        }
    </style>
    """, unsafe_allow_html=True)

    _, center_col, _ = st.columns([1, 1.6, 1])
    
    with center_col:
        # Header & Branding Badge
        st.markdown("""
        <div class="auth-card-header">
            <div class="auth-logo-badge">🌱</div>
            <h1 class="auth-title">ResiliMind</h1>
            <p class="auth-subtitle">سامانه هوشمند پایش و ارزیابی چندبعدی تاب‌آوری روان‌شناختی</p>
            <div class="auth-badges-row">
                <span class="auth-badge">🧠 مدل تحلیلی تاب‌آوری</span>
                <span class="auth-badge">🕸️ گراف دانش پویا</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        tab_login, tab_register = st.tabs(["🔑 ورود به سامانه", "📝 ثبت‌نام کاربر جدید"])

        # Tab 1: Login
        with tab_login:
            st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)
            with st.form("login_form", clear_on_submit=False):
                login_user = st.text_input(
                    "👤 نام کاربری",
                    placeholder="نام کاربری خود را وارد کنید...",
                    key="log_user"
                )
                login_pass = st.text_input(
                    "🔒 رمز عبور",
                    type="password",
                    placeholder="رمز عبور...",
                    key="log_pass"
                )
                st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
                login_submit = st.form_submit_button("ورود به سامانه 🚀", use_container_width=True)

            if login_submit:
                logger.debug("[UI-Auth] Login attempt initiated for user: %s", login_user)
                clean_user = login_user.strip()
                clean_pass = login_pass.strip()
                if clean_user and clean_pass:
                    user_id: Optional[int] = authenticate_user(clean_user, clean_pass)
                    if user_id:
                        logger.info("[UI-Auth] User '%s' authenticated successfully (id=%s).", clean_user, user_id)
                        st.session_state.user_id = user_id
                        st.session_state.username = clean_user
                        st.success("ورود موفقیت‌آمیز بود! در حال انتقال به سامانه...")
                        st.rerun()
                    else:
                        logger.warning("[UI-Auth] Failed login attempt for user: %s", clean_user)
                        st.error("نام کاربری یا رمز عبور اشتباه است.")
                else:
                    logger.warning("[UI-Auth] Login attempted with empty fields.")
                    st.warning("لطفاً همه فیلدها را پر کنید.")

        # Tab 2: Register
        with tab_register:
            st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)
            with st.form("register_form", clear_on_submit=False):
                reg_user = st.text_input(
                    "👤 نام کاربری دلخواه",
                    placeholder="حداقل ۳ نویسه (مثال: mohammad)",
                    key="reg_user"
                )
                reg_pass = st.text_input(
                    "🔒 رمز عبور",
                    type="password",
                    placeholder="حداقل ۴ نویسه ایمن...",
                    key="reg_pass"
                )
                st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
                reg_submit = st.form_submit_button("ایجاد حساب کاربری ✨", use_container_width=True)

            if reg_submit:
                logger.debug("[UI-Auth] Registration attempt initiated for user: %s", reg_user)
                clean_reg_user = reg_user.strip()
                clean_reg_pass = reg_pass.strip()
                if clean_reg_user and clean_reg_pass:
                    if len(clean_reg_user) < 3:
                        st.warning("نام کاربری باید حداقل ۳ کاراکتر باشد.")
                    elif len(clean_reg_pass) < 4:
                        st.warning("رمز عبور باید حداقل ۴ کاراکتر باشد.")
                    elif register_user(clean_reg_user, clean_reg_pass):
                        logger.info("[UI-Auth] User '%s' registered successfully.", clean_reg_user)
                        st.success("حساب کاربری با موفقیت ساخته شد! اکنون از تب «ورود به سامانه» وارد شوید.")
                    else:
                        logger.warning("[UI-Auth] Registration failed: Username '%s' already exists.", clean_reg_user)
                        st.error("این نام کاربری قبلاً ثبت شده است. لطفاً نام دیگری انتخاب کنید.")
                else:
                    logger.warning("[UI-Auth] Registration attempted with empty fields.")
                    st.warning("لطفاً همه فیلدها را پر کنید.")

        # Footer Disclaimer & Helpline
        st.markdown("""
        <div class="auth-footer-disclaimer">
            🛡️ سامانه پژوهشی دانشگاهی ResiliMind • حریم خصوصی و امنیت داده‌های مکالمه محفوظ است.<br>
            در شرایط بحرانی فوری، با اورژانس اجتماعی (۱۲۳) یا صدای مشاور (۱۴۸۰) تماس بگیرید.
        </div>
        """, unsafe_allow_html=True)
