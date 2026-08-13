import logging
import streamlit as st
from typing import Any, Dict

from resilimind.core.workflow import build_workflow
from resilimind.core.database import init_db
from resilimind.ui.styles import inject_custom_css
from resilimind.ui.auth import init_session_state, render_auth_page
from resilimind.ui.dashboard import render_sidebar_dashboard
from resilimind.ui.chat import render_chat_interface
from resilimind.services.memory_service import sync_chat_history
from resilimind.core.config import settings

# 0. Dynamic Logging Setup based on AppConfig
log_level_str = settings.LOG_LEVEL.upper()
numeric_level = getattr(logging, log_level_str, logging.INFO)

# Conditional formatting: cleaner and focused on logger name during DEBUG
if log_level_str == "DEBUG":
    log_format = '%(asctime)s [%(levelname)s] %(name)s:\t%(message)s'
else:
    log_format = '%(asctime)s [%(levelname)s]:\t%(message)s'
date_format = '%Y-%m-%d %H:%M:%S'
logging.basicConfig(
    level=numeric_level,
    format=log_format,
    datefmt=date_format,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(settings.resolved_log_file_path, encoding="utf-8")
    ]
)

# Initialize module logger
logger = logging.getLogger(__name__)

# 1. Initialization and CSS
logger.info("Configuring Streamlit page settings...")
st.set_page_config(
    page_title="ResiliMind | AI Resilience Platform",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

logger.info("[Main] Initializing database and custom CSS...")
init_db()
inject_custom_css()
init_session_state()

# 2. Authentication Gate
if st.session_state.user_id is None:
    logger.warning("[Main] User unauthenticated. Rendering authentication gate...")
    render_auth_page()
    st.stop()

logger.info(f"[Main] User authenticated successfully (user_id={st.session_state.user_id}).")

# 3. Initialize Graph & State
@st.cache_resource
def load_graph_application() -> Any:
    """Loads and compiles the LangGraph workflow application."""
    logger.info("[Graph] Compiling LangGraph application instance...")
    return build_workflow()

logger.debug("[Main] Loading cached graph application...")
app = load_graph_application()
config: Dict[str, Any] = {"configurable": {"thread_id": str(st.session_state.user_id)}}
current_graph_state = app.get_state(config)

logger.debug("[Main] Synchronizing chat history with graph state...")
sync_chat_history(current_graph_state)

# 4. Render Main Interface
logger.info("[Main] Rendering main application interface (sidebar & chat)...")
render_sidebar_dashboard()
render_chat_interface(app, config)
logger.debug("[Main] Main application interface successfully rendered.")
