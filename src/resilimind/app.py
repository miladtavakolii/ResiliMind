import streamlit as st
from typing import Any, Dict

from resilimind.core.workflow import build_workflow
from resilimind.core.database import init_db
from resilimind.ui.styles import inject_custom_css
from resilimind.ui.auth import init_session_state, render_auth_page
from resilimind.ui.dashboard import render_sidebar_dashboard
from resilimind.ui.chat import render_chat_interface
from resilimind.services.memory_service import sync_chat_history

# 1. Initialization and CSS
st.set_page_config(
    page_title="ResiliMind | AI Resilience Platform",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)
init_db()
inject_custom_css()
init_session_state()

# 2. Authentication Gate
if st.session_state.user_id is None:
    render_auth_page()
    st.stop()

# 3. Initialize Graph & State
@st.cache_resource
def load_graph_application() -> Any:
    return build_workflow()

app = load_graph_application()
config: Dict[str, Any] = {"configurable": {"thread_id": str(st.session_state.user_id)}}
current_graph_state = app.get_state(config)

sync_chat_history(current_graph_state)

# 4. Render Main Interface
render_sidebar_dashboard()
render_chat_interface(app, config)
