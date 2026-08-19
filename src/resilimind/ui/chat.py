import logging
import streamlit as st
from typing import Any, Dict
from resilimind.services.resilience_service import process_user_message
from resilimind.schemas.models import ProcessResult

# Initialize module logger
logger = logging.getLogger(__name__)


def render_chat_interface(app: Any, config: Dict[str, Any]) -> None:
    """Renders the main chat interface and handles user state updates via decoupled service results."""
    logger.debug("[UI-Chat] Rendering main chat interface...")
    
    st.markdown("""
    <div class="main-header">
        <h2 style="margin:0; font-weight:700;">🌱 سامانه هوشمند مشاوره و تاب‌آوری ResiliMind</h2>
        <p style="margin:6px 0 0 0; opacity:0.85; font-size:0.95rem;">
            تحلیل چندبعدی وضعیت بر اساس شبکه گراف دانش و هوش مصنوعی
        </p>
    </div>
    """, unsafe_allow_html=True)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input("پیام خود را اینجا بنویسید..."):
        logger.info(f"[UI-Chat] Received user input: '{user_input[:30]}...' (length: {len(user_input)})")
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            
            status_container = st.empty()
            status_container.info("⏳ در حال تحلیل پیام، واکشی از گراف دانش و ارزیابی...")
            message_placeholder = st.empty()
            
            class StreamlitStreamHandler:
                def __init__(self, container, status_box):
                    self.container = container
                    self.status_box = status_box
                    self.text = ""
                    self.first_token = True

                def on_llm_new_token(self, token: str) -> None:
                    if self.first_token:
                        self.status_box.empty()
                        self.first_token = False
                    
                    self.text += token
                    self.container.markdown(self.text.replace("\n", "  \n") + " ▌")

            stream_handler = StreamlitStreamHandler(message_placeholder, status_container)
            
            if "configurable" not in config:
                config["configurable"] = {}
            config["configurable"]["stream_handler"] = stream_handler

            logger.debug("[UI-Chat] Invoking process_user_message service layer with streaming...")
            
            result = process_user_message(app, config, user_input, st.session_state.user_id)
            st.session_state.last_state = result.state
            
            response_text = result.final_response
            if not response_text:
                response_text = result.state.get("final_response", "")
                
            if not response_text:
                status_container.empty()
                response_text = "⚠️ **خطا در سیستم:** گراف پردازش را تمام کرد اما ایجنت مشاور متنی تولید نکرد."
                logger.error(f"[UI-Chat] Empty response text. Graph State Dump: {result.state.keys()}")

            formatted_response = response_text.replace("\n", "  \n")
            message_placeholder.markdown(formatted_response)
            
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            logger.info("[UI-Chat] Assistant streaming completed and saved.")
