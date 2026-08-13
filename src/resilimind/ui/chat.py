import streamlit as st
from typing import Any, Dict
from resilimind.services.resilience_service import process_user_message
from resilimind.schemas.models import ProcessResult

def render_chat_interface(app: Any, config: Dict[str, Any]) -> None:
    """Renders the main chat interface and handles user state updates via decoupled service results."""
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
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("در حال تحلیل پیام، واکشی از گراف دانش و ارزیابی..."):
                # Call pure service layer
                result: ProcessResult = process_user_message(app, config, user_input, st.session_state.user_id)
                
                # UI explicitly manages its own session state based on service output
                st.session_state.last_state = result.state
                
                st.markdown(result.final_response)
                st.session_state.messages.append({"role": "assistant", "content": result.final_response})
                st.rerun()
