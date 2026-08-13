import streamlit as st
from typing import Any, List

def sync_chat_history(current_graph_state: Any) -> None:
    """Syncs the persistent graph memory with Streamlit session state."""
    if "messages" not in st.session_state or not st.session_state.messages:
        stored_messages: List[Any] = current_graph_state.values.get("messages", [])
        if stored_messages:
            st.session_state.messages = []
            for msg in stored_messages:
                role = "user" if msg.type == "human" else "assistant"
                st.session_state.messages.append({"role": role, "content": msg.content})
        else:
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": f"سلام {st.session_state.username} عزیز 👋 من **ResiliMind** هستم؛ دستیار هوشمند ارزیابی و تقویت تاب‌آوری.\n\nامروز چه حسی داری یا دوست داری در مورد چه موضوعی با هم گفتگو کنیم؟"
                }
            ]
            
    st.session_state.last_state = current_graph_state.values if current_graph_state.values else None
