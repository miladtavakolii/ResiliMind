import logging
import streamlit as st
from typing import Any, List

# Initialize module logger
logger = logging.getLogger(__name__)


def sync_chat_history(current_graph_state: Any) -> None:
    """Syncs the persistent graph memory with Streamlit session state."""
    logger.debug("[UI-Sync] Synchronizing persistent graph memory with Streamlit session state...")
    
    if "messages" not in st.session_state or not st.session_state.messages:
        stored_messages: List[Any] = current_graph_state.values.get("messages", [])
        if stored_messages:
            logger.info(f"[UI-Sync] Restoring {len(stored_messages)} historical messages from graph checkpointer.")
            st.session_state.messages = []
            for msg in stored_messages:
                role = "user" if msg.type == "human" else "assistant"
                st.session_state.messages.append({"role": role, "content": msg.content})
        else:
            logger.info("[UI-Sync] No stored messages found in graph state. Initializing default greeting.")
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": f"سلام {st.session_state.username} عزیز 👋 من **ResiliMind** هستم؛ دستیار هوشمند ارزیابی و تقویت تاب‌آوری.\n\nامروز چه حسی داری یا دوست داری در مورد چه موضوعی با هم گفتگو کنیم؟"
                }
            ]
            
    st.session_state.last_state = current_graph_state.values if current_graph_state.values else None
    logger.debug("[UI-Sync] Chat history synchronization completed successfully.")
