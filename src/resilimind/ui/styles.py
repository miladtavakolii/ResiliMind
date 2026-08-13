import streamlit as st

def inject_custom_css() -> None:
    """Injects custom CSS for RTL support, typography, and modern UI components."""
    st.markdown("""
    <style>
        /* Load Google Vazirmatn Font */
        @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@300;400;600;700&display=swap');

        /* Global RTL and Typography Settings */
        html, body, [class*="css"], .stChatMessage, .stTextInput, .stMarkdown {
            font-family: 'Vazirmatn', sans-serif !important;
            direction: rtl;
            text-align: right;
        }

        /* Modern Glassmorphic Dashboard Cards */
        .metric-card {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 14px;
            padding: 16px 20px;
            margin-bottom: 14px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
            transition: all 0.2s ease-in-out;
        }
        .metric-card:hover {
            background: rgba(255, 255, 255, 0.05);
            border-color: rgba(255, 255, 255, 0.15);
        }

        /* Node Chip Badge */
        .node-chip {
            display: inline-block;
            background-color: rgba(14, 165, 233, 0.1);
            color: #38bdf8;
            border: 1px solid rgba(14, 165, 233, 0.3);
            border-radius: 20px;
            padding: 4px 14px;
            font-size: 0.85rem;
            font-weight: 600;
            margin: 4px;
            direction: ltr;
        }

        /* Resilience Status Badges */
        .badge-green {
            background-color: rgba(16, 185, 129, 0.15);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.3);
            padding: 4px 12px;
            border-radius: 8px;
            font-weight: 700;
            font-size: 0.85rem;
        }
        .badge-yellow {
            background-color: rgba(245, 158, 11, 0.15);
            color: #fbbf24;
            border: 1px solid rgba(245, 158, 11, 0.3);
            padding: 4px 12px;
            border-radius: 8px;
            font-weight: 700;
            font-size: 0.85rem;
        }
        .badge-red {
            background-color: rgba(239, 68, 68, 0.15);
            color: #f87171;
            border: 1px solid rgba(239, 68, 68, 0.3);
            padding: 4px 12px;
            border-radius: 8px;
            font-weight: 700;
            font-size: 0.85rem;
        }

        /* Chat Input Fixes for RTL */
        .stChatInput textarea {
            direction: rtl;
            text-align: right;
        }

        /* Header Accent Banner */
        .main-header {
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
            border-radius: 18px;
            padding: 28px;
            color: #ffffff;
            margin-bottom: 30px;
            border: 1px solid rgba(255, 255, 255, 0.05);
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
        }
           
        /* Styled Input Fields */
        .stTextInput > div > div > input {
            border-radius: 10px !important;
            border: 1px solid rgba(255,255,255,0.1) !important;
            background-color: rgba(255,255,255,0.03) !important;
            padding: 12px 16px !important;
            transition: all 0.3s ease !important;
        }
        .stTextInput > div > div > input:focus {
            border-color: #38bdf8 !important;
            box-shadow: 0 0 0 1px #38bdf8 !important;
            background-color: rgba(255,255,255,0.06) !important;
        }

        /* Styled Buttons */
        .stButton > button {
            border-radius: 10px !important;
            font-weight: 600 !important;
            padding: 8px 16px !important;
            background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%) !important;
            border: none !important;
            color: white !important;
            transition: all 0.3s ease !important;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 15px rgba(2, 132, 199, 0.4) !important;
        }

        /* Styled Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: rgba(255, 255, 255, 0.02);
            padding: 6px;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px !important;
            padding: 10px 20px !important;
            background-color: transparent;
            border: none !important;
            font-weight: 600 !important;
        }
        .stTabs [aria-selected="true"] {
            background-color: rgba(255, 255, 255, 0.08) !important;
            color: #38bdf8 !important;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }
    </style>
    """, unsafe_allow_html=True)
