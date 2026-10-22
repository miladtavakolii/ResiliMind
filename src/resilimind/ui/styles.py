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
        .stChatMessage div[data-testid="stMarkdownContainer"] p {
            white-space: pre-line !important;
            line-height: 1.7; /* برای خوانایی بهتر متن‌های فارسی */
        }

        /* Crisis & Safety Banners */
        .crisis-banner-safe {
            background: rgba(15, 23, 42, 0.7);
            border: 1px solid rgba(56, 189, 248, 0.25);
            border-radius: 12px;
            padding: 10px 16px;
            margin-bottom: 20px;
            font-size: 0.88rem;
            color: #94a3b8;
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 10px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }
        .crisis-banner-safe strong {
            color: #e2e8f0;
        }
        .crisis-banner-alert {
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.22) 0%, rgba(185, 28, 28, 0.35) 100%);
            border: 1.5px solid #ef4444;
            border-radius: 14px;
            padding: 16px 20px;
            margin-bottom: 24px;
            font-size: 0.95rem;
            color: #fecaca;
            box-shadow: 0 0 25px rgba(239, 68, 68, 0.35);
            animation: pulse-border 2s infinite;
        }
        @keyframes pulse-border {
            0%, 100% { border-color: #ef4444; box-shadow: 0 0 15px rgba(239, 68, 68, 0.3); }
            50% { border-color: #f87171; box-shadow: 0 0 25px rgba(239, 68, 68, 0.5); }
        }

        /* Prompt Starter Chips */
        .prompt-chips-wrapper {
            background: rgba(255, 255, 255, 0.02);
            border: 1px dashed rgba(255, 255, 255, 0.12);
            border-radius: 14px;
            padding: 16px 18px;
            margin: 15px 0 25px 0;
        }
        .prompt-chips-title {
            font-size: 0.9rem;
            color: #94a3b8;
            margin-bottom: 12px;
            font-weight: 600;
        }

        /* Streamlit Status Widget RTL Fixes */
        [data-testid="stStatusWidget"] {
            direction: rtl !important;
            text-align: right !important;
            font-family: 'Vazirmatn', sans-serif !important;
            border-radius: 12px !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            background: rgba(15, 23, 42, 0.6) !important;
            margin-bottom: 15px !important;
        }
        [data-testid="stStatusWidget"] summary {
            font-family: 'Vazirmatn', sans-serif !important;
            font-weight: 600 !important;
            font-size: 0.92rem !important;
        }
        [data-testid="stStatusWidget"] p {
            font-size: 0.88rem !important;
            line-height: 1.6 !important;
            margin: 4px 0 !important;
        }

        /* Streamlit Expander RTL Fixes */
        [data-testid="stExpander"] {
            direction: rtl !important;
            text-align: right !important;
            border-radius: 12px !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            background: rgba(255, 255, 255, 0.02) !important;
            margin-top: 10px !important;
        }
        [data-testid="stExpander"] summary {
            font-family: 'Vazirmatn', sans-serif !important;
            font-size: 0.9rem !important;
            font-weight: 600 !important;
            color: #38bdf8 !important;
        }

        /* Evidence Cards inside Expander */
        .evidence-card {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.07);
            border-radius: 10px;
            padding: 12px 16px;
            margin-bottom: 10px;
        }
        .evidence-quote {
            border-right: 3px solid #38bdf8;
            padding-right: 12px;
            margin: 8px 0;
            color: #cbd5e1;
            font-style: italic;
            font-size: 0.88rem;
            background: rgba(14, 165, 233, 0.06);
            border-radius: 0 6px 6px 0;
            padding: 6px 12px;
        }
    </style>
    """, unsafe_allow_html=True)
