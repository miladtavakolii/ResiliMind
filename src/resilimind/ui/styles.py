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

        /* Page Container Spacing & Full Width */
        .block-container {
            padding-top: 4.2rem !important;
            padding-bottom: 1.2rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
            max-width: 100% !important;
        }

        /* Header Accent Banner */
        .main-header {
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
            border-radius: 14px;
            padding: 12px 18px;
            color: #ffffff;
            margin-bottom: 10px;
            border: 1px solid rgba(255, 255, 255, 0.05);
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
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

        /* Styled Tabs - Sticky & Permanently Visible */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: rgba(15, 23, 42, 0.85);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            padding: 6px;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            margin-bottom: 10px;
            direction: rtl !important;
            justify-content: flex-start !important;
            display: flex !important;
            width: 100% !important;
            position: sticky !important;
            top: 3.5rem !important;
            z-index: 100 !important;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px !important;
            padding: 8px 18px !important;
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

        /* Dedicated Chat Scrollable Container & Responsive Viewport Height */
        div[data-testid="stVerticalBlockBorderWrapper"]:has(div[data-testid="stChatMessage"]),
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.prompt-chips-wrapper) {
            height: calc(100vh - 290px) !important;
            max-height: calc(100vh - 290px) !important;
            min-height: 440px !important;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(div[data-testid="stChatMessage"]) > div,
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.prompt-chips-wrapper) > div {
            height: 100% !important;
        }

        /* Sleek Scrollbar */
        div[data-testid="stVerticalBlockBorderWrapper"] > div[style*="overflow: auto"],
        div[data-testid="stVerticalBlockBorderWrapper"] > div[style*="overflow-y: auto"] {
            scrollbar-width: thin;
            scrollbar-color: rgba(56, 189, 248, 0.35) rgba(15, 23, 42, 0.4);
            scroll-behavior: smooth;
            padding-right: 8px !important;
            padding-left: 8px !important;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] > div[style*="overflow: auto"]::-webkit-scrollbar,
        div[data-testid="stVerticalBlockBorderWrapper"] > div[style*="overflow-y: auto"]::-webkit-scrollbar {
            width: 6px;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] > div[style*="overflow: auto"]::-webkit-scrollbar-track,
        div[data-testid="stVerticalBlockBorderWrapper"] > div[style*="overflow-y: auto"]::-webkit-scrollbar-track {
            background: rgba(15, 23, 42, 0.4);
            border-radius: 4px;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] > div[style*="overflow: auto"]::-webkit-scrollbar-thumb,
        div[data-testid="stVerticalBlockBorderWrapper"] > div[style*="overflow-y: auto"]::-webkit-scrollbar-thumb {
            background: rgba(56, 189, 248, 0.35);
            border-radius: 4px;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] > div[style*="overflow: auto"]::-webkit-scrollbar-thumb:hover,
        div[data-testid="stVerticalBlockBorderWrapper"] > div[style*="overflow-y: auto"]::-webkit-scrollbar-thumb:hover {
            background: rgba(56, 189, 248, 0.65);
        }

        /* Crisis & Safety Banners */
        .crisis-banner-safe {
            background: rgba(15, 23, 42, 0.7);
            border: 1px solid rgba(56, 189, 248, 0.25);
            border-radius: 12px;
            padding: 8px 14px;
            margin-bottom: 12px;
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

        /* ----------------------------------------------------
           Hide Streamlit default "Press Enter to apply" hint
           ---------------------------------------------------- */
        div[data-testid="InputInstructions"],
        div[data-testid="stTextInputInstructions"],
        .stTextInput small,
        [data-testid="stTextInput"] [data-testid="InputInstructions"] {
            display: none !important;
        }

        /* ----------------------------------------------------
           Modern Authentication Portal & Glassmorphic Styling
           ---------------------------------------------------- */
        .auth-card-header {
            text-align: center;
            margin-bottom: 2rem;
        }
        .auth-logo-badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 72px;
            height: 72px;
            background: radial-gradient(circle, rgba(56, 189, 248, 0.25) 0%, rgba(14, 165, 233, 0.05) 100%);
            border: 1.5px solid rgba(56, 189, 248, 0.4);
            border-radius: 22px;
            font-size: 2.3rem;
            margin-bottom: 14px;
            box-shadow: 0 0 30px rgba(56, 189, 248, 0.25);
            animation: pulse-logo 3.5s ease-in-out infinite;
        }
        @keyframes pulse-logo {
            0%, 100% { transform: translateY(0); box-shadow: 0 0 25px rgba(56, 189, 248, 0.2); }
            50% { transform: translateY(-4px); box-shadow: 0 0 35px rgba(56, 189, 248, 0.4); }
        }
        .auth-title {
            font-weight: 800;
            color: #f8fafc;
            font-size: 2.2rem;
            margin: 0 0 6px 0;
            letter-spacing: -0.5px;
            background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .auth-subtitle {
            color: #94a3b8;
            font-size: 0.96rem;
            margin-top: 0;
            margin-bottom: 16px;
            line-height: 1.6;
        }
        .auth-badges-row {
            display: flex;
            justify-content: center;
            gap: 8px;
            flex-wrap: wrap;
            margin-bottom: 12px;
        }
        .auth-badge {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 20px;
            padding: 4px 12px;
            font-size: 0.78rem;
            color: #cbd5e1;
        }
        .auth-footer-disclaimer {
            text-align: center;
            color: #64748b;
            font-size: 0.8rem;
            margin-top: 28px;
            line-height: 1.6;
        }

        /* Form submit button hover & sizing */
        div[data-testid="stForm"] {
            border: none !important;
            padding: 0 !important;
            background: transparent !important;
        }
    </style>
    """, unsafe_allow_html=True)
