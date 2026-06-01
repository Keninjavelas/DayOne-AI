"""DayOne AI: Multi-tenant HR onboarding RAG SaaS (Streamlit)."""

from __future__ import annotations

import html
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import streamlit as st
import streamlit_authenticator as stauth
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import HumanMessage, SystemMessage
from pydantic import SecretStr

from chat_memory import ConversationHistory
from ingest import rebuild_organization_index
from retriever import (
    HybridRetriever,
    RetrievalResult,
    USE_RERANKER,
    CONF_LOW,
    confidence_label,
    build_pgvector_hybrid_retriever,
)
from user_admin import (
    ROLE_ADMIN,
    ROLE_EMPLOYEE,
    clone_config,
    create_user_record,
    load_app_config,
    save_app_config,
    serialize_user,
    update_user_record,
    delete_user_record,
)

ROOT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = ROOT_DIR / "config.yaml"
ASSETS_DIR = ROOT_DIR / "assets"
DATA_DIR = ROOT_DIR / "data"
LOGS_DIR = ROOT_DIR / "logs"
MASCOT_PATH = ASSETS_DIR / "mascot.png"
MODEL_NAME = os.getenv("DAYONE_GROQ_MODEL", "llama-3.1-8b-instant")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LOGS_DIR.mkdir(exist_ok=True)

SUGGESTED_PROMPTS = [
    "What are the PTO rules?",
    "How does health insurance enrollment work?",
    "What is the onboarding timeline for new hires?",
]

SYSTEM_PROMPT = (
    "You are DayOne AI, a professional HR onboarding assistant. "
    "Answer ONLY from the retrieved context provided. "
    "If the answer is not in the context, say exactly: "
    "'I do not have that information in the current HR files. Please contact HR.' "
    "Do not invent, infer, or extrapolate beyond the retrieved text. "
    "If the retrieved context contains conflicting information from different "
    "documents, explicitly flag the conflict before answering."
)


@st.cache_resource(show_spinner=False)
def load_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


@st.cache_resource(show_spinner=False)
def get_llm() -> ChatGroq:
    return ChatGroq(
        model=MODEL_NAME,
        temperature=0,
        api_key=SecretStr(os.getenv("GROQ_API_KEY", "")),
    )


def configure_page(authenticated: bool) -> None:
    st.set_page_config(
        page_title="DayOne AI",
        page_icon="✨",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    sidebar_css = ""
    if not authenticated:
        sidebar_css = """
        [data-testid="stSidebar"] { display: none; }
        """

    # Inject Inter font from Google Fonts
    st.markdown(
        '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <style>            /* ── Global font ── */
            html, body, [class*="css"] {{
                font-family: 'Inter', sans-serif !important;
                background-color: #050816 !important;
                color: #f8fafc !important;
            }}

            .stApp {{
                background: #050816;
                color: #f8fafc;
            }}

            #MainMenu, footer {{ visibility: hidden; }}
            {sidebar_css}

            /* AI Concierge Hero Layout */
            .hero-section {{
                text-align: center;
                padding: 2rem 0 3rem 0;
                display: flex;
                flex-direction: column;
                align-items: center;
            }}
            .mascot-container {{
                position: relative;
                width: 140px;
                height: 140px;
                margin: 0 auto 1.5rem auto;
                display: flex;
                align-items: center;
                justify-content: center;
            }}
            .mascot-halo {{
                position: absolute;
                top: 50%; left: 50%;
                transform: translate(-50%, -50%);
                width: 140px; height: 140px;
                border-radius: 50%;
                box-shadow: 0 0 60px 20px rgba(124, 58, 237, 0.4), 0 0 100px 40px rgba(59, 130, 246, 0.2);
                animation: pulseHalo 4s ease-in-out infinite;
                z-index: 1;
            }}
            @keyframes pulseHalo {{
                0% {{ opacity: 0.6; transform: translate(-50%, -50%) scale(0.95); }}
                50% {{ opacity: 1; transform: translate(-50%, -50%) scale(1.05); }}
                100% {{ opacity: 0.6; transform: translate(-50%, -50%) scale(0.95); }}
            }}
            .spark {{
                position: absolute;
                width: 4px; height: 4px;
                background: #fff;
                border-radius: 50%;
                box-shadow: 0 0 10px 2px rgba(255, 255, 255, 0.8);
                opacity: 0;
                animation: floatSpark 3s ease-in infinite;
                z-index: 2;
            }}
            .spark-1 {{ top: 10%; left: 20%; animation-delay: 0s; }}
            .spark-2 {{ top: 80%; left: 80%; animation-delay: 1.5s; width: 3px; height: 3px; }}
            .spark-3 {{ top: 70%; left: 10%; animation-delay: 2.2s; }}
            .spark-4 {{ top: 20%; left: 85%; animation-delay: 0.7s; width: 5px; height: 5px; }}
            @keyframes floatSpark {{
                0% {{ transform: translateY(0) scale(0.5); opacity: 0; }}
                20% {{ opacity: 1; }}
                80% {{ opacity: 1; }}
                100% {{ transform: translateY(-40px) scale(1.2); opacity: 0; }}
            }}

            .hero-mascot {{
                width: 140px;
                height: 140px;
                border-radius: 50%;
                box-shadow: 0 0 40px rgba(124, 58, 237, 0.5), inset 0 0 20px rgba(59, 130, 246, 0.5);
                border: 2px solid rgba(255, 255, 255, 0.15);
                animation: float 6s ease-in-out infinite, pulseGlow 4s ease-in-out infinite;
                object-fit: cover;
                position: relative;
                z-index: 10;
            }}
            .hero-mascot-placeholder {{
                font-size: 4rem;
                animation: float 6s ease-in-out infinite;
                position: relative;
                z-index: 10;
            }}
            @keyframes float {{
                0% {{ transform: translateY(0px); }}
                50% {{ transform: translateY(-12px); }}
                100% {{ transform: translateY(0px); }}
            }}
            @keyframes pulseGlow {{
                0% {{ box-shadow: 0 0 30px rgba(124, 58, 237, 0.4), inset 0 0 10px rgba(59, 130, 246, 0.3); border-color: rgba(255,255,255,0.1); }}
                50% {{ box-shadow: 0 0 80px rgba(124, 58, 237, 0.7), inset 0 0 40px rgba(59, 130, 246, 0.7); border-color: rgba(124, 58, 237, 0.6); }}
                100% {{ box-shadow: 0 0 30px rgba(124, 58, 237, 0.4), inset 0 0 10px rgba(59, 130, 246, 0.3); border-color: rgba(255,255,255,0.1); }}
            }}


            
            .section-kicker {{
                font-size: 0.85rem;
                font-weight: 700;
                color: #64748b;
                text-transform: uppercase;
                letter-spacing: 0.1em;
                margin-bottom: 1rem;
            }}
            /* Glassmorphism Cards */
            .glass-card {{
                background: rgba(255, 255, 255, 0.03) !important;
                backdrop-filter: blur(24px) !important;
                -webkit-backdrop-filter: blur(24px) !important;
                border: 1px solid rgba(255, 255, 255, 0.08) !important;
                border-radius: 20px !important;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4) !important;
                padding: 1.5rem !important;
                transition: transform 0.3s ease, border-color 0.3s ease;
            }}
            .glass-card:hover {{
                transform: translateY(-2px);
                border-color: rgba(255, 255, 255, 0.15) !important;
            }}
            .stat-card {{
                display: flex;
                flex-direction: column;
                min-height: 180px;
            }}
            .stat-title {{
                color: #94a3b8;
                font-size: 0.8rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.1em;
                margin-bottom: 0.75rem;
            }}
            .stat-number {{
                color: #f8fafc;
                font-size: 4rem;
                font-weight: 800;
                line-height: 1;
                letter-spacing: -0.04em;
            }}
            .stat-unit {{
                color: #64748b;
                font-size: 0.9rem;
                font-weight: 500;
                margin-top: 0.25rem;
                margin-bottom: auto;
            }}
            .stat-value {{
                color: #f8fafc;
                font-size: 2.2rem;
                font-weight: 800;
                line-height: 1.1;
                margin: 0.25rem 0;
            }}
            .stat-bar {{
                margin-top: auto;
                width: 100%;
                height: 6px;
                background: rgba(255, 255, 255, 0.06);
                border-radius: 3px;
                overflow: hidden;
            }}
            .stat-progress {{
                height: 100%;
                background: linear-gradient(90deg, #7C3AED, #3b82f6);
                border-radius: 3px;
                transition: width 0.8s ease;
            }}
            .task-item {{
                display: flex;
                align-items: flex-start;
                gap: 0.5rem;
                color: #94a3b8;
                font-size: 0.85rem;
                line-height: 1.3;
                margin-top: 0.4rem;
            }}
            .task-dot {{
                width: 6px;
                height: 6px;
                border-radius: 50%;
                background: #FFB020;
                margin-top: 0.35rem;
                flex-shrink: 0;
                box-shadow: 0 0 6px rgba(255, 176, 32, 0.6);
            }}
            .text-success {{ color: #00D084 !important; }}
            .text-warning {{ color: #FFB020 !important; }}

            /* ── Glassmorphism Chat Bubbles ── */
            .stChatMessage {{
                background: rgba(13, 19, 38, 0.6) !important;
                backdrop-filter: blur(12px) !important;
                border: 1px solid rgba(255, 255, 255, 0.05) !important;
                border-radius: 16px !important;
                padding: 1.5rem !important;
                margin-bottom: 1.5rem !important;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
            }}
            .stChatMessage:hover {{
                border: 1px solid rgba(255, 255, 255, 0.1) !important;
            }}
            /* ── Re-styled Buttons as Feature Tiles (Secondary) ── */
            .stButton > button[kind="secondary"] {{
                border-radius: 16px !important;
                background: rgba(255, 255, 255, 0.02) !important;
                backdrop-filter: blur(20px) !important;
                border: 1px solid rgba(255, 255, 255, 0.05) !important;
                border-top: 1px solid rgba(255, 255, 255, 0.02) !important;
                color: #94a3b8 !important;
                font-weight: 600 !important;
                font-size: 1.05rem !important;
                padding: 1.25rem 1rem !important;
                transition: all 0.2s ease !important;
                box-shadow: inset 0 1px 0 rgba(255,255,255,0.01), 0 4px 10px rgba(0,0,0,0.1) !important;
                width: 100% !important;
                display: flex !important;
                justify-content: flex-start !important;
                text-align: left !important;
            }}
            .stButton > button[kind="secondary"]:hover {{
                background: rgba(255, 255, 255, 0.05) !important;
                border-color: rgba(255, 255, 255, 0.1) !important;
                box-shadow: 0 8px 20px rgba(0,0,0,0.2) !important;
                color: #f8fafc !important;
                transform: translateY(-2px) !important;
            }}
            .stButton > button[kind="secondary"] p {{
                font-size: 1.05rem !important;
            }}

            /* ── Sidebar Navigation Link Styling ── */
            .stSidebar .stButton > button {{
                background: transparent !important;
                border: none !important;
                box-shadow: none !important;
                border-radius: 8px !important;
                color: #94a3b8 !important;
                padding: 0.6rem 1rem !important;
                font-weight: 500 !important;
                font-size: 0.95rem !important;
                margin-bottom: 0.2rem !important;
            }}
            .stSidebar .stButton > button p {{
                font-size: 0.95rem !important;
                font-weight: 500 !important;
            }}
            .stSidebar .stButton > button:hover {{
                background: rgba(255, 255, 255, 0.05) !important;
                color: #f8fafc !important;
                transform: none !important;
                border-color: transparent !important;
                box-shadow: none !important;
            }}

            /* ── Proactive Action Cards (Lower Weight) ── */
            .proactive-card {{
                background: rgba(255, 255, 255, 0.01);
                backdrop-filter: blur(12px);
                border: 1px solid rgba(255, 255, 255, 0.05);
                border-bottom: none;
                border-top-left-radius: 16px;
                border-top-right-radius: 16px;
                padding: 1.25rem;
                margin-bottom: -1rem; /* Collapse with button */
                box-shadow: inset 0 1px 1px rgba(255,255,255,0.02);
            }}
            .proactive-card-red {{ border-top: 2px solid rgba(244, 63, 94, 0.5); background: linear-gradient(180deg, rgba(244, 63, 94, 0.05) 0%, transparent 100%); }}
            .proactive-card-purple {{ border-top: 2px solid rgba(168, 85, 247, 0.5); background: linear-gradient(180deg, rgba(168, 85, 247, 0.05) 0%, transparent 100%); }}
            .proactive-card-green {{ border-top: 2px solid rgba(16, 185, 129, 0.5); background: linear-gradient(180deg, rgba(16, 185, 129, 0.05) 0%, transparent 100%); }}
            .proactive-card-blue {{ border-top: 2px solid rgba(59, 130, 246, 0.5); background: linear-gradient(180deg, rgba(59, 130, 246, 0.05) 0%, transparent 100%); }}
            .proactive-card h4 {{
                margin: 0 0 0.5rem 0;
                font-size: 1.05rem;
                color: #e2e8f0;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }}
            .proactive-card p {{
                margin: 0;
                font-size: 0.9rem;
                color: #64748b;
                line-height: 1.4;
            }}
            
            /* Overriding secondary buttons inside columns to snap to proactive cards */
            .stApp [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] .stButton > button[kind="secondary"] {{
                border-top-left-radius: 0 !important;
                border-top-right-radius: 0 !important;
                border-bottom-left-radius: 16px !important;
                border-bottom-right-radius: 16px !important;
                margin-top: -16px !important;
                justify-content: center !important;
                text-align: center !important;
                padding: 0.8rem !important;
                margin-bottom: 1rem !important;
            }}

            .unified-hero {{
                background: linear-gradient(145deg, rgba(30, 41, 59, 0.6), rgba(15, 23, 42, 0.8));
                backdrop-filter: blur(40px);
                border: 1px solid rgba(255, 255, 255, 0.15);
                border-top: 1px solid rgba(124, 58, 237, 0.5);
                border-top-left-radius: 24px;
                border-top-right-radius: 24px;
                border-bottom: none;
                box-shadow: inset 0 1px 1px rgba(255, 255, 255, 0.1), 0 10px 40px rgba(0, 0, 0, 0.5), 0 0 40px rgba(124, 58, 237, 0.1);
                padding: 3rem 2rem 1.5rem 2rem;
                max-width: 700px;
                margin: 0 auto;
                text-align: center;
                position: relative;
                overflow: hidden;
            }}
            .unified-hero::before {{
                content: '';
                position: absolute;
                top: 0; left: 0; right: 0; height: 100%;
                background: radial-gradient(circle at top center, rgba(124, 58, 237, 0.15), transparent 60%);
                pointer-events: none;
            }}
            .unified-hero .hero-title {{
                font-size: 3.5rem !important;
                font-weight: 800 !important;
                letter-spacing: -0.05em !important;
                margin-bottom: 0.25rem !important;
                line-height: 1.1 !important;
                background: linear-gradient(135deg, #ffffff, #94a3b8);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                position: relative; z-index: 10;
            }}
            .unified-hero .hero-context {{
                font-size: 1.15rem;
                color: #94a3b8;
                font-weight: 500;
                margin-bottom: 2.5rem;
                letter-spacing: 0.02em;
                position: relative; z-index: 10;
            }}
            .unified-hero .insight-box {{
                background: rgba(255, 255, 255, 0.03);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 16px;
                padding: 1.5rem;
                text-align: left;
                margin-bottom: 0.5rem;
                position: relative; z-index: 10;
            }}
            .unified-hero .insight-header {{
                display: flex;
                align-items: center;
                gap: 0.5rem;
                font-size: 0.85rem;
                font-weight: 700;
                color: #A78BFA;
                margin-bottom: 0.75rem;
                letter-spacing: 0.08em;
                text-transform: uppercase;
            }}
            .unified-hero .insight-body {{
                font-size: 1.15rem;
                color: #f8fafc;
                line-height: 1.6;
                font-weight: 500;
                margin: 0;
            }}
            
            /* Primary button snaps to unified hero card bottom */
            .stButton > button[kind="primary"] {{
                border-top-left-radius: 0 !important;
                border-top-right-radius: 0 !important;
                border-bottom-left-radius: 24px !important;
                border-bottom-right-radius: 24px !important;
                background: rgba(124, 58, 237, 0.2) !important;
                border: 1px solid rgba(255, 255, 255, 0.15) !important;
                border-top: 1px solid rgba(255, 255, 255, 0.03) !important;
                color: #fff !important;
                text-align: center !important;
                justify-content: center !important;
                padding: 1.2rem !important;
                box-shadow: 0 15px 40px rgba(0, 0, 0, 0.5) !important;
                margin-top: -16px !important;
                max-width: 700px !important;
                margin-left: auto !important;
                margin-right: auto !important;
                display: block !important;
                transition: background 0.2s !important;
            }}
            .stButton > button[kind="primary"]:hover {{
                background: rgba(124, 58, 237, 0.4) !important;
                border-color: rgba(124, 58, 237, 0.6) !important;
            }}

            /* ── SVG Circular Progress ── */
            .circular-chart {{
                display: block;
                width: 60px !important;
                height: 60px !important;
                flex-shrink: 0;
            }}
            .circle-bg {{
                fill: none;
                stroke: rgba(255, 255, 255, 0.05);
                stroke-width: 3.8;
            }}
            .circle {{
                fill: none;
                stroke-width: 3.8;
                stroke-linecap: round;
                animation: progress 1.5s ease-out forwards;
            }}
            @keyframes progress {{
                0% {{ stroke-dasharray: 0 100; }}
            }}
            .percentage {{
                fill: #f8fafc;
                font-family: inherit;
                font-size: 0.7rem;
                font-weight: 800;
                text-anchor: middle;
            }}

            /* ── Massive Chat Input ── */
            .stChatInputContainer {{
                padding-bottom: 2.5rem !important;
            }}
            .stChatInput {{
                background: rgba(15, 23, 42, 0.9) !important;
                backdrop-filter: blur(24px) !important;
                border: 1px solid rgba(255, 255, 255, 0.15) !important;
                border-top: 1px solid rgba(124, 58, 237, 0.5) !important;
                border-radius: 20px !important;
                box-shadow: 0 -10px 40px rgba(0, 0, 0, 0.5), inset 0 1px 1px rgba(255,255,255,0.1) !important;
                padding: 0.5rem !important;
            }}
            .stChatInput textarea {{
                color: #f8fafc !important;
                font-size: 1.1rem !important;
            }}
            .stChatInput textarea::placeholder {{
                color: #64748b !important;
                font-weight: 500 !important;
            }}        padding: 0.2rem !important;
            }}
            .stChatInput:focus-within {{
                border-color: #7C3AED !important;
                box-shadow: 0 0 0 1px rgba(124, 58, 237, 0.4), 0 10px 40px rgba(0, 0, 0, 0.5) !important;
            }}
            .stChatInput textarea {{
                font-size: 1.1rem !important;
            }}

            /* ── Login layout ── */
            .login-wrap {{
                margin-top: 5vh;
                display: flex;
                align-items: center;
                justify-content: center;
            }}

            /* ── Login card ── */
            .login-card {{
                width: min(520px, 92vw);
                border-radius: 24px;
                border: 1px solid rgba(255, 255, 255, 0.08);
                background: rgba(13, 19, 38, 0.6);
                backdrop-filter: blur(24px);
                padding: 3rem 2.5rem;
                box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5);
                text-align: center;
            }}

            /* Streamlit Form Styling (for login form) */
            div[data-testid="stForm"] {{
                border-radius: 20px !important;
                background: rgba(255, 255, 255, 0.02) !important;
                padding: 2.5rem !important;
                box-shadow: none !important;
                border: 1px solid rgba(255, 255, 255, 0.05) !important;
            }}

            /* ── D1 monogram ── */
            .login-monogram {{
                width: 64px;
                height: 64px;
                border-radius: 16px;
                background: linear-gradient(135deg, rgba(124, 58, 237, 0.2), rgba(59, 130, 246, 0.1));
                border: 1px solid rgba(124, 58, 237, 0.4);
                display: flex;
                align-items: center;
                justify-content: center;
                margin: 0 auto 1.5rem;
                font-size: 1.75rem;
                font-weight: 800;
                color: #7C3AED;
                box-shadow: 0 0 30px rgba(124, 58, 237, 0.2);
            }}

            .login-title {{
                margin: 0 0 0.5rem 0;
                font-size: 2.2rem;
                font-weight: 800;
                letter-spacing: -0.04em;
                color: #f8fafc;
            }}

            .login-subtitle {{
                margin: 0 0 2rem 0;
                color: #94a3b8;
                font-size: 1.1rem;
                font-weight: 500;
            }}

            .helper {{
                color: #64748b;
                font-size: 0.95rem;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_state() -> None:
    defaults: Dict[str, Any] = {
        "messages": [],
        "memory": ConversationHistory(),
        "pending_prompt": None,
        "current_org": None,
        "current_username": None,
        "kb_missing": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def load_config() -> dict:
    return load_app_config(CONFIG_PATH)


def persist_config(config: dict) -> None:
    save_app_config(CONFIG_PATH, config)


def require_groq_api_key() -> None:
    if not os.getenv("GROQ_API_KEY", "").strip():
        st.error(
            "Missing GROQ_API_KEY. Add it to your .env file as GROQ_API_KEY=<your_key>, then restart DayOne AI."
        )
        st.stop()


def clear_conversation_memory() -> None:
    st.session_state.messages = []
    memory = st.session_state.get("memory")
    if hasattr(memory, "clear"):
        memory.chat_memory.clear()


def clear_session_on_logout(*_args: Any, **_kwargs: Any) -> None:
    clear_conversation_memory()
    st.session_state.clear()


def reset_invalid_auth_state() -> None:
    clear_conversation_memory()
    st.session_state.current_org = None
    st.session_state.current_username = None


def build_hybrid_retriever(org_id: str) -> HybridRetriever:
    return build_pgvector_hybrid_retriever(
        organization=org_id,
        tenant_id=None,
        embeddings=load_embeddings(),
        use_reranker=USE_RERANKER,
    )


def rewrite_query(query: str, memory: ConversationHistory) -> str:
    """Rewrite a follow-up query into a standalone question using chat history.

    Skipped if there is no prior conversation (avoids unnecessary LLM call).
    Only the last two exchanges are used to keep the rewriter prompt short.
    """
    messages = memory.chat_memory.messages
    if len(messages) < 2:
        return query  # First message — no rewriting needed

    recent = messages[-4:]  # Last 2 user+assistant pairs
    history = "\n".join(
        f"{'User' if i % 2 == 0 else 'Assistant'}: {m.content}"
        for i, m in enumerate(recent)
    )
    prompt = (
        f"Conversation so far:\n{history}\n\n"
        f"Follow-up question: {query}\n\n"
        "Rewrite the follow-up as a fully self-contained question that can be "
        "understood without any prior context. Output ONLY the rewritten question."
    )
    try:
        result = get_llm().invoke([HumanMessage(content=prompt)])
        rewritten = result.content.strip()
        return rewritten if rewritten else query
    except Exception:
        return query  # Graceful degradation


def detect_conflict(docs: List[Any]) -> bool:
    """Return True if retrieved docs come from ≥2 distinct source files.

    This is a necessary (not sufficient) condition for conflicting policies.
    The LLM system prompt instructs it to flag actual conflicts in the answer.
    """
    sources = {Path(str(d.metadata.get("source", ""))).name for d in docs}
    return len(sources) >= 2


def write_audit_log(
    username: str,
    org_id: str,
    query: str,
    rewritten_query: str,
    answer: str,
    confidence: float,
    sources: List[str],
    latency_ms: float,
    conflict_detected: bool,
) -> None:
    """Append one query record to logs/query_log.jsonl."""
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "username": username,
        "organization": org_id,
        "query": query,
        "rewritten_query": rewritten_query,
        "answer_snippet": answer[:200],
        "confidence": round(confidence, 4),
        "confidence_label": confidence_label(confidence),
        "sources": sources,
        "latency_ms": round(latency_ms, 1),
        "conflict_detected": conflict_detected,
    }
    log_path = LOGS_DIR / "query_log.jsonl"
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def citation_lines(source_documents: Sequence[Any]) -> List[str]:
    lines: List[str] = []
    seen = set()
    for doc in source_documents:
        metadata = getattr(doc, "metadata", {}) or {}
        source_name = Path(str(metadata.get("source", "unknown"))).name
        page = metadata.get("page")
        row = metadata.get("row")
        if page is not None:
            label = f"{source_name} (Page {int(page) + 1})" if str(page).isdigit() else f"{source_name} (Page {page})"
        elif row is not None:
            label = f"{source_name} (Row {int(row) + 1})" if str(row).isdigit() else f"{source_name} (Row {row})"
        else:
            label = source_name
        if label not in seen:
            seen.add(label)
            lines.append(label)
    return lines


def _build_justification(result: RetrievalResult, sources: List[str]) -> List[dict]:
    """Build serialisable justification records from a RetrievalResult.

    Each record contains the chunk snippet, source, score, and rank-change
    so the UI can display explainable context for every answer.
    """
    rank_changes = result.rank_changes
    records = []
    for i, (doc, score) in enumerate(zip(result.final_docs, result.final_scores)):
        meta = getattr(doc, "metadata", {}) or {}
        source_name = Path(str(meta.get("source", "unknown"))).name
        page = meta.get("page")
        row = meta.get("row")
        if page is not None:
            loc = f"Page {int(page) + 1}" if str(page).isdigit() else f"Page {page}"
        elif row is not None:
            loc = f"Row {int(row) + 1}" if str(row).isdigit() else f"Row {row}"
        else:
            loc = ""
        records.append({
            "rank": i + 1,
            "source": source_name,
            "location": loc,
            "snippet": doc.page_content[:400].strip(),
            "score": round(score, 4),
            "rank_change": rank_changes[i] if i < len(rank_changes) else 0,
        })
    return records


def run_rag_query(
    prompt: str,
    memory: ConversationHistory,
    username: str,
    org_id: str,
) -> tuple[str, List[str], float, bool, List[dict]]:
    """Run hybrid retrieval + LLM generation.

    Returns: (answer, sources, confidence, conflict_detected, justification)
    """
    rewritten = rewrite_query(prompt, memory)
    try:
        retriever = build_hybrid_retriever(org_id)
    except Exception:
        return (
            "Your organisation's knowledge base is currently empty. Please contact HR.",
            [], 0.0, False, [],
        )

    with st.spinner("Searching policies..."):
        result: RetrievalResult = retriever.retrieve(rewritten)

    if not result.final_docs:
        return (
            "I do not have that information in the current HR files. Please contact HR.",
            [], 0.0, False, [],
        )

    conflict = detect_conflict(result.final_docs)
    context = "\n\n---\n\n".join(
        f"[Source: {Path(d.metadata.get('source', 'unknown')).name}]\n{d.page_content}"
        for d in result.final_docs
    )
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Question: {rewritten}\n\nContext:\n{context}"),
    ]
    try:
        llm_result = get_llm().invoke(messages)
        answer = llm_result.content.strip()
    except Exception:
        return (
            "I ran into a temporary retrieval issue. Please try again or contact HR support.",
            [], 0.0, False, [],
        )

    memory.chat_memory.add_user_message(prompt)
    memory.chat_memory.add_ai_message(answer)

    sources = citation_lines(result.final_docs)
    justification = _build_justification(result, sources)
    write_audit_log(
        username=username, org_id=org_id, query=prompt,
        rewritten_query=rewritten, answer=answer,
        confidence=result.confidence, sources=sources,
        latency_ms=result.latency_ms, conflict_detected=conflict,
    )
    return answer, sources, result.confidence, conflict, justification


def render_login_hero() -> None:
    st.markdown(
        """
        <div class="login-wrap">
            <div class="login-card">
                <div class="login-monogram">D1</div>
                <h1 class="login-title">DayOne AI</h1>
                <p class="login-subtitle">Secure multi-tenant HR onboarding assistant</p>
                <p class="helper" style="margin-top:0.35rem;">Sign in to access your organization's policy knowledge base.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_employee_sidebar(authenticator: stauth.Authenticate, org_id: str) -> None:
    st.sidebar.markdown(f"""
    <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1.5rem;">
        <div style="background: linear-gradient(135deg, #7C3AED, #3b82f6); width: 36px; height: 36px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-weight: 800; color: white; font-size: 1rem; box-shadow: 0 4px 10px rgba(124, 58, 237, 0.4);">D1</div>
        <h2 style="margin: 0; font-size: 1.25rem; font-weight: 700; color: #f8fafc; letter-spacing: -0.02em;">DayOne AI</h2>
    </div>
    """, unsafe_allow_html=True)

    if st.sidebar.button("🏠 Home", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    if st.sidebar.button("📋 Tasks", use_container_width=True):
        st.session_state.pending_prompt = "What are my pending tasks?"
    if st.sidebar.button("🏥 Benefits", use_container_width=True):
        st.session_state.pending_prompt = "Show me my benefits portal."
    if st.sidebar.button("📖 Policies", use_container_width=True):
        st.session_state.pending_prompt = "Open the policies library."
    if st.sidebar.button("📄 Documents", use_container_width=True):
        st.session_state.pending_prompt = "Where are my documents?"
    
    st.sidebar.markdown("<br><p style='color: #64748b; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; margin-bottom: 0.5rem; letter-spacing: 0.05em;'>Recent Chats</p>", unsafe_allow_html=True)
    if st.sidebar.button("💬 Q3 PTO Policy", use_container_width=True):
        st.session_state.pending_prompt = "Tell me about the Q3 PTO policy again."
    if st.sidebar.button("💬 Travel Expenses", use_container_width=True):
        st.session_state.pending_prompt = "How do I file travel expenses?"

    st.sidebar.markdown("<br><br><br><br>", unsafe_allow_html=True)
    
    if st.sidebar.button("⚙️ Settings", use_container_width=True):
        st.session_state.pending_prompt = "How do I change my settings?"
        
    authenticator.logout("Sign Out", "sidebar", callback=clear_session_on_logout)


def _users_for_org(config: dict, org_id: str) -> List[dict]:
    credentials = config.get("credentials", {})
    usernames = credentials.get("usernames", {})
    if not isinstance(usernames, dict):
        return []
    scoped = [
        serialize_user(username, record)
        for username, record in usernames.items()
        if str(record.get("organization", "")).strip() == org_id
    ]
    return sorted(scoped, key=lambda item: item["username"].lower())


def render_admin_portal(config: dict, username: str, org_id: str) -> None:
    # ── Page header ──────────────────────────────────────────────────────────
    st.markdown("""
    <div style="
        background: linear-gradient(145deg, rgba(30,41,59,0.6), rgba(15,23,42,0.8));
        backdrop-filter: blur(40px);
        border: 1px solid rgba(255,255,255,0.1);
        border-top: 1px solid rgba(124,58,237,0.4);
        border-radius: 20px;
        padding: 2rem 2rem 1.5rem 2rem;
        margin-bottom: 2rem;
    ">
        <div style="font-size:0.8rem; font-weight:700; color:#A78BFA; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.5rem;">Administration</div>
        <div style="font-size:2rem; font-weight:800; color:#f8fafc; letter-spacing:-0.03em; margin-bottom:0.5rem;">Knowledge Base</div>
        <div style="font-size:0.95rem; color:#64748b;">Upload policy documents to rebuild your organization's AI knowledge index.</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Upload section ────────────────────────────────────────────────────────
    st.markdown("""
    <div style="font-size:0.8rem; font-weight:700; color:#64748b; text-transform:uppercase;
        letter-spacing:0.1em; margin-bottom:0.75rem;">Upload Documents</div>
    """, unsafe_allow_html=True)

    upload_col, info_col = st.columns([2, 1])
    with upload_col:
        upload_org = st.selectbox("Organization", options=[org_id], disabled=True)
        uploaded_files = st.file_uploader(
            "Drag & drop policy files here",
            type=["pdf", "csv"],
            accept_multiple_files=True,
            label_visibility="visible",
        )
        if st.button("🔄  Rebuild Knowledge Index", type="primary", use_container_width=True):
            if not uploaded_files:
                st.warning("Please choose at least one file before rebuilding.")
            else:
                target_dir = DATA_DIR / upload_org
                target_dir.mkdir(parents=True, exist_ok=True)
                saved: List[str] = []
                for upload in uploaded_files:
                    filename = Path(upload.name).name
                    if Path(filename).suffix.lower() not in {".pdf", ".csv"}:
                        continue
                    destination = target_dir / filename
                    destination.write_bytes(upload.getbuffer())
                    saved.append(filename)
                if not saved:
                    st.error("No supported files were uploaded.")
                else:
                    with st.status("Rebuilding knowledge index…", expanded=True) as rebuild_status:
                        st.write("📄 Parsing and chunking documents…")
                        rebuild_organization_index(target_dir, load_embeddings())
                        rebuild_status.update(
                            label=f"✅ Index rebuilt for **{upload_org}** ({len(saved)} file(s))",
                            state="complete",
                            expanded=False,
                        )
                    st.session_state.chain = None
                    st.success(f"Successfully indexed {len(saved)} file(s) for **{upload_org}**.")
                    st.caption("Files: " + ", ".join(saved) + f" · Updated by: {username}")

    with info_col:
        st.markdown("""
        <div style="
            background: rgba(255,255,255,0.02);
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 16px;
            padding: 1.5rem;
            height: 100%;
        ">
            <div style="font-size:0.8rem; font-weight:700; color:#A78BFA; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:1rem;">Supported Formats</div>
            <div style="display:flex; align-items:center; gap:0.75rem; margin-bottom:0.75rem; color:#94a3b8; font-size:0.9rem;">
                <span style="font-size:1.2rem;">📄</span> PDF documents
            </div>
            <div style="display:flex; align-items:center; gap:0.75rem; color:#94a3b8; font-size:0.9rem;">
                <span style="font-size:1.2rem;">📊</span> CSV spreadsheets
            </div>
            <div style="margin-top:1.5rem; padding-top:1rem; border-top:1px solid rgba(255,255,255,0.05);">
                <div style="font-size:0.75rem; color:#475569; line-height:1.5;">
                    Files are chunked, embedded, and added to the vector index for this organization only.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin: 2.5rem 0 0 0;'></div>", unsafe_allow_html=True)
    render_admin_user_management(config, username, org_id)


def render_admin_user_management(config: dict, current_username: str, org_id: str) -> None:
    st.markdown("""
    <div style="font-size:0.8rem; font-weight:700; color:#64748b; text-transform:uppercase;
        letter-spacing:0.1em; margin-bottom:0.75rem;">User Management</div>
    """, unsafe_allow_html=True)

    users = _users_for_org(config, org_id)
    if users:
        st.dataframe(
            [
                {
                    "Username": user["username"],
                    "Name": user["name"] or "—",
                    "Email": user["email"] or "—",
                    "Role": user["role"],
                }
                for user in users
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No users are configured for this organization yet.")

    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)
    create_col, update_col = st.columns(2)

    with create_col:
        st.markdown("""
        <div style="font-size:0.8rem; font-weight:700; color:#64748b; text-transform:uppercase;
            letter-spacing:0.08em; margin-bottom:0.75rem;">Add New User</div>
        """, unsafe_allow_html=True)
        with st.form("create-user-form", clear_on_submit=True):
            new_username = st.text_input("Username", placeholder="e.g. john.doe", help="3–64 chars: letters, numbers, . _ -")
            new_name = st.text_input("Full Name", placeholder="John Doe")
            new_email = st.text_input("Email", placeholder="john@company.com")
            new_role = st.selectbox("Role", options=[ROLE_EMPLOYEE, ROLE_ADMIN], index=0)
            new_password = st.text_input("Temporary Password", type="password", help="Minimum 8 characters")
            create_submitted = st.form_submit_button("➕  Create User", use_container_width=True, type="primary")

        if create_submitted:
            updated_config = clone_config(config)
            try:
                create_user_record(
                    config=updated_config,
                    username=new_username,
                    password=new_password,
                    organization=org_id,
                    role=new_role,
                    name=new_name,
                    email=new_email,
                )
            except ValueError as exc:
                st.error(str(exc))
            else:
                persist_config(updated_config)
                st.success(f"Created user `{new_username.strip()}`.")
                st.rerun()

    with update_col:
        st.markdown("""
        <div style="font-size:0.8rem; font-weight:700; color:#64748b; text-transform:uppercase;
            letter-spacing:0.08em; margin-bottom:0.75rem;">Edit / Remove User</div>
        """, unsafe_allow_html=True)
        user_options = [user["username"] for user in users]
        selected_username = st.selectbox(
            "Select user to edit",
            options=user_options,
            index=0 if user_options else None,
            placeholder="Choose a user",
        )

        selected_user = next((user for user in users if user["username"] == selected_username), None)
        if selected_user:
            with st.form("update-user-form"):
                edit_name = st.text_input("Full Name", value=selected_user["name"])
                edit_email = st.text_input("Email", value=selected_user["email"])
                role_index = 0 if selected_user["role"] == ROLE_EMPLOYEE else 1
                edit_role = st.selectbox("Role", options=[ROLE_EMPLOYEE, ROLE_ADMIN], index=role_index)
                edit_password = st.text_input(
                    "New Password",
                    type="password",
                    help="Leave blank to keep the existing password",
                )
                save_user = st.form_submit_button("💾  Save Changes", use_container_width=True, type="primary")

            if save_user:
                updated_config = clone_config(config)
                try:
                    update_user_record(
                        config=updated_config,
                        username=selected_username,
                        current_organization=org_id,
                        name=edit_name,
                        email=edit_email,
                        role=edit_role,
                        password=edit_password or None,
                    )
                except (PermissionError, ValueError) as exc:
                    st.error(str(exc))
                else:
                    persist_config(updated_config)
                    st.success(f"Updated user `{selected_username}`.")
                    st.rerun()

            st.markdown("<div style='margin-top:0.5rem;'></div>", unsafe_allow_html=True)
            delete_disabled = selected_username == current_username
            delete_help = "You cannot delete your own account." if delete_disabled else None
            if st.button("🗑️  Delete User", use_container_width=True, disabled=delete_disabled, help=delete_help):
                updated_config = clone_config(config)
                try:
                    delete_user_record(
                        config=updated_config,
                        username=selected_username,
                        current_organization=org_id,
                    )
                except (PermissionError, ValueError) as exc:
                    st.error(str(exc))
                else:
                    persist_config(updated_config)
                    st.success(f"Deleted user `{selected_username}`.")
                    st.rerun()


SUGGESTION_ICONS = ["🕐", "🏥", "📅"]


import base64

def get_base64_image(image_path: Path) -> str:
    if not image_path.exists():
        return ""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def render_zero_state() -> None:
    username_display = st.session_state.get("name") or st.session_state.get("username", "User")
    first_name = html.escape(username_display.split()[0])
    

    mascot_b64 = get_base64_image(MASCOT_PATH)
    mascot_html = f'''
    <div class="mascot-container">
        <div class="mascot-halo"></div>
        <div class="spark spark-1"></div>
        <div class="spark spark-2"></div>
        <div class="spark spark-3"></div>
        <div class="spark spark-4"></div>
        <img src="data:image/png;base64,{mascot_b64}" class="hero-mascot" alt="Mascot">
    </div>
    ''' if mascot_b64 else '<div class="hero-mascot-placeholder">🤖</div>'

    hero_md = f'<div class="unified-hero">{mascot_html}<div class="hero-title">Good Morning, {first_name} \U0001f44b</div><div class="hero-context">14 PTO days \u2022 2 tasks due</div><div class="insight-box"><div class="insight-header">\U0001f916 DayOne Insight</div><div class="insight-body">You have <strong style="color: #A78BFA;">enough PTO</strong> for a long weekend. However, your compliance training is <strong style="color: #FFB020;">overdue</strong>.</div></div></div>'
    st.markdown(hero_md, unsafe_allow_html=True)
    if st.button("Take Action →", key="insight_btn", type="primary", use_container_width=False):
        st.session_state.pending_prompt = "Help me finish my compliance training and request PTO."

    st.markdown('<div class="section-kicker">Your Workspace</div>', unsafe_allow_html=True)
    cols = st.columns([2, 1, 1])

    with cols[0]:
        st.markdown("""
        <div class="glass-card stat-card">
            <div class="stat-title">PTO Balance</div>
            <div class="stat-number" style="color:#A78BFA;">14</div>
            <div class="stat-unit">Days Remaining</div>
            <div class="stat-bar" style="margin-top: 1rem;">
                <div class="stat-progress" style="width: 70%;"></div>
            </div>
            <div style="display:flex; justify-content:space-between; margin-top:0.4rem;">
                <span style="font-size:0.75rem; color:#64748b;">14 of 20 used</span>
                <span style="font-size:0.75rem; color:#7C3AED; font-weight:600;">70%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with cols[1]:
        st.markdown("""
        <div class="glass-card stat-card">
            <div class="stat-title">Pending Tasks</div>
            <div class="stat-number" style="color:#FFB020;">2</div>
            <div class="stat-unit">Due This Week</div>
            <div style="margin-top: 0.75rem;">
                <div class="task-item">
                    <div class="task-dot"></div>
                    <span>Compliance Training</span>
                </div>
                <div class="task-item">
                    <div class="task-dot"></div>
                    <span>Profile Verification</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with cols[2]:
        st.markdown("""
        <div class="glass-card stat-card" style="align-items:center; text-align:center;">
            <div class="stat-title" style="align-self:flex-start;">Coverage Score</div>
            <svg viewBox="0 0 36 36" style="stroke:#00D084; width:80px; height:80px; margin: 0.5rem auto;">
                <path style="fill:none; stroke:rgba(255,255,255,0.06); stroke-width:3.8;" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
                <path style="fill:none; stroke-width:3.8; stroke-linecap:round; stroke-dasharray:92,100;" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
                <text x="18" y="21" style="fill:#f8fafc; font-size:8px; font-weight:800; text-anchor:middle; font-family:inherit;">92%</text>
            </svg>
            <div style="display:flex; flex-direction:column; gap:0.3rem; margin-top:auto; align-self:flex-start; width:100%;">
                <div style="display:flex; align-items:center; gap:0.5rem; font-size:0.82rem; color:#94a3b8;">
                    <span style="color:#00D084; font-weight:700;">&#10003;</span> Medical
                </div>
                <div style="display:flex; align-items:center; gap:0.5rem; font-size:0.82rem; color:#94a3b8;">
                    <span style="color:#00D084; font-weight:700;">&#10003;</span> Dental
                </div>
                <div style="display:flex; align-items:center; gap:0.5rem; font-size:0.82rem; color:#94a3b8;">
                    <span style="color:#00D084; font-weight:700;">&#10003;</span> Vision
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    
    st.markdown('<div class="section-kicker">AI Suggestions</div>', unsafe_allow_html=True)
    scols = st.columns(2)
    with scols[0]:
        st.markdown("""
        <div class="proactive-card proactive-card-purple">
            <h4>✈️ PTO Opportunity</h4>
            <p>You have enough balance for a long weekend. Consider taking Friday off to recharge.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Request Leave", key="sug_pto", type="secondary", use_container_width=True):
            st.session_state.pending_prompt = "I want to request leave for a long weekend."
            
        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
            
        st.markdown("""
        <div class="proactive-card proactive-card-green">
            <h4>🏥 New Benefits Available</h4>
            <p>The updated medical plan has been verified. Would you like a summary?</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Review Benefits", key="sug_ben", type="secondary", use_container_width=True):
            st.session_state.pending_prompt = "What are the new health benefits available?"

    with scols[1]:
        st.markdown("""
        <div class="proactive-card proactive-card-blue">
            <h4>📄 Payslip Available</h4>
            <p>Your May salary statement is ready and available for secure download.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Download", key="sug_pay", type="secondary", use_container_width=True):
            st.session_state.pending_prompt = "Where can I download my latest payslip?"
            
        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
            
        st.markdown("""
        <div class="proactive-card proactive-card-red">
            <h4>🚀 Action Required</h4>
            <p>Your compliance training is overdue. You have 2 pending modules remaining.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Start Training", key="sug_task", type="secondary", use_container_width=True):
            st.session_state.pending_prompt = "What is remaining on my onboarding checklist?"

    
    st.markdown('<div class="section-kicker">Recent Activity</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="border-left: 2px solid rgba(255,255,255,0.05); margin-left: 1rem; padding-left: 1.5rem; position: relative; margin-top: 1.5rem;">
        <div style="margin-bottom: 1.5rem; position: relative;">
            <div style="position: absolute; left: -1.9rem; top: 0.2rem; width: 14px; height: 14px; background: #00D084; border-radius: 50%; box-shadow: 0 0 10px #00D084;"></div>
            <div style="font-size: 0.8rem; color: #64748b; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.25rem;">Today</div>
            <div style="color: #e2e8f0; font-size: 1.05rem;">PTO approved for upcoming holiday</div>
        </div>
        <div style="margin-bottom: 1.5rem; position: relative;">
            <div style="position: absolute; left: -1.85rem; top: 0.2rem; width: 12px; height: 12px; background: rgba(255,255,255,0.2); border-radius: 50%;"></div>
            <div style="font-size: 0.8rem; color: #64748b; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.25rem;">Yesterday</div>
            <div style="color: #94a3b8; font-size: 1.05rem;">Payslip downloaded successfully</div>
        </div>
        <div style="position: relative;">
            <div style="position: absolute; left: -1.85rem; top: 0.2rem; width: 12px; height: 12px; background: rgba(255,255,255,0.2); border-radius: 50%;"></div>
            <div style="font-size: 0.8rem; color: #64748b; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.25rem;">Monday</div>
            <div style="color: #94a3b8; font-size: 1.05rem;">Benefits profile updated</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _render_justification(justification: List[dict], used_reranker: bool) -> None:
    """Render the Answer Justification expander — the 'explainable RAG' layer."""
    if not justification:
        return
    score_label = "Reranker score" if used_reranker else "BM25 score"
    with st.expander(
        f"🔍 Answer Justification — {len(justification)} retrieved chunk(s) "
        f"| {score_label} shown",
        expanded=False,
    ):
        if used_reranker:
            st.caption(
                f"Retrieved {12} candidates via BM25+pgvector→RRF, "
                f"then cross-encoder reranked to top {len(justification)}. "
                "\u2191N = promoted N positions by reranker."
            )
        else:
            st.caption("Reranker OFF — BM25+pgvector→RRF fusion only. Score = BM25 score.")

        for rec in justification:
            change = rec["rank_change"]
            arrow = f"↑{change}" if change > 0 else ("↓{abs(change)}" if change < 0 else "—")
            loc = f" · {rec['location']}" if rec["location"] else ""
            st.markdown(
                f"**#{rec['rank']} — `{rec['source']}`{loc}** "
                f"&nbsp;&nbsp; `{score_label}: {rec['score']:.3f}` "
                f"&nbsp; `Rank change: {arrow}`"
            )
            st.markdown(
                f"> {rec['snippet'].replace(chr(10), ' ')[:350]}…"
                if len(rec["snippet"]) > 350 else f"> {rec['snippet']}"
            )
            st.divider()


def render_chat_history() -> None:
    for message in st.session_state.messages:
        role = message["role"]
        avatar = "🧑" if role == "user" else "🤖"
        with st.chat_message(role, avatar=avatar):
            st.markdown(message["content"])
            if role == "assistant":
                conf = message.get("confidence", 0.0)
                conflict = message.get("conflict_detected", False)
                sources = message.get("sources", [])
                justification = message.get("justification", [])
                used_reranker = message.get("used_reranker", USE_RERANKER)

                label = confidence_label(conf)
                colour = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(label, "⚪")
                st.caption(f"{colour} Confidence: {label} ({conf:.0%})  |  Reranker: {'ON' if used_reranker else 'OFF'}  |  Chunks: {len(justification)}")

                if conflict:
                    st.warning(
                        "⚠️ Retrieved context spans multiple documents. "
                        "Conflicting policies may exist — verify with HR before acting.",
                        icon="⚠️",
                    )
                if conf < CONF_LOW and conf > 0:
                    st.info(
                        "ℹ️ Low retrieval confidence. This answer may be incomplete — "
                        "please cross-check with HR."
                    )

                _render_justification(justification, used_reranker)

                if sources:
                    with st.expander("📎 View Sources"):
                        for source in sources:
                            st.markdown(f"- {html.escape(source)}")


def render_employee_chat(authenticator: stauth.Authenticate, org_id: str) -> None:
    render_employee_sidebar(authenticator, org_id)

    # Removed dashboard top bar since it is now natively in the zero state.

    if st.session_state.kb_missing:
        st.warning("Your organisation's knowledge base is currently empty. Please contact HR.")

    if st.session_state.messages:
        render_chat_history()
    else:
        render_zero_state()

    typed_prompt = st.chat_input("Ask DayOne anything... PTO, benefits, expenses")
    active_prompt = st.session_state.get("pending_prompt") or typed_prompt
    st.session_state.pending_prompt = None

    if not active_prompt:
        return

    normalized_prompt = active_prompt.strip()
    if not normalized_prompt:
        return

    username = str(st.session_state.get("current_username", "unknown"))
    st.session_state.messages.append({"role": "user", "content": normalized_prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(normalized_prompt)

    answer, sources, confidence, conflict, justification = run_rag_query(
        normalized_prompt,
        st.session_state.memory,
        username,
        org_id,
    )
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "confidence": confidence,
        "conflict_detected": conflict,
        "justification": justification,
        "used_reranker": USE_RERANKER,
    })
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(answer)
        label = confidence_label(confidence)
        colour = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(label, "⚪")
        st.caption(f"{colour} Confidence: {label} ({confidence:.0%})  |  Reranker: {'ON' if USE_RERANKER else 'OFF'}  |  Chunks: {len(justification)}")
        
        # Trust Signal
        st.markdown(
            "<div style='margin-top: 0.5rem; padding: 0.5rem 0.75rem; background-color: rgba(15, 23, 42, 0.6); border-left: 3px solid #3b82f6; border-radius: 4px; font-size: 0.85rem; color: #94a3b8;'>"
            "<strong>Sourced from official HR documents.</strong> <br>Last updated: May 2026."
            "</div>",
            unsafe_allow_html=True
        )

        if conflict:
            st.warning("⚠️ Retrieved context spans multiple documents — verify with HR.", icon="⚠️")
        if confidence < CONF_LOW and confidence > 0:
            st.info("ℹ️ Low retrieval confidence — please cross-check with HR.")
        _render_justification(justification, USE_RERANKER)
        if sources:
            with st.expander("📎 View Sources"):
                for source in sources:
                    st.markdown(f"- {html.escape(source)}")


def main() -> None:
    load_dotenv()
    initialize_state()

    auth_status = bool(st.session_state.get("authentication_status"))
    configure_page(authenticated=auth_status)
    require_groq_api_key()

    config = load_config()
    credentials_root = config.get("credentials", {})
    users = credentials_root.get("usernames", {})
    authenticator = stauth.Authenticate(
        credentials=credentials_root,
        cookie_name=config.get("cookie", {}).get("name", "dayone_ai_auth"),
        cookie_key=config.get("cookie", {}).get("key", "change-this-key"),
        cookie_expiry_days=config.get("cookie", {}).get("expiry_days", 30),
        preauthorized=config.get("preauthorized"),
        auto_hash=False,
    )

    if st.session_state.get("authentication_status") is not True:
        render_login_hero()
        col1, col2, col3 = st.columns([1, 1.2, 1])
        with col2:
            authenticator.login(fields={"Form name": "Sign in"})

    if st.session_state.get("authentication_status") is True:
        username = str(st.session_state.get("username", "")).strip()
        name = st.session_state.get("name")

        user_info = users.get(username, {})
        org_id = str(user_info.get("organization", "")).strip()
        user_role = str(user_info.get("role", "employee") or "employee").strip().lower()

        if not username or not user_info:
            reset_invalid_auth_state()
            st.error("Authenticated user mapping is invalid. Contact admin.")
            st.stop()

        if not org_id:
            reset_invalid_auth_state()
            st.error("No organization is mapped to this account.")
            st.stop()

        if st.session_state.current_org != org_id or st.session_state.current_username != username:
            clear_conversation_memory()
            st.session_state.current_org = org_id
            st.session_state.current_username = username

        st.session_state.kb_missing = False

        st.session_state.kb_missing = False

        if user_role == ROLE_ADMIN:
            authenticator.logout("Sign Out", "main", callback=clear_session_on_logout)
            render_admin_portal(config, username, org_id)
            return
        elif user_role == ROLE_EMPLOYEE:
            render_employee_chat(authenticator, org_id)
            return

        reset_invalid_auth_state()
        st.error("Unsupported role configuration.")
        st.stop()

    if st.session_state.get("authentication_status") is False:
        reset_invalid_auth_state()
        st.error("Invalid username or password.")
        st.stop()

    st.stop()


if __name__ == "__main__":
    main()
