"""
Mana Image — AI Image Generator
Streamlit app powered by HuggingFace FLUX.1-schnell
"""

import io
import os
import base64
import time
from datetime import datetime
from pathlib import Path

import streamlit as st
from PIL import Image
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
HF_MODEL_URL = (
    "https://api-inference.huggingface.co/models/"
    "black-forest-labs/FLUX.1-schnell"
)

APP_DIR = Path(__file__).parent
CSS_PATH = APP_DIR / "assets" / "style.css"

# ---------------------------------------------------------------------------
# Page Config — must be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Mana Image — AI Image Generator",
    page_icon="🖼️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# =========================
# STARTUP SPLASH SCREEN
# =========================

if "startup_done" not in st.session_state:

    splash = st.empty()

    boot_id = f"{__import__('random').randint(100000000, 999999999)}"

    status_lines = [
        "[+] Starting ...",
        "[+] Loading model weights ...",
        "[+] Connecting to inference API ...",
        "[+] Warming up renderer ...",
        "[+] Ready.",
    ]

    splash_css = (
        "<style>"
        ".splash-wrap{height:100vh;display:flex;align-items:center;"
        "justify-content:center;}"
        ".splash-box{width:100%;max-width:560px;background:rgba(3,3,5,.92);"
        "border:1px solid var(--border-purple, #6c63ff);padding:25px;"
        "position:relative;box-shadow:0 0 30px rgba(122,31,162,.2),"
        "inset 0 0 15px rgba(122,31,162,.1);}"
        ".splash-box::before{content:'';position:absolute;top:0;left:0;"
        "right:0;height:3px;background:var(--primary-pink, #ff6ec7);"
        "box-shadow:0 0 15px var(--primary-pink, #ff6ec7);}"
        ".splash-top{display:flex;justify-content:space-between;"
        "color:var(--text-muted, #8fa3c7);font-size:.75rem;margin-bottom:20px;}"
        ".splash-title{color:var(--primary-cyan, #00e5ff);font-size:1.6rem;"
        "font-weight:700;letter-spacing:2px;"
        "text-shadow:0 0 15px rgba(0,240,255,.5);margin-bottom:20px;}"
        ".splash-bar-track{width:100%;height:6px;background:rgba(108,99,255,.18);"
        "border-radius:4px;overflow:hidden;}"
        ".splash-bar-fill{height:100%;background:var(--primary-cyan, #00e5ff);"
        "box-shadow:0 0 10px var(--primary-cyan, #00e5ff);"
        "transition:width 0.05s linear;}"
        ".splash-status{color:var(--text-main, #e6f1ff);font-size:.82rem;"
        "letter-spacing:1px;margin-top:16px;}"
        "</style>"
    )

    def render_splash(pct: int, status: str) -> str:
        return (
            splash_css +
            '<div class="splash-wrap"><div class="splash-box">'
            '<div class="splash-top">'
            f'<span>Created by Maneesh</span><span>{boot_id}</span>'
            '</div>'
            '<div class="splash-title">&gt; INITIALIZING MANA ...</div>'
            f'<div class="splash-bar-track"><div class="splash-bar-fill" '
            f'style="width:{pct}%;"></div></div>'
            f'<div class="splash-status">{status}</div>'
            '</div></div>'
        )

    for i in range(100):
        pct = i + 1
        status = status_lines[min(i // 21, len(status_lines) - 1)]
        splash.markdown(render_splash(pct, status), unsafe_allow_html=True)
        time.sleep(0.012)

    splash.empty()

    st.session_state.startup_done = True

    st.rerun()


# ---------------------------------------------------------------------------
# Inject Custom CSS
# ---------------------------------------------------------------------------

def load_css():
    if CSS_PATH.exists():
        with open(CSS_PATH, "r", encoding="utf-8") as f:
            css = f.read()

        st.markdown(
            f"<style>{css}</style>",
            unsafe_allow_html=True
        )

load_css()

st.markdown(
    '<div class="mana-header"><div class="terminal-box">'
    '<div class="terminal-top"><span>Created by Maneesh</span>'
    '<span>&gt; MANA IMAGE v2</span></div>'
    '<div class="terminal-title">&gt; MANA IMAGE</div>'
    '</div></div>',
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------

if "gallery" not in st.session_state:
    st.session_state.gallery = []  # list of dicts: {prompt, image_bytes, timestamp}

if "generating" not in st.session_state:
    st.session_state.generating = False

# ---------------------------------------------------------------------------
# API Helper
# ---------------------------------------------------------------------------

from huggingface_hub import InferenceClient

token_configured = bool(HF_API_TOKEN) and HF_API_TOKEN != "your_huggingface_token_here"

client = None
if token_configured:
    client = InferenceClient(
        provider="hf-inference",
        api_key=HF_API_TOKEN
    )


def generate_image(prompt: str) -> bytes:
    """Call the HF Inference API and return PNG bytes. Raises on failure."""
    if client is None:
        raise ValueError(
            "HuggingFace API token not configured. "
            "Add your token to the `.env` file and restart the app."
        )

    try:
        image = client.text_to_image(
            prompt,
            model="black-forest-labs/FLUX.1-schnell"
        )
    except Exception as exc:  # network / API errors from huggingface_hub
        raise RuntimeError(f"Image generation failed: {exc}") from exc

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def image_to_base64(image_bytes: bytes) -> str:
    """Convert raw image bytes to a base64-encoded string."""
    return base64.b64encode(image_bytes).decode("utf-8")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    # Brand
    st.markdown("# MANA IMAGE")
    st.markdown("---")

    # API Status
    st.markdown("##### System Status")

    if token_configured:
        masked_token = HF_API_TOKEN[:4] + "••••" + HF_API_TOKEN[-4:]
        st.markdown(f" &nbsp; Token: `{masked_token}`")
    else:
        st.markdown(" &nbsp; Token: **Not configured**")
        st.markdown(
            '<span style="color: #8fa3c7; font-size: 0.78rem;">'
            "Add your HF token to the `.env` file and restart."
            "</span>",
            unsafe_allow_html=True,
        )

    st.markdown(f" &nbsp; Model: `FLUX.1-schnell`")
    st.markdown("---")

    # Gallery History
    st.markdown("##### Generation History")

    gallery = st.session_state.gallery
    if gallery:
        for i, item in enumerate(reversed(gallery)):
            idx = len(gallery) - 1 - i
            prompt_preview = item["prompt"][:40] + ("..." if len(item["prompt"]) > 40 else "")
            with st.expander(f"🖼 {prompt_preview}", expanded=False):
                st.image(item["image_bytes"], use_container_width=True)
                st.caption(f"_{item['timestamp']}_")
                st.download_button(
                    label="⬇ Download",
                    data=item["image_bytes"],
                    file_name=f"mana_image_{idx}.png",
                    mime="image/png",
                    key=f"sidebar_dl_{idx}",
                )
    else:
        st.markdown(
            '<span style="color: #5a6b8a; font-size: 0.78rem;">No generations yet.</span>',
            unsafe_allow_html=True,
        )

    # Clear History
    if st.button("[ Clear History ]", use_container_width=True, key="clear_btn"):
        st.session_state.gallery = []
        st.rerun()

# ---------------------------------------------------------------------------
# Main Content
# ---------------------------------------------------------------------------

# Header
st.markdown(
    """
    <div style="text-align: center; padding: 20px 0 10px 0;">
        <h1 style="margin-bottom: 4px;">MANA IMAGE</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# Prompt Input
prompt = st.text_area(
    "Describe the image you want to generate",
    placeholder="Enter prompt for image",
    height=100,
    key="prompt_input",
    label_visibility="collapsed",
)

# Generate Button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    generate_clicked = st.button(
        "Generate",
        use_container_width=True,
        key="generate_btn",
        disabled=st.session_state.generating,
    )

st.markdown("---")

# ---------------------------------------------------------------------------
# Generation Logic
# ---------------------------------------------------------------------------

if generate_clicked:
    if not prompt or not prompt.strip():
        st.error("⚠ Please enter a prompt before generating.")
    elif not token_configured:
        st.error(
            "⚠ HuggingFace API token not configured. "
            "Add your token to the `.env` file and restart the app."
        )
    else:
        try:
            st.session_state.generating = True

            with st.spinner("Generating image..."):
                image_bytes = generate_image(prompt.strip())

                # Validate image bytes are a real, openable image
                img = Image.open(io.BytesIO(image_bytes))
                img.verify()

            # Store in session gallery
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            st.session_state.gallery.append({
                "prompt": prompt.strip(),
                "image_bytes": image_bytes,
                "timestamp": timestamp,
            })

            st.session_state.generating = False
            st.rerun()

        except ValueError as ve:
            st.session_state.generating = False
            st.error(f"⚠ Configuration Error: {ve}")

        except RuntimeError as rte:
            st.session_state.generating = False
            st.error(f"⚠ {rte}")

        except Exception as e:
            st.session_state.generating = False
            st.exception(e)

# ---------------------------------------------------------------------------
# Display Latest Generated Image
# ---------------------------------------------------------------------------

if st.session_state.gallery:

    latest = st.session_state.gallery[-1]

    st.markdown(
        '<div class="section-title">&gt; LATEST GENERATION</div>',
        unsafe_allow_html=True,
    )

    st.image(latest["image_bytes"], use_container_width=True)

    st.markdown(
        f'<div class="prompt-card"><div class="prompt-label">PROMPT</div>'
        f'<div>{latest["prompt"]}</div></div>',
        unsafe_allow_html=True,
    )

    st.download_button(
        "⬇ DOWNLOAD IMAGE",
        latest["image_bytes"],
        file_name="mana_image.png",
        mime="image/png",
        use_container_width=True
    )

    # Full gallery below
    if len(st.session_state.gallery) > 1:
        st.markdown("---")
        st.markdown(
            """
            <div style="text-align: center; margin-bottom: 16px;">
                <span style="color: #ff6ec7; font-size: 0.8rem; letter-spacing: 1px; text-transform: uppercase;">
                    Previous Generations
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Show in reverse order (newest first), skip the latest (already shown above)
        previous_items = list(reversed(st.session_state.gallery[:-1]))
        cols_per_row = 2

        for row_start in range(0, len(previous_items), cols_per_row):
            row_items = previous_items[row_start: row_start + cols_per_row]
            cols = st.columns(cols_per_row)

            for col_idx, item in enumerate(row_items):
                with cols[col_idx]:
                    st.image(item["image_bytes"], use_container_width=True)
                    prompt_preview = item["prompt"][:60] + ("..." if len(item["prompt"]) > 60 else "")
                    st.caption(f"_{prompt_preview}_")
                    # Calculate the original index for the filename
                    original_idx = st.session_state.gallery.index(item)
                    st.download_button(
                        label="⬇ Download",
                        data=item["image_bytes"],
                        file_name=f"mana_image_{original_idx}.png",
                        mime="image/png",
                        key=f"gallery_dl_{row_start}_{col_idx}",
                        use_container_width=True,
                    )

else:
    # Empty state
    st.markdown(
        """
        <div style="text-align: center; padding: 60px 20px; color: #8fa3c7;">
            <p style="font-size: 2.5rem; margin-bottom: 8px;">🖼️</p>
            <p style="font-size: 1rem; letter-spacing: 1px;">
                Enter a prompt above and click <span style="color: #ff6ec7;">GENERATE</span>
            </p>
            <p style="font-size: 0.78rem; margin-top: 8px; color: #5a6b8a;">
                Your generated images will appear here
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )