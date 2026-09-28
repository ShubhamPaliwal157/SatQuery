"""
SatQuery AI — Frontend (Step 3.2: theme fix + demo mode + map + overview)
====================================================================
Step 3.1 shipped a chat-style rebuild but had two real bugs: the sidebar
stayed hardcoded to a light theme while the rest of the app rendered
dark (whatever the viewer's OS/browser preference happened to be), and
several buttons had no explicit width so they sized to their own text
instead of lining up with each other. This pass:

  - Fixes theming at the root cause: a `.streamlit/config.toml` dark
    theme so every native widget (sidebar included) is dark by
    construction, instead of a pile of CSS overrides that only ever
    patched some elements. The custom CSS below only adds project-
    specific components (cards, bars, the confidence ring) on top.
  - Every button that should line up now explicitly sets
    `use_container_width=True`; the one button that's deliberately
    small (the per-chat delete "×") is deliberately NOT full-width,
    sitting in a narrow column — an intentional size difference
    instead of an accidental one.
  - Bundled demo scenes (`demo_assets/`, procedurally generated —
    see `tools/generate_demo_assets.py`) so Analyze and Compare both
    have a "load a demo scene" one-click path that needs no upload,
    no internet, and no real dataset on hand.
  - Example-query chips on a fresh Analyze conversation, sharing one
    `submit_query()` helper with the chat input box so both paths
    behave identically.
  - A new **Map** view: shows a georeferenced upload's real footprint
    (extracted server-side via rasterio, reprojected to EPSG:4326 —
    see `backend/image_utils.py`) on a Leaflet/OpenStreetMap panel.
    A plain JPEG/PNG has no coordinates to show, and the empty state
    says exactly that instead of guessing.
  - A new **Overview** view: what's actually built vs. still roadmap,
    stated plainly rather than implied.

Run (with the backend already running on :8000):
    streamlit run app.py
"""

import base64
import os
import time
import uuid

import requests
import streamlit as st
from PIL import Image, ImageDraw

# st.iframe (added in newer Streamlit) replaces the older
# st.components.v1.html for embedding raw HTML/JS — used below for the
# Leaflet map. Falling back keeps this working on an older pinned install
# too, since requirements.txt only floors the version at >=1.40.
if hasattr(st, "iframe"):
    def _embed_html(html: str, height: int) -> None:
        st.iframe(html, height=height)
else:  # pragma: no cover - exercised only on older Streamlit installs
    import streamlit.components.v1 as _components

    def _embed_html(html: str, height: int) -> None:
        _components.html(html, height=height)

BACKEND_URL = os.environ.get("SATQUERY_BACKEND_URL", "http://localhost:8000")
ANALYZE_TIMEOUT_S = 300  # generous — first call per model can trigger a multi-GB download
DEMO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_assets")

USER_AVATAR = ":material/person:"
ASSISTANT_AVATAR = ":material/satellite_alt:"
TIER_COLOR = {"high": "#22c55e", "medium": "#eab308", "heuristic_fallback": "#ef4444"}
TIER_LABEL = {
    "high": "confident",
    "medium": "moderate confidence",
    "heuristic_fallback": "low confidence — keyword fallback",
}

NAV_ITEMS = [
    ("Overview", ":material/rocket_launch:"),
    ("Analyze", ":material/forum:"),
    ("Compare", ":material/compare_arrows:"),
    ("Map", ":material/map:"),
    ("Audit trail", ":material/verified_user:"),
]

DEMO_SCENES = [
    {"key": "farmland", "label": ":material/grass: Farmland (georeferenced)", "file": "farmland.tif", "mime": "image/tiff"},
    {"key": "coastal", "label": ":material/location_city: Coastal urban scene", "file": "coastal_urban.png", "mime": "image/png"},
]
DEMO_CHANGE_PAIR = {"before_file": "change_before.tif", "after_file": "change_after.tif"}

EXAMPLE_QUERIES = [
    "Describe this scene",
    "What type of land use is this?",
    "Are there any buildings visible?",
    "Classify this area",
]


def _make_favicon() -> Image.Image:
    """A small flat, geometric mark (rounded square + orbit ring + node)
    instead of an emoji favicon — generated at runtime so there's no
    external asset to ship."""
    size = 64
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([0, 0, size - 1, size - 1], radius=14, fill="#111318")
    draw.ellipse([15, 15, 49, 49], outline="#ffffff", width=3)
    draw.ellipse([28, 9, 37, 18], fill="#2563eb")
    return img


st.set_page_config(page_title="SatQuery AI", page_icon=_make_favicon(), layout="wide")

# ---------------------------------------------------------------------------
# Visual layer — the dark palette itself lives in .streamlit/config.toml so
# every native widget picks it up automatically (including the sidebar).
# Everything here only styles this project's own custom HTML (cards, bars,
# nav, chips) using the same colors, and a few small consistency fixes for
# native widgets that CSS alone still needs to touch (button radius, the
# disabled state, chat message spacing).
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
    /* Only hiding the hamburger menu and the "made with Streamlit" footer.
       The previous two attempts at also hiding the header/toolbar (and a
       config.toml toolbarMode setting) kept taking the sidebar's re-open
       arrow down with them — leaving the header and toolbar completely
       untouched, with no config.toml override either, is the reliable fix. */
    #MainMenu, footer { visibility: hidden; }

    :root {
        --sq-accent: #2563eb;
        --sq-accent-soft: rgba(37, 99, 235, 0.16);
        --sq-bg-soft: #171a21;      /* matches secondaryBackgroundColor in config.toml */
        --sq-bg-soft-2: #1d212b;
        --sq-border: #2a2e38;
        --sq-text: #e8e9ec;         /* matches textColor in config.toml */
        --sq-text-dim: #8b90a0;
    }

    .block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 980px; }

    /* No hardcoded sidebar background here on purpose — secondaryBackgroundColor
       from config.toml already applies to it, and to every other native widget,
       so it can never drift out of sync with the rest of the app again. */
    section[data-testid="stSidebar"] { border-right: 1px solid var(--sq-border); }
    section[data-testid="stSidebar"] .block-container { padding-top: 1.3rem; }

    .sq-brand-row { display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.15rem; }
    .sq-logo-mark {
        width: 32px; height: 32px; border-radius: 8px; background: var(--sq-accent);
        color: #fff; display: flex; align-items: center; justify-content: center;
        font-weight: 700; font-size: 1rem; flex-shrink: 0;
    }
    .sq-brand { font-size: 1.15rem; font-weight: 700; letter-spacing: -0.02em; color: var(--sq-text); }
    .sq-subtitle { color: var(--sq-text-dim); font-size: 0.8rem; margin: 0.1rem 0 1.1rem 2.6rem; }
    .sq-section-label {
        color: var(--sq-text-dim); font-size: 0.7rem; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.07em; margin: 1rem 0 0.4rem 0;
    }

    .sq-card {
        background: var(--sq-bg-soft); border: 1px solid var(--sq-border);
        border-radius: 14px; padding: 0.9rem 1.1rem; margin-bottom: 0.7rem;
    }
    .sq-metric-label { color: var(--sq-text-dim); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; }
    .sq-metric-value { font-size: 1.4rem; font-weight: 650; color: var(--sq-text); margin-top: 0.15rem; }

    .sq-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 0.45rem; }

    .sq-bar-chart { margin: 0.5rem 0 0.3rem 0; }
    .sq-bar-row { display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.4rem; }
    .sq-bar-label { flex: 0 0 150px; font-size: 0.85rem; color: var(--sq-text); text-overflow: ellipsis; overflow: hidden; white-space: nowrap; }
    .sq-bar-track { flex: 1; background: var(--sq-border); border-radius: 6px; height: 9px; overflow: hidden; }
    .sq-bar-fill { height: 100%; background: var(--sq-accent); border-radius: 6px; }
    .sq-bar-pct { flex: 0 0 46px; font-size: 0.8rem; color: var(--sq-text-dim); text-align: right; }

    .sq-ring { display: inline-flex; align-items: center; gap: 0.6rem; margin: 0.3rem 0; }

    .sq-empty-state {
        text-align: center; padding: 2.2rem 1.2rem; color: var(--sq-text-dim);
        border: 1px dashed var(--sq-border); border-radius: 16px; margin: 0.8rem 0;
    }

    .sq-req-table { width: 100%; border-collapse: collapse; margin: 0.5rem 0 1.3rem 0; font-size: 0.86rem; }
    .sq-req-table th, .sq-req-table td { text-align: left; padding: 0.55rem 0.6rem; border-bottom: 1px solid var(--sq-border); vertical-align: top; }
    .sq-req-table th { color: var(--sq-text-dim); font-weight: 600; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em; }
    .sq-req-table td:first-child { color: var(--sq-text); font-weight: 550; white-space: nowrap; }
    .sq-badge { display: inline-block; padding: 0.15rem 0.6rem; border-radius: 999px; font-size: 0.72rem; font-weight: 600; white-space: nowrap; }
    .sq-badge-done { background: rgba(34,197,94,0.15); color: #22c55e; }
    .sq-badge-roadmap { background: rgba(234,179,8,0.15); color: #eab308; }

    /* Buttons: width is set explicitly in Python (use_container_width) per
       button, everywhere it should line up with its neighbors — this CSS
       only makes every button look like it belongs to the same family. */
    .stButton>button { border-radius: 10px; font-weight: 550; }
    .stButton>button:disabled { opacity: 0.4; }
    div[data-testid="stChatMessage"] { padding: 0.4rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


def backend_status() -> dict | None:
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=3)
        return r.json() if r.ok else None
    except requests.exceptions.RequestException:
        return None


def render_label_bars(top_labels: list[dict]) -> None:
    """Flat horizontal bar chart for classify-intent scores — replaces
    a bare bullet list with something actually skimmable at a glance."""
    if not top_labels:
        return
    top_score = max(item["score"] for item in top_labels) or 1.0
    rows = []
    for item in top_labels:
        width = max(4.0, item["score"] / top_score * 100)
        rows.append(
            f'<div class="sq-bar-row">'
            f'<div class="sq-bar-label">{item["label"]}</div>'
            f'<div class="sq-bar-track"><div class="sq-bar-fill" style="width:{width:.1f}%"></div></div>'
            f'<div class="sq-bar-pct">{item["score"] * 100:.1f}%</div>'
            f'</div>'
        )
    st.markdown(f'<div class="sq-bar-chart">{"".join(rows)}</div>', unsafe_allow_html=True)


def render_confidence_ring(score: float, tier: str | None) -> None:
    """Flat SVG ring (no external chart library) showing routing
    confidence, colored by tier."""
    pct = max(0.0, min(1.0, score))
    color = TIER_COLOR.get(tier, "#8b90a0")
    circumference = 2 * 3.14159 * 22
    dash = circumference * pct
    svg = (
        f'<svg width="52" height="52" viewBox="0 0 52 52">'
        f'<circle cx="26" cy="26" r="22" fill="none" stroke="#2a2e38" stroke-width="5"/>'
        f'<circle cx="26" cy="26" r="22" fill="none" stroke="{color}" stroke-width="5" '
        f'stroke-dasharray="{dash:.1f} {circumference:.1f}" stroke-linecap="round" '
        f'transform="rotate(-90 26 26)"/>'
        f'<text x="26" y="31" text-anchor="middle" font-size="11" font-weight="600" fill="#e8e9ec">{int(pct * 100)}%</text>'
        f'</svg>'
    )
    st.markdown(
        f'<div class="sq-ring">{svg}<span style="font-size:0.85rem; color:var(--sq-text-dim);">'
        f'routing confidence</span></div>',
        unsafe_allow_html=True,
    )


def render_pipeline_status(result: dict) -> None:
    """Flat step strip for the Compare view — align / diff / regions /
    review-flag / audit-log, each with a check or warning icon. Mirrors
    the idea of a validation pipeline without pretending to be a formal
    certification system."""
    steps = [
        ("Align", result["aligned"]),
        ("Diff (SSIM)", result.get("ssim_score") is not None),
        ("Regions", len(result["regions"]) > 0),
        ("Review flag", not result["needs_human_review"]),
        ("Audit log", bool(result.get("audit"))),
    ]
    cols = st.columns(len(steps))
    for col, (label, ok) in zip(cols, steps):
        with col:
            st.markdown(":material/check_circle:" if ok else ":material/warning:")
            st.caption(label)


# ---------------------------------------------------------------------------
# Chat-history state (used by the Analyze view)
# ---------------------------------------------------------------------------
def _new_chat() -> str:
    chat_id = str(uuid.uuid4())
    st.session_state.chats[chat_id] = {
        "title": "New conversation",
        "messages": [],
        "image_bytes": None,
        "image_name": None,
        "image_type": None,
        "created": time.time(),
        "geo_bounds": None,  # [west, south, east, north] EPSG:4326, once known
        "crs": None,
    }
    st.session_state.active_chat_id = chat_id
    return chat_id


def _load_demo_into_chat(chat: dict, scene: dict) -> None:
    """Populates a chat's image state directly from a bundled demo file —
    same effect as an upload, without needing one."""
    with open(os.path.join(DEMO_DIR, scene["file"]), "rb") as f:
        chat["image_bytes"] = f.read()
    chat["image_name"] = scene["file"]
    chat["image_type"] = scene["mime"]
    chat["messages"] = []
    chat["title"] = scene["file"]
    chat["geo_bounds"] = None
    chat["crs"] = None


def submit_query(chat: dict, query: str) -> None:
    """Runs one Analyze turn against `/analyze` and appends both the user
    and assistant messages to chat state. Shared by the chat input box and
    the example-query chips so both behave identically — previously the
    chips didn't exist and this logic lived inline in one place only."""
    if not chat["image_bytes"] or not query or not query.strip():
        return
    if chat["title"] == "New conversation":
        chat["title"] = query[:40]
    chat["messages"].append({"role": "user", "content": query})

    with st.spinner("Routing query and running inference..."):
        try:
            files = {"image": (chat["image_name"], chat["image_bytes"], chat["image_type"])}
            data = {"query": query}
            resp = requests.post(f"{BACKEND_URL}/analyze", files=files, data=data, timeout=ANALYZE_TIMEOUT_S)
        except requests.exceptions.RequestException as e:
            chat["messages"].append({"role": "assistant", "content": f"Could not reach the backend: {e}"})
            return

    if resp.ok:
        result = resp.json()
        meta = f"intent: `{result['intent']}` · {TIER_LABEL.get(result.get('confidence_tier'), '')}"
        if result.get("geo_bounds"):
            chat["geo_bounds"] = result["geo_bounds"]
            chat["crs"] = result.get("crs")
        chat["messages"].append({
            "role": "assistant",
            "content": result["answer"],
            "top_labels": result.get("top_labels"),
            "confidence": result.get("confidence"),
            "confidence_tier": result.get("confidence_tier"),
            "meta": meta,
            "trace": result["evidence_trace"],
        })
    else:
        try:
            err = resp.json().get("detail", "Something went wrong.")
        except ValueError:
            err = f"Backend error: {resp.status_code}"
        chat["messages"].append({"role": "assistant", "content": f"Error: {err}"})


if "chats" not in st.session_state:
    st.session_state.chats = {}
    _new_chat()
if st.session_state.get("active_chat_id") not in st.session_state.chats:
    _new_chat()
if "page" not in st.session_state:
    st.session_state.page = "Overview"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div class="sq-brand-row"><div class="sq-logo-mark">S</div>'
        '<div class="sq-brand">SatQuery AI</div></div>'
        '<div class="sq-subtitle">SIH26167 remote sensing assistant</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sq-section-label">Navigate</div>', unsafe_allow_html=True)
    for name, icon in NAV_ITEMS:
        active = st.session_state.page == name
        if st.button(f"{icon} {name}", key=f"nav_{name}", use_container_width=True,
                     type="primary" if active else "secondary"):
            st.session_state.page = name
            st.rerun()

    if st.session_state.page == "Analyze":
        st.markdown('<div class="sq-section-label">Conversations</div>', unsafe_allow_html=True)
        if st.button(":material/edit_square: New chat", use_container_width=True):
            _new_chat()
            st.rerun()

        ordered = sorted(st.session_state.chats.items(), key=lambda kv: -kv[1]["created"])
        for chat_id, chat in ordered:
            is_active = chat_id == st.session_state.active_chat_id
            title = chat["title"] if len(chat["title"]) <= 26 else chat["title"][:25] + "…"
            row = st.columns([6, 1])
            with row[0]:
                if st.button(
                    f":material/chat_bubble: {title}",
                    key=f"select_{chat_id}",
                    type="primary" if is_active else "secondary",
                    use_container_width=True,
                ):
                    st.session_state.active_chat_id = chat_id
                    st.rerun()
            with row[1]:
                if st.button("×", key=f"delete_{chat_id}", help="Delete this conversation"):
                    del st.session_state.chats[chat_id]
                    if not st.session_state.chats:
                        _new_chat()
                    elif st.session_state.active_chat_id == chat_id:
                        st.session_state.active_chat_id = next(iter(st.session_state.chats))
                    st.rerun()

    st.markdown('<div class="sq-section-label">System</div>', unsafe_allow_html=True)
    status = backend_status()
    dot_color = "#22c55e" if status else "#ef4444"
    st.markdown(
        f'<span class="sq-dot" style="background:{dot_color}"></span>'
        f'{"Backend online" if status else "Backend unreachable"}',
        unsafe_allow_html=True,
    )
    st.caption(status.get("step", "") if status else "Start it with `uvicorn main:app --reload`")

    with st.expander(":material/settings: Model warmup"):
        st.caption("Pre-loads BLIP + the classifier once (~2-3GB) so the first real query isn't stuck behind a download.")
        if st.button(":material/bolt: Warm up now", use_container_width=True):
            with st.spinner("Loading models — first time can take a few minutes..."):
                try:
                    r = requests.post(f"{BACKEND_URL}/warmup", timeout=600)
                    if r.ok:
                        st.success(f"Loaded: {', '.join(r.json()['loaded'])}")
                    else:
                        st.error(r.json().get("detail", "Warmup failed."))
                except requests.exceptions.RequestException as e:
                    st.error(f"Could not reach the backend: {e}")


# ---------------------------------------------------------------------------
# View: Overview
# ---------------------------------------------------------------------------
def render_overview():
    st.subheader("SatQuery AI — SIH26167")
    st.caption(
        "An interactive vision-language assistant for remote-sensing image analysis through "
        "text queries — single-image VQA / captioning / classification, bi-temporal change "
        "detection, and a cryptographically verifiable audit trail, on free/local models."
    )

    st.markdown('<div class="sq-section-label">Coverage</div>', unsafe_allow_html=True)
    rows = [
        ("Natural-language VQA", "BLIP VQA behind a calibrated 3-tier intent router", "done"),
        ("Automated scene captioning", "BLIP image captioning", "done"),
        ("Scene / land-cover classification", "Zero-shot classifier over a scene-type label set", "done"),
        ("Bi-temporal change detection", "ORB+RANSAC co-registration, SSIM diff, boxed regions", "done"),
        ("Query understanding / routing", "Sentence-embedding router with a heuristic fallback tier", "done"),
        ("Explainability (XAI)", "Step-by-step evidence trace + confidence tier on every answer", "done"),
        ("Verifiable audit trail", "SHA-256 hash-chained ledger, independently re-verifiable", "done"),
        ("Geolocation / map view", "Footprint map for georeferenced uploads", "done"),
        ("Optical + SAR sensor fusion", "Two-branch CNN, optional SAR band on /analyze — next up", "roadmap"),
        ("3D globe visualization", "Would need a React/Cesium frontend alongside this one", "roadmap"),
    ]
    table_rows = "".join(
        f'<tr><td>{name}</td><td style="color:var(--sq-text-dim)">{detail}</td>'
        f'<td><span class="sq-badge sq-badge-{status}">{"Done" if status == "done" else "Roadmap"}</span></td></tr>'
        for name, detail, status in rows
    )
    st.markdown(
        f'<table class="sq-req-table"><thead><tr><th>Capability</th><th>How</th><th>Status</th></tr></thead>'
        f'<tbody>{table_rows}</tbody></table>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sq-section-label">Pipeline</div>', unsafe_allow_html=True)
    st.caption(
        "Analyze: input validation → query routing → specialist model (caption / VQA / "
        "classify) → evidence trace → audit log. Compare: co-registration → SSIM diff → "
        "region extraction → optional per-region labeling → audit log. Every number in "
        "Compare comes from OpenCV / scikit-image math, never from a language model."
    )

    st.markdown('<div class="sq-section-label">Stack</div>', unsafe_allow_html=True)
    st.caption(
        "FastAPI + PyTorch/HuggingFace backend (BLIP captioning/VQA, a zero-shot scene "
        "classifier, sentence-transformers for routing), rasterio/GDAL for GeoTIFF handling, "
        "OpenCV + scikit-image for change detection, a local SHA-256 hash chain for the audit "
        "trail, and this Streamlit frontend. No paid APIs, nothing sent to a third-party model."
    )

    st.info(
        "New here? Open **Analyze** or **Compare** and load a bundled demo scene — "
        "no upload, dataset, or setup needed.",
        icon=":material/lightbulb:",
    )


# ---------------------------------------------------------------------------
# View: Analyze (chat-style, per-conversation image + history)
# ---------------------------------------------------------------------------
def render_analyze():
    chat_id = st.session_state.active_chat_id
    chat = st.session_state.chats[chat_id]

    st.subheader("Analyze")
    st.caption("Upload a remote-sensing image, then ask follow-up questions about it as a running chat.")

    uploaded = st.file_uploader(
        "Upload a remote-sensing image",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        help="Multi-band / 16-bit GeoTIFFs are supported.",
        key=f"uploader_{chat_id}",
    )
    if uploaded is not None and uploaded.getvalue() != chat["image_bytes"]:
        chat["image_bytes"] = uploaded.getvalue()
        chat["image_name"] = uploaded.name
        chat["image_type"] = uploaded.type
        chat["messages"] = []
        chat["geo_bounds"] = None
        chat["crs"] = None
        if chat["title"] == "New conversation":
            chat["title"] = uploaded.name

    if not chat["image_bytes"]:
        st.markdown('<div class="sq-section-label">Or try a demo scene — no upload needed</div>', unsafe_allow_html=True)
        dcols = st.columns(len(DEMO_SCENES))
        for col, scene in zip(dcols, DEMO_SCENES):
            with col:
                if st.button(scene["label"], key=f"demo_{scene['key']}_{chat_id}", use_container_width=True):
                    _load_demo_into_chat(chat, scene)
                    st.rerun()

    if chat["image_bytes"]:
        cols = st.columns([1, 3])
        with cols[0]:
            st.image(chat["image_bytes"], caption="Current image")
        with cols[1]:
            st.caption(
                "Ask about this image below. Uploading a new file here replaces the image "
                "in this conversation; use \"New chat\" in the sidebar to start a separate one."
            )
            if chat.get("geo_bounds"):
                st.caption(":material/location_on: Georeferenced — see the **Map** tab for its footprint.")

    for turn in chat["messages"]:
        avatar = USER_AVATAR if turn["role"] == "user" else ASSISTANT_AVATAR
        with st.chat_message(turn["role"], avatar=avatar):
            st.write(turn["content"])
            if turn.get("top_labels"):
                render_label_bars(turn["top_labels"])
            if turn.get("confidence") is not None:
                render_confidence_ring(turn["confidence"], turn.get("confidence_tier"))
            if turn.get("meta"):
                st.caption(turn["meta"])
            if turn.get("trace"):
                with st.expander(":material/manage_search: Evidence trace"):
                    for step in turn["trace"]:
                        st.markdown(f"- **{step['step']}** — {step['detail']}")

    if chat["image_bytes"] and not chat["messages"]:
        st.markdown('<div class="sq-section-label">Try asking</div>', unsafe_allow_html=True)
        chip_cols = st.columns(len(EXAMPLE_QUERIES))
        for col, eq in zip(chip_cols, EXAMPLE_QUERIES):
            with col:
                if st.button(eq, key=f"chip_{chat_id}_{eq}"):
                    submit_query(chat, eq)
                    st.rerun()

    query = st.chat_input(
        "Ask a question about the image..." if chat["image_bytes"] else "Upload an image (or load a demo scene) above first",
        disabled=not chat["image_bytes"],
    )
    if query:
        submit_query(chat, query)
        st.rerun()


# ---------------------------------------------------------------------------
# View: Compare (bi-temporal change detection)
# ---------------------------------------------------------------------------
def render_compare():
    st.subheader("Compare")
    st.caption(
        "Upload a before/after pair of the same scene. Every metric below is computed "
        "deterministically (OpenCV/scikit-image); the classifier only ever labels a region "
        "already found by that math, never decides whether something counts as change."
    )

    for key in ("cd_before_bytes", "cd_before_name", "cd_before_type",
                "cd_after_bytes", "cd_after_name", "cd_after_type", "cd_last_result"):
        st.session_state.setdefault(key, None)

    c1, c2 = st.columns(2)
    with c1:
        before = st.file_uploader("Before image", type=["png", "jpg", "jpeg", "tif", "tiff"], key="cd_before_uploader")
        if before is not None and before.getvalue() != st.session_state.cd_before_bytes:
            st.session_state.cd_before_bytes = before.getvalue()
            st.session_state.cd_before_name = before.name
            st.session_state.cd_before_type = before.type
            st.session_state.cd_last_result = None
        if st.session_state.cd_before_bytes:
            st.image(st.session_state.cd_before_bytes, caption="Before")
    with c2:
        after = st.file_uploader("After image", type=["png", "jpg", "jpeg", "tif", "tiff"], key="cd_after_uploader")
        if after is not None and after.getvalue() != st.session_state.cd_after_bytes:
            st.session_state.cd_after_bytes = after.getvalue()
            st.session_state.cd_after_name = after.name
            st.session_state.cd_after_type = after.type
            st.session_state.cd_last_result = None
        if st.session_state.cd_after_bytes:
            st.image(st.session_state.cd_after_bytes, caption="After")

    if not (st.session_state.cd_before_bytes and st.session_state.cd_after_bytes):
        st.markdown('<div class="sq-section-label">Or try the bundled demo pair — no upload needed</div>', unsafe_allow_html=True)
        if st.button(":material/bolt: Load demo pair — vacant lot to new construction", use_container_width=True):
            with open(os.path.join(DEMO_DIR, DEMO_CHANGE_PAIR["before_file"]), "rb") as f:
                st.session_state.cd_before_bytes = f.read()
            st.session_state.cd_before_name = DEMO_CHANGE_PAIR["before_file"]
            st.session_state.cd_before_type = "image/tiff"
            with open(os.path.join(DEMO_DIR, DEMO_CHANGE_PAIR["after_file"]), "rb") as f:
                st.session_state.cd_after_bytes = f.read()
            st.session_state.cd_after_name = DEMO_CHANGE_PAIR["after_file"]
            st.session_state.cd_after_type = "image/tiff"
            st.session_state.cd_last_result = None
            st.rerun()

    label_regions = st.checkbox("Label change regions with the scene classifier", value=True)

    can_compare = bool(st.session_state.cd_before_bytes and st.session_state.cd_after_bytes)
    if st.button(":material/compare_arrows: Compare", key="run_compare_btn", type="primary",
                 disabled=not can_compare, use_container_width=True):
        with st.spinner("Aligning frames and computing the diff..."):
            try:
                files = {
                    "image_before": (st.session_state.cd_before_name, st.session_state.cd_before_bytes, st.session_state.cd_before_type),
                    "image_after": (st.session_state.cd_after_name, st.session_state.cd_after_bytes, st.session_state.cd_after_type),
                }
                data = {"query": "what changed between these two images?", "label_regions": str(label_regions)}
                resp = requests.post(f"{BACKEND_URL}/analyze-change", files=files, data=data, timeout=ANALYZE_TIMEOUT_S)
            except requests.exceptions.RequestException as e:
                st.error(f"Could not reach the backend: {e}")
                resp = None

        if resp is not None:
            if resp.ok:
                st.session_state.cd_last_result = resp.json()
            else:
                st.session_state.cd_last_result = None
                try:
                    st.error(resp.json().get("detail", "Something went wrong."))
                except ValueError:
                    st.error(f"Backend error: {resp.status_code}")

    result = st.session_state.get("cd_last_result")
    if result:
        render_pipeline_status(result)
        st.markdown("")

        m1, m2, m3 = st.columns(3)
        for col, label, value in (
            (m1, "SSIM similarity", f"{result['ssim_score']:.3f}"),
            (m2, "Changed area", f"{result['changed_area_fraction'] * 100:.1f}%"),
            (m3, "Reprojection error", f"{result['reprojection_error_px']}px" if result["reprojection_error_px"] is not None else "n/a"),
        ):
            with col:
                st.markdown(
                    f'<div class="sq-card"><div class="sq-metric-label">{label}</div>'
                    f'<div class="sq-metric-value">{value}</div></div>',
                    unsafe_allow_html=True,
                )

        st.image(
            base64.b64decode(result["heatmap_png_base64"]),
            caption="Change heatmap — red overlay marks detected change; numbered boxes match the region list below.",
        )

        st.write(result["summary"])

        if result["regions"]:
            st.markdown('<div class="sq-section-label">Change regions</div>', unsafe_allow_html=True)
            for i, r in enumerate(result["regions"], start=1):
                x, y, w, h = r["bbox"]
                line = f"**{i}.** {r['area_fraction'] * 100:.2f}% of frame, box ({x}, {y}, {w}×{h})"
                if r.get("label_before") and r.get("label_after"):
                    line += f" — *{r['label_before']}* → *{r['label_after']}*"
                st.markdown(line)

        with st.expander(":material/manage_search: Evidence trace"):
            for step in result["evidence_trace"]:
                st.markdown(f"- **{step['step']}** — {step['detail']}")
        if result.get("audit"):
            st.caption(f"Logged to audit trail — record `{result['audit']['record_id'][:8]}…`")


# ---------------------------------------------------------------------------
# View: Map (footprint of a georeferenced upload)
# ---------------------------------------------------------------------------
def render_map():
    st.subheader("Map")
    st.caption(
        "Shows the real footprint of the current Analyze image, extracted from its embedded "
        "geo-reference — never estimated or guessed. A plain JPEG/PNG carries no coordinates, "
        "and that's shown honestly rather than faked."
    )

    chat = st.session_state.chats.get(st.session_state.active_chat_id)
    geo_bounds = chat.get("geo_bounds") if chat else None

    if not geo_bounds:
        st.markdown(
            '<div class="sq-empty-state">'
            '<b>No embedded location data on the current Analyze image.</b><br>'
            "Plain JPEG/PNG files don't carry coordinates. Open <b>Analyze</b>, ask at least one "
            'question about a georeferenced GeoTIFF — or load the bundled "Farmland" demo scene — '
            "and its footprint will appear here."
            '</div>',
            unsafe_allow_html=True,
        )
        return

    west, south, east, north = geo_bounds
    lat_c, lon_c = (south + north) / 2, (west + east) / 2
    leaflet_html = f"""
    <div id="sq-map" style="height: 460px; border-radius: 14px; overflow: hidden;"></div>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
      var map = L.map('sq-map').setView([{lat_c}, {lon_c}], 15);
      L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
        attribution: '&copy; OpenStreetMap contributors', maxZoom: 19
      }}).addTo(map);
      var bounds = [[{south}, {west}], [{north}, {east}]];
      L.rectangle(bounds, {{color: '#2563eb', weight: 2, fillOpacity: 0.12}}).addTo(map);
      map.fitBounds(bounds);
    </script>
    """
    _embed_html(leaflet_html, height=470)

    crs_note = f" · source CRS: {chat.get('crs')}" if chat.get("crs") else ""
    st.caption(
        f"Image: `{chat.get('image_name')}` · bounds (WGS84): "
        f"{west:.5f}, {south:.5f} → {east:.5f}, {north:.5f}{crs_note}"
    )


# ---------------------------------------------------------------------------
# View: Audit Trail
# ---------------------------------------------------------------------------
def render_audit():
    st.subheader("Audit trail")
    st.caption(
        "Every analysis is appended to a local SHA-256 hash-chained ledger. Each record's "
        "hash covers the previous record's hash, so editing any past entry breaks the chain "
        "from that point on — this recomputes the whole chain rather than just asserting the "
        "log is trustworthy."
    )

    if st.button(":material/verified_user: Verify chain integrity", type="primary", use_container_width=True):
        try:
            resp = requests.get(f"{BACKEND_URL}/audit/verify", timeout=15)
            if resp.ok:
                v = resp.json()
                if v["valid"]:
                    st.markdown(
                        f'<span class="sq-dot" style="background:#22c55e;"></span>'
                        f'Chain intact — {v["records_checked"]} record(s) verified from genesis.',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<span class="sq-dot" style="background:#ef4444;"></span>'
                        f'Chain broken at record `{v["broken_at"]}` — {v["reason"]}',
                        unsafe_allow_html=True,
                    )
            else:
                st.error("Could not verify — backend error.")
        except requests.exceptions.RequestException as e:
            st.error(f"Could not reach the backend: {e}")

    st.divider()
    st.caption("Recent ledger entries (newest first). `input_hash` is a fingerprint of the uploaded image, not the image itself.")
    try:
        resp = requests.get(f"{BACKEND_URL}/audit/recent", timeout=15)
        if resp.ok:
            records = resp.json()["records"]
            if not records:
                st.info("No analyses logged yet — run something in Analyze or Compare first.")
            for r in records:
                st.markdown(
                    f'<div class="sq-card">'
                    f'<b>{r["endpoint"]}</b> · intent: {r["intent"]}<br>'
                    f'<span style="color:var(--sq-text-dim); font-size:0.85rem;">query: {r["query"]}</span><br>'
                    f'<span style="color:var(--sq-text-dim); font-size:0.8rem;">'
                    f'record_hash: {r["record_hash"][:16]}… · input_hash: {r["input_hash"][:16]}…</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.error("Could not load ledger — backend error.")
    except requests.exceptions.RequestException as e:
        st.error(f"Could not reach the backend: {e}")


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
_PAGES = {
    "Overview": render_overview,
    "Analyze": render_analyze,
    "Compare": render_compare,
    "Map": render_map,
    "Audit trail": render_audit,
}
_PAGES[st.session_state.page]()
