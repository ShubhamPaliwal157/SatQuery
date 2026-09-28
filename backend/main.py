"""
SatQuery AI — Backend Orchestrator (Step 3)
=================================================
Step 2 gave a single-image caption/VQA/classify pipeline with calibrated
routing and real GeoTIFF support. Step 3 adds the two features flagged
as "not built yet" in Step 2's routing notes, plus one bonus feature,
still without changing the existing `/analyze` contract:

  - **`POST /analyze-change`** — bi-temporal change detection. Takes a
    before/after image pair and returns a deterministic diff (SSIM,
    changed-area %, per-region bounding boxes, an optional zero-shot
    label for what a region looked like before vs. after) plus a
    red-overlay heatmap image. See `models/change_detector.py` for why
    the numbers here are computed, never generated.
  - **Router honesty update** — a single-image query that reads like a
    change-detection question now tells the user to switch to the
    compare flow, instead of the old "not available until Step 3" note.
  - **Cryptographic audit trail** (`audit.py`) — every `/analyze` and
    `/analyze-change` call is appended to a local SHA-256 hash-chained
    ledger. `GET /audit/recent` and `GET /audit/verify` expose it, so a
    judge (or anyone) can independently confirm the log hasn't been
    edited after the fact, not just trust a claim that it hasn't.

Run:
    uvicorn main:app --reload --port 8000
"""

from __future__ import annotations
import logging
import time

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from router import route_query
from image_utils import load_image_any
from models import captioner, classifier, change_detector
import audit
import scenes

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("satquery.main")

app = FastAPI(title="SatQuery AI", version="0.3.0-step3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _classify_label(image) -> str:
    """Thin adapter so change_detector doesn't need to know classifier's
    return shape — it just wants "the top label as a string"."""
    return classifier.classify_scene(image, top_k=1).top_labels[0][0]


def _log_audit(endpoint: str, raw_image_bytes: bytes, query: str, intent: str, answer: str) -> dict | None:
    """Best-effort audit logging — must never break the user-facing
    response if the ledger write fails for some local-disk reason."""
    try:
        record = audit.append_record(endpoint, raw_image_bytes, query, intent, answer)
        return {"record_id": record.record_id, "record_hash": record.record_hash}
    except Exception:
        logger.exception("Audit logging failed; continuing without it.")
        return None


@app.get("/health")
def health():
    return {
        "status": "ok",
        "step": "3 - +bi-temporal change detection, cryptographic audit trail",
    }


@app.post("/warmup")
def warmup():
    """Pre-load every model (triggering weight downloads if needed) ahead
    of the first real analysis request. Safe to call more than once —
    everything behind it is cached via lru_cache."""
    loaded: list[str] = []
    try:
        captioner.preload()
        loaded.append("blip-caption+vqa")
        classifier.preload()
        loaded.append("classifier")
    except Exception as e:
        logger.exception("Warmup failed")
        raise HTTPException(
            status_code=500,
            detail=f"Warmup failed after loading {loaded}: {e}",
        )
    return {"status": "warmed_up", "loaded": loaded}


@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    query: str = Form(...),
):
    """Single entry point for the single-image pipeline.

    Returns:
        intent          — task branch the router selected
        confidence      — router's similarity score for that intent
        confidence_tier — "high" | "medium" | "heuristic_fallback"
        answer          — the specialist model's natural-language answer
        top_labels      — present only for the "classify" intent
        evidence_trace  — ordered list of pipeline steps, for transparency
        audit           — {record_id, record_hash} if ledger logging succeeded
        crs             — source CRS string if the upload carried one, else null
        geo_bounds      — [west, south, east, north] in EPSG:4326 if
                           georeferenced, else null (never fabricated — see
                           image_utils.py)
    """
    trace = []
    t0 = time.time()

    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        loaded = load_image_any(raw)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    trace.append({
        "step": "input_validation",
        "detail": (
            f"size={loaded.image.size} bands={loaded.source_bands} "
            f"dtype={loaded.source_dtype} crs={loaded.crs or 'none'} "
            f"stretched_for_display={loaded.stretched}"
        ),
    })

    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="Query text is required.")

    routing = route_query(query)
    trace.append({
        "step": "query_routing",
        "detail": (
            f"intent={routing.intent} confidence={routing.confidence} "
            f"tier={routing.confidence_tier} raw_best={routing.raw_best_intent}"
        ),
    })
    if routing.note:
        trace.append({"step": "routing_note", "detail": routing.note})

    try:
        if routing.intent == "vqa":
            result = captioner.run_vqa(loaded.image, query)
            payload = {"answer": result.answer}
        elif routing.intent == "classify":
            result = classifier.classify_scene(loaded.image)
            payload = {
                "answer": result.top_labels[0][0],
                "top_labels": [{"label": l, "score": s} for l, s in result.top_labels],
            }
        else:
            result = captioner.run_caption(loaded.image)
            payload = {"answer": result.answer}
    except Exception as e:
        logger.exception("Specialist inference failed")
        raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")

    trace.append({
        "step": "specialist_inference",
        "detail": f"model={result.model_used} device={result.device}",
    })

    audit_info = _log_audit("/analyze", raw, query, routing.intent, payload["answer"])
    if audit_info:
        trace.append({"step": "audit_log", "detail": f"record_id={audit_info['record_id']}"})

    elapsed = round(time.time() - t0, 2)
    trace.append({"step": "done", "detail": f"total_time_s={elapsed}"})

    return {
        "intent": routing.intent,
        "confidence": routing.confidence,
        "confidence_tier": routing.confidence_tier,
        **payload,
        "evidence_trace": trace,
        "audit": audit_info,
        "crs": loaded.crs,
        "geo_bounds": loaded.geo_bounds,  # [west, south, east, north] in EPSG:4326, or null
    }


@app.post("/analyze-change")
async def analyze_change(
    image_before: UploadFile = File(...),
    image_after: UploadFile = File(...),
    query: str = Form("what changed between these two images?"),
    label_regions: bool = Form(True),
):
    """Bi-temporal change detection over an image pair.

    Returns deterministic metrics (never LLM-generated numbers — see
    `models/change_detector.py`), a base64 PNG heatmap, per-region
    bounding boxes with optional before/after scene labels, a plain-text
    summary built from those numbers, and a `needs_human_review` flag
    for cases where registration was shaky or dissimilarity was extreme.
    """
    import base64

    trace = []
    t0 = time.time()

    raw_before = await image_before.read()
    raw_after = await image_after.read()
    if not raw_before or not raw_after:
        raise HTTPException(status_code=400, detail="Both before and after images are required.")

    try:
        loaded_before = load_image_any(raw_before)
        loaded_after = load_image_any(raw_after)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    trace.append({
        "step": "input_validation",
        "detail": (
            f"before={loaded_before.image.size} bands={loaded_before.source_bands}; "
            f"after={loaded_after.image.size} bands={loaded_after.source_bands}"
        ),
    })

    w0, h0 = loaded_before.image.size
    w1, h1 = loaded_after.image.size
    aspect_before, aspect_after = w0 / h0, w1 / h1
    if abs(aspect_before - aspect_after) / aspect_before > 0.15:
        trace.append({
            "step": "sanity_check_warning",
            "detail": (
                f"Aspect ratios differ noticeably (before={aspect_before:.2f}, "
                f"after={aspect_after:.2f}) — pair may not be the same scene/framing."
            ),
        })

    try:
        result = change_detector.detect_change(
            loaded_before.image,
            loaded_after.image,
            classify_fn=_classify_label if label_regions else None,
        )
    except Exception as e:
        logger.exception("Change detection failed")
        raise HTTPException(status_code=500, detail=f"Change detection failed: {e}")

    trace.append({
        "step": "change_detection",
        "detail": (
            f"model={result.model_used} aligned={result.aligned} "
            f"reprojection_error_px={result.reprojection_error_px} "
            f"ssim={result.ssim_score} changed_area_fraction={result.changed_area_fraction}"
        ),
    })
    if result.needs_human_review:
        trace.append({
            "step": "human_escalation_flag",
            "detail": "Registration confidence or dissimilarity is outside the comfortable range — recommend a human sanity-check before acting on this result.",
        })

    combined_raw = raw_before + raw_after  # single fingerprint covering both inputs
    audit_info = _log_audit("/analyze-change", combined_raw, query, "change_detection", result.summary)
    if audit_info:
        trace.append({"step": "audit_log", "detail": f"record_id={audit_info['record_id']}"})

    elapsed = round(time.time() - t0, 2)
    trace.append({"step": "done", "detail": f"total_time_s={elapsed}"})

    return {
        "intent": "change_detection",
        "aligned": result.aligned,
        "reprojection_error_px": result.reprojection_error_px,
        "ssim_score": result.ssim_score,
        "changed_area_fraction": result.changed_area_fraction,
        "needs_human_review": result.needs_human_review,
        "summary": result.summary,
        "regions": [
            {
                "bbox": r.bbox,
                "area_fraction": r.area_fraction,
                "label_before": r.label_before,
                "label_after": r.label_after,
            }
            for r in result.regions
        ],
        "heatmap_png_base64": base64.b64encode(result.change_mask_png).decode("ascii"),
        "evidence_trace": trace,
        "audit": audit_info,
    }


@app.get("/audit/recent")
def audit_recent(limit: int = 20):
    """Most recent audit-ledger entries, newest first. Each entry stores
    a fingerprint of the input image, not the image itself."""
    return {"records": audit.read_records(limit=limit)}


@app.get("/audit/verify")
def audit_verify():
    """Recomputes the whole hash chain from genesis and reports whether
    it's intact — and if not, exactly which record broke it."""
    return audit.verify_chain()


# ---------------------------------------------------------------------------
# Scene registry (roadmap step 1) — real data the user dropped into data/aois
# ---------------------------------------------------------------------------
@app.get("/scenes")
def scenes_list():
    """Summaries of every AOI folder under data/aois, incl. which analyses
    each one supports. Reads file names and raster headers only."""
    return {"scenes": [scenes.scene_summary(s) for s in scenes.list_scenes()]}


@app.get("/scenes/{scene_id}")
def scenes_detail(scene_id: str):
    """Full detail for one AOI: acquisitions, band inventory, grid info,
    capability map (with plain-language 'requires' text), and any
    naming/format warnings found while scanning."""
    scene = scenes.get_scene(scene_id)
    if scene is None:
        raise HTTPException(status_code=404, detail=f"No scene named '{scene_id}' under data/aois.")
    return scenes.scene_to_dict(scene)


@app.get("/quicklooks")
def quicklooks_list():
    """Single-image files in data/quicklooks (simple one-image demos)."""
    return {"files": scenes.list_quicklooks()}
