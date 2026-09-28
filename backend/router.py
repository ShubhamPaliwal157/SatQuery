"""
Query Router — Step 2
-------------------------
Classifies an incoming natural-language query into a task intent so the
orchestrator knows which specialist branch to call.

Active intents:
    - "caption"   : general "describe / what is in this image" requests
    - "vqa"       : specific yes/no or counting-style questions
    - "classify"  : "what type of area/land cover is this" style queries,
                    answered via RemoteCLIP zero-shot classification (new
                    in Step 2)

Still-future intents (kept as exemplars so routing quality doesn't need
reworking when they go live — they just move from "detected but
unsupported" to "active"):
    - "change_detection" : bi-temporal queries. As of Step 3 this is a
                            real feature — but it needs a *second* image,
                            which `/analyze`'s single-image contract can't
                            carry. So it's still "detected but redirected"
                            here: the router recognizes the intent and
                            points the user at `/analyze-change` / the
                            "Compare" tab instead of answering it wrong.
    - "sar_fusion"        : optical+SAR queries (needs an SAR band — Step 4)

Step 1 used one hard similarity threshold and always fell back to
"caption" below it, which quietly discarded information (a clearly
question-shaped query would still just get captioned, with no signal
that the router wasn't sure). Step 2 replaces that with a
confidence-calibrated chain:

    1. High confidence  (>= HIGH_CONF) -> trust the embedding router
    2. Medium confidence (>= LOW_CONF) -> use it, but flag the lower
       confidence in the evidence trace
    3. Low confidence                  -> fall back to a cheap keyword
       heuristic (question words -> vqa, otherwise -> caption) instead of
       always defaulting to caption

It's also honest when the *best-scoring* intent overall isn't active yet
(change_detection / sar_fusion) — that's surfaced explicitly in `note`
rather than silently swallowed, so the evidence trace can say "this looks
like a change-detection question, which lands in Step 3" instead of just
quietly captioning it and hoping nobody notices.
"""

from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache

from sentence_transformers import SentenceTransformer, util

INTENT_EXEMPLARS: dict[str, list[str]] = {
    "caption": [
        "describe this image",
        "what is in this image",
        "give me a summary of this scene",
        "what does this satellite image show",
        "provide a caption for this image",
    ],
    "vqa": [
        "how many buildings are in this image",
        "is there a road near the river",
        "what color is the vegetation",
        "does this image contain water bodies",
        "count the number of ships in the image",
    ],
    "classify": [
        "what type of land cover is this",
        "is this urban or agricultural",
        "what kind of terrain is shown here",
        "classify this scene",
        "what category of area is this",
    ],
    "change_detection": [
        "what changed between these two images",
        "compare before and after images",
        "has the land use changed over time",
    ],
    "sar_fusion": [
        "combine radar and optical data to identify the object",
        "use SAR imagery to confirm this feature",
    ],
}

ACTIVE_INTENTS = {"caption", "vqa", "classify"}

# Purely for user-facing messaging in the evidence trace.
INTENT_ETA = {
    "change_detection": "the 'Compare (Change Detection)' tab — it needs a second image",
    "sar_fusion": "Step 4",
}

HIGH_CONF = 0.5
LOW_CONF = 0.32

_QUESTION_WORDS = ("is ", "are ", "does ", "how many", "how much", "count", "?")


@dataclass
class RoutingResult:
    intent: str
    confidence: float
    confidence_tier: str          # "high" | "medium" | "heuristic_fallback"
    raw_best_intent: str          # whatever scored highest, even if inactive
    note: str | None = None       # human-readable aside for the evidence trace


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")


@lru_cache(maxsize=1)
def _get_exemplar_embeddings():
    model = _get_model()
    flat_texts, flat_labels = [], []
    for intent, phrases in INTENT_EXEMPLARS.items():
        for p in phrases:
            flat_texts.append(p)
            flat_labels.append(intent)
    embeddings = model.encode(flat_texts, convert_to_tensor=True)
    return embeddings, flat_labels


def _keyword_fallback(query: str) -> str:
    q = query.lower().strip()
    if q.startswith(("what", "describe", "summarize", "summarise")):
        return "caption"
    if any(w in q for w in _QUESTION_WORDS):
        return "vqa"
    return "caption"


def route_query(query: str) -> RoutingResult:
    model = _get_model()
    exemplar_embeddings, labels = _get_exemplar_embeddings()

    query_embedding = model.encode(query, convert_to_tensor=True)
    sims = util.cos_sim(query_embedding, exemplar_embeddings)[0]

    # Best score per intent (each intent has several exemplars).
    best_per_intent: dict[str, float] = {}
    for score, label in zip(sims.tolist(), labels):
        if score > best_per_intent.get(label, -1.0):
            best_per_intent[label] = score

    raw_best_intent = max(best_per_intent, key=best_per_intent.get)
    raw_best_score = best_per_intent[raw_best_intent]

    active_scored = {k: v for k, v in best_per_intent.items() if k in ACTIVE_INTENTS}
    best_active_intent = max(active_scored, key=active_scored.get)
    best_active_score = active_scored[best_active_intent]

    note = None
    if raw_best_intent not in ACTIVE_INTENTS and raw_best_score >= LOW_CONF:
        eta = INTENT_ETA.get(raw_best_intent, "a later step")
        if raw_best_intent == "change_detection":
            note = (
                "This reads like a change-detection question, which needs two "
                f"images. Use {eta} rather than a single-image upload — "
                f"answering with '{best_active_intent}' here instead."
            )
        else:
            note = (
                f"This reads like a '{raw_best_intent}' question, which isn't "
                f"available until {eta} yet — answering with '{best_active_intent}' instead."
            )

    if best_active_score >= HIGH_CONF:
        return RoutingResult(
            intent=best_active_intent,
            confidence=round(best_active_score, 3),
            confidence_tier="high",
            raw_best_intent=raw_best_intent,
            note=note,
        )
    if best_active_score >= LOW_CONF:
        return RoutingResult(
            intent=best_active_intent,
            confidence=round(best_active_score, 3),
            confidence_tier="medium",
            raw_best_intent=raw_best_intent,
            note=note,
        )

    heuristic_intent = _keyword_fallback(query)
    return RoutingResult(
        intent=heuristic_intent,
        confidence=round(best_active_score, 3),
        confidence_tier="heuristic_fallback",
        raw_best_intent=raw_best_intent,
        note=note or "Router confidence was low; used a keyword heuristic instead.",
    )
