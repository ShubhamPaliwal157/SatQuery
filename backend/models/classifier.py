"""
Zero-shot scene classifier — Step 2's new specialist
-------------------------------------------------------
Wraps RemoteCLIP (a CLIP model fine-tuned on remote-sensing image-text
pairs) for zero-shot classification / retrieval-style queries such as
"what type of land cover is this" or "is this urban or agricultural" —
things BLIP (a general-purpose captioner/VQA model, not remote-sensing
specific) is weaker at.

RemoteCLIP's checkpoints are published as raw PyTorch state-dicts on the
HuggingFace Hub (chendelong/RemoteCLIP), loaded on top of the `open_clip`
ViT-B/32 architecture rather than as a `transformers` model. Because that
download can fail (network hiccup, the repo/filename changing, a version
mismatch between this code and whatever ships next), this module degrades
gracefully: if the RemoteCLIP checkpoint can't be loaded, it falls back to
stock OpenCLIP weights (LAION-2B) so the "classify" intent still works —
with lower domain accuracy on satellite imagery — instead of taking the
whole app down. Whichever path is used is reported back in `model_used` so
it's visible in the evidence trace, not silently hidden.
"""

from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
import logging

import torch
from PIL import Image

logger = logging.getLogger("satquery.classifier")

REMOTECLIP_REPO = "chendelong/RemoteCLIP"
REMOTECLIP_FILENAME = "RemoteCLIP-ViT-B-32.pt"
OPEN_CLIP_ARCH = "ViT-B-32"
FALLBACK_PRETRAINED = "laion2b_s34b_b79k"  # stock weights if RemoteCLIP can't be fetched

PROMPT_TEMPLATE = "a satellite image of {label}"

# Generic remote-sensing scene / land-cover vocabulary. This is a plain
# list fed through CLIP's text encoder, not a trained classification head
# — extending it needs no retraining, just add a phrase.
CANDIDATE_LABELS = [
    "an airport", "bare land", "a beach", "a bridge", "a commercial area",
    "a dense residential area", "desert", "farmland", "a forest",
    "an industrial area", "a meadow", "mountains", "a parking lot",
    "a park", "a pond", "a port", "a railway station", "a river",
    "a sparse residential area", "a stadium", "storage tanks",
    "an urban area", "wetland", "clouds", "snow", "ocean", "a lake",
    "a highway", "agricultural land", "a construction site",
]


@dataclass
class ClassificationResult:
    top_labels: list[tuple[str, float]]  # (label, similarity score), sorted desc
    model_used: str
    device: str


@lru_cache(maxsize=1)
def _load_model():
    import open_clip

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(OPEN_CLIP_ARCH)
    tokenizer = open_clip.get_tokenizer(OPEN_CLIP_ARCH)

    model_used = "RemoteCLIP-ViT-B-32"
    try:
        from huggingface_hub import hf_hub_download

        ckpt_path = hf_hub_download(repo_id=REMOTECLIP_REPO, filename=REMOTECLIP_FILENAME)
        state_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state_dict)
        logger.info("Loaded RemoteCLIP domain-tuned weights.")
    except Exception as e:  # noqa: BLE001 — deliberately broad: this is a graceful-degrade path
        logger.warning(
            "Could not load RemoteCLIP checkpoint (%s). Falling back to stock "
            "OpenCLIP weights (%s). Classification will still work, just less "
            "specialized for satellite imagery.",
            e, FALLBACK_PRETRAINED,
        )
        model, _, preprocess = open_clip.create_model_and_transforms(
            OPEN_CLIP_ARCH, pretrained=FALLBACK_PRETRAINED
        )
        model_used = f"OpenCLIP-{OPEN_CLIP_ARCH}-{FALLBACK_PRETRAINED} (RemoteCLIP fallback)"

    model = model.to(device).eval()

    # Pre-encode the label bank once; reused across every request.
    text_tokens = tokenizer([PROMPT_TEMPLATE.format(label=l) for l in CANDIDATE_LABELS]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    return model, preprocess, text_features, device, model_used


def preload() -> None:
    """Force the model (and its checkpoint download) to load now, rather
    than on the first real request."""
    _load_model()


def classify_scene(image: Image.Image, top_k: int = 3) -> ClassificationResult:
    model, preprocess, text_features, device, model_used = _load_model()

    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)[0]

    scores, indices = similarity.topk(top_k)
    top_labels = [(CANDIDATE_LABELS[i], round(float(s), 3)) for s, i in zip(scores, indices)]

    return ClassificationResult(top_labels=top_labels, model_used=model_used, device=device)
