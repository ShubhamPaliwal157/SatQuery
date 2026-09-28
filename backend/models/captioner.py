"""
Vision-Language specialist — Step 1 MVP
-----------------------------------------
Wraps BLIP (Salesforce/blip-image-captioning-base and
Salesforce/blip-vqa-base) to serve both the "caption" and "vqa" intents.

This is deliberately the ONE model the MVP depends on so the full pipeline
(upload -> route -> infer -> answer) works end-to-end on day one. RemoteCLIP,
the two-branch optical+SAR fusion CNN, and a dedicated change-detection model
are dropped in as additional specialists in later build steps behind the
same `run_caption` / `run_vqa` style interface — the orchestrator and
frontend will not need to change when that happens.

Models download from the HuggingFace Hub on first run, so this file needs
an internet connection the first time it's executed (e.g. on Colab or any
machine with normal internet access) — after that they're cached locally.
"""

from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from PIL import Image

import torch
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    BlipForQuestionAnswering,
)

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class InferenceResult:
    answer: str
    model_used: str
    device: str


@lru_cache(maxsize=1)
def _get_caption_pipeline():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(_DEVICE)
    return processor, model


@lru_cache(maxsize=1)
def _get_vqa_pipeline():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained(
        "Salesforce/blip-vqa-base"
    ).to(_DEVICE)
    return processor, model


def preload() -> None:
    """Force both BLIP pipelines (and their weight downloads) to load now,
    rather than on the first real request."""
    _get_caption_pipeline()
    _get_vqa_pipeline()


def run_caption(image: Image.Image) -> InferenceResult:
    processor, model = _get_caption_pipeline()
    inputs = processor(image, return_tensors="pt").to(_DEVICE)
    out = model.generate(**inputs, max_new_tokens=40)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return InferenceResult(
        answer=caption,
        model_used="blip-image-captioning-base",
        device=_DEVICE,
    )


def run_vqa(image: Image.Image, question: str) -> InferenceResult:
    processor, model = _get_vqa_pipeline()
    inputs = processor(image, question, return_tensors="pt").to(_DEVICE)
    out = model.generate(**inputs, max_new_tokens=20)
    answer = processor.decode(out[0], skip_special_tokens=True)
    return InferenceResult(
        answer=answer,
        model_used="blip-vqa-base",
        device=_DEVICE,
    )
