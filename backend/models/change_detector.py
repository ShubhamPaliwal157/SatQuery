"""
Bi-temporal change detection specialist — Step 3
=====================================================
Answers "what changed between these two images" style queries given a
before/after image pair.

Design principle this module holds itself to: **every number comes from
a deterministic, inspectable computation, never from a language model.**
SSIM, the changed-area percentage, region bounding boxes and the
alignment error are all plain OpenCV / scikit-image math over the pixel
arrays. The only place a learned model is allowed to speak is labeling
*what* a specific already-detected region looks like (via the existing
RemoteCLIP classifier), and even then the two labels are just shown
side by side — the model never gets asked to invent the percentage or
decide whether something "counts" as change.

Pipeline:
    1. Resize "after" to "before"'s dimensions.
    2. Co-register with ORB keypoint matching + RANSAC homography, so a
       simple camera/orbit shift between passes doesn't get mistaken for
       ground change. If registration doesn't find enough confident
       matches, this is reported honestly (`aligned=False`) instead of
       silently producing a noisy diff.
    3. Structural similarity (SSIM) between the aligned grayscale pair
       gives both a single similarity score and a per-pixel dissimilarity
       map, which is thresholded (Otsu) and cleaned up morphologically
       into a binary change mask.
    4. Connected components on that mask become discrete "change
       regions", ranked by area; each region is optionally cropped from
       both frames and passed through the existing zero-shot scene
       classifier so the report can say e.g. "farmland -> a construction
       site" for the largest region, not just "14.8% of the frame changed".
    5. A red-overlay heatmap image is produced for the UI.

Known limitations (flagged, not hidden): this is intensity/structure-based
change detection, not a trained change-detection network — it will flag
illumination and seasonal differences as "change" alongside real ones,
and homography-based registration assumes a roughly planar scene (it will
struggle with very hilly terrain or very large time gaps with heavy cloud
cover). Good enough to be a real, honest Step 3; a learned bi-temporal
model would be the natural next iteration.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import io
import logging

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

logger = logging.getLogger("satquery.change_detector")

MIN_REGION_AREA_FRACTION = 0.002   # ignore speckle smaller than 0.2% of the frame
MAX_REPORTED_REGIONS = 5


@dataclass
class ChangeRegion:
    bbox: tuple[int, int, int, int]     # (x, y, w, h) in "before" pixel coords
    area_fraction: float
    label_before: str | None = None
    label_after: str | None = None


@dataclass
class ChangeResult:
    aligned: bool
    reprojection_error_px: float | None
    ssim_score: float
    changed_area_fraction: float
    change_mask_png: bytes              # PNG bytes, red overlay of the change mask
    regions: list[ChangeRegion] = field(default_factory=list)
    summary: str = ""
    needs_human_review: bool = False    # flagged when registration/quality is shaky
    model_used: str = "opencv-orb-ransac+ssim-diff"


def _align(gray_before: np.ndarray, gray_after: np.ndarray):
    """ORB + RANSAC homography to warp `gray_after` onto `gray_before`'s
    frame. Returns (warped_after, aligned_ok, reprojection_error_px)."""
    orb = cv2.ORB_create(nfeatures=2000)
    kp1, des1 = orb.detectAndCompute(gray_before, None)
    kp2, des2 = orb.detectAndCompute(gray_after, None)
    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return gray_after, False, None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(matcher.match(des1, des2), key=lambda m: m.distance)
    if len(matches) < 10:
        return gray_after, False, None

    good = matches[: max(30, int(len(matches) * 0.3))]
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    homography, inlier_mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
    if homography is None:
        return gray_after, False, None

    h, w = gray_before.shape
    warped = cv2.warpPerspective(gray_after, homography, (w, h))

    reproj_err = None
    if inlier_mask is not None and inlier_mask.ravel().astype(bool).any():
        inliers = inlier_mask.ravel().astype(bool)
        p1 = pts1[inliers].reshape(-1, 2)
        p2 = pts2[inliers].reshape(-1, 2)
        p2_proj = cv2.perspectiveTransform(p2.reshape(-1, 1, 2), homography).reshape(-1, 2)
        reproj_err = float(np.mean(np.linalg.norm(p1 - p2_proj, axis=1)))

    return warped, True, reproj_err


def _change_mask(gray_before: np.ndarray, gray_after: np.ndarray):
    score, diff = ssim(gray_before, gray_after, full=True)
    dissimilarity = ((1.0 - diff) * 255).astype("uint8")
    _, mask = cv2.threshold(dissimilarity, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return float(score), mask


def _extract_regions(mask: np.ndarray, img_before: Image.Image, img_after: Image.Image, classify_fn):
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    img_area = mask.shape[0] * mask.shape[1]
    candidates = sorted(range(1, num_labels), key=lambda i: -stats[i, cv2.CC_STAT_AREA])

    regions: list[ChangeRegion] = []
    for i in candidates:
        area = int(stats[i, cv2.CC_STAT_AREA])
        area_fraction = area / img_area
        if area_fraction < MIN_REGION_AREA_FRACTION:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        region = ChangeRegion(bbox=(x, y, w, h), area_fraction=round(area_fraction, 4))

        if classify_fn is not None:
            try:
                region.label_before = classify_fn(img_before.crop((x, y, x + w, y + h)))
                region.label_after = classify_fn(img_after.crop((x, y, x + w, y + h)))
            except Exception:
                logger.exception("Region-level classification failed; leaving labels unset.")

        regions.append(region)
        if len(regions) >= MAX_REPORTED_REGIONS:
            break
    return regions


def _heatmap_png(base_image: Image.Image, mask: np.ndarray, regions: list[ChangeRegion]) -> bytes:
    """Red change-mask overlay, plus a numbered white box drawn on each
    reported region — so the region list in the UI (which uses the same
    1-based numbering, in the same order) can be visually located on the
    image instead of read off as raw pixel coordinates."""
    base = np.array(base_image).copy()
    overlay = base.copy()
    overlay[mask > 0] = [255, 60, 60]
    blended = cv2.addWeighted(base, 0.55, overlay, 0.45, 0)

    for i, region in enumerate(regions, start=1):
        x, y, w, h = region.bbox
        cv2.rectangle(blended, (x, y), (x + w, y + h), (255, 255, 255), 2)
        label_pos = (x + 4, max(18, y - 6))
        # black outline + white fill so the number stays legible on any background
        cv2.putText(blended, str(i), label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(blended, str(i), label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

    buf = io.BytesIO()
    Image.fromarray(blended).save(buf, format="PNG")
    return buf.getvalue()


def _summarize(ssim_score, changed_fraction, regions, aligned_ok, reproj_err) -> tuple[str, bool]:
    needs_review = False
    lines = []
    if not aligned_ok:
        lines.append(
            "Could not confidently co-register the two frames (too few matched "
            "keypoints) — the diff below may include shift/orbit artifacts, not "
            "just real ground change."
        )
        needs_review = True
    else:
        lines.append(
            f"Frames aligned with ORB+RANSAC homography (mean reprojection error "
            f"{reproj_err:.2f}px)."
        )
        if reproj_err is not None and reproj_err > 3.0:
            needs_review = True

    lines.append(f"Structural similarity (SSIM) between the two scenes: {ssim_score:.3f} (1.0 = identical).")
    lines.append(f"Approximately {changed_fraction * 100:.1f}% of the frame shows significant pixel-level change.")

    if regions:
        lines.append(
            f"{len(regions)} distinct change region(s) found; the largest covers "
            f"{regions[0].area_fraction * 100:.2f}% of the frame."
        )
        shifted = [r for r in regions if r.label_before and r.label_after and r.label_before != r.label_after]
        if shifted:
            r = shifted[0]
            lines.append(f"Largest labeled shift: '{r.label_before}' \u2192 '{r.label_after}'.")
    else:
        lines.append("No region exceeded the minimum size threshold to be reported individually.")

    if changed_fraction < 0.005 and ssim_score > 0.97:
        needs_review = False
    elif ssim_score < 0.4:
        needs_review = True  # extreme dissimilarity is itself worth a human glance

    return " ".join(lines), needs_review


def detect_change(img_before: Image.Image, img_after: Image.Image, classify_fn=None) -> ChangeResult:
    """classify_fn: optional callable(PIL.Image) -> str, used to label
    individual change regions (wire in classifier.classify_scene's top
    label). Left None, regions are still detected, just unlabeled."""
    w, h = img_before.size
    img_after_r = img_after.resize((w, h))

    gray_before = cv2.cvtColor(np.array(img_before), cv2.COLOR_RGB2GRAY)
    gray_after = cv2.cvtColor(np.array(img_after_r), cv2.COLOR_RGB2GRAY)

    aligned_after, aligned_ok, reproj_err = _align(gray_before, gray_after)
    score, mask = _change_mask(gray_before, aligned_after)
    changed_fraction = float(np.count_nonzero(mask)) / mask.size

    regions = _extract_regions(mask, img_before, img_after_r, classify_fn)
    heatmap = _heatmap_png(img_before, mask, regions)
    summary, needs_review = _summarize(score, changed_fraction, regions, aligned_ok, reproj_err)

    model_used = "opencv-orb-ransac+ssim-diff"
    if classify_fn is not None:
        model_used += "+remoteclip-region-labels"

    return ChangeResult(
        aligned=aligned_ok,
        reprojection_error_px=round(reproj_err, 2) if reproj_err is not None else None,
        ssim_score=round(score, 4),
        changed_area_fraction=round(changed_fraction, 4),
        change_mask_png=heatmap,
        regions=regions,
        summary=summary,
        needs_human_review=needs_review,
        model_used=model_used,
    )
