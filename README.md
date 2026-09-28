# SatQuery AI — SIH26167

An interactive vision-language assistant for remote-sensing image analysis
through text queries, built **top-down**: every step below is a complete,
runnable project. Later steps add specialists and modalities on top of the
same API contracts and UI — nothing gets rewritten from scratch.

## Step 3 (this drop)

Step 2 flagged two things as "detected but not built yet" in its routing
notes: change detection and SAR fusion. Step 3 builds the first one for
real, plus a cryptographic audit trail, plus a full UI pass — without
touching the existing `/analyze` contract.

**New:**
- **Bi-temporal change detection (`POST /analyze-change`)** — upload a
  before/after image pair and get back a deterministic diff: images are
  co-registered with ORB keypoint matching + RANSAC homography (so an
  orbit/camera shift between passes isn't mistaken for ground change),
  then compared with structural similarity (SSIM) to produce a changed-area
  percentage, a red-overlay heatmap, and per-region bounding boxes. Each
  region can optionally be labeled by the existing zero-shot classifier on
  both frames (e.g. `farmland → a construction site`) — see the design
  note at the top of `backend/models/change_detector.py` for why every
  *number* here comes from OpenCV/scikit-image math and never from a
  language model, with the classifier only ever describing an already-
  detected region, never deciding whether something counts as change.
  Registration failures and extreme dissimilarity set a `needs_human_review`
  flag rather than silently reporting a shaky result as confident.
- **Cryptographic audit trail (`backend/audit.py`)** — every `/analyze`
  and `/analyze-change` call is appended to a local SHA-256 hash-chained
  ledger (`GET /audit/recent`, `GET /audit/verify`). Each record's hash
  covers the previous record's hash, so editing any past entry breaks the
  chain from that point on — `verify_chain()` recomputes the whole thing
  from genesis and reports exactly which record it would break at, rather
  than just asserting the log is trustworthy. Only a SHA-256 fingerprint
  of each uploaded image is stored, not the image itself. (Being precise
  about scope: this is a local, inspectable hash chain, not hardware TEE
  attestation — an honest claim beats an inflated one.)
- **Frontend rebuilt around a chat-style layout** — closer to a
  ChatGPT/Claude conversation surface than a form-and-button dashboard.
  Three views from the sidebar: **Analyze** (upload once, then ask
  follow-up questions about the same image as a running chat), **Compare**
  (the new change-detection flow, with metric cards and the heatmap), and
  **Audit Trail** (chain-verify button + recent ledger entries). Custom
  CSS trims Streamlit's default chrome and alert-box styling down to a
  quiet, mostly-monochrome look with one accent color.
- **Router honesty update** — a single-image query that reads like a
  change-detection question ("what changed between these...") now tells
  the user to switch to the Compare tab, instead of the old "not
  available until Step 3" message (see `router.py`).

### Step 3.1 — sidebar + visualization pass (same drop)

Follow-up UI pass on top of the above, no backend contract changes:

- **Chat history** — the Analyze view now keeps multiple saved
  conversations (one per uploaded image), listed and switchable from the
  sidebar, with a "New chat" action and per-conversation delete. Previously
  there was only one running chat that silently reset on re-upload.
- **No emoji, anywhere** — nav, buttons, and expanders use Streamlit's
  built-in Material Symbols shortcode icons (`:material/name:`) instead;
  the browser tab icon is a small flat geometric mark generated at
  runtime with Pillow (a ring + node on a rounded square) instead of an
  emoji favicon.
- **Actual visual grounding, not just numbers in a list** —
  `change_detector.py` now burns numbered white boxes onto the heatmap
  itself (server-side, via OpenCV), matching the numbering in the region
  list underneath, so a region can be looked at rather than only read as
  raw pixel coordinates. The Analyze and Compare views also gained a flat
  SVG confidence ring, flat CSS bar charts for classify-intent scores,
  and a 5-step pipeline-status strip (align / diff / regions / review
  flag / audit log) in Compare.

**Verified this pass** (same no-network constraint as before): the
region-numbering change to `change_detector.py` was re-run against the
same synthetic before/after pair from Step 3 and the output PNG was
visually inspected — the injected change region is correctly boxed and
labeled "1". `frontend/app.py`'s new logic (the favicon generator, the
bar-chart/ring HTML builders, the pipeline-status renderer, and the
chat-history session-state helpers) was exercised against a small hand-written
stub of the `streamlit` module (mirroring how `router.py` was unit-tested
in Step 2 with a stubbed embedding layer) — confirmed the module imports
cleanly, all three views run without exceptions, and the generated HTML/SVG
is well-formed. **Not yet verified:** how this actually looks and behaves
in a real browser — that still needs your machine.

**Still future** (interfaces already shaped so these are additions, not
rewrites):
- Optical + SAR fusion (two-branch CNN, extend `/analyze` to accept an
  optional SAR band)
- A geolocated map view. A full 3D globe (CesiumJS) is a genuinely
  different frontend stack (React/TypeScript, not Streamlit) — worth
  doing as its own step rather than bolted awkwardly onto this one; a
  Leaflet 2D map is the more natural next increment on the current stack.
- Deployment to HuggingFace Spaces / a public URL

### Step 3.2 — theme fix, demo mode, map view, overview tab (same drop)

Feedback after actually opening Step 3.1 in a browser: the sidebar rendered
with a hardcoded light background while the rest of the app was dark
(whatever the viewer's OS/browser preference happened to trigger for
Streamlit's own chrome), and several buttons had no explicit width so they
sized to their own text instead of lining up with their neighbors. Root
cause and fix:

- **Theme fixed at the root, not patched per-element.** Added
  `frontend/.streamlit/config.toml` with an explicit dark theme, so every
  native widget — sidebar, file uploader, chat input, buttons, expanders —
  is dark by construction and can't drift out of sync with the rest of the
  app again. The old CSS block only ever styled *some* elements and
  hardcoded a light sidebar background regardless of theme; that rule is
  gone, and the remaining custom CSS (cards, bars, the confidence ring, the
  new coverage table) reads from the same color palette as the config.
- **Button widths fixed explicitly.** Every button that should line up now
  passes `use_container_width=True` in Python — the sidebar nav, "New
  chat", each saved conversation. The one button that's deliberately small
  (the per-chat delete "×") sits in a narrow column on purpose, not by
  accident.
- **Bundled demo scenes** (`frontend/demo_assets/`, generated by
  `tools/generate_demo_assets.py` — procedurally drawn, so there's no
  copyright question and no internet dependency): a georeferenced farmland
  scene, a non-georeferenced coastal urban scene, and a before/after
  "vacant lot → new construction" pair. One-click "load demo scene"
  buttons in both Analyze and Compare mean a judge (or you, mid-demo)
  never needs a real dataset on hand. The before/after pair was run
  through the actual `change_detector.py` pipeline (not faked) —
  SSIM 0.729, the new building correctly boxed, ORB+RANSAC alignment
  under 1px reprojection error.
- **Example-query chips** on a fresh Analyze conversation ("Describe this
  scene", "What type of land use is this?", etc.), sharing one
  `submit_query()` helper with the real chat input so both paths behave
  identically — this also removed some duplicated turn-handling logic
  that Step 3.1 had inline.
- **New Map view.** `image_utils.py` now extracts real geo bounds
  (reprojected to EPSG:4326 via `rasterio.warp.transform_bounds`) when a
  GeoTIFF carries a CRS, returned from `/analyze` as `geo_bounds` /
  `crs`. The Map tab shows that footprint on a Leaflet/OpenStreetMap
  panel. A plain JPEG/PNG has no coordinates, and the empty state says
  exactly that instead of guessing — nothing here is fabricated.
- **New Overview tab.** A plain "what's actually built vs. still roadmap"
  coverage table, so the app's own landing screen states its scope
  honestly rather than implying more than what's shipped.

**Verified this pass**: every file changed passes a syntax compile check.
`change_detector.py` was re-run directly (not through the API) against
the newly generated demo before/after pair and produced a correct result
(see above). The full frontend — all five views (Overview / Analyze /
Compare / Map / Audit trail), the demo-scene buttons, the example-query
chips, a full Compare run through to a rendered result, both the
georeferenced and non-georeferenced Map states, and the audit
verify button — was exercised end-to-end with Streamlit's real
`streamlit.testing.v1.AppTest` harness (with `requests` mocked so it
runs with no backend needed), not just a hand-rolled stub: zero
exceptions across every path. **Not yet verified:** pixel-level visual
polish in an actual browser (AppTest confirms the script runs correctly
and renders the right components, not exact spacing/colors) — worth a
quick look on your machine before a live demo, though the theme is now
driven by Streamlit's own config mechanism rather than ad-hoc CSS, which
is a much smaller surface for anything to visually drift.

## What's been verified

Same constraint as Step 2: this build environment has no live internet
access to download BLIP/RemoteCLIP weights or install `fastapi`/`torch`/
`streamlit`, so the full pipeline hasn't been run end-to-end here — that
still needs to happen on your machine (which is where Step 2 was actually
manually tested). What *could* be verified without network access, was:

- All Python files pass a syntax compile check.
- `backend/models/change_detector.py` was run directly against synthetic
  before/after NumPy image pairs (not through the API): confirmed it
  correctly (a) detects and corrects a deliberate pixel shift via
  ORB+RANSAC, (b) finds and boxes an injected "changed" region at
  approximately the right location and size, (c) correctly wires a stub
  classifier into `label_before`/`label_after` for that region, and
  (d) falls back to `aligned=False` + `needs_human_review=True` on a pair
  of flat, featureless images where ORB can't find enough keypoints,
  instead of crashing or silently producing a meaningless diff.
- `backend/audit.py` was exercised directly: appended three records,
  confirmed `verify_chain()` reports valid; then hand-edited one record's
  `answer` field directly in the ledger file and confirmed `verify_chain()`
  correctly reports the chain broken at that exact record.

Not yet verified (needs your machine, same as Step 2): `/analyze-change`
and `/audit/*` through actual HTTP calls, the classifier wired to real
RemoteCLIP weights on real change regions, and the new Streamlit UI
rendering (chat history state, the Compare tab's metric cards, image
upload widgets) in an actual browser.

## Running it

Requires two terminals (backend + frontend) and Python 3.10+.

```bash
# Terminal 1 — backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Terminal 2 — frontend
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

Open the Streamlit URL it prints (usually http://localhost:8501). Click
**"Warm up models"** in the sidebar before your first real query — or,
while that's loading, click into **Overview** and load a bundled demo
scene in Analyze/Compare so there's something to look at immediately.

> **New dependencies this step:** `opencv-python-headless` and
> `scikit-image`, added to `backend/requirements.txt`. Both ship prebuilt
> wheels for common platforms, so this shouldn't reintroduce the
> "no C compiler" problem hit with Python 3.14 earlier — but if you're
> still on 3.14 rather than the 3.11 venv from that fix, that's worth
> double-checking first.

> **No new dependencies as of Step 3.2** — the theme fix, demo mode, and
> Map view all reuse packages already in the two `requirements.txt`
> files (rasterio was already there for GeoTIFF support; the Map tab's
> Leaflet/OpenStreetMap panel loads from a CDN in the browser, same as
> the Leaflet.js choice already made for this stack). The demo scenes
> under `frontend/demo_assets/` are pre-generated and committed — you
> don't need to run `tools/generate_demo_assets.py` yourself unless you
> want to regenerate or customize them (needs `numpy`, `pillow`,
> `rasterio`, already covered by `backend/requirements.txt`).

## Manual test plan for this step

1. Take any two photos of roughly the same scene (a NASA/Unsplash aerial
   shot works, or literally two photos of your desk a few minutes apart)
   and run them through **Compare**. You should see a heatmap, an SSIM
   score noticeably below 1.0, and at least one region boxed.
2. Run the exact same image as both "before" and "after" — SSIM should
   land very close to 1.0 and changed-area should be near 0%.
3. Open **Audit Trail** after a few queries and click "Verify chain
   integrity" — should report intact. Then (for fun / to prove the
   point) manually open `backend/data/audit_ledger.jsonl` in a text
   editor, change one character in any `"answer"` field, save, and
   verify again — it should now report exactly which record broke.

## Project layout

```
satquery-mvp/
├── backend/
│   ├── main.py                # FastAPI orchestrator — /analyze, /analyze-change,
│   │                           #   /warmup, /health, /audit/recent, /audit/verify
│   ├── router.py               # Query intent routing (calibrated fallback chain)
│   ├── image_utils.py          # Robust image loading — GeoTIFF/multi-band/16-bit aware
│   ├── audit.py                # SHA-256 hash-chained audit ledger
│   ├── scenes.py               # Scene registry: scans data/aois, gates analyses on what files exist
│   ├── ingest.py               # Raw downloads -> one-grid AOI bundle + validation report
│   ├── tests/                  # python -m unittest tests.test_scenes tests.test_ingest -v
│   ├── models/
│   │   ├── captioner.py        # BLIP caption + VQA specialist
│   │   ├── classifier.py       # RemoteCLIP zero-shot scene classifier
│   │   └── change_detector.py  # Bi-temporal change detection (deterministic)
│   ├── data/                    # audit_ledger.jsonl lives here (gitignored)
│   └── requirements.txt
├── frontend/
│   ├── .streamlit/config.toml   # Dark theme — single source of truth for the palette
│   ├── app.py                   # Streamlit UI — Overview / Analyze / Compare / Map / Audit trail
│   ├── demo_assets/              # Bundled procedurally-generated demo scenes (pre-built)
│   │   ├── farmland.tif           # Georeferenced — powers the Map-tab demo too
│   │   ├── coastal_urban.png       # Not georeferenced on purpose (Map's honest fallback)
│   │   ├── change_before.tif       # Before/after pair for the Compare demo
│   │   └── change_after.tif
│   └── requirements.txt
├── data/                        # YOUR real satellite data — see data/README.md
├── tools/
│   ├── make_test_geotiff.py
│   ├── generate_demo_assets.py  # Regenerates frontend/demo_assets/ if you ever need to
│   ├── check_scenes.py          # What the app sees under data/aois (CAN DO / MISSING)
│   └── ingest_scene.py          # inbox/ -> data/aois/<aoi>/  (use --dry-run first)
└── README.md
```

## Working with real satellite data (analyst-grade build plan)

Alongside the chat app there is a data pipeline being built step by step
(status table in `SESSION_SUMMARY.md`). What exists today, all command-line:

1. Put downloads (Sentinel-2 L2A, Sentinel-1 backscatter, a DEM) in `data/inbox/`.
2. `python tools/ingest_scene.py <aoi_name> --dry-run`, read the report, then run it without
   `--dry-run`. This builds `data/aois/<aoi_name>/` on a single shared grid.
3. `python tools/check_scenes.py` lists which analyses that data unlocks and what each missing
   one needs; the same information is served at `GET /scenes`.

Nothing in the browser UI reads these scenes yet; that arrives with the layer/canvas steps.
Details and limits: `data/README.md`.

## Roadmap (each step ships a complete app)

1. ✅ MVP — single image, caption + VQA
2. ✅ RemoteCLIP scene classification, robust GeoTIFF loading, calibrated routing
3. ✅ Bi-temporal change detection, cryptographic audit trail, chat-style UI,
   consistent dark theme, bundled demo mode, Leaflet map view, overview/
   coverage tab (this drop, Steps 3–3.2)
4. Optical + SAR fusion — two-branch CNN, extend `/analyze` to accept an
   optional SAR band
5. A CesiumJS 3D globe, if there's appetite for a React frontend alongside
   this one — the 2D Leaflet map already covers the geolocated-footprint
   need on the current stack
6. Package + deploy to a public URL (HuggingFace Spaces / Render /
   Vercel); W&B experiment tracking for any fine-tuning done on Colab

Tell me when you want step 4 and I'll build directly on top of this.
