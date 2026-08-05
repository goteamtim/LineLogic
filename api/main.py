"""
LineLogic Classifier — FastAPI Inference Server

Loads the trained CNN + label map once at startup.
POST /predict  — base64 PNG → top-k predictions
GET  /categories — list of all 345 category names
GET  /health   — liveness check
Static files   — serves ../frontend/ at /
"""

import base64
import io
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
RUN_DIR = ROOT / "linelogic_output" / "20260729_220240"
MODEL_PATH = RUN_DIR / "linelogic_cnn.keras"
LABEL_MAP_PATH = RUN_DIR / "label_map.json"
FRONTEND_DIR = ROOT / "frontend"

# ---------------------------------------------------------------------------
# Module-level singletons (loaded once at startup)
# ---------------------------------------------------------------------------
_model = None
_label_map: dict[str, str] = {}
_categories: list[str] = []

# ---------------------------------------------------------------------------
# Model contract constants (must match training)
# ---------------------------------------------------------------------------
IMG_SIZE = 28
TARGET_FILL = 0.80


# ---------------------------------------------------------------------------
# Preprocessing (ported VERBATIM from linelogic_cnn_fromscratch.py step 9)
# ---------------------------------------------------------------------------
def preprocess_canvas_capture(img: Image.Image, ink_is_dark: bool = True, target_fill: float = TARGET_FILL) -> np.ndarray:
    """
    Convert raw canvas image to canonical model input.

    Raw canvas (any size, dark ink on light/transparent)
    → (28,28,1) float32, white-on-black, [0,1], centered

    Args:
        img: PIL.Image (RGBA/RGB/L), any size
        ink_is_dark: True if strokes are dark on light background
        target_fill: fraction of frame the drawing should fill (0.8 = 80%)

    Returns:
        (28,28,1) float32 array in [0,1]
    """
    # 1. Extract ink mask (high = ink)
    if img.mode == 'RGBA':
        # Transparent background: alpha channel = ink mask
        ink = np.array(img.split()[-1], dtype=np.float32) / 255.0
    else:
        g = np.array(img.convert('L'), dtype=np.float32) / 255.0
        ink = (1.0 - g) if ink_is_dark else g

    # 2. Find bounding box
    ys, xs = np.where(ink > 0.1)
    if len(xs) == 0:
        return np.zeros((IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)  # blank
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    crop = ink[y0:y1 + 1, x0:x1 + 1]

    # 3. Pad to square (preserve aspect ratio)
    h, w = crop.shape
    side = max(h, w)
    square = np.zeros((side, side), dtype=np.float32)
    square[(side - h) // 2:(side - h) // 2 + h,
           (side - w) // 2:(side - w) // 2 + w] = crop

    # 4. Resize to target fill, then center in 28x28
    inner = max(1, int(round(IMG_SIZE * target_fill)))
    resized = np.array(
        Image.fromarray((square * 255).astype(np.uint8))
             .resize((inner, inner), Image.LANCZOS),
        dtype=np.float32) / 255.0

    out = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    off = (IMG_SIZE - inner) // 2
    out[off:off + inner, off:off + inner] = resized

    return out[..., np.newaxis]


# ---------------------------------------------------------------------------
# Lifespan — load model + label map once
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _label_map, _categories

    import tensorflow as tf  # imported here to keep startup log clean
    print(f"Loading model from {MODEL_PATH} ...")
    _model = tf.keras.models.load_model(str(MODEL_PATH))
    print("Model loaded.")

    with open(LABEL_MAP_PATH, "r") as f:
        _label_map = json.load(f)

    # Build an ordered list of category names (index 0 … N-1)
    _categories = [_label_map[str(i)] for i in range(len(_label_map))]
    print(f"Label map loaded: {len(_categories)} categories.")

    yield  # app runs here


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="LineLogic Classifier", lifespan=lifespan)

# CORS — allow all origins (lock down per-origin via env var for production)
allowed_origins_env = os.environ.get("ALLOWED_ORIGINS", "*")
allowed_origins = allowed_origins_env.split(",") if allowed_origins_env != "*" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    image: str          # base64-encoded PNG (no data URI prefix)
    top_k: Optional[int] = 5


class Prediction(BaseModel):
    label: str
    confidence: float


class PredictResponse(BaseModel):
    blank: bool
    predictions: list[Prediction]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "categories": len(_categories)}


@app.get("/categories")
def categories():
    return {"categories": _categories}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # 1. Decode base64 → PIL Image
    try:
        # Strip data URI prefix if the client accidentally included it
        image_data = req.image
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]
        raw = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(raw))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image data")

    # 2. Preprocess (verbatim contract function)
    canvas = preprocess_canvas_capture(img, ink_is_dark=True)

    # 3. Blank canvas detection
    if canvas.max() < 0.05:
        return PredictResponse(blank=True, predictions=[])

    # 4. Inference
    x = canvas[np.newaxis, ...]           # (1, 28, 28, 1)
    probs = _model.predict(x, verbose=0)[0]  # (345,)

    # 5. Top-k
    top_k = max(1, min(req.top_k or 5, len(_categories)))
    idxs = np.argsort(probs)[::-1][:top_k]
    predictions = [
        Prediction(label=_label_map[str(int(i))], confidence=float(probs[i]))
        for i in idxs
    ]

    return PredictResponse(blank=False, predictions=predictions)


# ---------------------------------------------------------------------------
# Static files — serve frontend at /
# (mounted last so API routes take priority)
# ---------------------------------------------------------------------------
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
