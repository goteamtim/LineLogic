# LineLogic!

LineLogic is a real-time drawing-recognition game with a convolutional neural network (CNN) as a judge: you draw, the CNN guesses what it is each time you stop drawing, and you score points for being recognized fast. This is the same feel as Google's [Quick, Draw!](https://quickdraw.withgoogle.com/), built from scratch on the same [dataset](https://github.com/googlecreativelab/quickdraw-dataset).

The purpose of this project is to learn deep learning fundamentals by implementing the whole pipeline from data prep to a served model rather than fine-tuning something pretrained.

## Demo

Draw the prompted word on the canvas. The model re-guesses after every stroke and once its confidence for the target word clears a threshold, you win the round and it moves to the next word. Faster recognition = more points.

## Why a CNN trained from scratch

The first version of this project fine-tuned MobileNetV3 (ImageNet weights, 224×224 RGB input) on [QuickDraw sketches](https://github.com/googlecreativelab/quickdraw-dataset). I quickly realized this method wouldn't be realistic as my validation accuracy never made it above anything other than random chance. The problem was a domain gap: ImageNet features are tuned for photographic textures and color, and MobileNet's frozen BatchNorm statistics didn't adapt to sparse black-and-white line drawings. Upscaling 28×28 bitmaps to 224×224 didn't fix the underlying mismatch.

The fix was to stop trying to reuse a photo-classification model for a sketch-classification problem and instead train a small CNN directly on the native 28×28 grayscale bitmaps.

## Results

- **345 categories**, the full [QuickDraw label](https://github.com/googlecreativelab/quickdraw-dataset) set
- **55% top-1 test accuracy** across all 345 classes — for scale, random guessing is ~0.3%. Accuracy is naturally lower on the full label set than on a handful of categories (a 5–10 category subset gets ~85–90%), which is why the game scores on whether the target appears in the model's **top-k** guesses rather than requiring an exact top-1 match.
- Hyperparameters were tuned with [Optuna](https://optuna.org/) on a category subset, then reused for the final training run on all 345 classes.
- Per-category accuracy breakdown is saved with each training run for finding weak/confusable classes (`linelogic_output/<run>/per_category_accuracy.json`).

## How it works

**Model contract** — the model only accepts a specific canonical input, and training and serving share the exact same preprocessing function so there's no train/serve skew:

- `(28, 28, 1)` float32, values in `[0, 1]`
- White ink on a black background (inverse of a typical canvas)
- Drawing cropped to its bounding box, padded to square, centered, and scaled to fill ~80% of the frame

**Game loop** (all client-side, server stays a stateless predictor):

- Inference fires when a stroke ends (pointer-up), not on a timer; rate limited to avoid spamming the API mid-scribble
- Live guesses are hidden until confidence clears a floor, so the first meaningless strokes don't show a random guess
- The round is won the instant the target word appears in the model's top-k above that confidence floor; score is based on time-to-recognition

## Architecture

```
frontend/index.html      Canvas UI + game loop (vanilla JS, no build step)
api/main.py               FastAPI inference server (/predict, /categories, /health)
linelogic_cnn_fromscratch.py   Training script: data prep, Optuna tuning, final training
linelogic_output/<run>/   Trained model + label map + metrics per run
Dockerfile                 Multi-stage build: lean runtime image, no training deps
```

The training script and the API server both use the same `preprocess_canvas_capture` function so a canvas drawing at inference time is preprocessed identically to how training images were prepared.

## Running it

**With Docker:**

```bash
docker build -t linelogic .
docker run -p 8000:8000 linelogic
```

Then open `http://localhost:8000`.

**Locally, for development:**

```bash
python -m venv venv && source venv/bin/activate
pip install -r api/requirements.txt
uvicorn api.main:app --reload
```

`ALLOWED_ORIGINS` controls CORS and defaults to `*`; override it for a production deployment (`ALLOWED_ORIGINS="https://yourdomain.com"`).

## Retraining

```bash
pip install -r requirements.txt   # includes training deps (Optuna, TF, etc.)
python linelogic_cnn_fromscratch.py
```

Produces a new timestamped run under `linelogic_output/`. To serve a different run, update `RUN_DIR` in `api/main.py` and the model paths in the `Dockerfile`.
