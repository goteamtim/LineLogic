#!/usr/bin/env python3
"""
LineLogic Classifier - CNN From Scratch (Scalable + Canvas-Robust)

A small CNN trained directly on native 28x28 QuickDraw bitmaps.

Designed to:
- Scale to all 345 categories (just edit the category list)
- Tune with Optuna on a small subset (fast), then train the full set
- Survive real user input from a drawing canvas via preprocessing

The model ONLY ever sees inputs in this exact canonical format:
- 28 x 28, single grayscale channel
- White ink (high values) on a black background (0)
- Values normalized to [0, 1]
- Drawing centered and scaled to fill ~80% of the frame

QuickDraw bitmaps already match this. preprocess_canvas_capture() (Step 9)
converts raw canvas images to match it. Training augmentation makes the model
tolerant of the residual gap.

BEFORE RUNNING:
    pip install -r requirements.txt
    
Or install manually:
    pip install tensorflow optuna scikit-learn pandas pillow matplotlib
"""

import os
import json
import gc
import urllib.request
import urllib.parse
from datetime import datetime
from collections import defaultdict

# Suppress TensorFlow C++ backend noise (CUDA not found, etc.) before importing TF
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import optuna
from optuna.trial import TrialState
import pandas as pd
from PIL import Image, ImageDraw

# =====================================================================
# SETUP: GPU Configuration & Project Paths
# =====================================================================

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

print("=" * 70)
print("LineLogic CNN Training")
print("=" * 70)
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {'Yes' if len(gpus) > 0 else 'No (CPU mode - still fast!)'}")
print(f"Optuna version: {optuna.__version__}")

# Local checkpoint directory - each run gets its own timestamped subdirectory
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_ID = datetime.now().strftime('%Y%m%d_%H%M%S')
CKPT_DIR = os.path.join(PROJECT_DIR, 'linelogic_output', RUN_ID)
os.makedirs(CKPT_DIR, exist_ok=True)
print(f"Run ID: {RUN_ID}")
print(f"Checkpoint directory: {CKPT_DIR}\n")


# =====================================================================
# STEP 1: Configuration
# =====================================================================
# Edit these to scale training or change behavior

# Category selection
USE_ALL_345 = True  # Set True to download all 345 QuickDraw categories
CUSTOM_CATEGORIES = [
    'apple', 'cat', 'dog', 'house', 'tree', 
    'star', 'car', 'fish', 'bird', 'flower'
]

if USE_ALL_345:
    cats_url = 'https://raw.githubusercontent.com/googlecreativelab/quickdraw-dataset/master/categories.txt'
    cats_file = os.path.join(PROJECT_DIR, 'categories.txt')
    if not os.path.exists(cats_file):
        print("Downloading official QuickDraw category list...")
        urllib.request.urlretrieve(cats_url, cats_file)
    else:
        print("Using cached categories.txt")
    with open(cats_file) as f:
        CATEGORIES = [line.strip() for line in f if line.strip()]
else:
    CATEGORIES = CUSTOM_CATEGORIES

# Tuning subset - Optuna searches on this (faster)
# Note for tuning/testing. I wonder what the difference it makes 
TUNING_CATEGORIES = CATEGORIES[:50] if len(CATEGORIES) >= 50 else CATEGORIES[:10]

# Model input contract
IMG_SIZE = 28
TARGET_FILL = 0.80  # drawing fills ~80% of the 28x28 frame

# Data loading
MAX_IMAGES_PER_CATEGORY = 1500  # Reduce for faster testing, raise for production

# Hyperparameter tuning
N_TRIALS = 20          # Number of Optuna trials
TUNE_EPOCHS = 7       # Epochs per trial (keep modest - tuning is expensive)
SKIP_TUNING = False    # Set True to reuse previously saved hyperparameters
LOAD_PARAMS_RUN_ID = None  # When SKIP_TUNING=True, set to a RUN_ID string e.g. '20260729_143022'

# Final training
FINAL_EPOCHS = 60      # Epochs for final model
FINAL_PATIENCE = 8     # Early stopping patience (epochs without improvement)

# Print config summary
print("CONFIGURATION")
print("-" * 70)
print(f"Total categories to train:  {len(CATEGORIES)}")
print(f"Tuning subset:              {len(TUNING_CATEGORIES)}")
print(f"Images per category:        {MAX_IMAGES_PER_CATEGORY}")
est_total = len(CATEGORIES) * MAX_IMAGES_PER_CATEGORY
est_gb = est_total * 784 / 1e9
print(f"Estimated total images:     {est_total:,} (~{est_gb:.2f} GB)")
print(f"Optuna trials:              {N_TRIALS}")
print()

# Save training configuration
TRAINING_CONFIG = {
    'run_id': RUN_ID,
    'categories': CATEGORIES,
    'num_categories': len(CATEGORIES),
    'tuning_categories': TUNING_CATEGORIES,
    'num_tuning_categories': len(TUNING_CATEGORIES),
    'max_images_per_category': MAX_IMAGES_PER_CATEGORY,
    'img_size': IMG_SIZE,
    'target_fill': TARGET_FILL,
    'n_trials': N_TRIALS,
    'tune_epochs': TUNE_EPOCHS,
    'final_epochs': FINAL_EPOCHS,
    'final_patience': FINAL_PATIENCE,
    'skip_tuning': SKIP_TUNING,
    'load_params_run_id': LOAD_PARAMS_RUN_ID,
}
os.makedirs(CKPT_DIR, exist_ok=True)
with open(f'{CKPT_DIR}/training_config.json', 'w') as f:
    json.dump(TRAINING_CONFIG, f, indent=2)
print(f"✓ Training config saved to {CKPT_DIR}/training_config.json")


# =====================================================================
# STEP 2: Download Data
# =====================================================================
# Downloads native 28x28 QuickDraw bitmaps as compact uint8 arrays

os.makedirs('linelogic_data', exist_ok=True)

def download_category(category):
    """
    Download one QuickDraw category and return the sliced uint8 array (N, 784).
    
    Optimizations:
    1. Handles space-in-name categories by percent-encoding the URL
    2. Avoids memory spike: memory-maps the full file, then copies only the
       slice we need. Deletes the raw file after.
    3. Caches the sliced version so re-runs are instant
    """
    safe = category.replace(' ', '_')
    cache = f"linelogic_data/{safe}_slice.npy"
    
    if os.path.exists(cache):
        return np.load(cache)

    url = ("https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"
           f"{urllib.parse.quote(category)}.npy")
    raw = f"linelogic_data/{safe}_raw.npy"
    
    print(f"  Downloading {category}...", end=" ", flush=True)
    urllib.request.urlretrieve(url, raw)
    print("✓")

    # Memory-map to avoid loading entire file into RAM
    mm = np.load(raw, mmap_mode='r')
    if mm.ndim == 3 and mm.shape[1] == 28:
        mm = mm.reshape(-1, 784)
    
    # Copy only the slice we need
    sliced = np.array(mm[:MAX_IMAGES_PER_CATEGORY], dtype=np.uint8)
    del mm

    # Cache the slice for fast re-runs
    np.save(cache, sliced)
    os.remove(raw)  # Free the large raw file
    return sliced


def load_categories(category_list):
    """
    Load a list of categories into compact uint8 arrays + label map.
    
    Returns:
        X: (N, 784) uint8 array of images
        y: (N,) int32 array of labels
        label_map: dict mapping str(idx) -> category_name
    """
    imgs, lbls, label_map = [], [], {}
    
    for idx, cat in enumerate(category_list):
        label_map[str(idx)] = cat
        try:
            arr = download_category(cat)
            imgs.append(arr)
            lbls.extend([idx] * len(arr))
        except Exception as e:
            print(f"  ✗ Skipped {cat}: {str(e)[:50]}")
        
        # Periodic cleanup
        if (idx + 1) % 25 == 0:
            gc.collect()
            print(f"  ...{idx + 1}/{len(category_list)} categories loaded")
    
    X = np.concatenate(imgs, axis=0).astype(np.uint8)
    y = np.array(lbls, dtype=np.int32)
    del imgs
    gc.collect()
    
    print(f"Total: {len(X)} images | {X.nbytes/1e6:.1f} MB uint8\n")
    return X, y, label_map


print("STEP 2: Downloading training data")
print("-" * 70)
X_full, y_full, label_map = load_categories(CATEGORIES)


# =====================================================================
# STEP 3: Streaming Datasets + Augmentation
# =====================================================================
# Augmentation makes the model robust to real canvas input variation

# Augmentation applied ONLY to training data
augment = models.Sequential([
    layers.RandomRotation(0.06),           # ±~11 degrees
    layers.RandomTranslation(0.10, 0.10),  # ±10% shift
    layers.RandomZoom(0.15),               # ±15% scale
], name='augment')


def stroke_jitter(batch):
    """
    Randomly thicken/thin strokes to mimic finger vs mouse drawing.
    
    Dilation (thicken) ~ max_pool
    Erosion (thin) ~ -max_pool(-x)
    
    Applied to entire batch for efficiency.
    """
    def same():
        return batch
    
    def thicken():
        return tf.nn.max_pool2d(batch, ksize=2, strides=1, padding='SAME')
    
    def thin():
        return -tf.nn.max_pool2d(-batch, ksize=2, strides=1, padding='SAME')
    
    choice = tf.random.uniform([], maxval=3, dtype=tf.int32)
    return tf.switch_case(choice, branch_fns=[same, thicken, thin])


def _to_canonical(img, lbl):
    # (784,) uint8 -> (28,28,1) float32 in [0,1]
    img = tf.reshape(img, (IMG_SIZE, IMG_SIZE, 1))
    img = tf.cast(img, tf.float32) / 255.0
    return img, lbl


def make_dataset(images, labels, batch_size, training=False):
    """
    Create a tf.data pipeline with optional augmentation.
    
    Pipeline:
    1. Load from tensors
    2. Shuffle + repeat (if training)
    3. Reshape & normalize to [0,1]
    4. Batch
    5. Apply augmentation (if training)
    6. Apply stroke jitter (if training)
    7. Prefetch for parallel loading
    """
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    
    if training:
        ds = ds.shuffle(buffer_size=min(4000, len(images))).repeat()

    ds = ds.map(_to_canonical, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    
    if training:
        ds = ds.map(lambda x, y: (augment(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.map(lambda x, y: (stroke_jitter(x), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    
    return ds.prefetch(tf.data.AUTOTUNE)


def split_and_build(X, y, batch_size):
    """
    Stratified 80/10/10 train/val/test split -> streaming datasets.
    """
    X_tmp, X_te, y_tmp, y_te = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tmp, y_tmp, test_size=0.11, random_state=42, stratify=y_tmp)
    del X_tmp, y_tmp
    gc.collect()
    
    return (
        make_dataset(X_tr, y_tr, batch_size, training=True),
        make_dataset(X_va, y_va, batch_size, training=False),
        make_dataset(X_te, y_te, batch_size, training=False),
        len(X_tr), len(X_va), len(X_te),
    )

print("STEP 3: Dataset pipeline + augmentation")
print("-" * 70)
print("✓ Pipeline ready\n")


# =====================================================================
# STEP 3b: Sanity-Check Augmentation (Optional)
# =====================================================================
# Uncomment to visualize what the model actually sees

def show_augmented_samples():
    """Visualize augmented training samples."""
    print("STEP 3b: Augmentation sanity check")
    print("-" * 70)
    
    _dbg = make_dataset(X_full[:256], y_full[:256], 16, training=True)
    for imgs, lbls in _dbg.take(1):
        break

    fig, axes = plt.subplots(2, 6, figsize=(14, 5))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(imgs[i].numpy().squeeze(), cmap='gray', vmin=0, vmax=1)
        ax.set_title(label_map[str(int(lbls[i]))], fontsize=9)
        ax.axis('off')
    plt.suptitle('What the model sees (augmented, white-on-black)', fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    del _dbg
    gc.collect()
    print()

# Uncomment to run:
# show_augmented_samples()


# =====================================================================
# STEP 4: Model Builder
# =====================================================================
# Parameterized so Optuna can tune the architecture

def build_model(num_classes, conv1=32, conv2=64, conv3=128,
                dense_units=128, dropout=0.3, learning_rate=1e-3):
    """
    Build a compact CNN.
    
    Architecture:
    - 3 conv blocks (Conv2D -> BatchNorm -> MaxPool)
    - GlobalAveragePooling
    - 2 dense layers with dropout
    - Softmax output
    
    All parameters are tunable by Optuna.
    Output layer automatically sizes to num_classes.
    """
    model = models.Sequential([
        layers.Input((IMG_SIZE, IMG_SIZE, 1)),
        
        layers.Conv2D(conv1, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        
        layers.Conv2D(conv2, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        
        layers.Conv2D(conv3, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        
        layers.GlobalAveragePooling2D(),
        layers.Dropout(dropout),
        
        layers.Dense(dense_units, activation='relu'),
        layers.Dropout(dropout),
        
        layers.Dense(num_classes, activation='softmax'),
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

print("STEP 4: Model builder")
print("-" * 70)
print("✓ Model builder ready\n")


# =====================================================================
# STEP 5: Optuna Hyperparameter Search (on tuning subset)
# =====================================================================

if not SKIP_TUNING:
    print("STEP 5: Optuna hyperparameter search")
    print("-" * 70)
    print(f"Loading {len(TUNING_CATEGORIES)} tuning categories...")
    X_tune, y_tune, _ = load_categories(TUNING_CATEGORIES)

    def objective(trial):
        """Optuna objective: train a model variant and return val accuracy."""
        tf.keras.backend.clear_session()
        gc.collect()

        # Suggest hyperparameters
        lr = trial.suggest_float('learning_rate', 1e-4, 3e-3, log=True)
        conv1 = trial.suggest_categorical('conv1', [32, 64])
        conv2 = trial.suggest_categorical('conv2', [64, 128])
        conv3 = trial.suggest_categorical('conv3', [128, 256])
        dense_units = trial.suggest_categorical('dense_units', [256, 512])
        dropout = trial.suggest_float('dropout', 0.2, 0.5, step=0.1)
        batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])

        try:
            tr, va, _, n_tr, _, _ = split_and_build(X_tune, y_tune, batch_size)
            model = build_model(len(TUNING_CATEGORIES), conv1, conv2, conv3,
                                dense_units, dropout, lr)
            
            hist = model.fit(
                tr, validation_data=va, epochs=TUNE_EPOCHS,
                steps_per_epoch=n_tr // batch_size,
                callbacks=[EarlyStopping('val_accuracy', patience=2,
                                        restore_best_weights=True)],
                verbose=0)
            
            val_acc = max(hist.history['val_accuracy'])
            del model, hist
            tf.keras.backend.clear_session()
            gc.collect()
            return val_acc
            
        except Exception as e:
            tf.keras.backend.clear_session()
            gc.collect()
            print(f"Trial error: {str(e)[:60]}")
            return 0.0

    # Run the search
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    BEST_PARAMS = study.best_params

    print("\n" + "=" * 70)
    print(f"Best tuning accuracy: {study.best_value:.2%}")
    print("Best hyperparameters:")
    for k, v in BEST_PARAMS.items():
        print(f"  {k:16} = {v}")
    print("=" * 70)

    # Save for future runs
    with open(f'{CKPT_DIR}/best_hyperparameters.json', 'w') as f:
        json.dump(BEST_PARAMS, f, indent=2)
    print(f"✓ Saved to {CKPT_DIR}/best_hyperparameters.json")

    # Save all trial results
    trials_df = study.trials_dataframe()
    trials_df.to_csv(f'{CKPT_DIR}/optuna_trials.csv', index=False)
    print(f"✓ All {len(trials_df)} Optuna trials saved to {CKPT_DIR}/optuna_trials.csv\n")

    # Check if params hit search ceiling
    ceilings = {'conv1': 64, 'conv2': 128, 'conv3': 256, 'dense_units': 512}
    pinned = [k for k, cap in ceilings.items() if BEST_PARAMS.get(k) == cap]
    if pinned:
        print(f"⚠ Warning: Params hit search ceiling on {pinned}")
        print("  Consider widening those ranges for better optimization.\n")

else:
    print("STEP 5: Optuna search")
    print("-" * 70)
    print("SKIP_TUNING=True - Loading saved hyperparameters...\n")
    params_dir = (os.path.join(PROJECT_DIR, 'linelogic_output', LOAD_PARAMS_RUN_ID)
                  if LOAD_PARAMS_RUN_ID else CKPT_DIR)
    with open(f'{params_dir}/best_hyperparameters.json') as f:
        BEST_PARAMS = json.load(f)
    
    print("Loaded hyperparameters:")
    for k, v in BEST_PARAMS.items():
        print(f"  {k:16} = {v}")
    print()


# =====================================================================
# STEP 6: Train Final Model on Full Dataset
# =====================================================================

print("STEP 6: Final training on full dataset")
print("-" * 70)

best = BEST_PARAMS
batch_size = best['batch_size']

train_ds, val_ds, test_ds, n_tr, n_va, n_te = split_and_build(
    X_full, y_full, batch_size)
print(f"Train: {n_tr} | Val: {n_va} | Test: {n_te}")

model = build_model(
    len(CATEGORIES),
    conv1=best['conv1'], conv2=best['conv2'], conv3=best['conv3'],
    dense_units=best['dense_units'], dropout=best['dropout'],
    learning_rate=best['learning_rate']
)
model.summary()

# Save metadata before training (so interrupted runs aren't lost)
with open(f'{CKPT_DIR}/label_map.json', 'w') as f:
    json.dump(label_map, f, indent=2)
with open(f'{CKPT_DIR}/best_hyperparameters.json', 'w') as f:
    json.dump(best, f, indent=2)

print("\nTraining...\n")
history = model.fit(
    train_ds, validation_data=val_ds, epochs=FINAL_EPOCHS,
    steps_per_epoch=n_tr // batch_size,
    callbacks=[
        ModelCheckpoint(f'{CKPT_DIR}/best_model.keras',
                       monitor='val_accuracy', save_best_only=True,
                       verbose=0),
        tf.keras.callbacks.CSVLogger(f'{CKPT_DIR}/training_log.csv', append=True),
        EarlyStopping('val_accuracy', patience=FINAL_PATIENCE,
                     restore_best_weights=True, verbose=1),
    ],
    verbose=1)

print("\n✓ Training complete!")
print(f"✓ Best checkpoint saved: {CKPT_DIR}/best_model.keras\n")


# =====================================================================
# STEP 6b: Recover from Checkpoint (if training was interrupted)
# =====================================================================
# Uncomment to recover a previous run

RECOVER = False  # Set True if you need to reload a checkpoint

if RECOVER:
    print("STEP 6b: Recovering from checkpoint")
    print("-" * 70)
    
    with open(f'{CKPT_DIR}/label_map.json') as f:
        label_map = json.load(f)
    with open(f'{CKPT_DIR}/best_hyperparameters.json') as f:
        best = json.load(f)
    batch_size = best['batch_size']

    model = tf.keras.models.load_model(f'{CKPT_DIR}/best_model.keras')
    train_ds, val_ds, test_ds, n_tr, n_va, n_te = split_and_build(
        X_full, y_full, batch_size)

    # Reconstruct history from CSV log
    log_path = f'{CKPT_DIR}/training_log.csv'
    if os.path.exists(log_path):
        _log = pd.read_csv(log_path)
        class _H:
            pass
        history = _H()
        history.history = {c: _log[c].tolist() for c in _log.columns if c != 'epoch'}
        print(f"✓ Recovered {len(_log)} epochs of history\n")

    print(f"✓ Model reloaded from {CKPT_DIR}/best_model.keras\n")


# =====================================================================
# STEP 7: Evaluate + Diagnostics
# =====================================================================

print("STEP 7: Evaluation")
print("-" * 70)

test_loss, test_acc = model.evaluate(test_ds, verbose=0)
print(f"Test Accuracy: {test_acc:.2%} | Test Loss: {test_loss:.4f}\n")

# Save test results
test_results = {
    'test_accuracy': float(test_acc),
    'test_loss': float(test_loss),
    'num_test_samples': n_te,
}
with open(f'{CKPT_DIR}/test_results.json', 'w') as f:
    json.dump(test_results, f, indent=2)
print(f"✓ Test results saved to {CKPT_DIR}/test_results.json")

# Diagnose overfitting vs underfitting
tr_acc = history.history['accuracy'][-1]
va_acc = history.history['val_accuracy'][-1]
gap = tr_acc - va_acc

print("Capacity & Convergence Diagnostic:")
print(f"  Final train accuracy: {tr_acc:.2%}")
print(f"  Final val accuracy:   {va_acc:.2%}")
print(f"  Train-val gap:        {gap:+.2%}")

if gap < 0.05 and tr_acc < 0.70:
    print("  → UNDERFITTING: Increase model capacity (more conv filters/dense units)")
elif gap > 0.15:
    print("  → OVERFITTING: Use more data, augmentation, or higher dropout")
else:
    print("  → Reasonably balanced")

stopped_early = len(history.history['loss']) < FINAL_EPOCHS
epochs_run = len(history.history['loss'])
if stopped_early:
    print(f"\n  Early stopped at epoch {epochs_run}/{FINAL_EPOCHS}")
else:
    print(f"\n  Ran {epochs_run}/{FINAL_EPOCHS} epochs (hit limit - consider raising)")

# Plot training curves
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].plot(history.history['accuracy'], label='Train', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Val', linewidth=2)
axes[0].set_title('Accuracy', fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(history.history['loss'], label='Train', linewidth=2)
axes[1].plot(history.history['val_loss'], label='Val', linewidth=2)
axes[1].set_title('Loss', fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{CKPT_DIR}/training_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"✓ Training curves saved to {CKPT_DIR}/training_curves.png\n")


# =====================================================================
# STEP 8: Per-Category Accuracy (useful at scale)
# =====================================================================

print("STEP 8: Per-category accuracy")
print("-" * 70)

correct = defaultdict(int)
total = defaultdict(int)

for imgs, lbls in test_ds:
    preds = np.argmax(model.predict(imgs, verbose=0), axis=1)
    for t, p in zip(lbls.numpy(), preds):
        total[int(t)] += 1
        correct[int(t)] += int(t == p)

per_cat_results = {}
print("Accuracy by category:")
for idx in sorted(total):
    name = label_map[str(idx)]
    acc = correct[idx] / total[idx]
    per_cat_results[name] = {'accuracy': float(acc), 'correct': correct[idx], 'total': total[idx]}
    print(f"  {name:15} {acc:.1%}")
print()

with open(f'{CKPT_DIR}/per_category_accuracy.json', 'w') as f:
    json.dump(per_cat_results, f, indent=2)
print(f"✓ Per-category accuracy saved to {CKPT_DIR}/per_category_accuracy.json\n")


# =====================================================================
# STEP 9: Serve-Time Preprocessing (Critical for real-user accuracy)
# =====================================================================
# This converts raw canvas images to the canonical format

def preprocess_canvas_capture(img, ink_is_dark=True, target_fill=TARGET_FILL):
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


def predict_drawing(img, ink_is_dark=True, top_k=3):
    """
    End-to-end: raw canvas image → top-k predictions.
    
    Returns:
        List of (label, confidence) tuples, sorted by confidence
    """
    x = preprocess_canvas_capture(img, ink_is_dark=ink_is_dark)[None, ...]
    probs = model.predict(x, verbose=0)[0]
    idxs = np.argsort(probs)[::-1][:top_k]
    return [(label_map[str(int(i))], float(probs[i])) for i in idxs]

print("STEP 9: Serve-time preprocessing")
print("-" * 70)
print("✓ Preprocessing functions ready\n")


# =====================================================================
# STEP 9b: Test Preprocessing on a Simulated Canvas
# =====================================================================

print("STEP 9b: Testing preprocessing pipeline")
print("-" * 70)

# Take a real QuickDraw sample and simulate how it would look from a canvas
sample = X_full[0].reshape(28, 28)  # white-on-black source
true_name = label_map[str(int(y_full[0]))]

# Simulate canvas: dark ink on white, larger, off-center
canvas = Image.new('L', (256, 256), color=255)  # white background
small = Image.fromarray((255 - sample).astype(np.uint8)).resize(
    (120, 120), Image.LANCZOS)
canvas.paste(small, (30, 60))  # off-center

# Process through the pipeline
processed = preprocess_canvas_capture(canvas, ink_is_dark=True)

# Visualize
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].imshow(canvas, cmap='gray')
ax[0].set_title('Simulated canvas capture')
ax[0].axis('off')

ax[1].imshow(processed.squeeze(), cmap='gray', vmin=0, vmax=1)
ax[1].set_title('After preprocessing (model input)')
ax[1].axis('off')

plt.tight_layout()
plt.savefig(f'{CKPT_DIR}/preprocessing_demo.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Ground truth: {true_name}")
print(f"Top-3 predictions: {predict_drawing(canvas, ink_is_dark=True)}")
print(f"✓ Demo saved to {CKPT_DIR}/preprocessing_demo.png\n")


# =====================================================================
# STEP 10: Save Model Artifacts
# =====================================================================

print("STEP 10: Saving artifacts")
print("-" * 70)

os.makedirs(CKPT_DIR, exist_ok=True)

# Save Keras model (full format)
model.save(f'{CKPT_DIR}/linelogic_cnn.keras')
print(f"✓ Keras model: {CKPT_DIR}/linelogic_cnn.keras")

# Convert to TFLite for mobile/edge deployment
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
with open(f'{CKPT_DIR}/linelogic_cnn.tflite', 'wb') as f:
    f.write(converter.convert())
tflite_size = os.path.getsize(f'{CKPT_DIR}/linelogic_cnn.tflite') / 1e6
print(f"✓ TFLite model: {CKPT_DIR}/linelogic_cnn.tflite ({tflite_size:.2f} MB)")

# Save label map
with open(f'{CKPT_DIR}/label_map.json', 'w') as f:
    json.dump(label_map, f, indent=2)
print(f"✓ Label map: {CKPT_DIR}/label_map.json")

# Save hyperparameters
with open(f'{CKPT_DIR}/best_hyperparameters.json', 'w') as f:
    json.dump(BEST_PARAMS, f, indent=2)
print(f"✓ Hyperparameters: {CKPT_DIR}/best_hyperparameters.json")

# Document the input contract (so backends know what to send)
with open(f'{CKPT_DIR}/input_contract.json', 'w') as f:
    json.dump({
        'img_size': IMG_SIZE,
        'channels': 1,
        'ink': 'white (high) on black (0) background',
        'value_range': [0, 1],
        'centering': 'crop to bounding box, pad to square, center',
        'target_fill': TARGET_FILL,
        'resize_filter': 'LANCZOS (anti-aliased, not nearest)',
        'note': 'Canvas captures MUST be converted via preprocess_canvas_capture()'
    }, f, indent=2)
print(f"✓ Input contract: {CKPT_DIR}/input_contract.json")

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)
print(f"All artifacts saved to: {CKPT_DIR}\n")
print("Next steps:")
print("1. Wrap predict_drawing() in a FastAPI endpoint")
print("2. Build the canvas frontend (match brush width from Step 9)")
print("3. Collect real user drawings to improve the model")
print()
