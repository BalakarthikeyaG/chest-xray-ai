"""
Debug script: diagnose prediction accuracy per subgroup (Normal, Bacteria, Virus).
Loads checkpoint exactly as app.py does, checks class order, model state,
runs inference on sampled test images, and reports per-group accuracy + confusion matrix.
"""

import os
import sys
import random
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import create_model
from src.dataset import get_transforms, ChestXRayDataset

# ---------------------------------------------------------------------------
# 1. Load checkpoint EXACTLY like app.py
# ---------------------------------------------------------------------------
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_model.pt"
device = torch.device("cpu")

print("=" * 70)
print("STEP 1: Load checkpoint (same as app.py)")
print("=" * 70)

if not CHECKPOINT_PATH.is_file():
    sys.exit(f"ERROR: Checkpoint not found at {CHECKPOINT_PATH}")

model = create_model(num_classes=2, pretrained=False, device=device)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

# strict=True to surface mismatches
load_result = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
print(f"  Missing keys : {load_result.missing_keys}")
print(f"  Unexpected keys: {load_result.unexpected_keys}")

model = model.float()
model.eval()
print(f"  model.training = {model.training}  (expected False)")

# Class names stored in the checkpoint
stored_class_names = checkpoint.get("class_names", None)
print(f"  Checkpoint class_names: {stored_class_names}")

# Class names as used by app.py display (Title case)
if stored_class_names:
    app_class_names = [n.title() for n in stored_class_names]
else:
    app_class_names = ["Normal", "Pneumonia"]
    stored_class_names = ["NORMAL", "PNEUMONIA"]
print(f"  App display names   : {app_class_names}")

# ---------------------------------------------------------------------------
# 2. Check dataset class_to_idx matches
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 2: Dataset class_to_idx consistency")
print("=" * 70)

DATA_DIR = PROJECT_ROOT / "data" / "chest_xray"
test_dataset = ChestXRayDataset(
    data_dir=str(DATA_DIR), split="test", image_size=(224, 224),
    transform=get_transforms((224, 224), augment=False)["test"],
)
print(f"  test_dataset.class_names  = {test_dataset.class_names}")
print(f"  test_dataset.class_to_idx = {test_dataset.class_to_idx}")
print(f"  checkpoint class_names    = {stored_class_names}")
if test_dataset.class_names != list(stored_class_names):
    print("  *** MISMATCH detected between dataset and checkpoint class order! ***")
else:
    print("  OK – class order matches.")

# ---------------------------------------------------------------------------
# 3. Preprocessing: same transform used in app.py
# ---------------------------------------------------------------------------
preprocess = get_transforms((224, 224), augment=False)["test"]

def predict_file(filepath: str):
    """Return (predicted_idx, confidence, probs_array, raw_logits)."""
    pil = Image.open(filepath).convert("RGB")
    tensor = preprocess(pil).unsqueeze(0).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)
    return (
        pred_idx.item(),
        confidence.item(),
        probs[0].cpu().numpy(),
        logits[0].cpu().numpy(),
    )

# ---------------------------------------------------------------------------
# 4. Single-image sanity check on the reported misprediction
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 3: Single-image sanity check (person1_virus_9.jpeg)")
print("=" * 70)

target_image = DATA_DIR / "test" / "PNEUMONIA" / "person1_virus_9.jpeg"
if target_image.is_file():
    pred_idx, conf, probs, logits = predict_file(str(target_image))
    print(f"  Logits  : {logits}")
    print(f"  Softmax : {probs}")
    print(f"  Predicted: {stored_class_names[pred_idx]} ({conf:.4f})")
    print(f"  True    : PNEUMONIA")
else:
    print(f"  Image not found: {target_image}")

# ---------------------------------------------------------------------------
# 5. Sample 20 images from each subgroup and predict
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 4: Per-group accuracy (20 images each)")
print("=" * 70)

random.seed(42)
SAMPLE = 20

normal_dir = DATA_DIR / "test" / "NORMAL"
pneumonia_dir = DATA_DIR / "test" / "PNEUMONIA"

normal_files = sorted([str(f) for f in normal_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])
pneumonia_files = sorted([str(f) for f in pneumonia_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])

bacteria_files = [f for f in pneumonia_files if "bacteria" in Path(f).name.lower()]
virus_files    = [f for f in pneumonia_files if "virus" in Path(f).name.lower()]

print(f"  Total NORMAL:   {len(normal_files)}")
print(f"  Total BACTERIA: {len(bacteria_files)}")
print(f"  Total VIRUS:    {len(virus_files)}")

sample_normal   = random.sample(normal_files, min(SAMPLE, len(normal_files)))
sample_bacteria = random.sample(bacteria_files, min(SAMPLE, len(bacteria_files)))
sample_virus    = random.sample(virus_files, min(SAMPLE, len(virus_files)))

def eval_group(files, true_class_name, group_label):
    true_idx = test_dataset.class_to_idx[true_class_name]
    correct = 0
    preds_list = []
    labels_list = []
    for fp in files:
        pred_idx, conf, probs, logits = predict_file(fp)
        preds_list.append(pred_idx)
        labels_list.append(true_idx)
        if pred_idx == true_idx:
            correct += 1
        else:
            fname = Path(fp).name
            print(f"    WRONG  {fname}: pred={stored_class_names[pred_idx]}({conf:.3f}), true={true_class_name}")
    acc = correct / len(files) if files else 0
    print(f"  {group_label}: {correct}/{len(files)} correct ({acc:.1%})")
    return preds_list, labels_list

preds_all, labels_all = [], []

p, l = eval_group(sample_normal, "NORMAL", "NORMAL")
preds_all.extend(p); labels_all.extend(l)

p, l = eval_group(sample_bacteria, "PNEUMONIA", "BACTERIA")
preds_all.extend(p); labels_all.extend(l)

p, l = eval_group(sample_virus, "PNEUMONIA", "VIRUS")
preds_all.extend(p); labels_all.extend(l)

# ---------------------------------------------------------------------------
# 6. Overall metrics on the 60 sampled images
# ---------------------------------------------------------------------------
preds_all = np.array(preds_all)
labels_all = np.array(labels_all)

print("\n" + "=" * 70)
print("STEP 5: Overall metrics (60 images)")
print("=" * 70)
print(classification_report(
    labels_all, preds_all,
    target_names=stored_class_names,
    labels=list(range(len(stored_class_names))),
    zero_division=0,
))
cm = confusion_matrix(labels_all, preds_all, labels=list(range(len(stored_class_names))))
print(f"Confusion matrix (rows=true, cols=pred):\n{cm}")

print(f"\nOverall accuracy : {accuracy_score(labels_all, preds_all):.4f}")
print(f"Precision (Pneumonia): {precision_score(labels_all, preds_all, zero_division=0):.4f}")
print(f"Recall    (Pneumonia): {recall_score(labels_all, preds_all, zero_division=0):.4f}")
print(f"F1        (Pneumonia): {f1_score(labels_all, preds_all, zero_division=0):.4f}")

# ---------------------------------------------------------------------------
# 7. Full test-set evaluation with threshold sweep
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 6: Full test-set evaluation + threshold sweep")
print("=" * 70)

all_files = [(f, 0) for f in normal_files] + [(f, 1) for f in pneumonia_files]
full_preds = []
full_labels = []
full_pneumonia_probs = []

for fp, true_label in all_files:
    pred_idx, conf, probs, _ = predict_file(fp)
    full_preds.append(pred_idx)
    full_labels.append(true_label)
    # Pneumonia probability is the probability for class index 1
    pneumonia_idx = test_dataset.class_to_idx.get("PNEUMONIA", 1)
    full_pneumonia_probs.append(probs[pneumonia_idx])

full_preds = np.array(full_preds)
full_labels = np.array(full_labels)
full_pneumonia_probs = np.array(full_pneumonia_probs)

print(f"\nFull test set ({len(all_files)} images) at default threshold=0.5:")
print(classification_report(
    full_labels, full_preds,
    target_names=stored_class_names,
    labels=[0, 1],
    zero_division=0,
))
cm_full = confusion_matrix(full_labels, full_preds, labels=[0, 1])
print(f"Confusion matrix:\n{cm_full}")

# Threshold sweep
print("\nThreshold sweep for Pneumonia prediction:")
print(f"{'Threshold':>10s}  {'Accuracy':>8s}  {'Precision':>9s}  {'Recall':>6s}  {'F1':>6s}")
for thr in [0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20]:
    preds_t = (full_pneumonia_probs >= thr).astype(int)
    acc = accuracy_score(full_labels, preds_t)
    prec = precision_score(full_labels, preds_t, zero_division=0)
    rec = recall_score(full_labels, preds_t, zero_division=0)
    f1 = f1_score(full_labels, preds_t, zero_division=0)
    print(f"  {thr:>8.2f}  {acc:>8.4f}  {prec:>9.4f}  {rec:>6.4f}  {f1:>6.4f}")

print("\n" + "=" * 70)
print("DIAGNOSIS COMPLETE")
print("=" * 70)
