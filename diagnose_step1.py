"""
STEP 1 Diagnosis Script for Chest X-Ray Medical Image Classifier.
Prints class counts, class_to_idx, class_weights, validation & test set evaluations,
and checks for patient leakage between train and val splits.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, balanced_accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from model import create_model
from dataset import ChestXRayDataset, get_data_loaders
from evaluate import run_evaluation


def extract_patient_id(filepath: str) -> str:
    """Extract patient ID from filename."""
    filename = Path(filepath).name
    # Handle personXXX_... format
    if filename.startswith("person"):
        parts = filename.split("_")
        return parts[0]  # e.g., person9
    # Handle IM-XXXX-XXXX or NORMAL2-IM-XXXX-XXXX format
    elif "IM-" in filename:
        parts = filename.split("-")
        if filename.startswith("NORMAL2-IM"):
            return f"{parts[0]}-{parts[1]}-{parts[2]}"  # NORMAL2-IM-XXXX
        else:
            return f"{parts[0]}-{parts[1]}"  # IM-XXXX
    else:
        # Fallback to filename without extension
        return Path(filename).stem


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("STEP 1: DIAGNOSIS OF EXISTING MODEL & DATASET SPLIT")
    print("=" * 70)

    data_dir = str(PROJECT_ROOT / "data" / "chest_xray")
    
    # a. Class to idx & class counts
    train_loader, val_loader, test_loader = get_data_loaders(
        data_dir=data_dir,
        image_size=(224, 224),
        batch_size=32,
        augment=False,
        include_val=True,
        val_fraction=0.2,
        split_seed=42,
        num_workers=0
    )
    
    train_ds = train_loader.dataset
    val_ds = val_loader.dataset
    full_train_ds = train_ds.dataset if hasattr(train_ds, 'dataset') else train_ds
    class_names = full_train_ds.class_names
    class_to_idx = full_train_ds.class_to_idx

    print("\n--- 1a. CLASS TO IDX & SPLIT COUNTS ---")
    print(f"class_to_idx: {class_to_idx}")
    
    train_labels = [full_train_ds.labels[i] for i in train_ds.indices]
    val_labels = [full_train_ds.labels[i] for i in val_ds.indices]

    train_counts = [sum(1 for l in train_labels if l == idx) for idx in range(len(class_names))]
    val_counts = [sum(1 for l in val_labels if l == idx) for idx in range(len(class_names))]

    print("Train class counts:")
    for cls, cnt in zip(class_names, train_counts):
        print(f"  {cls} (idx {class_to_idx[cls]}): {cnt}")

    print("Val class counts:")
    for cls, cnt in zip(class_names, val_counts):
        print(f"  {cls} (idx {class_to_idx[cls]}): {cnt}")

    # b. Class weights calculation check
    print("\n--- 1b. CLASS WEIGHTS VERIFICATION ---")
    counts_arr = np.array(train_counts, dtype=np.float32)
    weights = counts_arr.sum() / (len(counts_arr) * counts_arr)
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32)
    
    for cls in class_names:
        idx = class_to_idx[cls]
        w = class_weights_tensor[idx].item()
        print(f"  Class: {cls:<10} | idx: {idx} | Weight: {w:.4f}")
    
    normal_w = class_weights_tensor[class_to_idx['NORMAL']].item()
    pneu_w = class_weights_tensor[class_to_idx['PNEUMONIA']].item()
    if normal_w > pneu_w:
        print("  ✓ VERIFIED: Weight for NORMAL is HIGHER than PNEUMONIA.")
    else:
        print("  ❌ WARNING: Weight for NORMAL is NOT higher than PNEUMONIA!")

    # d. Check patient leakage between train and val
    print("\n--- 1d. PATIENT LEAKAGE CHECK (TRAIN vs VAL) ---")
    train_files = [full_train_ds.images[i] for i in train_ds.indices]
    val_files = [full_train_ds.images[i] for i in val_ds.indices]

    train_patients = set(extract_patient_id(f) for f in train_files)
    val_patients = set(extract_patient_id(f) for f in val_files)

    shared_patients = train_patients.intersection(val_patients)
    print(f"  Unique patients in Train: {len(train_patients)}")
    print(f"  Unique patients in Val:   {len(val_patients)}")
    print(f"  Shared patients between Train & Val: {len(shared_patients)}")
    if len(shared_patients) > 0:
        print(f"  ⚠️ PATIENT LEAKAGE DETECTED: {len(shared_patients)} patients appear in both Train and Val!")
        print(f"  Sample leaked patient IDs: {list(shared_patients)[:5]}")
    else:
        print("  ✓ No patient leakage detected.")

    # c. Run evaluation on Validation set and Test set
    print("\n--- 1c. EVALUATION ON VALIDATION AND TEST SETS ---")
    model_path = str(PROJECT_ROOT / "checkpoints" / "best_model.pt")
    checkpoint = torch.load(model_path, map_location=device)
    model = create_model(num_classes=2, pretrained=False, device=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    def eval_dataset(loader, split_name):
        print(f"Evaluating {split_name} loader ({len(loader.dataset)} samples)...", flush=True)
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, lbls in loader:
                imgs = imgs.to(device)
                logits = model(imgs)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(lbls.numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
        prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, labels=[0, 1], zero_division=0)
        b_acc = balanced_accuracy_score(all_labels, all_preds)
        
        print(f"\n=== {split_name.upper()} SET RESULTS ===", flush=True)
        print(f"Confusion Matrix (rows=True, cols=Pred):\n{cm}", flush=True)
        print(f"NORMAL    -> Precision: {prec[0]:.4f}, Recall: {rec[0]:.4f}, F1: {f1[0]:.4f}", flush=True)
        print(f"PNEUMONIA -> Precision: {prec[1]:.4f}, Recall: {rec[1]:.4f}, F1: {f1[1]:.4f}", flush=True)
        print(f"Balanced Accuracy: {b_acc:.4f} ({b_acc*100:.2f}%)", flush=True)

    eval_dataset(val_loader, "Validation")
    eval_dataset(test_loader, "Test")

if __name__ == "__main__":
    main()
