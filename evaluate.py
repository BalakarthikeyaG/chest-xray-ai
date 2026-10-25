"""
Independent evaluation script for Explainable AI-Based Medical Image Classification Using Deep Learning.
Evaluates best_model.pt on the full test dataset and checks predictions.
"""

import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from model import create_model
from dataset import ChestXRayDataset, get_transforms, preprocess_image_tensor


def run_evaluation(
    checkpoint_path: str = "checkpoints/best_model.pt",
    data_dir: str = "data/chest_xray",
    image_size=(224, 224)
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("INDEPENDENT MODEL EVALUATION")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Checkpoint path: {checkpoint_path}")
    
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.is_absolute():
        ckpt_file = PROJECT_ROOT / ckpt_file
    
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_file}")
    
    # 1. Load Checkpoint with strict state dict checking
    print("\n[1/4] Loading checkpoint state dict (strict=True)...")
    checkpoint = torch.load(ckpt_file, map_location=device)
    
    stored_keys = list(checkpoint.keys())
    print(f"  Checkpoint keys present: {stored_keys}")
    print(f"  Saved epoch: {checkpoint.get('epoch')}")
    print(f"  Best val F1: {checkpoint.get('best_f1')}")
    
    class_names = checkpoint.get('class_names', ['NORMAL', 'PNEUMONIA'])
    print(f"  Checkpoint class_names: {class_names}")
    
    model = create_model(num_classes=len(class_names), pretrained=False, freeze_backbone=False, device=device)
    
    incompatible = model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    print(f"  Missing keys: {incompatible.missing_keys}")
    print(f"  Unexpected keys: {incompatible.unexpected_keys}")
    print("  ✓ State dict loaded successfully with strict=True")
    
    model.eval()
    
    # 2. Load Test Dataset
    print("\n[2/4] Loading test dataset...")
    test_dir = PROJECT_ROOT / data_dir
    test_dataset = ChestXRayDataset(
        data_dir=str(test_dir),
        split='test',
        image_size=image_size,
        transform=get_transforms(image_size, augment=False)['test'],
        class_names=class_names
    )
    
    print(f"  Test class_to_idx mapping: {test_dataset.class_to_idx}")
    assert test_dataset.class_names == class_names, f"Class mismatch: {test_dataset.class_names} vs {class_names}"
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )
    
    decision_threshold = checkpoint.get('decision_threshold', 0.5)
    print(f"  Decision threshold from checkpoint: {decision_threshold:.4f}")
    
    # 3. Evaluate on full test set under torch.no_grad()
    print("\n[3/4] Running inference on full test dataset...")
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = (all_probs[:, 1] >= decision_threshold).astype(int)
    
    from sklearn.metrics import balanced_accuracy_score
    acc = accuracy_score(all_labels, all_preds)
    bacc = balanced_accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))
    
    normal_recall = recall_score(all_labels == 0, all_preds == 0, zero_division=0)
    pneu_recall = recall_score(all_labels == 1, all_preds == 1, zero_division=0)
    
    per_class_acc = {
        'NORMAL': float(normal_recall),
        'PNEUMONIA': float(pneu_recall)
    }
            
    print("\n" + "=" * 70)
    print(f"TEST SET EVALUATION METRICS (Threshold={decision_threshold:.4f})")
    print("=" * 70)
    print(f"Accuracy:          {acc:.4f} ({acc*100:.2f}%)")
    print(f"Balanced Accuracy: {bacc:.4f} ({bacc*100:.2f}%)")
    print(f"Precision:         {prec:.4f}")
    print(f"Recall (Pneu):     {rec:.4f}")
    print(f"F1-Score:          {f1:.4f}")
    print("Per-class Recall:")
    print(f"  NORMAL Recall:    {normal_recall:.4f} ({normal_recall*100:.2f}%)")
    print(f"  PNEUMONIA Recall: {pneu_recall:.4f} ({pneu_recall*100:.2f}%)")
    print(f"Confusion Matrix (rows=true, cols=pred):\n{cm}")
    print("\nClassification Report:\n" + classification_report(
        all_labels, all_preds, target_names=class_names, zero_division=0
    ))
    
    # 4. Print sample probabilities for 10 images per class
    print("\n[4/4] Sample Predictions (10 images per class):")
    print("-" * 70)
    for class_idx, class_name in enumerate(class_names):
        print(f"\n--- True Class: {class_name} (idx {class_idx}) ---")
        indices = np.where(all_labels == class_idx)[0][:10]
        for idx in indices:
            path_str = test_dataset.images[idx]
            filename = Path(path_str).name
            probs = all_probs[idx]
            pred_idx = all_preds[idx]
            pred_name = class_names[pred_idx]
            correct = "✓ CORRECT" if pred_idx == class_idx else "✗ MISCLASSIFIED"
            prob_str = ", ".join([f"{class_names[c]}: {probs[c]:.4f}" for c in range(len(class_names))])
            print(f"  File: {filename:<30} | Pred: {pred_name:<10} | Probs: [{prob_str}] | {correct}")
            
    return {
        'accuracy': acc,
        'balanced_accuracy': bacc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'confusion_matrix': cm,
        'per_class_acc': per_class_acc,
        'decision_threshold': decision_threshold
    }


if __name__ == "__main__":
    run_evaluation()
