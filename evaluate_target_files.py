"""
Script to evaluate the 12 specific batch app target files requested in Step 4c.
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from model import create_model
from app import load_model_and_gradcam, predict_batch, preprocess_image


def run_target_batch_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("STEP 4c: BATCH APP EVALUATION ON TARGET FILES")
    print("=" * 80)
    print(f"Device: {device}")
    
    ckpt_path = "checkpoints/best_model.pt"
    model, gradcam, dev, stored_class_names, threshold = load_model_and_gradcam(ckpt_path, device)
    
    if model is None:
        print("Error: Could not load model from checkpoint.")
        return
        
    class_names = [name.upper() for name in stored_class_names]
    print(f"Loaded checkpoint with classes {class_names} and decision threshold = {threshold:.4f}\n")
    
    target_stems = [
        "NORMAL2-IM-1427",
        "NORMAL2-IM-1430",
        "NORMAL2-IM-1431",
        "NORMAL2-IM-1436",
        "NORMAL2-IM-1437",
        "NORMAL2-IM-1438",
        "NORMAL2-IM-1440",
        "NORMAL2-IM-1442",
        "person9_bacteria_38",
        "person9_bacteria_39",
        "person9_bacteria_40",
        "person9_bacteria_41"
    ]
    
    data_dir = PROJECT_ROOT / "data" / "chest_xray"
    found_files = []
    
    for stem in target_stems:
        matches = list(data_dir.rglob(f"*{stem}*"))
        if matches:
            found_files.append((stem, matches[0]))
        else:
            print(f"Warning: Could not find file matching '{stem}' in {data_dir}")
            
    batch_tensors = []
    file_metadata = []
    
    for stem, path in found_files:
        with open(path, "rb") as f:
            tensor, pil_img, vis_np = preprocess_image(f, image_size=(224, 224), device=device)
            batch_tensors.append(tensor)
            
            # True label determination based on directory path
            if "NORMAL" in str(path).upper():
                true_label = "NORMAL"
            elif "PNEUMONIA" in str(path).upper():
                true_label = "PNEUMONIA"
            else:
                true_label = "UNKNOWN"
                
            file_metadata.append({
                "stem": stem,
                "filename": path.name,
                "path": str(path),
                "true_label": true_label
            })
            
    results = predict_batch(model, batch_tensors, decision_threshold=threshold)
    
    print(f"{'Filename':<32} | {'True Class':<10} | {'Pred Class':<10} | {'P(Pneu)':<8} | {'Conf (%)':<8} | Status")
    print("-" * 92)
    
    correct_count = 0
    for meta, (pred_idx, conf, probs) in zip(file_metadata, results):
        pred_class = class_names[pred_idx]
        p_pneu = float(probs[1]) if len(probs) > 1 else 0.0
        is_correct = (meta['true_label'] == pred_class)
        if is_correct:
            correct_count += 1
        status = "✓ CORRECT" if is_correct else "✗ MISCLASSIFIED"
        
        print(f"{meta['filename']:<32} | {meta['true_label']:<10} | {pred_class:<10} | {p_pneu:<8.4f} | {conf*100:<8.1f} | {status}")
        
    print("-" * 92)
    print(f"Accuracy on target batch files: {correct_count}/{len(file_metadata)} ({correct_count/len(file_metadata)*100:.1f}%)\n")


if __name__ == "__main__":
    run_target_batch_evaluation()
