"""
Compare inference outputs between app.py and evaluate.py on sample test images.
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from model import create_model
from dataset import preprocess_image_tensor
from app import load_model_and_gradcam, predict_batch, preprocess_image


def compare():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("COMPARING APP.PY AND EVALUATE.PY INFERENCE OUTPUTS")
    print("=" * 70)
    
    # 1. Load model via app.py loader
    model, gradcam, dev, stored_class_names = load_model_and_gradcam("checkpoints/best_model.pt", device)
    if model is None:
        print("Failed to load model in app.py!")
        return False
    
    class_names = [name.upper() for name in stored_class_names]
    print(f"Loaded model in app.py. Class names: {class_names}")
    
    # Pick sample images from test folder
    normal_img_path = PROJECT_ROOT / "data" / "chest_xray" / "test" / "NORMAL" / "IM-0001-0001.jpeg"
    pneumonia_img_path = PROJECT_ROOT / "data" / "chest_xray" / "test" / "PNEUMONIA" / "person100_bacteria_475.jpeg"
    
    test_paths = [normal_img_path, pneumonia_img_path]
    
    for path in test_paths:
        print(f"\nEvaluating: {path.name}")
        
        # Streamlit App preprocessing & inference
        with open(path, "rb") as f:
            app_tensor, app_pil, app_vis = preprocess_image(f, image_size=(224, 224), device=device)
        
        app_results = predict_batch(model, [app_tensor])
        app_pred_idx, app_conf, app_probs = app_results[0]
        
        # Evaluate.py / dataset preprocessing & inference
        eval_tensor, eval_pil, eval_vis = preprocess_image_tensor(path, image_size=(224, 224))
        eval_tensor = eval_tensor.to(device)
        
        with torch.no_grad():
            eval_logits = model(eval_tensor)
            eval_probs = F.softmax(eval_logits, dim=1)[0].cpu().numpy()
            eval_pred_idx = int(eval_logits.argmax(dim=1).item())
        
        print(f"  App.py   -> Pred: {class_names[app_pred_idx]:<10} | Conf: {app_conf:.4f} | Probs: {app_probs}")
        print(f"  Eval.py  -> Pred: {class_names[eval_pred_idx]:<10} | Conf: {eval_probs[eval_pred_idx]:.4f} | Probs: {eval_probs}")
        
        diff = abs(app_probs - eval_probs).max()
        if diff < 1e-5 and app_pred_idx == eval_pred_idx:
            print(f"  ✓ MATCH: Outputs are 100% identical (max prob diff: {diff:.8f})")
        else:
            print(f"  ✗ MISMATCH: Max prob diff = {diff:.8f}")
            return False
            
    print("\n" + "=" * 70)
    print("✓ VERIFICATION PASSED: Streamlit inference and evaluate.py produce identical results!")
    print("=" * 70)
    return True


if __name__ == "__main__":
    compare()
