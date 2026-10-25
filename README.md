# Explainable AI-Based Medical Image Classification Using Deep Learning

An end-to-end deep learning framework for chest X-ray binary classification (**Normal** vs. **Pneumonia**) using **DenseNet121 transfer learning** and **Grad-CAM explainability**, with an interactive **Streamlit UI**.

---

## 🌟 Key Features

- **Transfer Learning Backbone**: DenseNet121 pretrained on ImageNet, fine-tuned on chest X-rays.
- **Stratified 80/20 Train/Val Split**: Deterministic stratified split from the training dataset for reliable validation, retaining the test set for unbiased evaluation.
- **Class Imbalance Handling**: Weighted Cross-Entropy Loss to handle imbalance (~3x more Pneumonia than Normal).
- **Domain-Specific Augmentation**: Rotation (±7°), horizontal flip, slight zoom (0.95–1.05), and mild brightness adjustments.
- **Explainability (Grad-CAM)**: Gradient-weighted Class Activation Mapping targeting `features.norm5` to highlight influential diagnostic regions.
- **Streamlit Web Application**: User-friendly web app allowing image uploads, predicted class, confidence scores, probability distribution, and Grad-CAM heatmap overlays.

---

## 📁 Repository Structure

```
├── data/
│   └── chest_xray/
│       ├── train/
│       ├── test/
│       └── val/
├── src/
│   ├── dataset.py        # Dataset loader, transforms, stratified splitting, corruption handling
│   ├── model.py          # DenseNet121 architecture & fine-tuning logic
│   ├── train.py          # Training loop, evaluation, checkpointing, metric plotting
│   ├── gradcam.py        # Grad-CAM heatmap generation & visualization
│   └── app.py            # Streamlit web application
├── checkpoints/          # Saved model weights (best_model.pt)
├── outputs/              # Training curves and confusion matrix plots
├── test_dataset_quick.py # Quick test for dataset loading
├── test_train_quick.py   # Quick test for training pipeline (1 epoch on 200-image subset)
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites & Environment Setup

Ensure Python 3.9+ is installed. Install required packages:

```bash
pip install -r requirements.txt
```

### 2. Run Quick Unit & Pipeline Tests

Run dataset verification:
```bash
python test_dataset_quick.py
```

Run 1-epoch training pipeline test:
```bash
python test_train_quick.py
```

### 3. Model Training

Train the full DenseNet121 model:

```bash
python -m src.train
```

During training:
- Epochs 1–2: DenseNet backbone frozen, training 2-class linear classifier.
- Epochs 3+: Unfreezes `denseblock4` and `norm5` for fine-tuning.
- Best model checkpoint saved to `checkpoints/best_model.pt` based on validation F1 score.
- Evaluation metrics and confusion matrix generated in `outputs/`.

### 4. Run Streamlit Web Application

Launch the Streamlit app:

```bash
streamlit run src/app.py --server.port 8504
```

Navigate to `http://localhost:8504` in your browser.

---

## 📊 Evaluation Metrics

After training, the model evaluates on the held-out test set and reports:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **Confusion Matrix**

Artifacts saved:
- `outputs/training_curves.png`
- `outputs/confusion_matrix.png`
