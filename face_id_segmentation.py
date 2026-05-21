# 🧑 Face ID & Segmentation

> Two advanced computer vision systems: face verification via Triplet Loss embeddings, and pixel-wise segmentation via U-Net.

## Part 1 — Face Verification (Siamese + Triplet Loss)

**Loss Function:**
```
L = max(||f(anchor) - f(positive)||² - ||f(anchor) - f(negative)||² + margin, 0)
```

| Metric | Value |
|---|---|
| Architecture | Siamese CNN + Triplet Loss |
| Embedding dim | 128 |
| Similarity metric | Cosine similarity |
| Verification threshold | 0.7 |

## Part 2 — Image Segmentation (U-Net)

**Architecture:**
```
Input → [Conv→BN→Conv→BN → MaxPool] × 4 → Bottleneck
      → [TransposeConv → Concat(skip) → Conv→BN×2] × 4 → 1×1 Conv → Mask
```

| Metric | Value |
|---|---|
| Loss | Binary Cross-Entropy + Dice |
| Output | Pixel-wise binary mask |
| Skip connections | 4 levels |

## Quick Start
```bash
pip install -r requirements.txt
python face_id_segmentation.py
```

## What I Learned
- Triplet mining strategy (semi-hard negatives) is crucial for training stability
- L2-normalization of embeddings enables cosine similarity via dot product
- U-Net's skip connections preserve fine-grained spatial details lost in pooling
- Dice coefficient loss handles class imbalance in segmentation masks

## Tech Stack
`TensorFlow/Keras` · `NumPy` · `Matplotlib`
