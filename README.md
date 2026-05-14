Egyptian Hieroglyph Recognition: IGSM-CVV Framework
This repository implements a robust pipeline for the automated recognition of Egyptian hieroglyphs, as developed for the Artificial Intelligence (CS3002) course project. Our approach re-engineers the framework proposed by Fuentes-Ferrer et al. (2025), specifically optimized for accessibility on consumer-grade hardware.

🚀 Key Features
Hardware Optimized: Specifically tuned for 12GB VRAM environments (NVIDIA T4) using PyTorch Mixed Precision (AMP).

IGSM Segmentation: A hybrid segmentation model using Local Adaptive Gaussian Thresholding and the Segment Anything Model (SAM) to resolve dense hieroglyphic registers.

CVV Ensemble: A 5-slot Cross-Validation Voting system using a ConvNeXt-tiny backbone, achieving a peak accuracy of 95.8%.

Curated Dataset: High-fidelity training on 155 Gardiner classes with a strict stratification threshold to eliminate data imbalance.

📊 Performance at a Glance
Our ensemble (CVV-SV) significantly outperforms individual models by resolving ambiguous morphological features through probability averaging.

Metric	Value
Global Accuracy	0.958
Balanced Accuracy	0.899
F1 Score	0.898
Classes Supported	155
