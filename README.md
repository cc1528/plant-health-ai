**Project: Tomato Leaf Health Diagnostics (Computer Vision)**

Built an end-to-end plant health assistant for tomato leaves using the PlantVillage dataset (healthy / early blight / late blight).

Designed a reproducible data pipeline: automatic download, label normalization, and deterministic train/val/test splits.

Trained a ResNet18-based classifier (PyTorch) on GPU (Snellius HPC). Achieved ~96% test accuracy on held-out data.

Exposed the model in two ways:

A Streamlit web app where users upload a leaf photo and get a diagnosis, confidence score, and care advice (“healthy” vs “needs attention”).

A FastAPI endpoint (POST /predict) returning JSON, structured for integration into other systems (e.g. mobile app or dashboard).

Added user-facing messaging about limitations: performance can degrade in real garden photos (natural lighting, background clutter). Proposed collecting in-the-wild images and fine-tuning as next step.

Goal: early detection of crop disease to reduce pesticide overuse and limit crop loss.
