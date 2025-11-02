**Project: Tomato Leaf Health Diagnostics (Computer Vision)**

Built an end-to-end plant health assistant for tomato leaves using the PlantVillage dataset (healthy / early blight / late blight).

Designed a reproducible data pipeline: automatic download, label normalization, and deterministic train/val/test splits.

Trained a ResNet18-based classifier (PyTorch) on GPU (Snellius HPC). Achieved ~96% test accuracy on held-out data.

Exposed the model in two ways:

A Streamlit web app where users upload a leaf photo and get a diagnosis, confidence score, and care advice (“healthy” vs “needs attention”).

A FastAPI endpoint (POST /predict) returning JSON, structured for integration into other systems (e.g. mobile app or dashboard).

Added user-facing messaging about limitations: performance can degrade in real garden photos (natural lighting, background clutter). Proposed collecting in-the-wild images and fine-tuning as next step.

Goal: early detection of crop disease to reduce pesticide overuse and limit crop loss.

## Repository Structure

```text
plant-health-ai/
├─ data/
│  ├─ tmp/              # Kaggle raw download (auto-created)
│  ├─ raw/              # class folders normalized
│  └─ splits/           # train/val/test for model
│      ├─ train/
│      ├─ val/
│      └─ test/
├─ models/
│  ├─ model_best.pth    # created after training
│  └─ model_final.pth   # optional
├─ src/
│  ├─ download_and_prepare.py
│  ├─ train.py
│  ├─ inference.py      # optional CLI tester, not strictly needed for recruiters now
│  └─ utils.py          # helper functions (optional if you're already using it)
├─ app/
│  └─ app.py            # Streamlit user-facing demo
├─ api/
│  └─ app.py            # FastAPI inference service (JSON API)
├─ notebooks/
│  └─ evaluation.ipynb  # (confusion matrix *)
├─ environment.yml
├─ README.md
└─ .gitignore
