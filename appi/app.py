# api/app.py

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "model_best.pth"

CLASS_NAMES = [
    "early_blight",
    "healthy",
    "late_blight",
]

ADVICE = {
    "healthy": "Leaf looks healthy. Keep normal care.",
    "early_blight": "Possible early blight: remove affected leaves, avoid overhead watering.",
    "late_blight": "Possible late blight: isolate plant, reduce humidity, monitor nearby plants.",
}

eval_tfms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

def load_model(device):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for p in model.parameters():
        p.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, len(CLASS_NAMES))

    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(device)

app = FastAPI(
    title="Tomato Leaf Disease API",
    description="Upload an image and get plant health classification.",
    version="1.0.0",
)

def predict_from_pil(pil_img: Image.Image):
    x = eval_tfms(pil_img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()

    cls = CLASS_NAMES[pred_idx]
    is_sick = cls != "healthy"

    return {
        "predicted_class": cls,
        "confidence": float(confidence),
        "is_sick": is_sick,
        "advice": ADVICE.get(cls, "No advice."),
        "probs": {
            CLASS_NAMES[i]: float(probs[i].item())
            for i in range(len(CLASS_NAMES))
        }
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
    result = predict_from_pil(pil_img)
    return JSONResponse(result)
