import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "model_best.pth"

CLASS_NAMES = ["early_blight", "healthy", "late_blight"]

eval_tfms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

ADVICE = {
    "healthy": "Leaf looks healthy ✅. Keep normal watering and monitor regularly.",
    "early_blight": "Possible early blight. Remove affected leaves, avoid overhead watering, and increase airflow around the plant.",
    "late_blight": "Possible late blight. This spreads fast in humidity. Isolate the plant, reduce leaf moisture, and check nearby plants.",
}

@st.cache_resource(show_spinner=False)
def load_model(device: torch.device):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for p in model.parameters():
        p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device).eval()
    return model

def predict_image(pil_img: Image.Image, model, device: torch.device):
    x = eval_tfms(pil_img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        idx = torch.argmax(probs).item()
    return CLASS_NAMES[idx], probs[idx].item(), {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

def prob_bar_html(label: str, p: float):
    pct = p * 100
    return f"""
    <div style='display:flex;justify-content:space-between;font-weight:600;color:#1a1a1a;'>
        <span>{label.replace('_',' ').title()}</span><span>{pct:.1f}%</span>
    </div>
    <div style='background:#e9ecef;border-radius:999px;height:10px;overflow:hidden;margin-bottom:8px;'>
        <div style='width:{pct:.2f}%;background:linear-gradient(90deg,#8B5CF6,#00BFA6);height:100%;'></div>
    </div>
    """

# ------------------- UI -------------------
st.set_page_config(page_title="Tomato Leaf Health Check", page_icon="🍅", layout="wide")

# Sidebar
with st.sidebar:
    st.markdown("## 🍅 Tomato Leaf AI")
    st.caption("Computer vision model (ResNet18 fine-tuned) that looks at a tomato leaf and checks for early or late blight.")
    st.markdown("### How to use")
    st.write("- Take a clear photo of ONE leaf\n- Avoid shadows / glare\n- Upload and read advice 🌿")
    st.markdown("### Important")
    st.write("Trained on clean dataset (PlantVillage). Real-garden lighting or messy backgrounds may reduce accuracy. Use as guidance 🌱.")
    st.markdown("<small>Built with PyTorch · Runs locally · No cloud upload</small>", unsafe_allow_html=True)

# Styles
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to bottom right, #e7f8ec, #d9f2e0);
    color: #1a1a1a;
}
.hero-card {
    text-align: center;
    background: linear-gradient(90deg, #dff7e6 0%, #d8f0df 100%);
    border-radius: 1rem;
    padding: 2.5rem 2rem;
    margin-bottom: 2rem;
    box-shadow: 0 10px 40px rgba(0,0,0,0.08);
}
.hero-title {
    font-size: 3.5rem;
    font-weight: 800;
    margin-bottom: 0.5rem;
    color: #083b1c;
}
.hero-badge {
    display:inline-block;
    background:#fff;
    color:#006b44;
    border:1px solid rgba(0,0,0,0.1);
    padding:0.35rem 0.7rem;
    border-radius:0.5rem;
    font-size:0.9rem;
    font-weight:600;
    margin-left:0.5rem;
}
.hero-sub {
    font-size: 1.4rem;
    max-width: 900px;
    margin: 0 auto;
    color: rgba(0,0,0,0.75);
    line-height: 1.6;
}
.panel-card {
    background: white;
    border-radius: 1rem;
    padding: 1.5rem 2rem;
    box-shadow: 0 15px 40px rgba(0,0,0,0.08);
}
.status-chip-ok {
    background:#e5fbf6;
    color:#009e7c;
    font-weight:600;
    padding:0.35rem 0.6rem;
    border-radius:0.5rem;
}
.status-chip-warn {
    background:#fff6d5;
    color:#8b6b00;
    font-weight:600;
    padding:0.35rem 0.6rem;
    border-radius:0.5rem;
}
.footer-card {
    text-align:center;
    margin-top:2rem;
}
.social-row {
    display:flex;
    justify-content:center;
    gap:1.2rem;
    margin-top:1rem;
}
.social-link {
    display:flex;
    align-items:center;
    gap:0.4rem;
    background:#fff;
    border-radius:0.6rem;
    border:1px solid rgba(0,0,0,0.1);
    padding:0.5rem 0.8rem;
    color:#1a1a1a;
    font-weight:500;
    text-decoration:none;
    box-shadow:0 5px 20px rgba(0,0,0,0.06);
}
.social-link:hover { background:#f3fff6; }
.social-icon { width:18px; height:18px; }
.footer-note {
    font-size:0.85rem;
    color:rgba(0,0,0,0.6);
    margin-top:1rem;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class='hero-card'>
  <div class='hero-title'>Tomato Leaf Health Check
    <span class='hero-badge'>early blight / late blight detector 🌿</span>
  </div>
  <div class='hero-sub'>
    Upload a tomato leaf photo. The model will predict if it's healthy or showing signs of early / late blight, and tell you what to do next.
  </div>
</div>
""", unsafe_allow_html=True)

# Layout
col1, col2 = st.columns(2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(device)

with col1:
    st.markdown("<div class='panel-card'><b>1 · Upload Leaf Photo</b></div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload a clear close-up (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Uploaded leaf 🍃", use_container_width=True)

with col2:
    st.markdown("<div class='panel-card'><b>2 · AI Screening Result</b></div>", unsafe_allow_html=True)
    if uploaded:
        cls, conf, probs = predict_image(img, model, device)
        is_sick = cls != "healthy"
        status = "<span class='status-chip-warn'>Needs attention ⚠️</span>" if is_sick else "<span class='status-chip-ok'>Looks OK ✅</span>"
        st.markdown(f"<h4>Prediction: {cls.replace('_',' ').title()} {status}</h4>", unsafe_allow_html=True)
        st.write(f"**Confidence:** {conf*100:.2f}%")
        st.markdown("**Suggested Action**")
        st.write(ADVICE[cls])
        st.markdown("**Class Probabilities**")
        for k, p in probs.items():
            st.markdown(prob_bar_html(k, p), unsafe_allow_html=True)
    else:
        st.write("No image yet. 👇 Upload on the left to see prediction here.")

# Footer with social links
st.markdown("""
<div class='footer-card'>
  <div class='social-row'>
    <a class='social-link' href='https://www.linkedin.com/in/cinthya-nathaly-criollo-quiroz/' target='_blank'>
      <img class='social-icon' src='https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/linkedin.svg'/> LinkedIn
    </a>
    <a class='social-link' href='https://github.com/cc1528' target='_blank'>
      <img class='social-icon' src='https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/github.svg'/> GitHub
    </a>
    <a class='social-link' href='https://www.instagram.com/?hl=en' target='_blank'>
      <img class='social-icon' src='https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/instagram.svg'/> Instagram
    </a>
  </div>
  <div class='footer-note'>© 2025 Cinthya Criollo · Tomato Leaf Health AI · All rights reserved 🌱</div>
</div>
""", unsafe_allow_html=True)
