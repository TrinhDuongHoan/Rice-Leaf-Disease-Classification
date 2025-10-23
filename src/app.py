import io, torch, torchvision.transforms as T
from PIL import Image
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


CKPT = "../src/trained_model/mobilenetv3_eca.pth"
CLASS_NAMES = ['bacterial_leaf_blight', 'bacterial_leaf_streak', 'bacterial_panicle_blight','blast','brown_spot','dead_heart','downy_mildew','hispa','normal','tungro']
IMG_SIZE = 224
MEAN, STD = [0.485,0.456,0.406], [0.229,0.224,0.225]
# =================================

from models.backbones.mobilenet import MobileNetV3_Small_ECA

device = "cuda" if torch.cuda.is_available() else "cpu"
model = MobileNetV3_Small_ECA(num_classes=len(CLASS_NAMES))
state = torch.load(CKPT, map_location=device)

if isinstance(state, dict) and "model" in state: state = state["model"]
if any(k.startswith("module.") for k in state.keys()):
    state = {k.replace("module.","",1): v for k,v in state.items()}
model.load_state_dict(state, strict=False)
model.to(device).eval()

tfm = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(), T.Normalize(MEAN, STD)])


app = FastAPI(title="Rice Leaf Disease API")
BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
STATIC_DIR = WEB_DIR / "static"
TEMPLATES_DIR = WEB_DIR / "templates"

STATIC_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

@app.get("/", response_class=HTMLResponse)
def ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "classes": CLASS_NAMES})

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@torch.inference_mode()
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0].cpu().tolist()
    best = int(torch.argmax(logits, 1))
    return {
        "label": CLASS_NAMES[best],
        "confidence": float(probs[best]),
        "probs": {c: float(p) for c, p in zip(CLASS_NAMES, probs)},
    }
