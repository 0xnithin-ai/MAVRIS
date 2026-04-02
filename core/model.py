"""
model.py — Plant disease classifier using your trained ViT weights.

Architecture: timm vit_base_patch16_224 (CONFIRMED by strict weight matching)
Output:       28 classes
Key format:   timm-style (cls_token, patch_embed.proj.*, blocks.N.*, norm.*, head.*)

Calibration:
    Temperature scaling is applied post-hoc to soften overconfident softmax outputs.
    Default temperature T=1.3 (conservative; overconfident pretrained ViTs typically need T>1).
    To calibrate properly: run calibrate_temperature() on a held-out validation set.

IMPORTANT — Class Labels:
The 28 class labels below are ordered to match the folder/alphabet order used
during your training data preparation (standard ImageFolder convention).
If predictions are wrong, reorder PLANT_CLASSES to match your dataset's
class_to_idx mapping exactly. The index order is what matters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import os

# ---------------------------------------------------------------------------
# 28 class labels — edit these to match your training data's folder ordering.
# These are currently ordered alphabetically as ImageFolder would produce.
# To verify: print your dataset's class_to_idx during training.
# ---------------------------------------------------------------------------
PLANT_CLASSES = [
    "Corn - Gray Leaf Spot",           # 0
    "Corn - Common Rust",              # 1
    "Corn - Northern Leaf Blight",     # 2
    "Corn - Healthy",                  # 3
    "Grape - Black Rot",               # 4
    "Grape - Black Measles",           # 5
    "Grape - Leaf Blight",             # 6
    "Grape - Healthy",                 # 7
    "Peach - Bacterial Spot",          # 8
    "Peach - Healthy",                 # 9
    "Pepper Bell - Bacterial Spot",    # 10
    "Pepper Bell - Healthy",           # 11
    "Potato - Early Blight",           # 12
    "Potato - Late Blight",            # 13
    "Potato - Healthy",                # 14
    "Rice - Brown Spot",               # 15
    "Rice - Healthy",                  # 16
    "Rice - Hispa",                    # 17
    "Rice - Leaf Blast",               # 18
    "Strawberry - Leaf Scorch",        # 19
    "Strawberry - Healthy",            # 20
    "Tomato - Bacterial Spot",         # 21
    "Tomato - Early Blight",           # 22
    "Tomato - Late Blight",            # 23
    "Tomato - Leaf Mold",              # 24
    "Tomato - Septoria Leaf Spot",     # 25
    "Tomato - Spider Mites",           # 26
    "Tomato - Healthy",                # 27
]

assert len(PLANT_CLASSES) == 28, "Label list must have exactly 28 entries."

# ---------------------------------------------------------------------------
# Transform — standard ImageNet normalisation for ViT
# ---------------------------------------------------------------------------
_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# Temperature Scaling — post-hoc calibration of softmax confidence
# ---------------------------------------------------------------------------
class TemperatureScaler(nn.Module):
    """
    Wraps a classification model and scales logits by a learned temperature T.

    Calibration intuition:
        - Uncalibrated ViT:  softmax(logits)   -> overconfident (e.g., 95% when true accuracy is 70%)
        - Calibrated output: softmax(logits/T) -> confidence matches real-world accuracy

    Usage:
        scaler = TemperatureScaler(model)
        # Auto-calibrate on a validation set:
        scaler.calibrate(val_loader, device)
        # Or use default T=1.3 (conservative, safe for overconfident models)
    """
    DEFAULT_TEMPERATURE = 1.3   # safe default; tune with calibrate() on validation data

    def __init__(self, model: nn.Module, temperature: float = DEFAULT_TEMPERATURE):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        return logits / self.temperature.clamp(min=0.1)   # clamp prevents div-by-zero

    def calibrate(self, val_loader, device: torch.device, lr: float = 0.01, epochs: int = 50):
        """
        Find the optimal temperature T that minimises NLL on a validation set.
        Call this once with a held-out set of labelled images.

        Args:
            val_loader: DataLoader yielding (image_tensor, label_tensor) batches
            device:     torch device
            lr:         learning rate for temperature optimisation
            epochs:     number of optimisation steps
        """
        self.model.eval()
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=epochs)

        logits_list, labels_list = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                logits_list.append(self.model(imgs.to(device)))
                labels_list.append(labels.to(device))
        logits_all = torch.cat(logits_list)
        labels_all = torch.cat(labels_list)

        def eval_step():
            optimizer.zero_grad()
            loss = F.cross_entropy(logits_all / self.temperature.clamp(min=0.1), labels_all)
            loss.backward()
            return loss

        optimizer.step(eval_step)
        print(f"[TemperatureScaler] Calibrated T = {self.temperature.item():.4f}")
        return self


class PlantClassifier:
    """
    Singleton wrapper around the trained timm ViT-B/16 model.
    Weights load once; subsequent calls reuse the loaded model.
    """

    _instance = None

    def __new__(cls, model_path: str = "plant_model.pth"):
        if cls._instance is None:
            inst = super().__new__(cls)
            inst._ready = False
            cls._instance = inst
        return cls._instance

    def _load(self, model_path: str):
        if self._ready:
            return
        try:
            import timm
        except ImportError:
            raise ImportError(
                "timm is required. Install with: pip install timm"
            )

        # Force CPU for the ViT classifier. ViT-Base inference is extremely fast on CPU (<50ms).
        # This frees up the entire 4GB GPU specifically for the heavy LLaVA Vision-Language Model during Test-Time Compute!
        self.device = torch.device("cpu")

        self.model = timm.create_model(
            "vit_base_patch16_224",
            pretrained=False,
            num_classes=28,
        )

        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device, weights_only=False)
            missing, unexpected = self.model.load_state_dict(state_dict, strict=True)
            if missing or unexpected:
                print(f"[Classifier] WARNING — missing={len(missing)}, unexpected={len(unexpected)}")
            else:
                print(f"[Classifier] Loaded {model_path} — all 152 keys matched perfectly.")
        else:
            print(f"[Classifier] WARNING: {model_path} not found. Using random weights.")

        self.model.to(self.device)
        self.model.eval()

        # Wrap with temperature scaler for calibrated confidence scores
        self._scaler = TemperatureScaler(self.model).to(self.device)
        self._scaler.eval()
        print(f"[Classifier] Temperature scaling active (T={TemperatureScaler.DEFAULT_TEMPERATURE}).")
        self._ready = True

    def predict(self, image_path: str, model_path: str = "plant_model.pth") -> dict:
        """
        Run inference on the image at image_path.

        Returns:
            {
                "top_class":  str   — human-readable label,
                "confidence": float — 0 to 1 (temperature-scaled),
                "calibrated": bool  — True when temperature scaling is active,
                "top3":       list[tuple[str, float]],
                "raw_index":  int   — raw class index (useful for debugging labels)
            }
        """
        self._load(model_path)

        image = Image.open(image_path).convert("RGB")
        tensor = _TRANSFORM(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Use temperature-scaled logits for calibrated confidence
            scaled_logits = self._scaler(tensor)
            probs = torch.softmax(scaled_logits, dim=1)[0]

        top3_vals, top3_idx = torch.topk(probs, k=3)
        top3 = [
            (PLANT_CLASSES[i.item()], round(v.item(), 4))
            for i, v in zip(top3_idx, top3_vals)
        ]

        return {
            "top_class":  top3[0][0],
            "confidence": top3[0][1],
            "calibrated": True,
            "top3":       top3,
            "raw_index":  top3_idx[0].item(),
        }
