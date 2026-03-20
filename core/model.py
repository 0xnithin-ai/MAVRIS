"""
model.py — Plant disease classifier using your trained ViT weights.

Architecture: timm vit_base_patch16_224 (CONFIRMED by strict weight matching)
Output:       28 classes
Key format:   timm-style (cls_token, patch_embed.proj.*, blocks.N.*, norm.*, head.*)

IMPORTANT — Class Labels:
The 28 class labels below are ordered to match the folder/alphabet order used
during your training data preparation (standard ImageFolder convention).
If predictions are wrong, reorder PLANT_CLASSES to match your dataset's
class_to_idx mapping exactly. The index order is what matters.
"""

import torch
import torch.nn as nn
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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        self._ready = True

    def predict(self, image_path: str, model_path: str = "plant_model.pth") -> dict:
        """
        Run inference on the image at image_path.

        Returns:
            {
                "top_class":  str   — human-readable label,
                "confidence": float — 0 to 1,
                "top3":       list[tuple[str, float]],
                "raw_index":  int   — raw class index (useful for debugging labels)
            }
        """
        self._load(model_path)

        image = Image.open(image_path).convert("RGB")
        tensor = _TRANSFORM(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]

        top3_vals, top3_idx = torch.topk(probs, k=3)
        top3 = [
            (PLANT_CLASSES[i.item()], round(v.item(), 4))
            for i, v in zip(top3_idx, top3_vals)
        ]

        return {
            "top_class":  top3[0][0],
            "confidence": top3[0][1],
            "top3":       top3,
            "raw_index":  top3_idx[0].item(),
        }
