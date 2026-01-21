from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import platform
from importlib.util import find_spec

import requests
from PIL import Image

from .settings import AppSettings


settings = AppSettings()


@dataclass
class PlantIdentification:
    label: str
    score: float


_MODEL = None
_PROCESSOR = None


def _load_model() -> str | None:
    global _MODEL, _PROCESSOR
    if _MODEL is not None and _PROCESSOR is not None:
        return None
    try:
        if platform.system() == "Windows":
            try:
                spec = find_spec("torch")
                if spec and spec.origin:
                    dll_path = Path(spec.origin).parent / "lib" / "c10.dll"
                    if dll_path.exists():
                        import ctypes

                        ctypes.CDLL(str(dll_path))
            except Exception:
                pass
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        _PROCESSOR = AutoImageProcessor.from_pretrained(settings.plant_id_model)
        _MODEL = AutoModelForImageClassification.from_pretrained(settings.plant_id_model)
        _MODEL.eval()
        return None
    except Exception as exc:
        return str(exc)


def identify_plant(image_url: str) -> PlantIdentification | None:
    err = _load_model()
    if err:
        return None
    resp = requests.get(image_url, timeout=settings.request_timeout_s)
    resp.raise_for_status()
    image = Image.open(BytesIO(resp.content)).convert("RGB")
    inputs = _PROCESSOR(images=image, return_tensors="pt")
    outputs = _MODEL(**inputs)
    probs = outputs.logits.softmax(dim=1)[0]
    score, idx = probs.max(dim=0)
    label = _MODEL.config.id2label[int(idx)]
    return PlantIdentification(label=label, score=float(score))
