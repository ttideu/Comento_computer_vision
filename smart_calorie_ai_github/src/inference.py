from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image

from .data import eval_transform
from .model import load_checkpoint


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float


class TorchClassifier:
    def __init__(self, checkpoint_path: str | Path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.class_names, _ = load_checkpoint(str(checkpoint_path), self.device)
        self.transform = eval_transform()

    @torch.inference_mode()
    def predict(self, image: Image.Image, top_k: int = 3) -> list[Prediction]:
        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)[0]
        values, indices = torch.topk(probs, k=min(top_k, len(self.class_names)))
        return [
            Prediction(self.class_names[idx], float(value))
            for value, idx in zip(values.cpu(), indices.cpu())
        ]


class ONNXClassifier:
    def __init__(self, onnx_path: str | Path, class_names: Sequence[str]):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime이 설치되어 있지 않습니다.") from exc

        self.session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.class_names = list(class_names)
        self.transform = eval_transform()

    def predict(self, image: Image.Image, top_k: int = 3) -> list[Prediction]:
        x = self.transform(image.convert("RGB")).unsqueeze(0).numpy().astype(np.float32)
        logits = self.session.run(None, {self.input_name: x})[0][0]
        logits = logits - np.max(logits)
        probs = np.exp(logits) / np.exp(logits).sum()
        indices = np.argsort(probs)[::-1][:top_k]
        return [Prediction(self.class_names[i], float(probs[i])) for i in indices]
