from __future__ import annotations

import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from src.config import IMAGE_SIZE, MODEL_DIR, REPORT_DIR
from src.model import load_checkpoint


TORCH_MODEL = MODEL_DIR / "best_model.pth"
ONNX_MODEL = MODEL_DIR / "food_classifier.onnx"
TORCH_METRICS = REPORT_DIR / "test_metrics.json"
ONNX_METRICS = REPORT_DIR / "onnx_fp32_metrics.json"


def mb(n: int) -> float:
    return n / (1024 * 1024)


def torch_model_size(path: Path) -> float:
    return mb(path.stat().st_size)


def onnx_model_size(path: Path) -> float:
    # ONNX가 external data를 사용하는 경우 .onnx.data까지 합산
    total = path.stat().st_size

    candidates = [
        Path(str(path) + ".data"),       # food_classifier.onnx.data
        path.with_suffix(".data"),       # food_classifier.data
    ]

    seen = set()
    for candidate in candidates:
        if candidate.exists() and candidate not in seen:
            total += candidate.stat().st_size
            seen.add(candidate)

    return mb(total)


def load_accuracy(path: Path) -> float | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    value = data.get("accuracy")
    return float(value) if value is not None else None


def benchmark_torch(x_np: np.ndarray, runs: int, warmup: int):
    model, _, _ = load_checkpoint(str(TORCH_MODEL), torch.device("cpu"))
    x = torch.from_numpy(x_np)

    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(x)

        times = []
        for _ in range(runs):
            start = time.perf_counter()
            _ = model(x)
            times.append((time.perf_counter() - start) * 1000)

    return statistics.median(times), statistics.mean(times)


def benchmark_onnx(x_np: np.ndarray, runs: int, warmup: int):
    import onnxruntime as ort

    session = ort.InferenceSession(
        str(ONNX_MODEL),
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name

    for _ in range(warmup):
        _ = session.run(None, {input_name: x_np})

    times = []
    for _ in range(runs):
        start = time.perf_counter()
        _ = session.run(None, {input_name: x_np})
        times.append((time.perf_counter() - start) * 1000)

    return statistics.median(times), statistics.mean(times)


def fmt_acc(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100:.2f}%"


def main():
    if not TORCH_MODEL.exists():
        raise FileNotFoundError(f"PyTorch model not found: {TORCH_MODEL}")
    if not ONNX_MODEL.exists():
        raise FileNotFoundError(f"ONNX model not found: {ONNX_MODEL}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    runs = 200
    warmup = 30

    rng = np.random.default_rng(42)
    x_np = rng.standard_normal(
        (1, 3, IMAGE_SIZE, IMAGE_SIZE),
        dtype=np.float32,
    )

    print()
    print("=" * 82)
    print("                 SMART CALORIE AI | FINAL CPU BENCHMARK")
    print("=" * 82)
    print("Benchmarking... (model inference only, image preprocessing excluded)")
    print()

    torch_median, torch_mean = benchmark_torch(x_np, runs, warmup)
    onnx_median, onnx_mean = benchmark_onnx(x_np, runs, warmup)

    torch_acc = load_accuracy(TORCH_METRICS)
    onnx_acc = load_accuracy(ONNX_METRICS)

    torch_size = torch_model_size(TORCH_MODEL)
    onnx_size = onnx_model_size(ONNX_MODEL)

    speedup = torch_median / onnx_median if onnx_median > 0 else 0.0
    accuracy_change = None
    if torch_acc is not None and onnx_acc is not None:
        accuracy_change = (onnx_acc - torch_acc) * 100

    print("[ CLASSIFICATION ACCURACY ]")
    print("-" * 82)
    print(f"PyTorch FP32 Test Accuracy : {fmt_acc(torch_acc)}")
    print(f"ONNX FP32 Test Accuracy    : {fmt_acc(onnx_acc)}")
    if accuracy_change is not None:
        print(f"Accuracy Change            : {accuracy_change:+.2f} percentage points")
    print()

    print("[ MODEL SIZE ]")
    print("-" * 82)
    print(f"PyTorch FP32               : {torch_size:.3f} MB")
    print(f"ONNX FP32                  : {onnx_size:.3f} MB")
    print()

    print("[ CPU INFERENCE LATENCY ]")
    print("-" * 82)
    print(f"{'Backend':<22}{'Median':>14}{'Mean':>14}")
    print("-" * 50)
    print(f"{'PyTorch FP32':<22}{torch_median:>11.3f} ms{torch_mean:>11.3f} ms")
    print(f"{'ONNX Runtime FP32':<22}{onnx_median:>11.3f} ms{onnx_mean:>11.3f} ms")
    print()
    print(f"ONNX Runtime Speedup       : {speedup:.2f}x")
    print()

    print("[ FINAL DEPLOYMENT DECISION ]")
    print("-" * 82)
    print("Deployment model           : PyTorch FP32 / MobileNetV3-Small")
    print("Reason                     : INT8 experiments caused unacceptable")
    print("                             accuracy degradation; FP32 accuracy retained.")
    print()
    print(f"Runs / Warm-up             : {runs} / {warmup}")
    print(f"Platform                   : {platform.platform()}")
    print("=" * 82)

    result = {
        "benchmark_scope": "model inference only; preprocessing excluded",
        "runs": runs,
        "warmup": warmup,
        "pytorch": {
            "accuracy": torch_acc,
            "model_mb": round(torch_size, 4),
            "median_ms": round(torch_median, 4),
            "mean_ms": round(torch_mean, 4),
        },
        "onnx_fp32": {
            "accuracy": onnx_acc,
            "model_mb": round(onnx_size, 4),
            "median_ms": round(onnx_median, 4),
            "mean_ms": round(onnx_mean, 4),
        },
        "onnx_speedup": round(speedup, 4),
        "accuracy_change_percentage_points": (
            None if accuracy_change is None else round(accuracy_change, 4)
        ),
        "deployment_decision": (
            "Use PyTorch FP32 MobileNetV3-Small. "
            "INT8 PTQ/QAT excluded due severe accuracy degradation."
        ),
    }

    out = REPORT_DIR / "final_fp32_benchmark.json"
    out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved: {out.resolve()}")


if __name__ == "__main__":
    main()
