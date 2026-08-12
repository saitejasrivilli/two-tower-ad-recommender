"""
TorchScript Export + Latency Benchmark
Exports the two-tower model to TorchScript for production serving
and benchmarks p50/p95/p99 latency before and after export.
"""

import torch
import numpy as np
import time
import json
import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Dict


# ------------------------------------------------------------------ #
# Safe dummy input helpers — always index 0, guaranteed in-range
# ------------------------------------------------------------------ #

def _user_dummy(user_tower, batch_size: int, device: str):
    """Create safe dummy inputs for the user tower."""
    num_cat = len(user_tower.embedding_layer.embeddings)
    emb_dim = user_tower.embedding_layer.embedding_dim
    numerical_dim = user_tower.mlp[0].in_features - num_cat * emb_dim
    dummy_cat = torch.zeros(batch_size, num_cat, dtype=torch.long).to(device)
    dummy_num = torch.randn(batch_size, numerical_dim).to(device)
    return dummy_cat, dummy_num


def _ad_dummy(ad_tower, batch_size: int, device: str):
    """Create safe dummy inputs for the ad tower."""
    num_cat = len(ad_tower.embedding_layer.embeddings)
    dummy_cat = torch.zeros(batch_size, num_cat, dtype=torch.long).to(device)
    return (dummy_cat,)


# ------------------------------------------------------------------ #
# Export
# ------------------------------------------------------------------ #

def export_user_tower(model, save_dir: str, device: str = "cpu") -> str:
    """
    Trace and save the user tower as TorchScript.
    Uses real embedding vocab sizes from the trained model.
    """
    model.eval()
    model.to(device)

    dummy_cat, dummy_num = _user_dummy(model.user_tower, 1, device)

    print("Tracing user tower...")
    with torch.no_grad():
        traced = torch.jit.trace(model.user_tower, (dummy_cat, dummy_num))

    out_path = Path(save_dir) / "user_tower_scripted.pt"
    traced.save(str(out_path))
    print(f"User tower saved → {out_path}")
    return str(out_path)


def export_ad_tower(model, save_dir: str, device: str = "cpu") -> str:
    """Trace and save the ad tower."""
    model.eval()
    model.to(device)

    dummy_cat, = _ad_dummy(model.ad_tower, 1, device)

    print("Tracing ad tower...")
    with torch.no_grad():
        traced = torch.jit.trace(model.ad_tower, (dummy_cat,))

    out_path = Path(save_dir) / "ad_tower_scripted.pt"
    traced.save(str(out_path))
    print(f"Ad tower saved → {out_path}")
    return str(out_path)


def export_full_model(model, save_dir: str, device: str = "cpu") -> str:
    """Trace full TwoTowerModel (both towers together)."""
    model.eval()
    model.to(device)

    dummy_user_cat, dummy_user_num = _user_dummy(model.user_tower, 1, device)
    dummy_ad_cat, = _ad_dummy(model.ad_tower, 1, device)

    print("Tracing full two-tower model...")
    with torch.no_grad():
        traced = torch.jit.trace(
            model, (dummy_user_cat, dummy_user_num, dummy_ad_cat)
        )

    out_path = Path(save_dir) / "two_tower_scripted.pt"
    traced.save(str(out_path))
    print(f"Full model saved → {out_path}")
    return str(out_path)


# ------------------------------------------------------------------ #
# Benchmark helpers
# ------------------------------------------------------------------ #

def _percentile(arr: list, p: float) -> float:
    idx = int(len(arr) * p / 100)
    return sorted(arr)[min(idx, len(arr) - 1)]


def benchmark_model(
    model,
    user_tower_ref,          # used to derive input shapes
    batch_size: int = 1,
    n_warmup: int = 20,
    n_runs: int = 200,
    device: str = "cpu",
    label: str = "",
) -> Dict:
    """
    Measure inference latency for the user tower (hot path at serve time).
    Derives correct input shapes from the live user_tower reference.
    """
    dummy_cat, dummy_num = _user_dummy(user_tower_ref, batch_size, device)

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy_cat, dummy_num)

    if device == "cuda":
        torch.cuda.synchronize()

    latencies = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(dummy_cat, dummy_num)
            if device == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)

    result = {
        "label": label,
        "batch_size": batch_size,
        "n_runs": n_runs,
        "mean_ms":       round(sum(latencies) / len(latencies), 3),
        "p50_ms":        round(_percentile(latencies, 50), 3),
        "p95_ms":        round(_percentile(latencies, 95), 3),
        "p99_ms":        round(_percentile(latencies, 99), 3),
        "throughput_qps": round(
            batch_size / (sum(latencies) / len(latencies) / 1000), 1
        ),
    }
    return result


def run_benchmark(model, save_dir: str, device: str = "cpu") -> Dict:
    """
    Export user tower to TorchScript, benchmark eager vs scripted side by side.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # --- Eager baseline ---
    print("\n--- Benchmarking eager model ---")
    eager_results = benchmark_model(
        model.user_tower,
        user_tower_ref=model.user_tower,
        device=device,
        label="eager",
    )
    _print_result(eager_results)

    # --- Export ---
    scripted_path = export_user_tower(model, save_dir, device)
    scripted_model = torch.jit.load(scripted_path, map_location=device)

    # --- Scripted ---
    print("\n--- Benchmarking TorchScript model ---")
    scripted_results = benchmark_model(
        scripted_model,
        user_tower_ref=model.user_tower,   # shape reference from original
        device=device,
        label="scripted",
    )
    _print_result(scripted_results)

    # --- Comparison ---
    speedup = eager_results["p99_ms"] / max(scripted_results["p99_ms"], 1e-9)
    print(
        f"\nSpeedup p99: {speedup:.2f}x  "
        f"(eager {eager_results['p99_ms']}ms → "
        f"scripted {scripted_results['p99_ms']}ms)"
    )

    summary = {
        "eager": eager_results,
        "scripted": scripted_results,
        "p99_speedup": round(speedup, 2),
    }

    out_json = Path(save_dir) / "benchmark_results.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved → {out_json}")

    return summary


def _print_result(r: Dict):
    print(
        f"  [{r['label']}] batch={r['batch_size']} | "
        f"mean={r['mean_ms']}ms | p50={r['p50_ms']}ms | "
        f"p95={r['p95_ms']}ms | p99={r['p99_ms']}ms | "
        f"throughput={r['throughput_qps']} QPS"
    )


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(description="Export TwoTowerModel to TorchScript")
    parser.add_argument("--model_dir", default="./models")
    parser.add_argument("--save_dir",  default="./models")
    parser.add_argument("--device",    default="cpu")
    parser.add_argument("--benchmark", action="store_true", default=True)
    args = parser.parse_args()

    from two_tower_model import TwoTowerModel

    # Load real feature dims from saved preprocessor
    preprocessor_path = os.path.join(args.model_dir, "preprocessor.pkl")
    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(
            f"preprocessor.pkl not found in {args.model_dir}. "
            "Run train.py first."
        )

    with open(preprocessor_path, "rb") as f:
        preprocessor = pickle.load(f)

    # Handle both object and dict forms
    if isinstance(preprocessor, dict):
        feature_dims  = preprocessor["feature_dims"]
        numerical_cols = preprocessor["numerical_cols"]
    else:
        feature_dims  = preprocessor.feature_dims
        numerical_cols = preprocessor.numerical_cols

    user_feature_dims = {
        f"C{i}": feature_dims[f"C{i}"]
        for i in range(1, 7)
        if f"C{i}" in feature_dims
    }
    ad_feature_dims = {
        f"C{i}": feature_dims[f"C{i}"]
        for i in range(7, 27)
        if f"C{i}" in feature_dims
    }
    numerical_dim = len(numerical_cols)

    print(f"Loaded preprocessor: {len(user_feature_dims)} user cat features, "
          f"{len(ad_feature_dims)} ad cat features, {numerical_dim} numerical")

    # Build model with correct dims
    model = TwoTowerModel(
        user_feature_dims=user_feature_dims,
        ad_feature_dims=ad_feature_dims,
        numerical_dim=numerical_dim,
        embedding_dim=16,
        hidden_dims=[512, 256],
        output_dim=256,
    )

    # Load trained weights
    ckpt_path = os.path.join(args.model_dir, "two_tower_best.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(args.model_dir, "two_tower_final.pt")

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=args.device)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state)
        print(f"Loaded weights from {ckpt_path}")
    else:
        print("No checkpoint found — using random weights.")

    model.to(args.device).eval()

    # Export all towers
    export_user_tower(model, args.save_dir, args.device)
    export_ad_tower(model, args.save_dir, args.device)
    export_full_model(model, args.save_dir, args.device)

    # Benchmark if requested
    if args.benchmark:
        run_benchmark(model, args.save_dir, args.device)

    print("\n✓ TorchScript export complete!")