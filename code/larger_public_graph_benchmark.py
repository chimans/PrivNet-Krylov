#!/usr/bin/env python3
"""Executed larger public-data graph experiment for PrivNet-Krylov.

The script converts scikit-learn's bundled handwritten-digits observations into
a 10-nearest-neighbour graph. Because the 1,797 observations ship with
scikit-learn, the experiment can be reproduced without a network download. A
fixed orthogonal projection, independent of the observations and labels, maps
64 pixel channels to 32 channels before Krylov compilation. Labels are used
only for the train/validation/test split and the supervised objective.

The activation interval is calibrated from training and validation nodes only;
test preactivations never influence the polynomial fit.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.datasets import load_digits
from sklearn.neighbors import NearestNeighbors

from privnet_krylov import CompiledSpectralNet, fit_relu_polynomial


@dataclass
class DigitsGraphConfig:
    neighbors: int = 10
    feature_dim: int = 32
    projection_seed: int = 20260905
    krylov_degree: int = 2
    hidden_dim: int = 28
    activation_degree: int = 5
    train_per_class: int = 50
    val_per_class: int = 20
    epochs_relu: int = 450
    patience_relu: int = 70
    epochs_poly: int = 160
    patience_poly: int = 45
    lr_relu: float = 0.02
    lr_poly: float = 0.004
    weight_decay: float = 5e-4
    interval_padding: float = 1.08
    interval_penalty: float = 0.005
    poly_grid_size: int = 12001


def _stratified_split(labels: np.ndarray, seed: int, cfg: DigitsGraphConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train: List[int] = []
    val: List[int] = []
    test: List[int] = []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0].copy()
        rng.shuffle(idx)
        nt, nv = cfg.train_per_class, cfg.val_per_class
        train.extend(idx[:nt])
        val.extend(idx[nt:nt + nv])
        test.extend(idx[nt + nv:])
    return np.asarray(train), np.asarray(val), np.asarray(test)


def _fixed_projection(dim_in: int, dim_out: int, seed: int) -> np.ndarray:
    """Return a fixed orthonormal projection independent of the observations."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(dim_in, dim_out)))
    return q[:, :dim_out]


def load_digits_knn(cfg: DigitsGraphConfig):
    digits = load_digits()
    raw = np.asarray(digits.data, dtype=np.float64) / 16.0
    labels = np.asarray(digits.target, dtype=np.int64)

    # The graph uses only observed pixel values, not class labels.
    nn = NearestNeighbors(n_neighbors=cfg.neighbors + 1, metric="euclidean", algorithm="auto")
    nn.fit(raw)
    _, indices = nn.kneighbors(raw, return_distance=True)
    rows = np.repeat(np.arange(raw.shape[0]), cfg.neighbors)
    cols = indices[:, 1:].reshape(-1)
    data = np.ones(rows.shape[0], dtype=np.float64)
    a = sp.csr_matrix((data, (rows, cols)), shape=(raw.shape[0], raw.shape[0]))
    a = a.maximum(a.T)
    a.data[:] = 1.0
    a.eliminate_zeros()

    proj = _fixed_projection(raw.shape[1], cfg.feature_dim, cfg.projection_seed)
    x = raw @ proj
    x = x / (np.sqrt(np.mean(x * x, axis=0, keepdims=True)) + 1e-12)

    a_loop = a + sp.eye(a.shape[0], format="csr")
    degree = np.asarray(a_loop.sum(axis=1)).ravel()
    inv = np.zeros_like(degree)
    mask = degree > 0
    inv[mask] = 1.0 / np.sqrt(degree[mask])
    d = sp.diags(inv)
    shift = (d @ a_loop @ d).tocsr()
    return a, shift, x.astype(np.float64), labels, proj


def sparse_chebyshev_compile(shift: sp.csr_matrix, x: np.ndarray, degree: int) -> List[np.ndarray]:
    blocks = [np.asarray(x, dtype=np.float64).copy()]
    if degree == 0:
        return blocks
    blocks.append(np.asarray(shift @ blocks[0]))
    for _ in range(1, degree):
        blocks.append(2.0 * np.asarray(shift @ blocks[-1]) - blocks[-2])
    return blocks


def _poly_torch(x: torch.Tensor, coeff: np.ndarray) -> torch.Tensor:
    y = torch.zeros_like(x) + float(coeff[-1])
    for c in coeff[-2::-1]:
        y = y * x + float(c)
    return y


def _train_relu(seed: int, cfg: DigitsGraphConfig, blocks_np, labels_np, tr, va):
    torch.manual_seed(seed)
    np.random.seed(seed)
    blocks = torch.tensor(np.stack(blocks_np), dtype=torch.float32)
    labels = torch.tensor(labels_np, dtype=torch.long)
    model = CompiledSpectralNet(cfg.feature_dim, cfg.hidden_dim, 10, cfg.krylov_degree)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr_relu, weight_decay=cfg.weight_decay)
    best_state = None
    best_val = -1.0
    bad = 0
    for _ in range(cfg.epochs_relu):
        model.train()
        opt.zero_grad(set_to_none=True)
        logits = model(blocks)
        loss = torch.nn.functional.cross_entropy(logits[tr], labels[tr])
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            val_acc = float((model(blocks)[va].argmax(dim=1) == labels[va]).float().mean().item())
        if val_acc > best_val + 1e-9:
            best_val = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= cfg.patience_relu:
            break
    if best_state is None:
        raise RuntimeError("training did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    return model, best_val


def _poly_finetune(model, blocks_np, labels_np, tr, va, coeff, bound, cfg: DigitsGraphConfig):
    blocks = torch.tensor(np.stack(blocks_np), dtype=torch.float32)
    labels = torch.tensor(labels_np, dtype=torch.long)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr_poly, weight_decay=cfg.weight_decay)
    best_state = None
    best_val = -1.0
    bad = 0
    for _ in range(cfg.epochs_poly):
        model.train()
        opt.zero_grad(set_to_none=True)
        u = model.preactivation(blocks)
        logits = model.out(_poly_torch(u, coeff))
        ce = torch.nn.functional.cross_entropy(logits[tr], labels[tr])
        # Keep the learned representation inside the interval chosen without test nodes.
        interval = torch.relu(torch.abs(u[tr]) - float(bound)).pow(2).mean()
        loss = ce + cfg.interval_penalty * interval
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            z = model.out(_poly_torch(model.preactivation(blocks), coeff))
            val_acc = float((z[va].argmax(dim=1) == labels[va]).float().mean().item())
        if val_acc > best_val + 1e-9:
            best_val = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= cfg.patience_poly:
            break
    if best_state is None:
        raise RuntimeError("polynomial fine-tuning did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    return model, best_val


def run_seed(seed: int, cfg: DigitsGraphConfig, save_artifact: Path | None = None) -> Dict:
    a, shift, x, labels_np, proj = load_digits_knn(cfg)
    tr, va, te = _stratified_split(labels_np, seed, cfg)
    blocks_np = sparse_chebyshev_compile(shift, x, cfg.krylov_degree)
    model, relu_val = _train_relu(seed, cfg, blocks_np, labels_np, tr, va)

    blocks = torch.tensor(np.stack(blocks_np), dtype=torch.float32)
    labels = torch.tensor(labels_np, dtype=torch.long)
    model.eval()
    with torch.no_grad():
        u0 = model.preactivation(blocks)
        z_relu0 = model.out(torch.relu(u0))
    relu_test0 = float((z_relu0[te].argmax(dim=1) == labels[te]).float().mean().item())

    calib_idx = np.concatenate([tr, va])
    bound = float(torch.max(torch.abs(u0[calib_idx])).item() * cfg.interval_padding + 1e-12)
    coeff, eps_act = fit_relu_polynomial(cfg.activation_degree, bound, cfg.poly_grid_size)
    with torch.no_grad():
        z_poly0 = model.out(_poly_torch(u0, coeff))
    poly_before = float((z_poly0[te].argmax(dim=1) == labels[te]).float().mean().item())

    model, poly_val = _poly_finetune(model, blocks_np, labels_np, tr, va, coeff, bound, cfg)
    model.eval()
    with torch.no_grad():
        u = model.preactivation(blocks)
        z_poly = model.out(_poly_torch(u, coeff))
        z_relu = model.out(torch.relu(u))
    poly_test = float((z_poly[te].argmax(dim=1) == labels[te]).float().mean().item())
    relu_test = float((z_relu[te].argmax(dim=1) == labels[te]).float().mean().item())
    agreement = float((z_poly.argmax(dim=1) == z_relu.argmax(dim=1)).float().mean().item())
    test_interval_coverage = float((torch.abs(u[te]) <= bound).float().mean().item())

    row = {
        "seed": seed,
        "nodes": int(x.shape[0]),
        "undirected_edges": int(a.nnz // 2),
        "feature_dim": int(x.shape[1]),
        "classes": 10,
        "knn_neighbors": cfg.neighbors,
        "krylov_degree": cfg.krylov_degree,
        "activation_degree": cfg.activation_degree,
        "train_nodes": int(len(tr)),
        "validation_nodes": int(len(va)),
        "test_nodes": int(len(te)),
        "relu_val_accuracy": relu_val,
        "relu_test_accuracy_before_poly_finetune": relu_test0,
        "poly_test_accuracy_before_finetune": poly_before,
        "poly_val_accuracy_after_finetune": poly_val,
        "poly_test_accuracy_after_finetune": poly_test,
        "relu_test_accuracy_after_finetune": relu_test,
        "poly_relu_prediction_agreement_all_nodes": agreement,
        "activation_interval_bound_train_val_only": bound,
        "activation_uniform_error": float(eps_act),
        "test_preactivation_interval_coverage": test_interval_coverage,
    }

    if save_artifact is not None:
        save_artifact.parent.mkdir(parents=True, exist_ok=True)
        state = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
        np.savez_compressed(
            save_artifact,
            adjacency_data=a.data,
            adjacency_indices=a.indices,
            adjacency_indptr=a.indptr,
            adjacency_shape=np.asarray(a.shape, dtype=np.int64),
            shift_data=shift.data,
            shift_indices=shift.indices,
            shift_indptr=shift.indptr,
            shift_shape=np.asarray(shift.shape, dtype=np.int64),
            features=x,
            labels=labels_np,
            blocks=np.stack(blocks_np),
            train_idx=tr,
            val_idx=va,
            test_idx=te,
            projection=proj,
            activation_coeff=np.asarray(coeff, dtype=np.float64),
            activation_bound=np.asarray([bound]),
            theta=state["theta"],
            hidden_bias=state["bias"],
            out_weight=state["out.weight"].T,
            out_bias=state["out.bias"],
            plaintext_poly_logits=z_poly.detach().cpu().numpy(),
        )
    return row


def _stats(rows: List[Dict], key: str) -> Dict[str, float]:
    a = np.asarray([r[key] for r in rows], dtype=float)
    return {"mean": float(a.mean()), "sample_sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0}


def summarize(rows: List[Dict], cfg: DigitsGraphConfig) -> Dict:
    first = rows[0]
    return {
        "dataset": "scikit-learn Digits 10-NN graph",
        "provenance": "load_digits() public data; graph built deterministically from pixel-space nearest neighbours",
        "num_seeds": len(rows),
        "config": asdict(cfg),
        "nodes": first["nodes"],
        "undirected_edges": first["undirected_edges"],
        "feature_dim": first["feature_dim"],
        "train_validation_test": [first["train_nodes"], first["validation_nodes"], first["test_nodes"]],
        "relu_test_accuracy_after_finetune": _stats(rows, "relu_test_accuracy_after_finetune"),
        "poly_test_accuracy_after_finetune": _stats(rows, "poly_test_accuracy_after_finetune"),
        "poly_relu_prediction_agreement_all_nodes": _stats(rows, "poly_relu_prediction_agreement_all_nodes"),
        "test_preactivation_interval_coverage": _stats(rows, "test_preactivation_interval_coverage"),
        "evidence_note": (
            "The graph is derived from public observations rather than supplied as a canonical graph benchmark. "
            "It is used to add a substantially larger, fully executable public-data graph while preserving an offline reproduction path."
        ),
    }


def write_table(path: Path, summary: Dict) -> None:
    pct = lambda x: 100.0 * x
    s = summary
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Larger public-data graph validation on the 1,797-node Digits 10-NN graph. Values are mean $\pm$ sample standard deviation across five stratified splits. The polynomial interval is calibrated from train and validation nodes only.}",
        r"\label{tab:digits-knn}",
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"Quantity & Result \\",
        r"\midrule",
        f"Nodes / undirected edges & {s['nodes']} / {s['undirected_edges']} \\\\",
        f"Classes / compiled feature channels & 10 / {s['feature_dim']} \\\\",
        f"Train / validation / test nodes & {s['train_validation_test'][0]} / {s['train_validation_test'][1]} / {s['train_validation_test'][2]} \\\\",
        f"Krylov degree / activation degree & {s['config']['krylov_degree']} / {s['config']['activation_degree']} \\\\",
        f"ReLU test accuracy & {pct(s['relu_test_accuracy_after_finetune']['mean']):.2f} $\\pm$ {pct(s['relu_test_accuracy_after_finetune']['sample_sd']):.2f}\\% \\\\",
        f"Polynomial test accuracy & {pct(s['poly_test_accuracy_after_finetune']['mean']):.2f} $\\pm$ {pct(s['poly_test_accuracy_after_finetune']['sample_sd']):.2f}\\% \\\\",
        f"Polynomial/ReLU agreement & {pct(s['poly_relu_prediction_agreement_all_nodes']['mean']):.2f} $\\pm$ {pct(s['poly_relu_prediction_agreement_all_nodes']['sample_sd']):.2f}\\% \\\\",
        f"Test preactivations inside calibrated interval & {pct(s['test_preactivation_interval_coverage']['mean']):.2f} $\\pm$ {pct(s['test_preactivation_interval_coverage']['sample_sd']):.2f}\\% \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[3, 7, 11, 19, 29])
    ap.add_argument("--artifact-seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=Path("outputs/larger_public_graph"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = DigitsGraphConfig()

    rows: List[Dict] = []
    for seed in args.seeds:
        artifact = args.out / "digits_knn_seed7_ckks_artifact.npz" if seed == args.artifact_seed else None
        rows.append(run_seed(seed, cfg, artifact))

    with (args.out / "digits_knn_benchmark.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows, cfg)
    (args.out / "digits_knn_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_table(args.out / "digits_knn_table.tex", summary)

    seeds = [r["seed"] for r in rows]
    relu = [100.0 * r["relu_test_accuracy_after_finetune"] for r in rows]
    poly = [100.0 * r["poly_test_accuracy_after_finetune"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.plot(seeds, relu, marker="o", label="ReLU")
    ax.plot(seeds, poly, marker="s", label="Degree-5 polynomial")
    ax.set_xlabel("Split seed")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Digits 10-NN graph: larger public-data validation")
    ax.set_ylim(80, 101)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out / "digits_knn_accuracy.pdf")
    fig.savefig(args.out / "digits_knn_accuracy.png", dpi=200)
    plt.close(fig)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
