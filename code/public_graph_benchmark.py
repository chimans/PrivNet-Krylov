#!/usr/bin/env python3
"""Public-graph validation and CKKS artifact export.

The experiment uses NetworkX's Zachary Karate Club graph (34 nodes, 78
undirected edges) with identity node features. For each seed, the script builds
a stratified train/validation/test split, trains the K=2 spectral model, fits a
degree-5 activation polynomial, and performs polynomial-aware fine-tuning. It
also exports the deterministic seed-7 artifact used by the matched KGC/SK-HE
TenSEAL benchmark.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

from privnet_krylov import (
    CompiledSpectralNet,
    accuracy,
    chebyshev_krylov_compile,
    cyclic_diagonal_count,
    fit_relu_polynomial,
    normalized_adjacency,
)


@dataclass
class PublicGraphConfig:
    krylov_degree: int = 2
    hidden_dim: int = 12
    activation_degree: int = 5
    train_per_class: int = 8
    val_per_class: int = 4
    epochs_relu: int = 1200
    patience_relu: int = 200
    epochs_poly: int = 500
    patience_poly: int = 100
    lr_relu: float = 0.03
    lr_poly: float = 0.005
    weight_decay: float = 5e-4
    interval_padding: float = 1.05
    interval_penalty: float = 0.01
    poly_grid_size: int = 12001


def _stratified_split(labels: np.ndarray, seed: int, cfg: PublicGraphConfig):
    rng = np.random.default_rng(seed)
    train: List[int] = []
    val: List[int] = []
    test: List[int] = []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0].copy()
        rng.shuffle(idx)
        nt, nv = cfg.train_per_class, cfg.val_per_class
        train.extend(idx[:nt].tolist())
        val.extend(idx[nt:nt + nv].tolist())
        test.extend(idx[nt + nv:].tolist())
    return np.asarray(train), np.asarray(val), np.asarray(test)


def _poly_torch(x: torch.Tensor, coeff: np.ndarray) -> torch.Tensor:
    # Horner evaluation is numerically stable in plaintext.  The HE path uses
    # TenSEAL.polyval, which employs a depth-aware polynomial circuit.
    y = torch.zeros_like(x) + float(coeff[-1])
    for c in coeff[-2::-1]:
        y = y * x + float(c)
    return y


def load_karate():
    g = nx.karate_club_graph()
    a = nx.to_numpy_array(g, dtype=np.float64, weight=None)
    labels = np.asarray(
        [0 if g.nodes[i]["club"] == "Mr. Hi" else 1 for i in range(g.number_of_nodes())],
        dtype=np.int64,
    )
    # Featureless transductive representation: one basis vector per node.
    x = np.eye(g.number_of_nodes(), dtype=np.float64)
    s = normalized_adjacency(a, self_loops=True)
    return g, a, s, x, labels


def _train_relu(seed: int, cfg: PublicGraphConfig, blocks_np, labels_np, tr, va):
    torch.manual_seed(seed)
    np.random.seed(seed)
    blocks = torch.tensor(np.stack(blocks_np), dtype=torch.float32)
    labels = torch.tensor(labels_np, dtype=torch.long)
    model = CompiledSpectralNet(
        feature_dim=blocks.shape[2], hidden_dim=cfg.hidden_dim,
        num_classes=2, degree=cfg.krylov_degree,
    )
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
            val_acc = accuracy(model(blocks), labels, va)
        if val_acc > best_val + 1e-9:
            best_val = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= cfg.patience_relu:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    return model, float(best_val)


def _poly_finetune(model, blocks_np, labels_np, tr, va, coeff, bound, cfg):
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
    assert best_state is not None
    model.load_state_dict(best_state)
    return model, float(best_val)


def run_seed(seed: int, cfg: PublicGraphConfig, save_artifact: Path | None = None) -> Dict:
    g, a, s, x, labels_np = load_karate()
    tr, va, te = _stratified_split(labels_np, seed, cfg)
    blocks_np = chebyshev_krylov_compile(s, x, cfg.krylov_degree)
    model, relu_val = _train_relu(seed, cfg, blocks_np, labels_np, tr, va)

    blocks = torch.tensor(np.stack(blocks_np), dtype=torch.float32)
    labels = torch.tensor(labels_np, dtype=torch.long)
    model.eval()
    with torch.no_grad():
        u0 = model.preactivation(blocks)
        z_relu = model.out(torch.relu(u0))
    relu_test = float((z_relu[te].argmax(dim=1) == labels[te]).float().mean().item())

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
        z_relu_after = model.out(torch.relu(u))
    poly_test = float((z_poly[te].argmax(dim=1) == labels[te]).float().mean().item())
    relu_after = float((z_relu_after[te].argmax(dim=1) == labels[te]).float().mean().item())
    agreement = float((z_poly.argmax(dim=1) == z_relu_after.argmax(dim=1)).float().mean().item())

    row = {
        "seed": seed,
        "nodes": g.number_of_nodes(),
        "edges": g.number_of_edges(),
        "feature_dim": int(x.shape[1]),
        "classes": 2,
        "krylov_degree": cfg.krylov_degree,
        "activation_degree": cfg.activation_degree,
        "relu_val_accuracy": relu_val,
        "relu_test_accuracy_before_poly_finetune": relu_test,
        "poly_test_accuracy_before_finetune": poly_before,
        "poly_val_accuracy_after_finetune": poly_val,
        "poly_test_accuracy_after_finetune": poly_test,
        "relu_test_accuracy_after_finetune": relu_after,
        "poly_relu_prediction_agreement_all_nodes": agreement,
        "activation_interval_bound_train_val_only": bound,
        "test_preactivation_interval_coverage": float((torch.abs(u[te]) <= bound).float().mean().item()),
        "activation_uniform_error": float(eps_act),
        "cyclic_diagonal_count_normalized_adjacency": int(cyclic_diagonal_count(s, tol=1e-15)),
    }

    if save_artifact is not None:
        save_artifact.parent.mkdir(parents=True, exist_ok=True)
        state = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
        np.savez_compressed(
            save_artifact,
            adjacency=a,
            shift=s,
            features=x,
            labels=labels_np,
            blocks=np.stack(blocks_np),
            train_idx=tr,
            val_idx=va,
            test_idx=te,
            activation_coeff=np.asarray(coeff, dtype=np.float64),
            activation_bound=np.asarray([bound]),
            theta=state["theta"],
            hidden_bias=state["bias"],
            out_weight=state["out.weight"].T,
            out_bias=state["out.bias"],
            plaintext_poly_logits=z_poly.detach().cpu().numpy(),
        )
    return row


def _summary(rows: List[Dict], cfg: PublicGraphConfig) -> Dict:
    def stat(key):
        a = np.asarray([r[key] for r in rows], dtype=float)
        return {"mean": float(a.mean()), "sample_sd": float(a.std(ddof=1))}
    return {
        "dataset": "Zachary Karate Club",
        "dataset_source": "NetworkX karate_club_graph; Zachary (1977)",
        "num_seeds": len(rows),
        "config": asdict(cfg),
        "relu_test_accuracy_before_poly_finetune": stat("relu_test_accuracy_before_poly_finetune"),
        "poly_test_accuracy_before_finetune": stat("poly_test_accuracy_before_finetune"),
        "poly_test_accuracy_after_finetune": stat("poly_test_accuracy_after_finetune"),
        "relu_test_accuracy_after_finetune": stat("relu_test_accuracy_after_finetune"),
        "poly_relu_prediction_agreement_all_nodes": stat("poly_relu_prediction_agreement_all_nodes"),
        "cyclic_diagonal_count_normalized_adjacency": rows[0]["cyclic_diagonal_count_normalized_adjacency"],
        "interpretation": (
            "The public-graph experiment validates model behavior on observed network data. "
            "It is intentionally a small-node benchmark so that the same artifact can be used "
            "for real CKKS execution. It is not presented as a large-scale GNN benchmark."
        ),
    }


def _write_table(path: Path, summary: Dict):
    pct = lambda x: 100.0 * x
    r = summary
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Public-graph validation on Zachary's Karate Club over 20 stratified splits. Values are mean $\pm$ sample standard deviation.}",
        r"\label{tab:karate-public}",
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"Quantity & Result \\",
        r"\midrule",
        r"Nodes / edges & 34 / 78 \\",
        r"Classes / input features & 2 / 34 \\",
        r"Krylov degree / polynomial degree & 2 / 5 \\",
        f"ReLU test accuracy & {pct(r['relu_test_accuracy_before_poly_finetune']['mean']):.2f} $\\pm$ {pct(r['relu_test_accuracy_before_poly_finetune']['sample_sd']):.2f}\\% \\\\",
        f"Polynomial test accuracy before fine-tuning & {pct(r['poly_test_accuracy_before_finetune']['mean']):.2f} $\\pm$ {pct(r['poly_test_accuracy_before_finetune']['sample_sd']):.2f}\\% \\\\",
        f"Polynomial test accuracy after fine-tuning & {pct(r['poly_test_accuracy_after_finetune']['mean']):.2f} $\\pm$ {pct(r['poly_test_accuracy_after_finetune']['sample_sd']):.2f}\\% \\\\",
        f"Polynomial/ReLU prediction agreement & {pct(r['poly_relu_prediction_agreement_all_nodes']['mean']):.2f} $\\pm$ {pct(r['poly_relu_prediction_agreement_all_nodes']['sample_sd']):.2f}\\% \\\\",
        f"Nonzero cyclic diagonals of $S$ & {r['cyclic_diagonal_count_normalized_adjacency']} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 21)))
    ap.add_argument("--artifact-seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=Path("outputs/public_graph"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = PublicGraphConfig()

    rows = []
    for seed in args.seeds:
        artifact = args.out / "karate_seed7_ckks_artifact.npz" if seed == args.artifact_seed else None
        rows.append(run_seed(seed, cfg, artifact))

    with (args.out / "karate_public_benchmark.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    summary = _summary(rows, cfg)
    (args.out / "karate_public_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_table(args.out / "karate_public_table.tex", summary)

    seeds = [r["seed"] for r in rows]
    relu = [100*r["relu_test_accuracy_before_poly_finetune"] for r in rows]
    poly = [100*r["poly_test_accuracy_after_finetune"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.plot(seeds, relu, marker="o", label="ReLU")
    ax.plot(seeds, poly, marker="s", label="Degree-5 polynomial")
    ax.set_xlabel("Split seed")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Zachary Karate Club: public-graph robustness")
    ax.set_ylim(35, 105)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out / "karate_public_accuracy.pdf")
    fig.savefig(args.out / "karate_public_accuracy.png", dpi=180)
    plt.close(fig)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
