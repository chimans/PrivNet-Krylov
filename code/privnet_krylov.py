#!/usr/bin/env python3
"""Reference implementation for the PrivNet-Krylov experiments.

The module covers client-side Krylov/Chebyshev compilation, the matching
plaintext spectral model, decision-margin checks, bounded CKKS-style numerical
perturbations, and the expander stress test. The perturbation model is numerical
only; direct CKKS execution is implemented separately in `tenseal_backend.py`.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from numpy.polynomial import Chebyshev, Polynomial


@dataclass
class ExperimentConfig:
    seed: int = 7
    num_classes: int = 3
    nodes_per_class: int = 120
    feature_dim: int = 8
    hidden_dim: int = 20
    krylov_degree: int = 4
    train_fraction: float = 0.55
    val_fraction: float = 0.15
    p_in: float = 0.085
    p_out: float = 0.012
    feature_noise: float = 1.20
    epochs: int = 450
    lr: float = 0.025
    weight_decay: float = 1e-4
    activation_degrees: Tuple[int, ...] = (2, 3, 5, 7, 9)
    poly_grid_size: int = 12001
    pre_ckks_error: float = 1.5e-3
    out_ckks_error: float = 8.0e-4
    expander_n: int = 256
    expander_degree: int = 6
    expander_seed: int = 11


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def normalized_adjacency(a: np.ndarray, self_loops: bool = True) -> np.ndarray:
    """Return D^{-1/2}(A+I)D^{-1/2}."""
    a = np.asarray(a, dtype=np.float64)
    if self_loops:
        a = a + np.eye(a.shape[0], dtype=np.float64)
    d = a.sum(axis=1)
    inv_sqrt = np.zeros_like(d)
    mask = d > 0
    inv_sqrt[mask] = 1.0 / np.sqrt(d[mask])
    return inv_sqrt[:, None] * a * inv_sqrt[None, :]


def build_sbm(cfg: ExperimentConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a class-structured graph and class-informative noisy features."""
    sizes = [cfg.nodes_per_class] * cfg.num_classes
    probs = np.full((cfg.num_classes, cfg.num_classes), cfg.p_out, dtype=float)
    np.fill_diagonal(probs, cfg.p_in)
    g = nx.stochastic_block_model(sizes, probs, seed=cfg.seed)
    a = nx.to_numpy_array(g, dtype=np.float64)

    labels = np.repeat(np.arange(cfg.num_classes), cfg.nodes_per_class)
    rng = np.random.default_rng(cfg.seed)
    class_means = rng.normal(0.0, 1.0, size=(cfg.num_classes, cfg.feature_dim))
    # Separate means modestly so the graph still matters.
    class_means *= 1.15
    x = class_means[labels] + rng.normal(0.0, cfg.feature_noise, size=(len(labels), cfg.feature_dim))
    x = (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1e-12)
    return a, x.astype(np.float64), labels.astype(np.int64)


def split_indices(n: int, train_frac: float, val_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]


def chebyshev_krylov_compile(s: np.ndarray, x: np.ndarray, degree: int) -> List[np.ndarray]:
    """Client-side topology-private compilation B_k = T_k(S) X.

    The recurrence is T_0(S)X=X, T_1(S)X=SX,
    T_{k+1}(S)X=2S T_k(S)X-T_{k-1}(S)X.
    """
    if degree < 0:
        raise ValueError("degree must be nonnegative")
    blocks = [x.copy()]
    if degree == 0:
        return blocks
    blocks.append(s @ x)
    for _k in range(1, degree):
        blocks.append(2.0 * (s @ blocks[-1]) - blocks[-2])
    return blocks


def direct_chebyshev_apply(s: np.ndarray, x: np.ndarray, coeffs: Sequence[float]) -> np.ndarray:
    blocks = chebyshev_krylov_compile(s, x, len(coeffs) - 1)
    y = np.zeros_like(x, dtype=np.float64)
    for c, b in zip(coeffs, blocks):
        y += float(c) * b
    return y


class CompiledSpectralNet(torch.nn.Module):
    """A polynomial spectral layer expressed entirely through precompiled blocks."""
    def __init__(self, feature_dim: int, hidden_dim: int, num_classes: int, degree: int):
        super().__init__()
        self.degree = degree
        self.theta = torch.nn.Parameter(torch.empty(degree + 1, feature_dim, hidden_dim))
        self.bias = torch.nn.Parameter(torch.zeros(hidden_dim))
        self.out = torch.nn.Linear(hidden_dim, num_classes)
        torch.nn.init.xavier_uniform_(self.theta)
        torch.nn.init.xavier_uniform_(self.out.weight)

    def preactivation(self, blocks: torch.Tensor) -> torch.Tensor:
        # blocks: [K+1, N, F]; theta: [K+1, F, H]
        u = torch.einsum("knf,kfh->nh", blocks, self.theta)
        return u + self.bias

    def forward(self, blocks: torch.Tensor) -> torch.Tensor:
        u = self.preactivation(blocks)
        return self.out(torch.relu(u))


def accuracy(logits: torch.Tensor, labels: torch.Tensor, idx: np.ndarray) -> float:
    pred = logits[idx].argmax(dim=1)
    return float((pred == labels[idx]).float().mean().item())


def train_model(cfg: ExperimentConfig, blocks_np: Sequence[np.ndarray], labels_np: np.ndarray,
                train_idx: np.ndarray, val_idx: np.ndarray) -> CompiledSpectralNet:
    blocks = torch.tensor(np.stack(blocks_np), dtype=torch.float32)
    labels = torch.tensor(labels_np, dtype=torch.long)
    model = CompiledSpectralNet(cfg.feature_dim, cfg.hidden_dim, cfg.num_classes, cfg.krylov_degree)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_state = None
    best_val = -1.0
    patience = 80
    bad = 0
    for _epoch in range(cfg.epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        logits = model(blocks)
        loss = torch.nn.functional.cross_entropy(logits[train_idx], labels[train_idx])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_acc = accuracy(model(blocks), labels, val_idx)
        if val_acc > best_val + 1e-8:
            best_val = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def fit_relu_polynomial(degree: int, bound: float, grid_size: int) -> Tuple[np.ndarray, float]:
    """Fit a Chebyshev least-squares approximation on a declared interval.

    The caller chooses the interval.  Experimental scripts calibrate it without
    consulting held-out test labels or test preactivations.
    """
    x = np.linspace(-bound, bound, grid_size)
    y = np.maximum(x, 0.0)
    cheb = Chebyshev.fit(x, y, deg=degree, domain=[-bound, bound])
    power: Polynomial = cheb.convert(kind=Polynomial)
    coeff = np.asarray(power.coef, dtype=np.float64)
    err = float(np.max(np.abs(power(x) - y)))
    return coeff, err


def balanced_power(x: np.ndarray, exponent: int, cache: Dict[int, np.ndarray]) -> np.ndarray:
    if exponent in cache:
        return cache[exponent]
    if exponent % 2 == 0:
        half = balanced_power(x, exponent // 2, cache)
        cache[exponent] = half * half
    else:
        a = exponent // 2
        left = balanced_power(x, a, cache)
        right = balanced_power(x, exponent - a, cache)
        cache[exponent] = left * right
    return cache[exponent]


def eval_poly_balanced(x: np.ndarray, coeff: Sequence[float]) -> np.ndarray:
    """Evaluate a polynomial from a power cache suitable for a balanced HE circuit."""
    coeff = np.asarray(coeff, dtype=np.float64)
    out = np.zeros_like(x, dtype=np.float64) + coeff[0]
    cache: Dict[int, np.ndarray] = {0: np.ones_like(x), 1: x}
    for j in range(1, len(coeff)):
        if coeff[j] != 0.0:
            out += coeff[j] * balanced_power(x, j, cache)
    return out


def multiplicative_depth_for_degree(degree: int) -> int:
    if degree <= 1:
        return 0
    return int(math.ceil(math.log2(degree)))


def column_l1_operator_norm(w: np.ndarray) -> float:
    """max output-column L1 norm: max_j sum_i |W_{ij}|."""
    return float(np.max(np.sum(np.abs(w), axis=0)))


def top2_margins(logits: np.ndarray) -> np.ndarray:
    part = np.partition(logits, kth=logits.shape[1] - 2, axis=1)
    return part[:, -1] - part[:, -2]


def cyclic_diagonal_count(m: np.ndarray, tol: float = 0.0) -> int:
    """Number of nonzero cyclic diagonals in a square matrix.

    This is an operation-count proxy for diagonal-method encrypted matvecs,
    not a hardware/runtime benchmark.
    """
    n = m.shape[0]
    count = 0
    for shift in range(n):
        vals = m[np.arange(n), (np.arange(n) + shift) % n]
        if np.any(np.abs(vals) > tol):
            count += 1
    return count


def run_activation_experiment(cfg: ExperimentConfig, model: CompiledSpectralNet,
                              blocks_np: Sequence[np.ndarray], labels_np: np.ndarray,
                              test_idx: np.ndarray, calibration_idx: np.ndarray) -> Tuple[List[Dict], Dict]:
    model.eval()
    blocks = torch.tensor(np.stack(blocks_np), dtype=torch.float32)
    labels = torch.tensor(labels_np, dtype=torch.long)
    with torch.no_grad():
        u_t = model.preactivation(blocks)
        exact_logits_t = model.out(torch.relu(u_t))

    u = u_t.cpu().numpy().astype(np.float64)
    exact_logits = exact_logits_t.cpu().numpy().astype(np.float64)
    w_out = model.out.weight.detach().cpu().numpy().T.astype(np.float64)  # [H,C]
    b_out = model.out.bias.detach().cpu().numpy().astype(np.float64)
    out_norm = column_l1_operator_norm(w_out)

    # Calibration uses train/validation nodes only; held-out preactivations do not
    # determine the approximation interval.  Certificates are issued only for
    # nodes whose hidden coordinates remain inside that interval.
    bound = float(np.max(np.abs(u[calibration_idx])) * 1.01 + 1e-12)
    inside_interval = np.max(np.abs(u), axis=1) <= bound
    margins = top2_margins(exact_logits)
    exact_pred = exact_logits.argmax(axis=1)
    labels = labels.cpu().numpy()
    rng = np.random.default_rng(cfg.seed + 101)

    rows: List[Dict] = []
    for degree in cfg.activation_degrees:
        coeff, eps_act = fit_relu_polynomial(degree, bound, cfg.poly_grid_size)
        # Adversarially bounded synthetic CKKS-style perturbations: uniform in [-eps, eps].
        e_pre = rng.uniform(-cfg.pre_ckks_error, cfg.pre_ckks_error, size=u.shape)
        h_hat = eval_poly_balanced(u + e_pre, coeff)
        logits_hat = h_hat @ w_out + b_out
        e_out = rng.uniform(-cfg.out_ckks_error, cfg.out_ckks_error, size=logits_hat.shape)
        logits_hat = logits_hat + e_out

        poly_logits = eval_poly_balanced(u, coeff) @ w_out + b_out
        bound_logit = out_norm * (eps_act + cfg.pre_ckks_error) + cfg.out_ckks_error
        certified = inside_interval & (margins > 2.0 * bound_logit)
        measured_err = float(np.max(np.abs(exact_logits - logits_hat)))
        cert_is_valid = bool(np.all(exact_pred[certified] == logits_hat.argmax(axis=1)[certified])) if np.any(certified) else True
        row = {
            "degree": int(degree),
            "activation_uniform_error": float(eps_act),
            "multiplicative_depth": multiplicative_depth_for_degree(int(degree)),
            "test_accuracy_polynomial": float(np.mean(poly_logits[test_idx].argmax(axis=1) == labels[test_idx])),
            "test_accuracy_simulated_ckks": float(np.mean(logits_hat[test_idx].argmax(axis=1) == labels[test_idx])),
            "prediction_agreement_with_relu": float(np.mean(logits_hat.argmax(axis=1) == exact_pred)),
            "certified_fraction_all_nodes": float(np.mean(certified)),
            "certified_fraction_test_nodes": float(np.mean(certified[test_idx])),
            "theoretical_logit_error_bound": float(bound_logit),
            "measured_max_logit_error": measured_err,
            "all_certified_nodes_invariant": cert_is_valid,
            "power_coefficients": [float(c) for c in coeff],
        }
        rows.append(row)

    summary = {
        "preactivation_bound": bound,
        "output_column_l1_norm": out_norm,
        "min_margin": float(margins.min()),
        "median_margin": float(np.median(margins)),
        "mean_margin": float(margins.mean()),
        "exact_test_accuracy_relu": float(np.mean(exact_pred[test_idx] == labels[test_idx])),
        "calibration_interval_coverage_all_nodes": float(np.mean(inside_interval)),
        "calibration_interval_coverage_test_nodes": float(np.mean(inside_interval[test_idx])),
        "calibration_policy": "train and validation preactivations only",
    }
    return rows, summary


def run_expander_experiment(cfg: ExperimentConfig) -> Tuple[List[Dict], Dict]:
    """Measure contraction on a random regular graph and compare with the Ramanujan bound."""
    g = nx.random_regular_graph(cfg.expander_degree, cfg.expander_n, seed=cfg.expander_seed)
    a = nx.to_numpy_array(g, dtype=np.float64)
    p = a / float(cfg.expander_degree)
    eig = np.linalg.eigvalsh(p)
    # Exclude the eigenvalue 1; take max abs among the rest.
    idx_one = int(np.argmax(eig))
    nontriv = np.delete(eig, idx_one)
    rho = float(np.max(np.abs(nontriv)))
    ramanujan_ratio = float(2.0 * math.sqrt(cfg.expander_degree - 1) / cfg.expander_degree)

    rng = np.random.default_rng(cfg.seed + 333)
    x = rng.normal(size=(cfg.expander_n, 4))
    x = x - x.mean(axis=0, keepdims=True)
    norm0 = np.linalg.norm(x, ord="fro")
    y = x.copy()
    rows: List[Dict] = []
    for k in range(1, 13):
        y = p @ y
        rel = float(np.linalg.norm(y, ord="fro") / norm0)
        rows.append({
            "K": k,
            "measured_relative_mean_zero_norm": rel,
            "rho_power_bound": float(rho ** k),
            "ramanujan_power_bound": float(ramanujan_ratio ** k),
        })
    summary = {
        "n": cfg.expander_n,
        "degree": cfg.expander_degree,
        "measured_nontrivial_spectral_radius_ratio": rho,
        "ramanujan_ratio": ramanujan_ratio,
        "satisfies_ramanujan_eigenvalue_inequality_empirically": bool(rho <= ramanujan_ratio + 1e-10),
    }
    return rows, summary


def save_csv(path: Path, rows: Sequence[Dict]) -> None:
    if not rows:
        return
    keys = [k for k in rows[0].keys() if k != "power_coefficients"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in keys})


def plot_activation(rows: Sequence[Dict], path: Path) -> None:
    deg = [r["degree"] for r in rows]
    err = [r["activation_uniform_error"] for r in rows]
    cert = [r["certified_fraction_test_nodes"] for r in rows]

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.plot(deg, err, marker="o", label="Uniform ReLU approximation error")
    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel("Approximation error")
    ax.grid(True, alpha=0.25)
    ax2 = ax.twinx()
    ax2.plot(deg, cert, marker="s", linestyle="--", label="Certified test fraction")
    ax2.set_ylabel("Certified fraction")
    ax2.set_ylim(0, 1.04)
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="center right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_mixing(rows: Sequence[Dict], path: Path) -> None:
    k = [r["K"] for r in rows]
    measured = [r["measured_relative_mean_zero_norm"] for r in rows]
    rho = [r["rho_power_bound"] for r in rows]
    ram = [r["ramanujan_power_bound"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.semilogy(k, measured, marker="o", label="Measured contraction")
    ax.semilogy(k, rho, linestyle="--", label=r"Measured $\rho^K$ bound")
    ax.semilogy(k, ram, linestyle=":", label="Ramanujan envelope")
    ax.set_xlabel("Propagation / compilation degree K")
    ax.set_ylabel("Relative mean-zero Frobenius norm")
    ax.grid(True, alpha=0.25, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(out_dir: Path, cfg: ExperimentConfig) -> Dict:
    set_seed(cfg.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir.parent / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    a, x, labels = build_sbm(cfg)
    s = normalized_adjacency(a, self_loops=True)
    blocks = chebyshev_krylov_compile(s, x, cfg.krylov_degree)
    train_idx, val_idx, test_idx = split_indices(len(labels), cfg.train_fraction, cfg.val_fraction, cfg.seed)
    model = train_model(cfg, blocks, labels, train_idx, val_idx)

    blocks_t = torch.tensor(np.stack(blocks), dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.long)
    with torch.no_grad():
        logits = model(blocks_t)
    accs = {
        "train": accuracy(logits, labels_t, train_idx),
        "validation": accuracy(logits, labels_t, val_idx),
        "test": accuracy(logits, labels_t, test_idx),
    }

    calibration_idx = np.concatenate([train_idx, val_idx])
    activation_rows, activation_summary = run_activation_experiment(
        cfg, model, blocks, labels, test_idx, calibration_idx
    )
    mixing_rows, mixing_summary = run_expander_experiment(cfg)

    # Verify exact compilation against a direct polynomial spectral evaluation for random scalar filter coefficients.
    rng = np.random.default_rng(cfg.seed + 888)
    scalar_coeff = rng.normal(size=cfg.krylov_degree + 1)
    compiled_scalar = sum(c * b for c, b in zip(scalar_coeff, blocks))
    direct_scalar = direct_chebyshev_apply(s, x, scalar_coeff)
    compilation_equivalence_error = float(np.max(np.abs(compiled_scalar - direct_scalar)))

    n = a.shape[0]
    adjacency_plus_i = a + np.eye(n)
    cyclic_diags = cyclic_diagonal_count(adjacency_plus_i)
    operation_counts = {
        "n_nodes": int(n),
        "n_edges": int(np.count_nonzero(np.triu(a, 1))),
        "average_degree": float(a.sum() / n),
        "nonzero_cyclic_diagonals_of_A_plus_I": int(cyclic_diags),
        "diagonal_method_graph_dependent_rotations_proxy": int(max(cyclic_diags - 1, 0)),
        "privnet_krylov_server_graph_dependent_rotations": 0,
        "uploaded_krylov_blocks": int(cfg.krylov_degree + 1),
        "note": "Operation-count illustration only; not a wall-clock benchmark.",
    }

    results = {
        "configuration": asdict(cfg),
        "dataset": {
            "type": "synthetic stochastic block model",
            "nodes": int(n),
            "classes": cfg.num_classes,
            "features": cfg.feature_dim,
            "train_nodes": int(len(train_idx)),
            "validation_nodes": int(len(val_idx)),
            "test_nodes": int(len(test_idx)),
        },
        "plaintext_relu_accuracy": accs,
        "exact_compilation_equivalence_max_abs_error": compilation_equivalence_error,
        "activation_summary": activation_summary,
        "activation_tradeoff": activation_rows,
        "expander_summary": mixing_summary,
        "expander_mixing": mixing_rows,
        "operation_counts": operation_counts,
        "disclaimer": "CKKS values are bounded numerical perturbation simulations, not measured cryptographic runtime/security results.",
    }

    with (out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    save_csv(out_dir / "activation_tradeoff.csv", activation_rows)
    save_csv(out_dir / "mixing.csv", mixing_rows)
    with (out_dir / "operation_counts.json").open("w", encoding="utf-8") as f:
        json.dump(operation_counts, f, indent=2)

    plot_activation(activation_rows, figures_dir / "activation_tradeoff")
    plot_mixing(mixing_rows, figures_dir / "mixing_bound")

    # Save model weights for reproducibility/reference backend.
    torch.save(model.state_dict(), out_dir / "compiled_spectral_model.pt")
    np.savez_compressed(
        out_dir / "compiled_blocks_and_data.npz",
        adjacency=a,
        normalized_adjacency=s,
        features=x,
        labels=labels,
        blocks=np.stack(blocks),
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )

    # Human-readable representative console output.
    best_cert = max(activation_rows, key=lambda r: (r["certified_fraction_test_nodes"], -r["multiplicative_depth"]))
    text = (
        "PrivNet-Krylov reproducible simulator\n"
        f"nodes={n}, edges={operation_counts['n_edges']}, K={cfg.krylov_degree}\n"
        f"plaintext ReLU accuracy: train={accs['train']:.4f}, val={accs['validation']:.4f}, test={accs['test']:.4f}\n"
        f"exact compilation max |error|={compilation_equivalence_error:.3e}\n"
        f"preactivation interval=[-{activation_summary['preactivation_bound']:.4f}, {activation_summary['preactivation_bound']:.4f}]\n"
        f"best certification row: degree={best_cert['degree']}, depth={best_cert['multiplicative_depth']}, "
        f"eps_act={best_cert['activation_uniform_error']:.4f}, test_acc_sim={best_cert['test_accuracy_simulated_ckks']:.4f}, "
        f"certified_test_fraction={best_cert['certified_fraction_test_nodes']:.4f}\n"
        f"expander rho={mixing_summary['measured_nontrivial_spectral_radius_ratio']:.6f}, "
        f"Ramanujan envelope={mixing_summary['ramanujan_ratio']:.6f}\n"
        f"server graph-dependent rotations: compiled={operation_counts['privnet_krylov_server_graph_dependent_rotations']} vs "
        f"diagonal-proxy={operation_counts['diagonal_method_graph_dependent_rotations_proxy']}\n"
        "NOTE: CKKS perturbations are simulated; see tenseal_backend.py for a real-CKKS adapter.\n"
    )
    (out_dir / "console_output.txt").write_text(text, encoding="utf-8")
    print(text)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PrivNet-Krylov reference experiments")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "outputs")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    cfg = ExperimentConfig(seed=args.seed)
    run(args.out, cfg)


if __name__ == "__main__":
    main()
