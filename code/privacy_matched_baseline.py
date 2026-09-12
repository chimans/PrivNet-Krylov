#!/usr/bin/env python3
"""Exact operation accounting for matched graph-propagation baselines.

FS-ETHE hides the graph shift by padding the cyclic-diagonal representation to
all n public positions and encrypting every diagonal, so the evaluator receives
a graph-independent message shape and follows a fixed loop. The circuit is
intentionally conservative; it is a privacy-matched reference, not a claim to
reproduce an optimized published HE-GNN implementation.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class BaselineCounts:
    nodes: int
    feature_dim: int
    krylov_degree: int
    kgc_input_ciphertexts: int
    kgc_graph_ct_pt_multiplies: int
    kgc_graph_ct_ct_multiplies: int
    kgc_graph_rotations: int
    skhe_input_ciphertexts: int
    skhe_graph_ct_pt_multiplies: int
    skhe_graph_rotations: int
    fsethe_input_feature_ciphertexts: int
    fsethe_input_topology_ciphertexts: int
    fsethe_graph_ct_ct_multiplies: int
    fsethe_graph_rotations: int
    fsethe_graph_additions: int


def fixed_shape_counts(nodes: int, feature_dim: int, krylov_degree: int) -> BaselineCounts:
    n, f, k = int(nodes), int(feature_dim), int(krylov_degree)
    if n < 1 or f < 1 or k < 0:
        raise ValueError("nodes and feature_dim must be positive; degree must be nonnegative")
    recurrence_signals = k * f
    return BaselineCounts(
        nodes=n,
        feature_dim=f,
        krylov_degree=k,
        kgc_input_ciphertexts=(k + 1) * f,
        kgc_graph_ct_pt_multiplies=0,
        kgc_graph_ct_ct_multiplies=0,
        kgc_graph_rotations=0,
        skhe_input_ciphertexts=f,
        skhe_graph_ct_pt_multiplies=recurrence_signals * n,
        skhe_graph_rotations=recurrence_signals * max(n - 1, 0),
        fsethe_input_feature_ciphertexts=f,
        fsethe_input_topology_ciphertexts=n,
        fsethe_graph_ct_ct_multiplies=recurrence_signals * n,
        fsethe_graph_rotations=recurrence_signals * max(n - 1, 0),
        fsethe_graph_additions=recurrence_signals * max(n - 1, 0),
    )


def dense_cyclic_diagonal_matvec(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Evaluate Mx from a full cyclic-diagonal decomposition in plaintext.

    The routine mirrors the packing identity used by the encrypted-topology
    reference circuit and is used by regression tests to check exactness.
    """
    m = np.asarray(matrix, dtype=np.float64)
    x = np.asarray(vector, dtype=np.float64)
    n = m.shape[0]
    if m.shape != (n, n) or x.shape != (n,):
        raise ValueError("expected a square matrix and a matching vector")
    out = np.zeros(n, dtype=np.float64)
    idx = np.arange(n)
    for d in range(n):
        # diag_d[i] = M[i, i+d mod n]; rotated signal supplies x[i+d].
        diag = m[idx, (idx + d) % n]
        out += diag * np.roll(x, -d)
    return out


def _artifact_meta(path: Path) -> Dict[str, int]:
    with np.load(path, allow_pickle=False) as z:
        n = int(z["features"].shape[0])
        f = int(z["features"].shape[1])
        k = int(z["blocks"].shape[0] - 1)
    return {"nodes": n, "feature_dim": f, "krylov_degree": k}


def _write_table(path: Path, rows: list[tuple[str, BaselineCounts]]) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Matched server-side graph-propagation accounting. FS-ETHE is a fixed-shape encrypted-topology reference: all $n$ cyclic diagonals are encrypted and evaluated, including padded zero diagonals, so sparsity is not disclosed through message shape or branch count. Counts exclude the post-bank head shared by all paths.}",
        r"\label{tab:privacy-matched-baseline}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Artifact & KGC graph ct-ct mult. & SK-HE graph ct-pt mult. & FS-ETHE graph ct-ct mult. & FS-ETHE rotations \\",
        r"\midrule",
    ]
    for name, c in rows:
        lines.append(
            f"{name} & {c.kgc_graph_ct_ct_multiplies:,} & {c.skhe_graph_ct_pt_multiplies:,} & "
            f"{c.fsethe_graph_ct_ct_multiplies:,} & {c.fsethe_graph_rotations:,} \\\\".replace(",", "{,}")
        )
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--karate-artifact", type=Path, default=Path("outputs/public_graph/karate_seed7_ckks_artifact.npz"))
    ap.add_argument("--digits-artifact", type=Path, default=Path("outputs/larger_public_graph/digits_knn_seed7_ckks_artifact.npz"))
    ap.add_argument("--out", type=Path, default=Path("outputs/matched_baselines"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[str, BaselineCounts]] = []
    for label, path in [("Karate", args.karate_artifact), ("Digits-10NN", args.digits_artifact)]:
        meta = _artifact_meta(path)
        rows.append((label, fixed_shape_counts(**meta)))

    payload = {
        "baseline": "fixed-shape encrypted-topology HE (FS-ETHE)",
        "security_position": (
            "FS-ETHE encrypts every cyclic diagonal in a public n-diagonal padded representation, so the evaluator does not receive plaintext topology and the program shape does not reveal sparsity. "
            "It is a matched reference circuit, not an optimized reproduction of FicGCN, DESIGN, G-HEMP, or MAPP."
        ),
        "artifacts": {name: asdict(c) for name, c in rows},
    }
    (args.out / "privacy_matched_baseline_counts.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (args.out / "privacy_matched_baseline_counts.csv").open("w", newline="") as f:
        fieldnames = ["artifact"] + list(asdict(rows[0][1]).keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for name, c in rows:
            item = {"artifact": name}
            item.update(asdict(c))
            w.writerow(item)
    _write_table(args.out / "privacy_matched_baseline_table.tex", rows)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
