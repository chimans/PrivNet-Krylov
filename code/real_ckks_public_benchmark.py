#!/usr/bin/env python3
"""Run a genuine TenSEAL CKKS relocation benchmark from a saved model artifact.

The program intentionally keeps measured evidence separate from analytical or
calibrated estimates. KGC encrypts a client-compiled Chebyshev/Krylov bank.
SK-HE encrypts the feature matrix and reconstructs the same bank at the server
with a plaintext graph shift. Both paths then evaluate the identical trained
polynomial network under one CKKS context and on one machine.

This script does not emulate CKKS. If TenSEAL is unavailable, it exits without
creating any ``measured_ckks_*`` result file.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import statistics
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

try:
    import psutil
except ImportError:  # pragma: no cover - optional convenience only
    psutil = None

try:
    import tenseal as ts
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit(
        "TenSEAL is not installed. No measured files were written. Install the "
        "optional environment with `pip install -r requirements-ckks.txt`, then "
        "rerun this program on the machine whose timings you intend to report."
    ) from exc


@dataclass(frozen=True)
class CKKSParameters:
    """Reference CKKS parameter tuple used by the reproducibility harness.

    The tuple is recorded, not asserted to prove a security level by itself.
    Any publication using the timings should validate the concrete parameter
    set with the security guidance of the installed SEAL/TenSEAL release.
    """

    poly_modulus_degree: int = 16384
    coeff_mod_bit_sizes: tuple[int, ...] = (60, 40, 40, 40, 40, 40, 40, 40, 60)
    global_scale_bits: int = 40


def _now_ms() -> float:
    return 1e3 * time.perf_counter()


def _median(xs: Sequence[float]) -> float:
    return float(statistics.median(xs))


def _serialize_bytes(columns: Iterable) -> int:
    return int(sum(len(c.serialize()) for c in columns))


def _flatten(nested: Sequence[Sequence]) -> List:
    return [v for row in nested for v in row]


def _rss_bytes() -> int | None:
    if psutil is None:
        return None
    return int(psutil.Process(os.getpid()).memory_info().rss)


def _peak_delta(before: int | None, after: int | None) -> int | None:
    if before is None or after is None:
        return None
    return max(0, int(after - before))


class _PeakRSSMonitor:
    def __init__(self, interval: float = 0.005):
        self.interval = interval
        self.baseline = _rss_bytes()
        self.peak = self.baseline
        self._stop = threading.Event()
        self._thread = None

    def _sample(self) -> None:
        while not self._stop.wait(self.interval):
            rss = _rss_bytes()
            if rss is not None and (self.peak is None or rss > self.peak):
                self.peak = rss

    def __enter__(self):
        if self.baseline is not None:
            self._thread = threading.Thread(target=self._sample, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        rss = _rss_bytes()
        if rss is not None and (self.peak is None or rss > self.peak):
            self.peak = rss

    @property
    def delta_bytes(self) -> int | None:
        if self.baseline is None or self.peak is None:
            return None
        return max(0, int(self.peak - self.baseline))


def create_context(params: CKKSParameters):
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=params.poly_modulus_degree,
        coeff_mod_bit_sizes=list(params.coeff_mod_bit_sizes),
    )
    context.global_scale = 2 ** params.global_scale_bits
    context.generate_galois_keys()
    context.generate_relin_keys()
    return context


def encrypt_columns(context, matrix: np.ndarray) -> List:
    """Pack the node axis: an [n,f] matrix becomes f encrypted CKKS vectors."""
    matrix = np.asarray(matrix, dtype=np.float64)
    return [ts.ckks_vector(context, matrix[:, j].tolist()) for j in range(matrix.shape[1])]


def encrypt_block_bank(context, blocks: np.ndarray) -> List[List]:
    blocks = np.asarray(blocks, dtype=np.float64)
    return [encrypt_columns(context, blocks[k]) for k in range(blocks.shape[0])]


def encrypted_affine(enc_blocks: Sequence[Sequence], theta: np.ndarray, bias: np.ndarray) -> List:
    theta = np.asarray(theta, dtype=np.float64)
    bias = np.asarray(bias, dtype=np.float64)
    kp1, fdim, hdim = theta.shape
    if len(enc_blocks) != kp1 or any(len(row) != fdim for row in enc_blocks):
        raise ValueError("encrypted bank dimensions do not match theta")
    out: List = []
    for h in range(hdim):
        acc = None
        for k in range(kp1):
            for f in range(fdim):
                term = enc_blocks[k][f] * float(theta[k, f, h])
                acc = term if acc is None else acc + term
        out.append(acc + float(bias[h]))
    return out


def encrypted_poly(columns: Sequence, coeff: Sequence[float]) -> List:
    coeff = [float(x) for x in coeff]
    return [x.polyval(coeff) for x in columns]



def encrypted_folded_poly_head(
    preactivation: Sequence,
    coeff: Sequence[float],
    weight: np.ndarray,
    bias: np.ndarray,
) -> List:
    """Evaluate polynomial activation and linear head without an extra CKKS level."""
    coeff = np.asarray(coeff, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    bias = np.asarray(bias, dtype=np.float64)

    hdim, classes = weight.shape
    if len(preactivation) != hdim:
        raise ValueError("hidden dimension mismatch")

    out: List = []
    for c in range(classes):
        acc = None
        for h in range(hdim):
            term = preactivation[h].polyval((coeff * weight[h, c]).tolist())
            acc = term if acc is None else acc + term
        out.append(acc + float(bias[c]))
    return out


def encrypted_head(hidden: Sequence, weight: np.ndarray, bias: np.ndarray) -> List:
    weight = np.asarray(weight, dtype=np.float64)
    bias = np.asarray(bias, dtype=np.float64)
    hdim, classes = weight.shape
    if len(hidden) != hdim:
        raise ValueError("hidden dimension mismatch")
    out: List = []
    for c in range(classes):
        acc = None
        for h in range(hdim):
            term = hidden[h] * float(weight[h, c])
            acc = term if acc is None else acc + term
        out.append(acc + float(bias[c]))
    return out


def decrypt_logits(columns: Sequence, n: int) -> np.ndarray:
    return np.stack([np.asarray(c.decrypt()[:n], dtype=np.float64) for c in columns], axis=1)


def encrypted_chebyshev_bank(enc_x: Sequence, shift: np.ndarray, degree: int) -> List[List]:
    """Construct T_k(S)X under CKKS while S remains a plaintext server input."""
    shift_t = np.asarray(shift, dtype=np.float64).T.tolist()
    bank: List[List] = [list(enc_x)]
    if degree == 0:
        return bank
    bank.append([v.matmul(shift_t) for v in enc_x])
    for _k in range(1, degree):
        bank.append([(v.matmul(shift_t) * 2.0) - prev for v, prev in zip(bank[-1], bank[-2])])
    return bank


def _run_kgc(context, a: Dict[str, np.ndarray]):
    with _PeakRSSMonitor() as rss_monitor:
        t0 = _now_ms()
        enc_bank = encrypt_block_bank(context, a["blocks"])
        enc_ms = _now_ms() - t0

        upload_bytes = _serialize_bytes(_flatten(enc_bank))

        t1 = _now_ms()
        u = encrypted_affine(enc_bank, a["theta"], a["hidden_bias"])
        h = encrypted_poly(u, a["activation_coeff"])
        z = encrypted_head(h, a["out_weight"], a["out_bias"])
        server_ms = _now_ms() - t1

        t2 = _now_ms()
        logits = decrypt_logits(z, int(a["features"].shape[0]))
        dec_ms = _now_ms() - t2

    return {
        "encrypt_ms": enc_ms,
        "server_ms": server_ms,
        "decrypt_ms": dec_ms,
        "upload_bytes": upload_bytes,
        "output_bytes": _serialize_bytes(z),
        "rss_delta_bytes": rss_monitor.delta_bytes,
        "logits": logits,
    }


def _run_skhe(context, a: Dict[str, np.ndarray], degree: int):
    with _PeakRSSMonitor() as rss_monitor:
        t0 = _now_ms()
        enc_x = encrypt_columns(context, a["features"])
        enc_ms = _now_ms() - t0

        upload_bytes = _serialize_bytes(enc_x)

        t1 = _now_ms()
        bank = encrypted_chebyshev_bank(enc_x, a["shift"], degree)
        u = encrypted_affine(bank, a["theta"], a["hidden_bias"])
        z = encrypted_folded_poly_head(
            u,
            a["activation_coeff"],
            a["out_weight"],
            a["out_bias"],
        )
        server_ms = _now_ms() - t1

        t2 = _now_ms()
        logits = decrypt_logits(z, int(a["features"].shape[0]))
        dec_ms = _now_ms() - t2

    return {
        "encrypt_ms": enc_ms,
        "server_ms": server_ms,
        "decrypt_ms": dec_ms,
        "upload_bytes": upload_bytes,
        "output_bytes": _serialize_bytes(z),
        "rss_delta_bytes": rss_monitor.delta_bytes,
        "logits": logits,
    }

def _metrics(logits: np.ndarray, a: Dict[str, np.ndarray]) -> Dict[str, float]:
    ref = np.asarray(a["plaintext_poly_logits"], dtype=np.float64)
    test = np.asarray(a["test_idx"], dtype=int)
    labels = np.asarray(a["labels"], dtype=int)
    pred = logits.argmax(axis=1)
    ref_pred = ref.argmax(axis=1)
    return {
        "max_abs_logit_error": float(np.max(np.abs(logits - ref))),
        "mean_abs_logit_error": float(np.mean(np.abs(logits - ref))),
        "test_accuracy": float(np.mean(pred[test] == labels[test])),
        "prediction_agreement_with_plaintext": float(np.mean(pred == ref_pred)),
    }


def _csr_to_dense(a: Dict[str, np.ndarray], prefix: str) -> np.ndarray:
    """Reconstruct a saved CSR matrix without requiring SciPy at import time."""
    data = np.asarray(a[f"{prefix}_data"], dtype=np.float64)
    indices = np.asarray(a[f"{prefix}_indices"], dtype=np.int64)
    indptr = np.asarray(a[f"{prefix}_indptr"], dtype=np.int64)
    shape = tuple(int(v) for v in a[f"{prefix}_shape"])
    out = np.zeros(shape, dtype=np.float64)
    for i in range(shape[0]):
        lo, hi = int(indptr[i]), int(indptr[i + 1])
        out[i, indices[lo:hi]] = data[lo:hi]
    return out


def _load_artifact(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    if "shift" not in a and "shift_data" in a:
        a["shift"] = _csr_to_dense(a, "shift")
    if "adjacency" not in a and "adjacency_data" in a:
        a["adjacency"] = _csr_to_dense(a, "adjacency")
    return a


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _hardware() -> Dict[str, str | int | None]:
    vm = psutil.virtual_memory() if psutil is not None else None
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": _cpu_model(),
        "logical_cpu_count": os.cpu_count(),
        "physical_cpu_count": psutil.cpu_count(logical=False) if psutil is not None else None,
        "ram_bytes": int(vm.total) if vm is not None else None,
        "python": sys.version.replace("\n", " "),
        "numpy": np.__version__,
        "tenseal": getattr(ts, "__version__", "unknown"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
    }


def _edge_count(adjacency: np.ndarray) -> int:
    a = np.asarray(adjacency)
    return int(np.count_nonzero(np.triu(a, k=1)))


def _cyclic_diagonal_count(m: np.ndarray, tol: float = 1e-15) -> int:
    m = np.asarray(m)
    n = m.shape[0]
    count = 0
    for d in range(n):
        vals = np.asarray([m[i, (i + d) % n] for i in range(n)])
        if np.any(np.abs(vals) > tol):
            count += 1
    return count


def _write_csv(path: Path, result: Dict):
    rows = []
    for name in ("KGC", "SK-HE"):
        x = result["paths"][name]
        row = {"path": name}
        row.update({k: v for k, v in x.items() if k != "quality"})
        row.update({f"quality_{k}": v for k, v in x["quality"].items()})
        rows.append(row)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _write_tex(path: Path, result: Dict):
    kgc = result["paths"]["KGC"]
    base = result["paths"]["SK-HE"]
    c = result["comparison"]
    label = str(result["dataset"]["name"]).replace("_", r"\_")
    mib = lambda b: b / (1024.0 * 1024.0)
    br = r" \\"
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Measured real-CKKS relocation benchmark on {label}. Values are medians over the declared repetitions on the recorded hardware.}}",
        r"\label{tab:real-ckks}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        "Quantity & KGC & SK-HE baseline" + br,
        r"\midrule",
        f"Input ciphertext vectors & {kgc['input_ciphertext_vectors']} & {base['input_ciphertext_vectors']}" + br,
        f"Graph-dependent HE matvec calls & {kgc['graph_he_matvec_calls']} & {base['graph_he_matvec_calls']}" + br,
        f"Encryption time (ms) & {kgc['encrypt_ms_median']:.2f} & {base['encrypt_ms_median']:.2f}" + br,
        f"Server evaluation time (ms) & {kgc['server_ms_median']:.2f} & {base['server_ms_median']:.2f}" + br,
        f"Decryption time (ms) & {kgc['decrypt_ms_median']:.2f} & {base['decrypt_ms_median']:.2f}" + br,
        f"Upload (MiB) & {mib(kgc['upload_bytes_median']):.3f} & {mib(base['upload_bytes_median']):.3f}" + br,
        f"Max. logit error & {kgc['quality']['max_abs_logit_error']:.3e} & {base['quality']['max_abs_logit_error']:.3e}" + br,
        f"Test accuracy (\\%) & {100*kgc['quality']['test_accuracy']:.2f} & {100*base['quality']['test_accuracy']:.2f}" + br,
        r"\midrule",
        f"Server speedup $T_{{\\rm SKHE}}/T_{{\\rm KGC}}$ & \\multicolumn{{2}}{{c}}{{{c['server_speedup']:.3f}$\\times$}}" + br,
        f"Topology Relocation Gain (TRG) & \\multicolumn{{2}}{{c}}{{{100*c['topology_relocation_gain']:.2f}\\%}}" + br,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    path.write_text("\n".join(lines) + "\n")

def _aggregate_path(runs: List[Dict], artifact: Dict[str, np.ndarray], degree: int, feature_dim: int, name: str) -> Dict:
    q = _metrics(runs[-1]["logits"], artifact)
    rss = [r["rss_delta_bytes"] for r in runs if r["rss_delta_bytes"] is not None]
    return {
        "encrypt_ms_median": _median([r["encrypt_ms"] for r in runs]),
        "server_ms_median": _median([r["server_ms"] for r in runs]),
        "decrypt_ms_median": _median([r["decrypt_ms"] for r in runs]),
        "upload_bytes_median": int(_median([r["upload_bytes"] for r in runs])),
        "output_bytes_median": int(_median([r["output_bytes"] for r in runs])),
        "rss_delta_bytes_median": int(_median(rss)) if rss else None,
        "quality": q,
        "input_ciphertext_vectors": (degree + 1) * feature_dim if name == "KGC" else feature_dim,
        "graph_he_matvec_calls": 0 if name == "KGC" else degree * feature_dim,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifact", type=Path, default=Path("outputs/public_graph/karate_seed7_ckks_artifact.npz"))
    ap.add_argument("--out", type=Path, default=Path("outputs/real_ckks/measured"))
    ap.add_argument("--label", default="Zachary Karate Club")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--security-estimator-json", type=Path)
    args = ap.parse_args()
    if args.security_estimator_json is None:
        raise SystemExit(
            "--security-estimator-json is required for MEASURED_REAL_CKKS output; "
            "provide a real security estimator JSON result for the CKKS parameter set"
        )
    if not args.security_estimator_json.is_file():
        raise SystemExit(f"security estimator JSON not found: {args.security_estimator_json}")
    security_estimator = json.loads(args.security_estimator_json.read_text())
    if args.repeats < 1 or args.warmup < 0:
        raise SystemExit("--repeats must be >= 1 and --warmup must be >= 0")
    if not args.artifact.is_file():
        raise SystemExit(f"artifact not found: {args.artifact}")
    args.out.mkdir(parents=True, exist_ok=True)

    artifact = _load_artifact(args.artifact)
    degree = int(artifact["blocks"].shape[0] - 1)
    feature_dim = int(artifact["features"].shape[1])
    n = int(artifact["features"].shape[0])
    edges = _edge_count(artifact["adjacency"])
    cyclic_diags = _cyclic_diagonal_count(artifact["shift"])
    params = CKKSParameters()

    t0 = _now_ms()
    context = create_context(params)
    context_setup_ms = _now_ms() - t0

    for _ in range(args.warmup):
        _run_kgc(context, artifact)
        _run_skhe(context, artifact, degree)

    raw: Dict[str, List[Dict]] = {"KGC": [], "SK-HE": []}
    for _ in range(args.repeats):
        raw["KGC"].append(_run_kgc(context, artifact))
        raw["SK-HE"].append(_run_skhe(context, artifact, degree))

    paths = {
        name: _aggregate_path(runs, artifact, degree, feature_dim, name)
        for name, runs in raw.items()
    }

    kgc_s = paths["KGC"]["server_ms_median"]
    base_s = paths["SK-HE"]["server_ms_median"]
    speedup = float(base_s / kgc_s) if kgc_s > 0 else math.inf
    trg = float(1.0 - kgc_s / base_s) if base_s > 0 else float("nan")

    t_compile0 = _now_ms()
    s = np.asarray(artifact["shift"], dtype=np.float64)
    x = np.asarray(artifact["features"], dtype=np.float64)
    bank = [x]
    if degree >= 1:
        bank.append(s @ x)
    for _k in range(1, degree):
        bank.append(2.0 * (s @ bank[-1]) - bank[-2])
    compile_ms = _now_ms() - t_compile0

    numerator = compile_ms + paths["KGC"]["encrypt_ms_median"] - paths["SK-HE"]["encrypt_ms_median"]
    denominator = base_s - kgc_s
    q_star = None if denominator <= 0 else max(1, int(math.ceil(max(numerator, 0.0) / denominator)))

    result = {
        "evidence_status": "MEASURED_REAL_CKKS",
        "benchmark": "PrivNet-Krylov matched TenSEAL CKKS relocation benchmark",
        "dataset": {"name": args.label, "nodes": n, "edges": edges, "feature_dim": feature_dim},
        "artifact": str(args.artifact.resolve()),
        "hardware": _hardware(),
        "ckks_parameters": asdict(params),
        "security_estimator": security_estimator,
        "context_and_key_setup_ms": context_setup_ms,
        "client_plaintext_compile_ms": compile_ms,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "paths": paths,
        "comparison": {
            "server_speedup": speedup,
            "topology_relocation_gain": trg,
            "upload_ratio_kgc_to_skhe": float(paths["KGC"]["upload_bytes_median"] / paths["SK-HE"]["upload_bytes_median"]),
            "amortized_break_even_queries": q_star,
            "cyclic_diagonal_count": cyclic_diags,
        },
        "scope_note": (
            "SK-HE uses plaintext topology and is a matched computation-placement ablation, "
            "not a privacy-equivalent competitor. Privacy-equivalent graph recurrence is "
            "handled separately by the fixed-shape encrypted-topology operation-count baseline."
        ),
    }

    out_json = args.out / "measured_ckks_results.json"
    out_json.write_text(json.dumps(result, indent=2) + "\n")
    _write_csv(args.out / "measured_ckks_results.csv", result)
    _write_tex(args.out / "measured_ckks_table.tex", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
