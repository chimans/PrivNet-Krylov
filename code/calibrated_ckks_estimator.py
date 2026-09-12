#!/usr/bin/env python3
"""Calibrated CKKS sensitivity model for the PrivNet-Krylov artifact.

Outputs from this script are modeled estimates, not hardware measurements. The
operation-time anchors are taken from published TenSEAL CKKS microbenchmarks on
an AWS c4.2xlarge host and rescaled to the manuscript's N=16384 reference
context. Hardware measurements belong to `real_ckks_public_benchmark.py` and
should be generated on the machine whose performance is being reported.
"""
from __future__ import annotations
import csv, json, math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "outputs" / "real_ckks"
OUT.mkdir(parents=True, exist_ok=True)

# Published TenSEAL microbenchmarks, milliseconds, shape [256], N=8192,
# coefficient modulus=200 bits. Benaissa et al. 2021, complete benchmarks.
BASE = {
    "add_ms": 0.08,
    "ctct_mul_ms": 4.45,
    "add_plain_ms": 0.80,
    "mul_plain_ms": 1.75,
    "dot_ms": 20.15,
    "polyval_quadratic_ms": 10.55,
    "keygen_ms": 940.0,
}

N0 = 8192
N = 16384
PRIMES0 = 4       # ~200 bits / ~50-bit average limb, calibration abstraction
PRIMES = 9        # manuscript chain: (60,40,40,40,40,40,40,40,60)
K = 2
F = 34
H = 12
C = 2
DIAGONALS = 34

# Heavy NTT/key-switch dominated operations: O(N log N) times RNS-limb count.
ring_scale = (N * math.log2(N)) / (N0 * math.log2(N0))
rns_scale = PRIMES / PRIMES0
heavy_scale = ring_scale * rns_scale
# Linear add operations do not pay the full NTT factor.
linear_scale = (N / N0) * rns_scale

mul_plain_ms = BASE["mul_plain_ms"] * heavy_scale
add_ms = BASE["add_ms"] * linear_scale
add_plain_ms = BASE["add_plain_ms"] * linear_scale
ctct_mul_ms = BASE["ctct_mul_ms"] * heavy_scale

# Infer a rotation/key-switch anchor from TenSEAL's encrypted dot benchmark.
# For a 256-slot reduction, approximate dot ~= one ct-ct multiply + 8 rotations
# + 8 additions. This assumption is kept explicit in the model.
log_reduction = 8
rotation_base_ms = max(
    (BASE["dot_ms"] - BASE["ctct_mul_ms"] - log_reduction * BASE["add_ms"]) / log_reduction,
    0.0,
)
rotation_ms = rotation_base_ms * heavy_scale

# Degree-5 polynomial: use 2.2x the published quadratic polyval cost after
# target-context scaling. This approximates a balanced/power-basis evaluation
# and is explicitly not a TenSEAL measurement.
poly5_ms = BASE["polyval_quadratic_ms"] * heavy_scale * 2.2

# Shared post-bank head operation counts for the Karate artifact.
plain_mults = (K + 1) * F * H + H * C
cipher_adds = H * ((K + 1) * F - 1) + C * (H - 1)
plain_adds = H + C
activations = H

shared_ms = (
    plain_mults * mul_plain_ms
    + cipher_adds * add_ms
    + plain_adds * add_plain_ms
    + activations * poly5_ms
)

# SK-HE Chebyshev graph recurrence. A 34-diagonal plaintext-matrix multiply is
# approximated by D ct-plain multiplies + (D-1) rotations + (D-1) additions.
matvec_ms = (
    DIAGONALS * mul_plain_ms
    + (DIAGONALS - 1) * rotation_ms
    + (DIAGONALS - 1) * add_ms
)
graph_matvecs = K * F
graph_recurrence_ms = graph_matvecs * matvec_ms
# T_2 recurrence includes 2*S*B1 - B0 for F columns.
recurrence_linear_ms = F * (mul_plain_ms + add_ms)
sk_graph_ms = graph_recurrence_ms + recurrence_linear_ms

kgc_server_ms = shared_ms
skhe_server_ms = shared_ms + sk_graph_ms
trg = 1.0 - kgc_server_ms / skhe_server_ms
absolute_saving_ms = skhe_server_ms - kgc_server_ms
speedup = skhe_server_ms / kgc_server_ms

# Client encryption/decryption estimates. There is no public TenSEAL encryption
# microbenchmark in the cited table, so these are tied to the target plain-mul
# cost and intentionally receive a wider uncertainty band.
encrypt_per_ct_ms = 1.15 * mul_plain_ms
decrypt_per_ct_ms = 0.70 * mul_plain_ms
kgc_input_ct = (K + 1) * F
skhe_input_ct = F
kgc_encrypt_ms = kgc_input_ct * encrypt_per_ct_ms
skhe_encrypt_ms = skhe_input_ct * encrypt_per_ct_ms
output_ct = C
output_decrypt_ms = output_ct * decrypt_per_ct_ms
context_key_ms = BASE["keygen_ms"] * heavy_scale

# Ciphertext-volume proxy: raw RNS payload for a two-component CKKS ciphertext
# at the initial level, excluding serialization framing and evaluation keys.
bytes_per_full_ct = 2 * N * PRIMES * 8
mib_per_full_ct = bytes_per_full_ct / (1024**2)
kgc_upload_mib = kgc_input_ct * mib_per_full_ct
skhe_upload_mib = skhe_input_ct * mib_per_full_ct
# Output assumed to retain four RNS primes after the reference degree-5 path.
OUT_PRIMES = 4
bytes_per_out_ct = 2 * N * OUT_PRIMES * 8
output_mib = output_ct * bytes_per_out_ct / (1024**2)

# Peak ciphertext working-set estimates, excluding context/evaluation keys and
# Python/runtime memory. KGC holds the compiled bank plus head/polynomial
# temporaries; SK-HE additionally holds recurrence and matmul temporaries.
kgc_working_mib = 1.58 * kgc_upload_mib
skhe_working_mib = 1.38 * (kgc_upload_mib + skhe_upload_mib)

# Network sensitivity, upload only, using the raw-RNS payload proxy.
def network_seconds(mib: float, mbps: float) -> float:
    return (mib * 1024 * 1024 * 8) / (mbps * 1_000_000)

results = {
    "evidence_type": "calibrated_representative_estimate_not_measurement",
    "calibration_host": {
        "platform": "AWS EC2 c4.2xlarge",
        "cpu": "Intel Xeon E5-2666 v3, 8 vCPU, 2.9 GHz",
        "memory_gib": 15,
        "os": "Ubuntu Server 20.04",
        "python": "3.8 (TenSEAL paper calibration environment)",
        "source": "Benaissa et al., TenSEAL, arXiv:2104.03152 / ICLR 2021 DPML workshop",
    },
    "target_ckks_context": {
        "poly_modulus_degree": N,
        "coeff_mod_bit_sizes": [60,40,40,40,40,40,40,40,60],
        "coeff_mod_total_bits": 400,
        "global_scale_bits": 40,
        "slot_capacity": N // 2,
    },
    "calibration": {
        "ring_scale": ring_scale,
        "rns_scale": rns_scale,
        "heavy_operation_scale": heavy_scale,
        "linear_operation_scale": linear_scale,
        "target_mul_plain_ms": mul_plain_ms,
        "target_rotation_ms": rotation_ms,
        "target_add_ms": add_ms,
        "target_degree5_polyval_ms": poly5_ms,
    },
    "karate_matched_pair": {
        "K": K, "features": F, "hidden": H, "classes": C,
        "nonzero_cyclic_diagonals": DIAGONALS,
        "kgc_input_ciphertexts": kgc_input_ct,
        "skhe_input_ciphertexts": skhe_input_ct,
        "skhe_graph_he_matvecs": graph_matvecs,
        "kgc_server_latency_s": kgc_server_ms / 1000,
        "skhe_server_latency_s": skhe_server_ms / 1000,
        "absolute_server_time_saved_s": absolute_saving_ms / 1000,
        "trg_fraction": trg,
        "trg_percent": 100 * trg,
        "server_speedup_x": speedup,
        "kgc_input_encryption_s": kgc_encrypt_ms / 1000,
        "skhe_input_encryption_s": skhe_encrypt_ms / 1000,
        "output_decryption_s": output_decrypt_ms / 1000,
        "one_time_context_key_setup_s": context_key_ms / 1000,
        "kgc_upload_payload_mib": kgc_upload_mib,
        "skhe_upload_payload_mib": skhe_upload_mib,
        "encrypted_output_payload_mib": output_mib,
        "kgc_peak_ciphertext_working_set_mib": kgc_working_mib,
        "skhe_peak_ciphertext_working_set_mib": skhe_working_mib,
        "network_upload_seconds_1gbps": {
            "kgc": network_seconds(kgc_upload_mib, 1000),
            "skhe": network_seconds(skhe_upload_mib, 1000),
        },
        "network_upload_seconds_100mbps": {
            "kgc": network_seconds(kgc_upload_mib, 100),
            "skhe": network_seconds(skhe_upload_mib, 100),
        },
        "uncertainty_guidance": {
            "server_latency_relative": "±25% scenario band",
            "encryption_relative": "±35% scenario band",
            "ciphertext_working_set_relative": "±30% scenario band",
            "payload_formula": "raw RNS payload proxy; actual TenSEAL serialization may differ",
        },
    },
}

# Structured scale envelope. Counts are analytical, not runtime measurements.
datasets = [
    ("Karate", 34, 78, 34, "observed public baseline"),
    ("Cora", 2708, 5429, 1433, "Planetoid citation graph"),
    ("CiteSeer", 3327, 4732, 3703, "Planetoid citation graph"),
    ("PubMed", 19717, 44338, 500, "Planetoid citation graph"),
    ("ogbn-arxiv", 169343, 1166243, 128, "OGB citation graph"),
    ("ogbn-products", 2449029, 61859140, 100, "OGB co-purchase graph"),
]
slots = N // 2
compact_f = 32
scale_rows = []
for name, n, e, f_raw, note in datasets:
    batches = math.ceil(n / slots)
    raw_kgc_ct = batches * (K + 1) * f_raw
    compact_kgc_ct = batches * (K + 1) * compact_f
    compact_skhe_input_ct = batches * compact_f
    # Block-sharded SK-HE graph matvec count depends on nonempty S_rs blocks.
    # Lower envelope assumes one graph block per row-shard; upper is B^2.
    skhe_matvec_low = K * compact_f * batches
    skhe_matvec_high = K * compact_f * batches * batches
    scale_rows.append({
        "dataset": name, "nodes": n, "edges": e, "raw_features": f_raw,
        "slots": slots, "node_batches": batches,
        "kgc_ciphertexts_raw_features": raw_kgc_ct,
        "kgc_upload_gib_raw_features": raw_kgc_ct * mib_per_full_ct / 1024,
        "deployment_latent_width": compact_f,
        "kgc_ciphertexts_f32": compact_kgc_ct,
        "kgc_upload_gib_f32": compact_kgc_ct * mib_per_full_ct / 1024,
        "skhe_input_ciphertexts_f32": compact_skhe_input_ct,
        "skhe_graph_matvec_lower_f32": skhe_matvec_low,
        "skhe_graph_matvec_upper_f32": skhe_matvec_high,
        "note": note,
    })
results["scale_envelope"] = scale_rows

(OUT / "representative_ckks_estimates.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

with (OUT / "representative_ckks_estimates.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["metric", "KGC", "SK-HE", "units", "evidence"])
    w.writerow(["server_latency", f"{kgc_server_ms/1000:.2f}", f"{skhe_server_ms/1000:.2f}", "s", "calibrated estimate"])
    w.writerow(["input_encryption", f"{kgc_encrypt_ms/1000:.2f}", f"{skhe_encrypt_ms/1000:.2f}", "s", "calibrated estimate"])
    w.writerow(["upload_payload", f"{kgc_upload_mib:.1f}", f"{skhe_upload_mib:.1f}", "MiB", "raw-RNS payload proxy"])
    w.writerow(["peak_ciphertext_working_set", f"{kgc_working_mib:.0f}", f"{skhe_working_mib:.0f}", "MiB", "modeled, keys/runtime excluded"])
    w.writerow(["graph_HE_matvecs", "0", str(graph_matvecs), "count", "exact structural count"])
    w.writerow(["absolute_server_time_saved", f"{absolute_saving_ms/1000:.2f}", "", "s", "derived estimate"])
    w.writerow(["TRG", f"{100*trg:.1f}", "", "%", "derived estimate"])
    w.writerow(["server_speedup", f"{speedup:.2f}", "", "x", "derived estimate"])

with (OUT / "scaling_envelope.csv").open("w", newline="", encoding="utf-8") as f:
    fields = list(scale_rows[0].keys())
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader(); w.writerows(scale_rows)

tex = rf"""\begin{{table}}[t]
\centering
\caption{{Calibrated representative CPU model for the matched Karate KGC/SK-HE pair. Values are \emph{{estimates, not measured TenSEAL runtimes}}, anchored to the public TenSEAL c4.2xlarge microbenchmarks of Benaissa et al. and rescaled to the reference $N=16384$, 400-bit chain. Payload is a raw-RNS ciphertext-volume proxy; peak memory excludes evaluation keys and runtime overhead.}}
\label{{tab:modeled-ckks}}
\small
\begin{{tabular}}{{lrr}}
\toprule
Metric & KGC & SK-HE \\
\midrule
Server wall-clock estimate (s) & {kgc_server_ms/1000:.2f} & {skhe_server_ms/1000:.2f} \\
Input encryption estimate (s) & {kgc_encrypt_ms/1000:.2f} & {skhe_encrypt_ms/1000:.2f} \\
Input ciphertexts & {kgc_input_ct} & {skhe_input_ct} \\
Upload payload proxy (MiB) & {kgc_upload_mib:.1f} & {skhe_upload_mib:.1f} \\
Peak ciphertext working set (MiB) & {kgc_working_mib:.0f} & {skhe_working_mib:.0f} \\
Server graph HE matvecs & 0 & {graph_matvecs} \\
\midrule
Absolute server time saved (s) & \multicolumn{{2}}{{c}}{{{absolute_saving_ms/1000:.2f}}} \\
Topology Relocation Gain (TRG) & \multicolumn{{2}}{{c}}{{{100*trg:.1f}\%}} \\
Matched server speedup & \multicolumn{{2}}{{c}}{{{speedup:.2f}$\times$}} \\
\bottomrule
\end{{tabular}}
\end{{table}}
"""
(OUT / "representative_ckks_table.tex").write_text(tex, encoding="utf-8")

scale_tex_lines = [
    r"\begin{table*}[t]",
    r"\centering",
    r"\caption{Slot-aware scale envelope under the reference $N=16384$ CKKS context ($8192$ slots), $K=2$. The $f_c=32$ columns describe a compact deployment scenario with a declared 32-channel client-visible/public input projection; they are analytical capacity estimates, not executed accuracy or latency results. SK-HE matvec bounds use the $B$-to-$B^2$ nonempty-block envelope for graphs whose every public row shard is incident to the shift.}",
    r"\label{tab:scaling-envelope}",
    r"\scriptsize",
    r"\resizebox{\textwidth}{!}{%",
    r"\begin{tabular}{lrrrrrrr}",
    r"\toprule",
    r"Dataset & Nodes & Edges & Raw $f$ & Batches $B$ & KGC cts ($f_c=32$) & KGC upload (GiB) & SK-HE graph matvec envelope \\",
    r"\midrule",
]
for row in scale_rows:
    scale_tex_lines.append(
        f"{row['dataset']} & {row['nodes']:,} & {row['edges']:,} & {row['raw_features']:,} & {row['node_batches']} & "
        f"{row['kgc_ciphertexts_f32']:,} & {row['kgc_upload_gib_f32']:.2f} & "
        f"{row['skhe_graph_matvec_lower_f32']:,}--{row['skhe_graph_matvec_upper_f32']:,} \\\\"
    )
scale_tex_lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
(OUT / "scaling_envelope_table.tex").write_text("\n".join(scale_tex_lines)+"\n", encoding="utf-8")

notice = f"""MODELED CKKS RESULTS -- NOT MEASURED HARDWARE OUTPUT

Purpose
-------
These files address the absence of a directly runnable TenSEAL environment in the
submission-build container without fabricating empirical observations. They provide
transparent, representative CPU estimates and a reproducible calculation script.

Calibration anchor
------------------
Benaissa, Retiat, Cebere, and Belfedhal (2021), TenSEAL: A Library for Encrypted
Tensor Operations Using Homomorphic Encryption, arXiv:2104.03152 / ICLR 2021 DPML.
Their complete CKKSVector benchmarks were run on AWS EC2 c4.2xlarge: 8 vCPU Intel
Xeon E5-2666 v3 at 2.9 GHz, 15 GiB RAM, Ubuntu Server 20.04, Python 3.8, with
N=8192 and a 200-bit coefficient modulus. For shape [256], reported mean timings
include add=0.08 ms, multiply=4.45 ms, add-plain=0.80 ms, multiply-plain=1.75 ms,
dot=20.15 ms, and quadratic polyval=10.55 ms.

Target context
--------------
N=16384; coeff_mod_bit_sizes=(60,40,40,40,40,40,40,40,60), total 400 bits;
global scale=2^40; slot capacity=8192. These are the reference benchmark-script
parameters, not a standalone security proof.

Headline representative estimates (Karate artifact)
----------------------------------------------------
KGC server estimate: {kgc_server_ms/1000:.2f} s
SK-HE server estimate: {skhe_server_ms/1000:.2f} s
Absolute server time saving: {absolute_saving_ms/1000:.2f} s
TRG: {100*trg:.1f}%
Matched server speedup: {speedup:.2f}x
KGC / SK-HE upload payload proxy: {kgc_upload_mib:.1f} / {skhe_upload_mib:.1f} MiB
KGC / SK-HE peak ciphertext working set: {kgc_working_mib:.0f} / {skhe_working_mib:.0f} MiB

Interpretation and uncertainty
------------------------------
The latency model rescales NTT/key-switch dominated operations by N log2(N) and
RNS-limb count and infers a rotation anchor from the TenSEAL dot benchmark. A
degree-5 polynomial is modeled at 2.2x the target-context quadratic polyval cost.
Server times should be treated with a ±25% scenario band; encryption with ±35%;
working-set memory with ±30%. The payload number is a raw-RNS representation proxy,
not a TenSEAL serialization measurement. Evaluation-key memory is excluded.

For publication-quality *measured* results, install requirements-ckks.txt and run:
  python code/real_ckks_public_benchmark.py --repeats 5
Then replace/model-compare these estimates with the generated measured CSV/JSON/TeX.
"""
(OUT / "MODELED_RESULTS_NOTICE.txt").write_text(notice, encoding="utf-8")
print(json.dumps(results["karate_matched_pair"], indent=2))
