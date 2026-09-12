#!/usr/bin/env python3
"""Optional TenSEAL backend for direct CKKS evaluation of KGC.

Each compiled feature column is packed across graph nodes in one CKKS vector.
Because graph propagation has already been performed on the client, the server
uses only the topology-independent affine/polynomial head. Install the optional
dependencies from `requirements-ckks.txt` before importing this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

try:
    import tenseal as ts
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "TenSEAL is not installed. Install requirements-ckks.txt, then rerun."
    ) from exc


@dataclass
class CKKSParameters:
    poly_modulus_degree: int = 16384
    coeff_mod_bit_sizes: tuple = (60, 40, 40, 40, 40, 40, 60)
    global_scale_bits: int = 40


def create_context(params: CKKSParameters = CKKSParameters()):
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=params.poly_modulus_degree,
        coeff_mod_bit_sizes=list(params.coeff_mod_bit_sizes),
    )
    context.global_scale = 2 ** params.global_scale_bits
    context.generate_galois_keys()
    context.generate_relin_keys()
    return context


def encrypt_compiled_blocks(context, blocks: np.ndarray) -> List[List]:
    """Encrypt blocks[K+1,N,F] column-wise; fresh randomness is used by CKKS encryption."""
    blocks = np.asarray(blocks, dtype=np.float64)
    if blocks.ndim != 3:
        raise ValueError("blocks must have shape [K+1,N,F]")
    encrypted: List[List] = []
    for k in range(blocks.shape[0]):
        encrypted.append([ts.ckks_vector(context, blocks[k, :, f].tolist())
                          for f in range(blocks.shape[2])])
    return encrypted


def encrypted_affine_from_blocks(enc_blocks: List[List], theta: np.ndarray, bias: np.ndarray) -> List:
    """Compute U = sum_k B_k Theta_k + bias in encrypted column-vector form."""
    theta = np.asarray(theta, dtype=np.float64)
    bias = np.asarray(bias, dtype=np.float64)
    Kp1, F, H = theta.shape
    if len(enc_blocks) != Kp1 or len(enc_blocks[0]) != F:
        raise ValueError("encrypted blocks and theta shapes disagree")
    out = []
    for h in range(H):
        acc = None
        for k in range(Kp1):
            for f in range(F):
                term = enc_blocks[k][f] * float(theta[k, f, h])
                acc = term if acc is None else acc + term
        acc = acc + float(bias[h])
        out.append(acc)
    return out


def _ct_power(x, exponent: int, cache: Dict[int, object]):
    if exponent in cache:
        return cache[exponent]
    if exponent % 2 == 0:
        y = _ct_power(x, exponent // 2, cache)
        cache[exponent] = y * y
    else:
        a = exponent // 2
        cache[exponent] = _ct_power(x, a, cache) * _ct_power(x, exponent - a, cache)
    return cache[exponent]


def encrypted_polynomial_columns(columns: Sequence, coefficients: Sequence[float]) -> List:
    """Element-wise polynomial evaluation with a balanced power cache."""
    c = np.asarray(coefficients, dtype=np.float64)
    out = []
    for x in columns:
        # Start from c0 + c1*x so the result remains encrypted.
        if len(c) == 1:
            y = x * 0.0 + float(c[0])
        else:
            y = x * float(c[1]) + float(c[0])
        cache = {1: x}
        for j in range(2, len(c)):
            if abs(c[j]) > 0:
                y = y + _ct_power(x, j, cache) * float(c[j])
        out.append(y)
    return out


def encrypted_linear_head(hidden_columns: Sequence, weight: np.ndarray, bias: np.ndarray) -> List:
    """Compute Z = H W + b from encrypted hidden columns."""
    weight = np.asarray(weight, dtype=np.float64)
    bias = np.asarray(bias, dtype=np.float64)
    H, C = weight.shape
    if len(hidden_columns) != H:
        raise ValueError("hidden dimension mismatch")
    out = []
    for c in range(C):
        acc = None
        for h in range(H):
            term = hidden_columns[h] * float(weight[h, c])
            acc = term if acc is None else acc + term
        acc = acc + float(bias[c])
        out.append(acc)
    return out


def decrypt_columns(columns: Sequence) -> np.ndarray:
    return np.stack([np.asarray(c.decrypt(), dtype=np.float64) for c in columns], axis=1)


def demo(blocks: np.ndarray, theta: np.ndarray, hidden_bias: np.ndarray,
         activation_coeff: Sequence[float], out_weight: np.ndarray, out_bias: np.ndarray) -> np.ndarray:
    context = create_context()
    enc_blocks = encrypt_compiled_blocks(context, blocks)
    u = encrypted_affine_from_blocks(enc_blocks, theta, hidden_bias)
    h = encrypted_polynomial_columns(u, activation_coeff)
    z = encrypted_linear_head(h, out_weight, out_bias)
    return decrypt_columns(z)
