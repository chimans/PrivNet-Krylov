import importlib.util
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / 'code' / 'privnet_krylov.py'
spec = importlib.util.spec_from_file_location('privnet_krylov', MODULE)
pk = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pk
spec.loader.exec_module(pk)


def test_chebyshev_compilation_matches_direct():
    rng = np.random.default_rng(123)
    a = rng.normal(size=(8, 8))
    a = (a + a.T) / 2
    # Scale spectral norm below one to satisfy the intended graph-shift regime.
    a = a / max(np.linalg.norm(a, 2), 1.0)
    x = rng.normal(size=(8, 3))
    coeff = rng.normal(size=5)
    blocks = pk.chebyshev_krylov_compile(a, x, 4)
    compiled = sum(c * b for c, b in zip(coeff, blocks))
    direct = pk.direct_chebyshev_apply(a, x, coeff)
    assert np.allclose(compiled, direct, atol=1e-12, rtol=1e-12)


def test_balanced_polynomial_matches_numpy():
    x = np.linspace(-2, 2, 101)
    coeff = np.array([0.2, -0.3, 0.7, 0.0, -0.05])
    got = pk.eval_poly_balanced(x, coeff)
    expected = sum(c * x**j for j, c in enumerate(coeff))
    assert np.allclose(got, expected, atol=1e-12, rtol=1e-12)


def test_depth_formula():
    expected = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2, 5: 3, 7: 3, 8: 3, 9: 4}
    for degree, depth in expected.items():
        assert pk.multiplicative_depth_for_degree(degree) == depth
