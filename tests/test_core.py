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


def test_chebyshev_compilation_matches_independent_recurrence():
    rng = np.random.default_rng(123)
    a = rng.normal(size=(8, 8))
    a = (a + a.T) / 2
    a = a / max(np.linalg.norm(a, 2), 1.0)
    x = rng.normal(size=(8, 3))
    degree = 4

    blocks = pk.chebyshev_krylov_compile(a, x, degree)

    n = a.shape[0]
    t_matrices = [np.eye(n), a.copy()]

    for _ in range(1, degree):
        t_next = 2.0 * a @ t_matrices[-1] - t_matrices[-2]
        t_matrices.append(t_next)

    expected_blocks = [t @ x for t in t_matrices]

    assert len(blocks) == degree + 1

    for actual, expected in zip(blocks, expected_blocks):
        assert np.allclose(actual, expected, atol=1e-12, rtol=1e-12)


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

def test_chebyshev_degree_zero_returns_input():
    rng = np.random.default_rng(321)

    a = rng.normal(size=(6, 6))
    a = (a + a.T) / 2
    a = a / max(np.linalg.norm(a, 2), 1.0)

    x = rng.normal(size=(6, 3))

    blocks = pk.chebyshev_krylov_compile(a, x, 0)

    assert len(blocks) == 1
    assert np.allclose(blocks[0], x, atol=1e-12, rtol=1e-12)

def test_chebyshev_zero_matrix():
    x = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ])

    a = np.zeros((3, 3))

    blocks = pk.chebyshev_krylov_compile(a, x, 3)

    assert len(blocks) == 4

    expected_0 = x
    expected_1 = np.zeros_like(x)
    expected_2 = -x
    expected_3 = np.zeros_like(x)

    assert np.allclose(blocks[0], expected_0, atol=1e-12, rtol=1e-12)
    assert np.allclose(blocks[1], expected_1, atol=1e-12, rtol=1e-12)
    assert np.allclose(blocks[2], expected_2, atol=1e-12, rtol=1e-12)
    assert np.allclose(blocks[3], expected_3, atol=1e-12, rtol=1e-12)

def test_chebyshev_rejects_invalid_shapes():
    x = np.ones((3, 2))

    nonsquare_s = np.ones((3, 4))
    with np.testing.assert_raises_regex(ValueError, "square 2D matrix"):
        pk.chebyshev_krylov_compile(nonsquare_s, x, 2)

    bad_x = np.ones(3)
    s = np.eye(3)
    with np.testing.assert_raises_regex(ValueError, "x must be a 2D matrix"):
        pk.chebyshev_krylov_compile(s, bad_x, 2)

    mismatched_x = np.ones((4, 2))
    with np.testing.assert_raises_regex(ValueError, "same number of rows"):
        pk.chebyshev_krylov_compile(s, mismatched_x, 2)

def test_chebyshev_rejects_negative_degree():
    s = np.eye(3)
    x = np.ones((3, 2))

    with np.testing.assert_raises_regex(ValueError, "nonnegative"):
        pk.chebyshev_krylov_compile(s, x, -1)
