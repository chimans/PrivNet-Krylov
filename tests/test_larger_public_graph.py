import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import numpy as np

from larger_public_graph_benchmark import DigitsGraphConfig, load_digits_knn, sparse_chebyshev_compile


def test_digits_graph_dimensions_and_labels():
    cfg = DigitsGraphConfig()
    a, s, x, y, projection = load_digits_knn(cfg)
    assert a.shape == (1797, 1797)
    assert s.shape == (1797, 1797)
    assert x.shape == (1797, 32)
    assert y.shape == (1797,)
    assert len(np.unique(y)) == 10
    assert projection.shape == (64, 32)
    assert a.nnz > 1797 * cfg.neighbors


def test_sparse_chebyshev_recurrence():
    cfg = DigitsGraphConfig(krylov_degree=2)
    _, s, x, _, _ = load_digits_knn(cfg)
    blocks = sparse_chebyshev_compile(s, x, 2)
    assert len(blocks) == 3
    assert np.allclose(blocks[1], s @ x)
    assert np.allclose(blocks[2], 2.0 * (s @ blocks[1]) - blocks[0])
