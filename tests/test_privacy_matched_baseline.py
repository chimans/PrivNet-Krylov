import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import numpy as np

from privacy_matched_baseline import dense_cyclic_diagonal_matvec, fixed_shape_counts


def test_cyclic_diagonal_identity():
    rng = np.random.default_rng(44)
    m = rng.normal(size=(9, 9))
    x = rng.normal(size=9)
    assert np.allclose(dense_cyclic_diagonal_matvec(m, x), m @ x, atol=1e-12)


def test_fixed_shape_counts_karate_reference():
    c = fixed_shape_counts(nodes=34, feature_dim=34, krylov_degree=2)
    assert c.kgc_graph_ct_ct_multiplies == 0
    assert c.fsethe_graph_ct_ct_multiplies == 2312
    assert c.fsethe_input_topology_ciphertexts == 34
