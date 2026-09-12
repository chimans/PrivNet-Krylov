import importlib.util
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for modname, rel in [("privnet_krylov", "code/privnet_krylov.py"), ("public_graph_benchmark", "code/public_graph_benchmark.py")]:
    spec = importlib.util.spec_from_file_location(modname, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)

pk = sys.modules["privnet_krylov"]
pg = sys.modules["public_graph_benchmark"]


def test_karate_public_graph_dimensions():
    g, a, s, x, labels = pg.load_karate()
    assert g.number_of_nodes() == 34
    assert g.number_of_edges() == 78
    assert a.shape == s.shape == x.shape == (34, 34)
    assert set(np.unique(labels)) == {0, 1}


def test_server_recurrence_matches_client_compilation():
    _, _, s, x, _ = pg.load_karate()
    degree = 2
    client = np.stack(pk.chebyshev_krylov_compile(s, x, degree))
    server = [x.copy(), s @ x]
    for _ in range(1, degree):
        server.append(2.0 * (s @ server[-1]) - server[-2])
    server = np.stack(server)
    assert np.allclose(client, server, atol=1e-12, rtol=1e-12)


def test_matched_relocation_structural_counts():
    _, _, s, x, _ = pg.load_karate()
    degree = 2
    f = x.shape[1]
    d = pk.cyclic_diagonal_count(s, tol=1e-15)
    assert (degree + 1) * f == 102
    assert f == 34
    assert degree * f == 68
    assert d == 34
    assert degree * f * (d - 1) == 2244


def test_seed7_artifact_full_matched_plaintext_logits():
    artifact_path = ROOT / "outputs/public_graph/karate_seed7_ckks_artifact.npz"
    with np.load(artifact_path, allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    degree = a["blocks"].shape[0] - 1
    server = [a["features"].copy()]
    if degree >= 1:
        server.append(a["shift"] @ a["features"])
    for _ in range(1, degree):
        server.append(2.0 * (a["shift"] @ server[-1]) - server[-2])
    server = np.stack(server)
    assert np.allclose(server, a["blocks"], atol=1e-12, rtol=1e-12)

    u = sum(server[k] @ a["theta"][k] for k in range(degree + 1)) + a["hidden_bias"]
    coeff = a["activation_coeff"]
    h = np.zeros_like(u, dtype=np.float64)
    for c in coeff[::-1]:
        h = h * u + float(c)
    logits = h @ a["out_weight"] + a["out_bias"]
    assert np.allclose(logits, a["plaintext_poly_logits"], atol=3e-5, rtol=3e-5)
