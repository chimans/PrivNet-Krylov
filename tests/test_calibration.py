import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "real_ckks"


def load():
    return json.loads((OUT / "representative_ckks_estimates.json").read_text())


def test_representative_trg_identity():
    d = load()["karate_matched_pair"]
    expected = 1.0 - d["kgc_server_latency_s"] / d["skhe_server_latency_s"]
    assert abs(expected - d["trg_fraction"]) < 1e-12
    assert abs((d["skhe_server_latency_s"] - d["kgc_server_latency_s"]) - d["absolute_server_time_saved_s"]) < 1e-12
    assert d["kgc_input_ciphertexts"] == 102
    assert d["skhe_input_ciphertexts"] == 34
    assert d["skhe_graph_he_matvecs"] == 68


def test_slot_scale_envelope_batches():
    rows = {r["dataset"]: r for r in load()["scale_envelope"]}
    assert rows["Cora"]["node_batches"] == 1
    assert rows["CiteSeer"]["node_batches"] == 1
    assert rows["PubMed"]["node_batches"] == 3
    assert rows["ogbn-arxiv"]["node_batches"] == 21
    assert rows["ogbn-products"]["node_batches"] == 299
    assert rows["ogbn-arxiv"]["kgc_ciphertexts_f32"] == 2016


def test_payload_proxy_is_structural():
    d = load()
    n = d["target_ckks_context"]["poly_modulus_degree"]
    primes = len(d["target_ckks_context"]["coeff_mod_bit_sizes"])
    mib = 2 * n * primes * 8 / (1024**2)
    k = d["karate_matched_pair"]
    assert abs(k["kgc_upload_payload_mib"] - 102 * mib) < 1e-12
    assert abs(k["skhe_upload_payload_mib"] - 34 * mib) < 1e-12
