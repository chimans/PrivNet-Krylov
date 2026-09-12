from pathlib import Path
import ast


def _get_ckks_defaults():
    src = Path("code/real_ckks_public_benchmark.py").read_text()
    tree = ast.parse(src)

    values = {}

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "CKKSParameters":
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    if stmt.value is not None:
                        values[stmt.target.id] = ast.literal_eval(stmt.value)

    return values


def test_real_ckks_defaults_match_manuscript():
    params = _get_ckks_defaults()

    assert params["poly_modulus_degree"] == 16384
    assert params["global_scale_bits"] == 40
    assert params["coeff_mod_bit_sizes"] == (
        60, 40, 40, 40, 40, 40, 40, 40, 60
    )


def test_real_ckks_reference_chain_is_400_bits():
    params = _get_ckks_defaults()
    assert sum(params["coeff_mod_bit_sizes"]) == 400
