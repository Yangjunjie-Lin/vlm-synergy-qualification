from capability_gate.models.registry import load_model_specs


def test_three_independent_exact_revision_families() -> None:
    specs = load_model_specs()
    assert len(specs) == 3
    assert len({spec["family"] for spec in specs}) == 3
    assert all(len(spec["revision"]) == 40 for spec in specs)
    assert all(4 <= spec["parameters_billions"] <= 12 for spec in specs)
    assert all(not spec["selection_used_task_accuracy"] for spec in specs)
    assert all(spec["expected_weight_sha256"] for spec in specs)
