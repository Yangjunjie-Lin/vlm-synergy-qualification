from capability_gate.statistics.metrics import cohens_kappa, one_sided_exact_lower


def test_atomic_exact_gate_counts() -> None:
    assert one_sided_exact_lower(51, 64) < 0.70
    assert one_sided_exact_lower(52, 64) >= 0.70
    assert one_sided_exact_lower(57, 64) < 0.82
    assert one_sided_exact_lower(58, 64) >= 0.82


def test_degenerate_kappa_is_not_treated_as_agreement() -> None:
    assert cohens_kappa(["north"] * 8, ["north"] * 8, ["north", "south"]) is None


def test_perfect_non_degenerate_kappa() -> None:
    labels = ["north", "south", "east", "west"]
    assert cohens_kappa(labels * 2, labels * 2, labels) == 1.0
