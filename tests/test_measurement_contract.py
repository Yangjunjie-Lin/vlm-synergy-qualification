from capability_gate.models.adapters import PrefixTrie


def test_prefix_trie_allows_only_candidate_continuations_then_eos() -> None:
    trie = PrefixTrie([[10, 11], [10, 12], [20]], [2])
    assert trie.allowed([]) == [10, 20]
    assert trie.allowed([10]) == [11, 12]
    assert trie.allowed([10, 11]) == [2]
    assert trie.allowed([99]) == [2]
