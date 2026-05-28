"""Specification for the CMP pooling (Node.descendant_evals).

These encode exactly what the clade pool must contain. They fail until
``Node.descendant_evals`` is implemented (see the TODO in tree.py).
"""

from hgm.tree import Node


def _chain():
    """root -> child -> grandchild, with known outcomes."""
    reg: list[Node] = []
    root = Node(reg, "root")
    child = Node(reg, "child", parent=root)
    grand = Node(reg, "grandchild", parent=child)
    root.utility_measures = [1, 0]
    child.utility_measures = [1, 1, 1]
    grand.utility_measures = [0]
    return root, child, grand


def test_leaf_clade_is_just_its_own_evals():
    _, _, grand = _chain()
    # a leaf with fewer than num_pseudo evals contributes its raw outcomes, no descendants
    assert sorted(grand.descendant_evals(num_pseudo=10)) == [0]


def test_clade_pools_descendants_raw():
    root, _, _ = _chain()
    # root smoothed-own (2 evals < 10 -> raw [1,0]) + child raw [1,1,1] + grandchild raw [0]
    pooled = root.descendant_evals(num_pseudo=10)
    assert len(pooled) == 6
    assert sum(pooled) == 4  # 1 + 3 + 0


def test_pseudo_count_smoothing_caps_own_weight():
    reg: list[Node] = []
    root = Node(reg, "root")
    root.utility_measures = [1] * 100  # heavily evaluated, mean 1.0
    # with num_pseudo=10 the own contribution collapses to 10 copies of the mean
    assert root.descendant_evals(num_pseudo=10) == [1.0] * 10
