"""Spend guard — the hard <$20 ceiling. Pure logic, $0, no Docker/LLM."""

import pytest

from hgm.spend import SpendCapExceeded, SpendGuard


def test_accumulates_and_reports_remaining():
    g = SpendGuard(cap_usd=20.0)
    g.add(5.0)
    assert g.spent == 5.0
    assert g.remaining() == 15.0


def test_raises_when_cap_reached():
    g = SpendGuard(cap_usd=10.0)
    g.add(6.0)
    with pytest.raises(SpendCapExceeded):
        g.add(5.0)  # 11 >= 10
    assert g.spent == 11.0  # spend is still recorded before raising


def test_check_blocks_once_capped():
    g = SpendGuard(cap_usd=10.0)
    with pytest.raises(SpendCapExceeded):
        g.add(10.0)
    with pytest.raises(SpendCapExceeded):
        g.check()  # subsequent paid actions are blocked up-front


def test_per_run_cost_limit_shrinks_with_budget():
    g = SpendGuard(cap_usd=20.0)
    big = g.per_run_cost_limit()
    g.add(18.0)
    small = g.per_run_cost_limit()
    assert small < big
    assert small >= 0.01  # never zero/negative
