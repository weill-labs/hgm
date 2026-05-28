"""Global spend guard — a hard ceiling on real LLM cost across an entire HGM run.

mini-swe-agent reports per-call cost; we accumulate it here. Crossing the cap raises
``SpendCapExceeded`` so the search aborts loudly rather than quietly overspending. Each
agent run also gets a per-run ``cost_limit`` (native to mini-swe-agent) as a second line
of defence.
"""

from __future__ import annotations

import threading


class SpendCapExceeded(RuntimeError):
    pass


class SpendGuard:
    def __init__(self, cap_usd: float) -> None:
        self.cap_usd = float(cap_usd)
        self._spent = 0.0
        self._lock = threading.Lock()

    @property
    def spent(self) -> float:
        with self._lock:
            return self._spent

    def remaining(self) -> float:
        with self._lock:
            return max(0.0, self.cap_usd - self._spent)

    def check(self) -> None:
        """Raise if we're already at/over the cap (call before starting a paid action)."""
        with self._lock:
            if self._spent >= self.cap_usd:
                raise SpendCapExceeded(
                    f"spent ${self._spent:.2f} >= cap ${self.cap_usd:.2f}"
                )

    def add(self, amount_usd: float) -> None:
        """Record spend, then raise if the cap is now breached."""
        with self._lock:
            self._spent += float(amount_usd)
            if self._spent >= self.cap_usd:
                raise SpendCapExceeded(
                    f"spent ${self._spent:.2f} >= cap ${self.cap_usd:.2f} (cap reached)"
                )

    def per_run_cost_limit(self, max_fraction: float = 0.5) -> float:
        """A per-run cost_limit: never let one agent run spend more than this. Bounds a
        single runaway run to a fraction of the remaining budget."""
        return max(0.01, self.remaining() * max_fraction)
