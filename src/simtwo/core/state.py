from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass
class UIState:
    lock: threading.Lock = field(default_factory=threading.Lock)

    epochs: list[int] = field(default_factory=list)
    times: list[float] = field(default_factory=list)
    conditions: dict[str, Any] = field(default_factory=dict)
    poincare_state: Any = None
    poincare_states: list[Any] = field(default_factory=list)

    run_speed_ms: int = 100
    running: bool = False
    status: str = ""

    def reset(self) -> None:
        with self.lock:
            self.epochs.clear()
            self.times.clear()
            self.conditions.clear()
            self.poincare_state = None
            self.poincare_states.clear()
            self.running = False
            self.status = ""
