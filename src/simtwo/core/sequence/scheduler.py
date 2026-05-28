from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from sequence.kernel.event import Event
from sequence.kernel.process import Process

PlotCallback = Callable[[int, float], None]
ConditionsCallback = Callable[[dict[str, Any]], None]


@dataclass
class SequenceExperimentScheduler:
    """
    Schedules one long sequence timeline (can probably remove once sim control panel is fixed)
    """

    timeline: Any
    rows: list[dict[str, Any]]
    link_model_manager: Any
    requester: Any
    responder: Any

    request_start_offset_ps: int
    request_end_offset_ps: int
    row_spacing_ps: int

    memory_size: int
    target_fidelity: float

    # only request entanglement every N rows
    request_every_n_rows: int = 50

    results_rows: list[dict[str, Any]] = field(default_factory=list)
    cb_plot: PlotCallback | None = None
    cb_conditions: ConditionsCallback | None = None

    # internal state
    last_total_successes: int = 0
    current_model_info: dict[int, dict[str, Any]] = field(default_factory=dict)

    def schedule_all(self):
        for i in range(len(self.rows)):
            base_t = i * self.row_spacing_ps

            # always update modeled links every row
            self.timeline.schedule(
                Event(base_t, Process(self, "apply_link_update", [i], {}))
            )

            # only issue a heavy entanglement request every N rows
            if i % self.request_every_n_rows == 0:
                self.timeline.schedule(
                    Event(base_t + 1, Process(self, "issue_request", [i], {}))
                )

                self.timeline.schedule(
                    Event(
                        base_t + self.request_end_offset_ps + 1,
                        Process(self, "sample_attempt", [i], {}),
                    )
                )

    def apply_link_update(self, row_index: int):
        row = dict(self.rows[row_index])
        self.current_model_info[row_index] = self.link_model_manager.apply_to_registered_links(row)

    def issue_request(self, row_index: int):
        base_t = row_index * self.row_spacing_ps

        self.requester.network_manager.request(
            self.responder.name,
            start_time=base_t + self.request_start_offset_ps,
            end_time=base_t + self.request_end_offset_ps,
            memory_size=self.memory_size,
            target_fidelity=self.target_fidelity,
        )

    def sample_attempt(self, row_index: int):
        """
        count cum entangled memories on requester that point to responder
        success for sampled attempt is cum count increased since last sample
        """
        total_successes_now = 0
        for info in self.requester.resource_manager.memory_manager:
            state = getattr(info, "state", None)
            remote_node = getattr(info, "remote_node", None)
            if state == "ENTANGLED" and remote_node == self.responder.name:
                total_successes_now += 1

        success_this_attempt = 1 if total_successes_now > self.last_total_successes else 0
        self.last_total_successes = max(self.last_total_successes, total_successes_now)

        attempts_completed = row_index + 1
        success_pct = 100.0 * self.last_total_successes / attempts_completed

        row = dict(self.rows[row_index])
        model_info = self.current_model_info.get(row_index, {})

        result_row = {
            "attempt": attempts_completed,
            "success": success_this_attempt,
            "cumulative_successes": self.last_total_successes,
            "success_rate_pct": success_pct,
            "current_model": model_info.get("current_model"),
            "input_temperature": model_info.get("input_temperature"),
            "predicted_path_delay_ps": model_info.get("predicted_path_delay_ps"),
            "predicted_path_delay_ns": model_info.get("predicted_path_delay_ns"),
            "predicted_path_delay_s": model_info.get("predicted_path_delay_s"),
            "predicted_distance_m": model_info.get("predicted_distance_m"),
            "input_t_sec": row.get("t_sec"),
            "input_path_delay_ns": row.get("path_delay", row.get("path_delay_ns")),
            "sampled_row_index": row_index,
            "request_every_n_rows": self.request_every_n_rows,
        }
        self.results_rows.append(result_row)

        if self.cb_conditions is not None:
            self.cb_conditions(
                {
                    "attempt": attempts_completed,
                    "successful_entanglements": self.last_total_successes,
                    "success_rate_pct": success_pct,
                    "last_attempt_success": success_this_attempt,
                    "current_model": model_info.get("current_model"),
                    "input_temperature": model_info.get("input_temperature"),
                    "predicted_path_delay_ns": model_info.get("predicted_path_delay_ns"),
                    "predicted_path_delay_s": model_info.get("predicted_path_delay_s"),
                    "predicted_distance_m": model_info.get("predicted_distance_m"),
                    "sampled_row_index": row_index,
                    "request_every_n_rows": self.request_every_n_rows,
                }
            )

        if self.cb_plot is not None:
            # Keep plotting%
            self.cb_plot(row_index, success_pct)