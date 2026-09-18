"""Per-timestep trajectory logger, populated post-hoc from a SimulationResult."""

from __future__ import annotations

import csv
import os

import numpy as np

_STATE_COL_NAMES = {0: "x", 1: "y", 2: "theta", 3: "speed"}


def _state_col(i: int) -> str:
    return _STATE_COL_NAMES.get(i, f"x_{i}")


def _fmt_opt(series, vid) -> str:
    if series is None or vid >= len(series) or series[vid] is None:
        return ""
    value = float(series[vid])
    return f"{value:.6f}" if np.isfinite(value) else ""


def _fmt_flag(series, vid) -> str:
    if series is None or vid >= len(series) or series[vid] is None:
        return ""
    return str(int(bool(series[vid])))


def build_csv_fields(x_dim: int = 3, u_dim: int = 1) -> list[str]:
    """Build the ordered list of CSV column names for the given dimensions."""
    fields = ["sim_time", "vehicle_id"]
    for i in range(x_dim):
        fields.append(_state_col(i))
    for i in range(u_dim):
        fields.append(f"control_{i}")
    for i in range(u_dim):
        fields.append(f"control_ref_{i}")
    fields.extend(["priority_rank", "ttr_estimate", "in_corridor",
                    "within_corridor", "completed", "escaped", "reached_goal_time",
                    "ttr_target", "ttr_target_capped",
                    "critical_neighbor", "critical_h", "safety_override"])
    return fields


class SimulationLogger:
    """Accumulates per-timestep, per-vehicle records and writes to CSV."""

    def __init__(self, x_dim: int = 3, u_dim: int = 1):
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.records: list[dict] = []
        self._reached_goal_time: dict[int, float] = {}

    def log_timestep(
        self,
        t: float,
        state_list: list[np.ndarray],
        control_list: list[np.ndarray] | None,
        control_ref_list: list[np.ndarray] | None,
        priority_list: list[int] | None,
        ttr_list: list[float] | None,
        corridor_handle,
        within_corridor_list: list[bool] | None = None,
        completed_list: list[bool] | None = None,
        escaped_list: list[bool] | None = None,
        ttr_target_list: list[float] | None = None,
        ttr_target_capped_list: list[bool] | None = None,
        critical_neighbor_list: list[int | None] | None = None,
        critical_h_list: list[float | None] | None = None,
        safety_override_list: list[bool] | None = None,
    ):
        """Record data for every vehicle at simulation time *t*."""
        num_vehicles = len(state_list)

        priority_rank_map: dict[int, int] = {}
        if priority_list is not None:
            for rank, vid in enumerate(priority_list):
                priority_rank_map[vid] = rank

        for vid in range(num_vehicles):
            s = np.asarray(state_list[vid]).flatten()
            in_corridor = bool(corridor_handle.check_in_corridor(s))

            if in_corridor and vid not in self._reached_goal_time:
                self._reached_goal_time[vid] = t

            record: dict = {
                "sim_time": f"{t:.6f}",
                "vehicle_id": vid,
            }

            for i in range(min(len(s), self.x_dim)):
                record[_state_col(i)] = f"{s[i]:.6f}"

            if control_list is not None and vid < len(control_list):
                ctrl = np.asarray(control_list[vid]).flatten()
                for i in range(min(len(ctrl), self.u_dim)):
                    record[f"control_{i}"] = f"{ctrl[i]:.6f}"

            if control_ref_list is not None and vid < len(control_ref_list):
                ctrl_ref = np.asarray(control_ref_list[vid]).flatten()
                for i in range(min(len(ctrl_ref), self.u_dim)):
                    record[f"control_ref_{i}"] = f"{ctrl_ref[i]:.6f}"

            record["priority_rank"] = priority_rank_map.get(vid, "")
            record["ttr_estimate"] = (
                f"{ttr_list[vid]:.6f}" if ttr_list is not None else "")
            record["in_corridor"] = int(in_corridor)
            record["within_corridor"] = (
                int(within_corridor_list[vid])
                if within_corridor_list is not None else "")
            record["completed"] = (
                int(completed_list[vid])
                if completed_list is not None else "")
            record["escaped"] = (
                int(escaped_list[vid]) if escaped_list is not None else "")
            record["reached_goal_time"] = (
                f"{self._reached_goal_time[vid]:.6f}"
                if vid in self._reached_goal_time else "")

            record["ttr_target"] = _fmt_opt(ttr_target_list, vid)
            record["ttr_target_capped"] = _fmt_flag(ttr_target_capped_list, vid)
            record["critical_neighbor"] = (
                "" if (critical_neighbor_list is None
                       or vid >= len(critical_neighbor_list)
                       or critical_neighbor_list[vid] is None)
                else int(critical_neighbor_list[vid]))
            record["critical_h"] = _fmt_opt(critical_h_list, vid)
            record["safety_override"] = _fmt_flag(safety_override_list, vid)

            self.records.append(record)

    def to_csv(self, path: str):
        """Write all records to a CSV file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fieldnames = build_csv_fields(self.x_dim, self.u_dim)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.records)

    @staticmethod
    def load_csv(path: str) -> list[dict]:
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            return list(reader)


def build_trajectory_log(sim_result, corridor_handle) -> SimulationLogger:
    """Populate a SimulationLogger from a SimulationResult's rollout arrays and extras."""
    ts = sim_result.ts
    xs = sim_result.xs
    us = sim_result.us
    extras = sim_result.extras
    num_vehicles = sim_result.num_vehicles
    x_dim = sim_result.vehicle_x_dim
    u_dim = us.shape[0] // num_vehicles if us is not None and num_vehicles > 0 else 1
    n_steps = len(ts)

    logger = SimulationLogger(x_dim=x_dim, u_dim=u_dim)

    priority_series = extras.get("priority")
    ttr_series = extras.get("ttr")
    u_ref_series = extras.get("u_ref")
    within_corridor_series = extras.get("within_corridor")
    completed_series = extras.get("mission_complete")
    escaped_series = extras.get("escaped")
    diagnostic_series = {
        key: extras.get(key) for key in
        ("ttr_target", "ttr_target_capped", "safety_critical",
         "safety_critical_h", "safety_override")}

    def _row(key, k):
        series = diagnostic_series[key]
        if series is None or not isinstance(series, list) or k >= len(series):
            return None
        return series[k]

    for k in range(n_steps):
        t = float(ts[k])
        state_list = [xs[x_dim * i: x_dim * (i + 1), k] for i in range(num_vehicles)]

        control_list = None
        if us is not None and k < us.shape[1]:
            control_list = [us[u_dim * i: u_dim * (i + 1), k] for i in range(num_vehicles)]

        control_ref_list = None
        if u_ref_series is not None:
            u_ref_k = u_ref_series[k] if isinstance(u_ref_series, list) else u_ref_series
            if hasattr(u_ref_k, '__len__') and len(u_ref_k) >= num_vehicles * u_dim:
                control_ref_list = [u_ref_k[u_dim * i: u_dim * (i + 1)] for i in range(num_vehicles)]

        priority_k = None
        if priority_series is not None:
            priority_k = priority_series[k] if isinstance(priority_series, list) else priority_series

        ttr_k = None
        if ttr_series is not None:
            ttr_k = ttr_series[k] if isinstance(ttr_series, list) else ttr_series

        within_corridor_k = None
        if within_corridor_series is not None and isinstance(within_corridor_series, list):
            within_corridor_k = within_corridor_series[k]

        completed_k = None
        if completed_series is not None and isinstance(completed_series, list):
            completed_k = completed_series[k]

        escaped_k = None
        if escaped_series is not None and isinstance(escaped_series, list):
            escaped_k = escaped_series[k]

        logger.log_timestep(
            t, state_list, control_list, control_ref_list,
            priority_k, ttr_k, corridor_handle,
            within_corridor_list=within_corridor_k,
            completed_list=completed_k,
            escaped_list=escaped_k,
            ttr_target_list=_row("ttr_target", k),
            ttr_target_capped_list=_row("ttr_target_capped", k),
            critical_neighbor_list=_row("safety_critical", k),
            critical_h_list=_row("safety_critical_h", k),
            safety_override_list=_row("safety_override", k))

    return logger
