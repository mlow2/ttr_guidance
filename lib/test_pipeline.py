"""Evaluation pipeline: run (scenario, nominal_mode, safety_mode) combinations and collect metrics."""

from __future__ import annotations

import os
import traceback
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from lib.scenario import ScenarioConfig
from lib.simulation_runner import SimulationRunner, SimulationResult, VALID_SAFETY_MODES, VALID_NOMINAL_MODES
from lib.safety_filter import DEFAULT_HMS_STAGE_NAMES
from lib.simulation_logger import build_trajectory_log
from lib.metrics import compute_metrics
from lib.vis import (
    animate_vehicle_trajectories, build_single_corridor_title,
    create_trajectory_figure, plot_control_history, plot_traj,
    plot_ttr_history)


@dataclass
class AAMSimConfig:
    """Per-run parameters forwarded to SimulationRunner.run"""
    cbf_rate: float = 1.0
    cbf_h_margin: float = 0.01
    t_sim: float = 150.0
    dt: float = 0.1
    ttr_separation: float = 10.0
    enable_early_termination: bool = True
    enable_cbf_logging: bool = True
    verbose: bool = False
    show_plots: bool = True
    save_static_plots: bool = False
    animate_scenarios: bool = False
    animate_modes: bool = False
    enable_mission_complete: bool = True
    min_corridor_traverse_dist: float = 3.5
    corridor_final_threshold: float = 4.0
    animation_verbosity: int = 1


RUN_SEED = 0


class AAMSimPipeline:
    """Iterate over (scenario, nominal_mode, safety_mode) combinations.

    Parameters
    ----------
    runner : SimulationRunner
        Shared runner with pre-loaded HJ data.
    scenarios : list[ScenarioConfig]
        Scenario specifications.
    nominal_modes : list[str]
        Nominal control modes to evaluate.
    safety_modes : list[str]
        Safety modes to evaluate.
    output_dir : str
        Root directory for per-run logs and the summary CSV.
    run_config : AAMSimConfig | None
        Simulation parameters.  Defaults to :class:`AAMSimConfig` defaults.
    """

    def __init__(
        self,
        runner: SimulationRunner,
        scenarios: list[ScenarioConfig],
        safety_modes: list[str],
        nominal_modes: list[str] | None = None,
        output_dir: str = "data/pipeline_results",
        run_config: AAMSimConfig | None = None,
        hybrid_orders: list[str] | None = None,
    ):
        for mode in safety_modes:
            if mode not in VALID_SAFETY_MODES:
                raise ValueError(f"Unknown safety_mode: {mode}")
        if nominal_modes is None:
            nominal_modes = ["ttr_min"]
        for mode in nominal_modes:
            if mode not in VALID_NOMINAL_MODES:
                raise ValueError(f"Unknown nominal_mode: {mode}")
        self.runner = runner
        self.scenarios = scenarios
        self.nominal_modes = nominal_modes
        self.safety_modes = safety_modes
        self.output_dir = Path(output_dir)
        self.run_config = run_config or AAMSimConfig()
        self.hybrid_orders = hybrid_orders or []

        self._hms_configs: list[tuple[int | None, str | None]] = []
        hms_counter = 0
        for mode in self.safety_modes:
            if mode == "hybrid_multi_stage":
                order = (
                    self.hybrid_orders[hms_counter]
                    if hms_counter < len(self.hybrid_orders)
                    else "most_critical_safety_pair"
                )
                self._hms_configs.append((hms_counter, order))
                hms_counter += 1
            else:
                self._hms_configs.append((None, None))

    def run(self) -> pd.DataFrame:
        """Execute the full pipeline and return a summary DataFrame."""
        os.makedirs(self.output_dir, exist_ok=True)
        rows: list[dict] = []
        total = len(self.scenarios) * len(self.nominal_modes) * len(self.safety_modes)
        idx = 0

        for scenario in self.scenarios:
            for nominal_mode in self.nominal_modes:
                for mode_idx, safety_mode in enumerate(self.safety_modes):
                    hms_idx, hms_order = self._hms_configs[mode_idx]
                    idx += 1
                    display_mode = self._display_mode(safety_mode, hms_idx)
                    label = (f"[{idx}/{total}] {self.runner.dynamics_type} | "
                             f"{scenario.name} | {nominal_mode} | {display_mode} | seed={RUN_SEED}")
                    print(f"--- {label} ---")
                    try:
                        row = self._run_single(
                            scenario, nominal_mode, safety_mode,
                            hms_index=hms_idx,
                            hybrid_order=hms_order,
                        )
                        rows.append(row)
                        print(f"    OK  tau={row.get('kendall_tau')}  "
                              f"violations_pair={row.get('safety_violations_pair')}  "
                              f"violations_vehicle={row.get('safety_violations_vehicle')}  "
                              f"ttr_delay_avg={row.get('ttr_delay_avg')}")
                    except Exception:
                        print("    FAILED:")
                        traceback.print_exc()
                        rows.append({
                            "dynamics_type": self.runner.dynamics_type,
                            "scenario": scenario.name,
                            "nominal_mode": nominal_mode,
                            "safety_mode": display_mode,
                            "seed": RUN_SEED,
                            "error": traceback.format_exc(),
                        })

        runs = pd.DataFrame(rows)
        runs_path = self.output_dir / "runs.csv"
        runs.to_csv(runs_path, index=False)
        print(f"\nPer-run results saved to {runs_path}")
        return runs

    # -- internals -----------------------------------------------------------

    @staticmethod
    def _display_mode(safety_mode: str, hms_index: int | None) -> str:
        """Build a descriptive label for the safety_mode column / display."""
        if safety_mode != "hybrid_multi_stage" or hms_index is None:
            return safety_mode
        return "HMS+" + "+".join(DEFAULT_HMS_STAGE_NAMES)

    def _run_single(self, scenario: ScenarioConfig, nominal_mode: str,
                    safety_mode: str, *,
                    hms_index: int | None = None,
                    hybrid_order: str | None = None) -> dict:
        rc = self.run_config
        result: SimulationResult = self.runner.run(
            scenario=scenario,
            safety_mode=safety_mode,
            nominal_mode=nominal_mode,
            ttr_separation=rc.ttr_separation,
            cbf_rate=rc.cbf_rate,
            cbf_h_margin=rc.cbf_h_margin,
            t_sim=rc.t_sim,
            dt=rc.dt,
            enable_early_termination=rc.enable_early_termination,
            enable_cbf_logging=rc.enable_cbf_logging,
            seed=RUN_SEED,
            verbose=rc.verbose,
            hybrid_order=hybrid_order or "most_critical_safety_pair",
            enable_mission_complete=rc.enable_mission_complete,
            min_corridor_traverse_dist=rc.min_corridor_traverse_dist,
            corridor_final_threshold=rc.corridor_final_threshold,
        )

        dir_mode = f"HMS_{hms_index}" if hms_index is not None else safety_mode
        run_tag = f"{self.runner.dynamics_type}__{scenario.name}__{nominal_mode}__{dir_mode}__seed{RUN_SEED}"
        run_dir = self.output_dir / run_tag
        os.makedirs(run_dir, exist_ok=True)

        traj_csv = str(run_dir / "trajectory.csv")
        traj_logger = build_trajectory_log(result, result.corridor_handle)
        traj_logger.to_csv(traj_csv)

        cbf_csv = None
        if result.cbf_logger is not None and result.cbf_logger.records:
            cbf_csv = str(run_dir / "cbf_log.csv")
            result.cbf_logger.to_csv(cbf_csv)

        metrics = compute_metrics(
            trajectory_csv_path=traj_csv,
            initial_ttr=result.initial_ttr,
            initial_priority=result.initial_priority,
            num_vehicles=result.num_vehicles,
            separation_distance=result.separation_distance,
            initial_target_ttr=result.initial_target_ttr,
            dt=rc.dt,
            corridor_entrance=tuple(result.corridor_handle.entrance_position.tolist()),
            corridor_half_width=result.corridor_handle.half_width,
            corridor_heading=result.corridor_handle.heading,
            cbf_log_csv_path=cbf_csv,
        )

        save_dir = run_dir if rc.save_static_plots else None
        if rc.show_plots or save_dir:
            self._show_static_plots(result, scenario.name, nominal_mode, safety_mode,
                                    save_dir=save_dir)

        display_mode = self._display_mode(safety_mode, hms_index)
        if rc.animate_scenarios and rc.animate_modes:
            video_path = str(run_dir / "animation.mp4")
            print(f"    Saving animation to {video_path} ...")
            priority_series = result.extras.get("priority")
            rank_series = None
            if priority_series is not None:
                rank_series = []
                # No-priority filters log a blank per-step priority; use the initial ordering instead.
                for row in priority_series:
                    if row is None:
                        row = result.initial_priority
                    if row is None:
                        rank_series.append(None)
                        continue
                    ranks = [0] * result.num_vehicles
                    for r, vid in enumerate(row):
                        ranks[vid] = r
                    rank_series.append(ranks)
            within_series = result.extras.get("within_corridor")
            complete_series = result.extras.get("mission_complete")
            status_series = None
            if within_series is not None:
                status_series = []
                for k, within_row in enumerate(within_series):
                    done_row = (complete_series[k]
                                if complete_series is not None
                                and k < len(complete_series) else None)
                    status_series.append([
                        "completed" if done_row is not None and done_row[i]
                        else ("in_corridor" if within_row[i] else "free")
                        for i in range(result.num_vehicles)])
            title = build_single_corridor_title(
                scenario_name=scenario.name,
                nominal_mode=nominal_mode,
                safety_mode=display_mode,
                ttr_separation=rc.ttr_separation,
                seed=RUN_SEED,
                num_vehicles=result.num_vehicles)
            animate_vehicle_trajectories(
                result.vehicle_x_dim,
                result.xs,
                result.num_vehicles,
                output_path=video_path,
                fps=10,
                frame_stride=3,
                corridor_handle=result.corridor_handle,
                trace_alpha=0.45,
                title=title,
                verbosity=rc.animation_verbosity,
                fs_priority_per_step=rank_series,
                ttr_per_step=result.extras.get("ttr"),
                priority_list=priority_series,
                ts_sim=result.ts,
                separation_distance=result.separation_distance,
                mission_complete_list=complete_series,
                status_per_step=status_series,
                ttr_target_per_step=result.extras.get("ttr_target"),
                critical_neighbor_per_step=result.extras.get("safety_critical"),
                critical_h_per_step=result.extras.get("safety_critical_h"),
                safety_override_per_step=result.extras.get("safety_override"),
                ttr_target_capped_per_step=result.extras.get("ttr_target_capped"),
            )

        row = {
            "dynamics_type": self.runner.dynamics_type,
            "scenario": scenario.name,
            "nominal_mode": nominal_mode,
            "safety_mode": display_mode,
            "seed": RUN_SEED,
            "num_vehicles": result.num_vehicles,
        }
        row.update(metrics.as_flat_dict())
        pd.DataFrame([row]).to_csv(run_dir / "stats.csv", index=False)
        return row

    def _show_static_plots(self, result: SimulationResult,
                           scenario_name: str, nominal_mode: str,
                           safety_mode: str,
                           save_dir: Path | None = None):
        vehicle_x_dim = result.vehicle_x_dim
        num_vehicles = result.num_vehicles
        xs_sim = result.xs
        show = self.run_config.show_plots

        title_suffix = f"{scenario_name} | {nominal_mode} | {safety_mode}"

        fig, ax = create_trajectory_figure()
        for i in range(num_vehicles):
            car_state = xs_sim[vehicle_x_dim * i : vehicle_x_dim * (i + 1), :]
            plot_traj(ax, car_state, i)
        if result.corridor_handle is not None:
            ax = result.corridor_handle.visualize_corridor(ax)
        ax.set_title(title_suffix)
        if save_dir:
            fig.savefig(save_dir / "trajectory_xy.png", dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

        u_ref_list = result.extras.get("u_ref", None)
        fig2 = plot_control_history(
            num_vehicles, result.ts, result.us, u_ref_list, apply_smoothing=True)
        fig2.suptitle(f"Control history: {title_suffix}")
        if save_dir:
            fig2.savefig(save_dir / "control_history.png", dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig2)

        ttr_data = result.extras.get("ttr")
        ttr_target_data = result.extras.get("ttr_target")
        if ttr_data is not None and result.initial_priority is not None:
            import numpy as np
            ttr_array = np.asarray(ttr_data)
            ttr_target_array = np.asarray(ttr_target_data) if ttr_target_data is not None else None
            fig3 = plot_ttr_history(result.ts, ttr_array, result.initial_priority, ttr_target_array)
            fig3.suptitle(f"TTR history: {title_suffix}")
            if save_dir:
                fig3.savefig(save_dir / "ttr_history.png", dpi=150, bbox_inches="tight")
            if show:
                plt.show()
            plt.close(fig3)
