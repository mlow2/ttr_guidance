#!/usr/bin/env python
"""Reproduce the single-corridor evaluation over the frozen scenario set.

Loads the stored initial states from a scenario JSON file and runs every
(nominal_mode, safety_mode) arm on them, so the scenario distribution is held
fixed and reproducible across machines. The defaults reproduce the random
8-vehicle scenarios across the eight evaluation arms.

    python run_scenarios.py
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import pandas as pd

from lib.run_paths import resolve_run_dir
from lib.safety_filter import VALID_HYBRID_ORDERS
from lib.scenario import scenario_from_states
from lib.simulation_runner import SimulationRunner
from lib.test_pipeline import AAMSimConfig, AAMSimPipeline

DEFAULT_SCENARIOS = os.path.join("scenarios", "random_8v_cdc.json")

SUMMARY_METRICS = [
    "safety_violations_vehicle_normalized",
    "safety_violations_corridor_normalized",
    "safety_violations_global_normalized",
    "min_separation_distance_free",
    "min_separation_distance_corridor",
    "min_separation_distance",
    "kendall_tau",
    "norm_ttr_delay_std",
    "ttr_delay_avg",
    "total_sim_time",
    "wall_clock_solve_time",
]


def load_scenarios(path, prefix):
    with open(path) as f:
        entries = json.load(f)
    scenarios = []
    for e in entries:
        name = e.get("name", "")
        if not name.startswith(prefix):
            continue
        states = [np.asarray(s, dtype=np.float64) for s in e["initial_states"]]
        scenarios.append(scenario_from_states(name, states))
    return scenarios


def perfect_run_rate(group, t_sim, strict):
    """Fraction of finished runs with zero free-space violation time and an
    early stop; strict additionally requires arrivals in priority order."""
    complete = group[~group["has_unfinished_vehicles"].astype(bool)]
    if len(complete) == 0:
        return float("nan")
    ok = ((complete["safety_violations_vehicle_normalized"] == 0.0)
          & (complete["total_sim_time"] < t_sim))
    if strict:
        ok &= complete["kendall_tau"] == 1.0
    return float(ok.fillna(False).mean())


def summarize(df, t_sim):
    df = df.copy()
    df["passed"] = ((df["safety_violations_pair"] == 0)
                    & (~df["has_unfinished_vehicles"].astype(bool)))
    rows = []
    for (nm, sm), g in df.groupby(["nominal_mode", "safety_mode"]):
        row = {
            "nominal_mode": nm, "safety_mode": sm, "n": len(g),
            "pass_rate_pct": 100.0 * g["passed"].mean(),
            "perfect_run_rate": perfect_run_rate(g, t_sim, strict=False),
            "perfect_run_rate_strict": perfect_run_rate(g, t_sim, strict=True),
        }
        for m in SUMMARY_METRICS:
            row[f"{m}_mean"] = g[m].mean()
            row[f"{m}_std"] = g[m].std()
        rows.append(row)
    return pd.DataFrame(rows)


def print_summary(summary):
    pd.set_option("display.width", 220)
    print("\n== summary (pass = no free-space violation and all vehicles finish; "
          "perfect = zero violation time and early stop; strict adds tau = 1) ==")
    head = ["nominal_mode", "safety_mode", "n", "pass_rate_pct",
            "perfect_run_rate", "perfect_run_rate_strict"]
    print(summary[head].to_string(index=False))
    for m in SUMMARY_METRICS:
        print(f"\n-- {m} (mean / std) --")
        cols = ["nominal_mode", "safety_mode", f"{m}_mean", f"{m}_std"]
        print(summary[cols].to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default=DEFAULT_SCENARIOS,
                    help="JSON file of stored scenario initial states")
    ap.add_argument("--scenario-prefix", default="random_8v",
                    help="only run scenarios whose name starts with this")
    ap.add_argument("--nominal-modes", nargs="+",
                    default=["ttr_min", "ttr_track_bang"])
    ap.add_argument("--safety-modes", nargs="+",
                    default=["none", "decentralized_nearest_joint_hard_then_slack",
                             "hybrid_multi_stage", "centralized_all_slack_pair"])
    ap.add_argument("--ttr-separation", type=float, default=10.0)
    ap.add_argument("--cbf-h-margin", type=float, default=0.01,
                    help="enforce the pairwise CBF at the h = margin level set")
    ap.add_argument("--t-sim", type=float, default=300.0)
    ap.add_argument("--hybrid-order", default="priority_list",
                    choices=sorted(VALID_HYBRID_ORDERS))
    ap.add_argument("--animate", action="store_true")
    ap.add_argument("--animation-verbosity", type=int, default=1, choices=[0, 1, 2],
                    help="0 = vehicle id only, 1 = adds rank and TTR labels, "
                         "2 = adds the per-vehicle metrics panel")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    scenarios = load_scenarios(args.scenarios, args.scenario_prefix)
    if not scenarios:
        raise SystemExit(f"no scenarios matching {args.scenario_prefix!r} in {args.scenarios}")
    print(f"running {len(scenarios)} scenarios from {args.scenarios}")

    out_dir = resolve_run_dir(args.output_dir, "cdc", create=True)
    runner = SimulationRunner("kinematic_vehicle")
    cfg = AAMSimConfig(
        t_sim=args.t_sim,
        ttr_separation=args.ttr_separation,
        cbf_h_margin=args.cbf_h_margin,
        show_plots=False,
        save_static_plots=False,
        animate_scenarios=bool(args.animate),
        animate_modes=bool(args.animate),
        animation_verbosity=args.animation_verbosity,
    )
    pipeline = AAMSimPipeline(
        runner=runner,
        scenarios=scenarios,
        nominal_modes=args.nominal_modes,
        safety_modes=args.safety_modes,
        hybrid_orders=[args.hybrid_order] * args.safety_modes.count("hybrid_multi_stage"),
        output_dir=str(out_dir),
        run_config=cfg,
    )
    df = pipeline.run()

    summary = summarize(df, args.t_sim)
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    print_summary(summary)
    print(f"\nfull results: {out_dir}")


if __name__ == "__main__":
    main()
