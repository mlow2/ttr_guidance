#!/usr/bin/env python
"""Post-install check: imports, CUDA, the two value-function pickles, and one short simulation.

    python check_environment.py
"""
import importlib
import os
import sys

REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

PACKAGES = ["numpy", "torch", "cvxpy", "clarabel", "matplotlib", "pandas", "control", "skfmm",
            "hj_reachability_torch", "hj_reachability_torch.shape"]
LIB_MODULES = ["cbf_logger", "dubins_car", "metrics", "run_paths", "safety_filter", "scenario",
               "sim_and_ctrl", "simulation_logger", "simulation_runner", "test_pipeline", "vis"]


def fail(message):
    print(f"FAIL: {message}")
    sys.exit(1)


def main():
    for name in PACKAGES + [f"lib.{m}" for m in LIB_MODULES]:
        try:
            importlib.import_module(name)
        except Exception as exc:
            fail(f"cannot import {name}: {exc}")
    print(f"imports ok: {len(PACKAGES)} packages, {len(LIB_MODULES)} lib modules")

    import numpy as np
    import torch
    from lib.scenario import scenario_from_states
    from lib.simulation_runner import _DEFAULT_DATA_PATHS, SimulationRunner

    if torch.cuda.is_available():
        print(f"cuda available: {torch.cuda.get_device_name(0)}")
    else:
        print("cuda not available: running on cpu")

    paths = _DEFAULT_DATA_PATHS["kinematic_vehicle"]
    missing = [p for p in paths.values() if not os.path.isfile(p)]
    if missing:
        fail(f"missing value-function pickle(s) {missing}; "
             "generate them with the scripts in bin/precompute/")

    try:
        runner = SimulationRunner("kinematic_vehicle")
    except Exception as exc:
        fail(f"could not load value functions: {exc}")
    print(f"{paths['hj_data']}: grid shape {runner.values_ttr.shape}, ttr_max {runner.ttr_max}")
    print(f"{paths['safety_data']}: grid shape {runner.values_safety.shape}")

    states = [np.array([-3.0, 1.0, 0.0, 0.07]),
              np.array([-3.0, -1.0, 0.0, 0.07]),
              np.array([-4.5, 0.0, 0.0, 0.07])]
    try:
        result = runner.run(scenario_from_states("environment_check", states), t_sim=5.0)
    except Exception as exc:
        fail(f"short simulation failed: {exc}")
    print(f"simulation ok: {result.num_vehicles} vehicles, {result.ts[-1]:.1f} s of sim time, "
          f"safety mode {result.safety_mode}, nominal mode {result.nominal_mode}")
    print("environment check passed")


if __name__ == "__main__":
    main()
