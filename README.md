# TTR Guidance

Code for time-to-reach (TTR) priority-based guidance for vehicles approaching a
shared air corridor. 

## Citation

Accepted at 2026 IEEE Conference on Decision and Control

```bibtex
@inproceedings{low2026ttr,
  author    = {Low, Matthew and Aloor, Jasmine Jerry and Tuck, Victoria Marie and Nuzzo, Pierluigi and Choi, Jason J.},
  title     = {Time-To-Reach Separation and Safety Filtering for Safe, Fair, and Efficient Multi-Agent Coordination},
  booktitle = {IEEE Conference on Decision and Control (CDC)},
  year      = {2026},
  note      = {arXiv:2605.20625},
  url       = {https://arxiv.org/abs/2605.20625},
}
```

## Requirements

- Python 3.11
- `numpy`, `torch` (CUDA build for GPU), `cvxpy`, `clarabel`, `matplotlib`, `pandas`, `control`,
  `scikit-fmm`
- [`hj_reachability_torch`](https://github.com/SCIAutonomy/hj_reachability_torch) at commit
  `ae936ad` (HJ reachability grid, dynamics, solver, and rollout primitives).
- To reproduce grids on GPU with above package, 32 GB VRAM is needed

## Install

```bash
conda create -n ttr python=3.11 -y
conda activate ttr
pip install numpy torch cvxpy clarabel matplotlib pandas control scikit-fmm

git clone https://github.com/SCIAutonomy/hj_reachability_torch.git
git -C hj_reachability_torch checkout ae936ad
pip install -e hj_reachability_torch
```

## Value functions

The scripts in `bin/precompute/` write the two value functions to `data/`.

| Script | Output | Grid | Approx. size | GPU memory |
| --- | --- | --- | --- | --- |
| `bin/precompute/compute_kinematic_vehicle_ttr.py` | `data/ttr_grid.pkl` | `(271, 271, 60, 21)` over `[x, y, heading, speed]` | 1.1 GB | about 32 GB |
| `bin/precompute/compute_kinematic_vehicle_safety.py` | `data/relative_dynamics_cbf.pkl` | `(121, 121, 32, 10, 10)` over relative `[x, y, heading, v_a, v_b]` | 0.4 GB | about 21 GB |

`ttr_grid.pkl` is the reach-avoid TTR value function that drives the nominal controller (horizon
360 s). `relative_dynamics_cbf.pkl` is the pairwise avoidance value function in the relative
dynamics of a vehicle pair, used as the CBF by the safety filter (0.5 km separation, 100 s horizon).

```bash
python bin/precompute/compute_kinematic_vehicle_ttr.py
python bin/precompute/compute_kinematic_vehicle_safety.py
```

The grid shapes and solver settings in the scripts are the ones used for the reported results.
Changing them changes the value functions and therefore the simulation outcomes.

## Environment check

After installing and computing the value functions, the environment can be checked:

```bash
python check_environment.py
```

## Full evaluation

```bash
python run_scenarios.py
```

The scenario set is `scenarios/random_8v_cdc.json`.
This runs 800 simulations (100 scenarios x 2 nominal control x 4 filter modes).

A single seed can be run:
```bash
python run_scenarios.py --scenario-prefix random_8v_sep1.0_ttr10.0_seed0 --t-sim 200
```

Results are written to `summary.csv`, visualization scripts are also available to render
the trajectories.

