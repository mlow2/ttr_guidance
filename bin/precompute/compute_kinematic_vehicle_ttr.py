"""Precompute the reach-avoid time-to-reach (TTR) grid for the kinematic vehicle.

Writes ``data/ttr_grid.pkl``. Solving on this grid needs roughly 32 GB of GPU memory.

    python bin/precompute/compute_kinematic_vehicle_ttr.py
"""
import os
import pathlib
import pickle
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
# Must be set before `import torch` (and any package that imports torch).
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

from hj_reachability_torch.grid import HjTorchGrid, HjGridMetaData, HjTtrData
from hj_reachability_torch.shape import shape_rectangle_by_corners, shape_rectangle_by_corners_eikonal
from hj_reachability_torch.solver import solve_reach_avoid

from lib.dubins_car import KinematicVehicle

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)
if device == "cuda":
    torch.cuda.set_device(0)
torch.set_default_dtype(torch.float32)
torch.manual_seed(0)

SPEED_MIN = 0.03  # km/s
SPEED_MAX = 0.09  # km/s
SPEED_TARGET = 0.07  # km/s
SPEED_TARGET_THRESHOLD = 0.02  # km/s
ACCEL_MIN = -0.001  # km/s^2
ACCEL_MAX = 0.002  # km/s^2
ANGULAR_RATE_MAX = 0.1  # rad/s
CORRIDOR_WIDTH = 1.0  # km

T_MAX = 360.0  # s
DT = 0.09  # s

dynsys = KinematicVehicle(v_min=SPEED_MIN, v_max=SPEED_MAX,
                          accel_min=ACCEL_MIN, accel_max=ACCEL_MAX,
                          angular_rate_max=ANGULAR_RATE_MAX)

domain_lo = np.array([-7.5, -7.5, -np.pi, SPEED_MIN - 0.02])
domain_hi = np.array([7.5, 7.5, np.pi, SPEED_MAX + 0.02])
grid_shape = [271, 271, 60, 21]
grid = HjTorchGrid(domain_lo, domain_hi, grid_shape, pdims=[2])

target_function = shape_rectangle_by_corners_eikonal(
    grid,
    [0, -CORRIDOR_WIDTH / 2, -np.pi / 9, SPEED_TARGET - SPEED_TARGET_THRESHOLD],
    [CORRIDOR_WIDTH / 2, CORRIDOR_WIDTH / 2, np.pi / 9, SPEED_TARGET + SPEED_TARGET_THRESHOLD])

# Failure region: the corridor walls beyond the entrance, plus the domain boundary.
avoid_function = shape_rectangle_by_corners(
    grid,
    [CORRIDOR_WIDTH / 2, -CORRIDOR_WIDTH / 2, -np.inf, -np.inf],
    [np.inf, CORRIDOR_WIDTH / 2, np.inf, np.inf])
avoid_function_boundary = -shape_rectangle_by_corners(grid, domain_lo, domain_hi)
avoid_function = torch.min(avoid_function, avoid_function_boundary)
constraint_function = -avoid_function

total_timestep = int(round(T_MAX / DT))
ttr = torch.inf * torch.ones(grid.shape, device=device)
values = target_function.clone()
for k_timestep in range(total_timestep):
    if k_timestep % 100 == 0:
        print(f"Time step: {k_timestep} / {total_timestep}")
    values, _ = solve_reach_avoid(dynsys, grid, values, constraint_function, DT, DT, return_final=True)
    ttr[(ttr == torch.inf) & (values < 0)] = k_timestep * DT

info = {'angular_rate_max': ANGULAR_RATE_MAX,
        'speed_min': SPEED_MIN, 'speed_max': SPEED_MAX,
        'speed_target': SPEED_TARGET, 'speed_target_threshold': SPEED_TARGET_THRESHOLD,
        'accel_min': ACCEL_MIN, 'accel_max': ACCEL_MAX,
        'corridor_width': CORRIDOR_WIDTH}
grid_meta_data = HjGridMetaData(domain_lo, domain_hi, grid_shape, pdims=[2])
hj_ttr_data = HjTtrData(grid_meta_data, target_function, constraint_function,
                        total_timestep * DT, ttr, info)

ttr_file_name = 'data/ttr_grid.pkl'
os.makedirs(os.path.dirname(ttr_file_name), exist_ok=True)
with open(ttr_file_name, 'wb') as file:
    pickle.dump(hj_ttr_data, file)
print(f"wrote {ttr_file_name}")
