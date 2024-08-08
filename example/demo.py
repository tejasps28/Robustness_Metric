from evo.core import metrics
from evo.core.units import Unit
from evo.tools import log
log.configure_logging(verbose=True, debug=True, silent=False)
import pprint
import numpy as np
from evo.tools import plot
import matplotlib.pyplot as plt
from evo.tools.settings import SETTINGS
import torch
import sys
from evo.tools import file_interface
from evo.core import sync
sys.path.insert(1, '../script')
from robustness import RobustnessMetric


ref_file = "../test/data/freiburg1_xyz-groundtruth.txt"
est_file = "../test/data/freiburg1_xyz-rgbdslam_drift.txt"

traj_ref = file_interface.read_tum_trajectory_file(ref_file)
traj_est = file_interface.read_tum_trajectory_file(est_file)


# print(f"Length of reference trajectory: {len(traj_ref.positions_xyz)}")
# print(f"Length of estimated trajectory: {len(traj_est.positions_xyz)}")


max_diff = 0.01
traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est, max_diff)
# print(f"Length of reference trajectory: {len(traj_ref.positions_xyz)}")
# print(f"Length of estimated trajectory: {len(traj_est.positions_xyz)}")


# Convert the positions to the format expected by RobustnessMetric
trans_deriv1 = [torch.tensor(pos) for pos in traj_est.positions_xyz]
trans_deriv2 = [torch.tensor(pos) for pos in traj_ref.positions_xyz]

# For simplicity, assume rotational derivatives are zero (as they are not provided in the example)
rot_deriv1 = [torch.tensor([0.0, 0.0, 0.0]) for _ in traj_est.positions_xyz]
rot_deriv2 = [torch.tensor([0.0, 0.0, 0.0]) for _ in traj_ref.positions_xyz]

trans_threshold = 0.1
rot_threshold = 0.1


fscore_trans, fscore_rot = RobustnessMetric.calc_fscore(
    trans_deriv1, rot_deriv1, trans_deriv2, rot_deriv2, trans_threshold, rot_threshold
)

print(f"F-score for translation: {fscore_trans} for rotation: {fscore_rot}")

threshold_start = 0.0
threshold_end = 1.0
threshold_interval = 0.1

## calculate AUC
result = RobustnessMetric.eval_robustness_batch(
    trans_deriv1, rot_deriv1, trans_deriv2, rot_deriv2,
    threshold_start, threshold_end, threshold_interval
)

print(result)
