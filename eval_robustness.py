import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, 'script'))
sys.path.append(os.path.join(script_dir, 'python_wrapper'))

from evo.core import metrics
from evo.core.units import Unit
from evo.tools import log, plot, file_interface
from evo.tools.settings import SETTINGS
from evo.core import sync
from robustness import RobustnessMetric
from pyhocon import ConfigFactory
import matplotlib.pyplot as plt
import torch
import RobustMetricLib 
from scipy.spatial.transform import Rotation

# log.configure_logging(verbose=True, debug=True, silent=False)

print(f"Current working directory: {os.getcwd()}")
config = ConfigFactory.parse_file('config.conf')

max_diff = config.get_float('parameters.max_diff')
trans_threshold = config.get_float('parameters.trans_threshold')
rot_threshold = config.get_float('parameters.rot_threshold')
threshold_start = config.get_float('parameters.threshold_start')
threshold_end = config.get_float('parameters.threshold_end')
threshold_interval = config.get_float('parameters.threshold_interval')
interv_ns = config.get_int('parameters.interv_ns', 1e7)  # default to 10ms 
keep_freq = config.get_int('parameters.keep_freq', 10)  # default to 10 

def format_results(ref_file, est_file, fscore_trans, fscore_rot, auc_result):
    result = "-------------\n"
    result += "Thresholds | F-score (Trans) | F-score (Rot)\n"
    result += "-----------+------------------+--------------\n"
    for t, ft, fr in zip(auc_result['thresholds'], 
                         auc_result['fscore_transes'], 
                         auc_result['fscore_rots']):
        result += f"{t:9.2f} | {ft:16.4f} | {fr:12.4f}\n"
    result += "-----------+------------------+--------------\n"
    result += f"AUC (Trans): {auc_result['fscore_area_trans'].item():.4f}\n"
    result += f"AUC (Rot)  : {auc_result['fscore_area_rot'].item():.4f}\n"
    return result

def process_trajectory_pair(ref_file, est_file):
    traj_ref = file_interface.read_tum_trajectory_file(ref_file)
    traj_est = file_interface.read_tum_trajectory_file(est_file)
    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est, max_diff)
    # print(traj_est.orientations_quat_wxyz)
    # print(traj_ref)
    def create_pose_tuple(quat, trans):
        return (tuple(quat), tuple(trans))

    poses_ref = [create_pose_tuple(quat, trans) 
                 for quat, trans in zip(traj_ref.orientations_quat_wxyz, traj_ref.positions_xyz)]
    poses_est = [create_pose_tuple(quat, trans) 
                 for quat, trans in zip(traj_est.orientations_quat_wxyz, traj_est.positions_xyz)]

    trans_deriv_ref, rot_deriv_ref = RobustMetricLib.estimate_velo(interv_ns, poses_ref, keep_freq)
    trans_deriv_est, rot_deriv_est = RobustMetricLib.estimate_velo(interv_ns, poses_est, keep_freq)

    # trans_deriv1 = [torch.tensor(pos) for pos in traj_est.positions_xyz]
    # trans_deriv2 = [torch.tensor(pos) for pos in traj_ref.positions_xyz]
    # rot_deriv1 = [torch.tensor([0.0, 0.0, 0.0]) for _ in traj_est.positions_xyz]
    # rot_deriv2 = [torch.tensor([0.0, 0.0, 0.0]) for _ in traj_ref.positions_xyz]

    trans_deriv1 = [torch.tensor(v) for v in trans_deriv_est]
    trans_deriv2 = [torch.tensor(v) for v in trans_deriv_ref]
    rot_deriv1 = [torch.tensor(v) for v in rot_deriv_est]
    rot_deriv2 = [torch.tensor(v) for v in rot_deriv_ref]

    fscore_trans, fscore_rot = RobustnessMetric.calc_fscore(
        trans_deriv1, rot_deriv1, trans_deriv2, rot_deriv2, trans_threshold, rot_threshold
    )

    auc_result = RobustnessMetric.eval_robustness_batch(
        trans_deriv1, rot_deriv1, trans_deriv2, rot_deriv2,
        threshold_start, threshold_end, threshold_interval
    )

    return fscore_trans, fscore_rot, auc_result

trajectory_pairs = config.get_list('trajectory_pairs')
for i, pair in enumerate(trajectory_pairs):
    ref_file = pair.get_string('reference')
    est_file = pair.get_string('estimated')
    
    # print(f"\nProcessing trajectory pair {i+1}:")
    print(f"Reference: {ref_file}")
    print(f"Estimated: {est_file}")
    
    fscore_trans, fscore_rot, auc_result = process_trajectory_pair(ref_file, est_file)
    formatted_results = format_results(ref_file, est_file, fscore_trans, fscore_rot, auc_result)
    
    print("---------------------------------")
    # print(f"F-score for translation: {fscore_trans}")
    # print(f"F-score for rotation: {fscore_rot}")
    print("\nAUC Result:")
    print(formatted_results)
    
    RobustnessMetric.save_results(ref_file, est_file, fscore_trans, fscore_rot, auc_result)

print("\nScript execution completed. Results have been saved in the reference directories.")