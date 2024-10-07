import sys
import os
import argparse

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, 'script'))
sys.path.append(os.path.join(script_dir, 'python_wrapper'))

from evo.core import metrics, trajectory
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
from evo.core.metrics import PoseRelation
import evo.main_rpe as main_rpe
import numpy as np

print(f"Current working directory: {os.getcwd()}")

def load_config():
    return ConfigFactory.parse_file('config.conf')

def format_results(auc_result):
    result = "-------------\n"
    result += "Thresholds | F-score (Trans) | F-score (Rot)\n"
    result += "-----------+------------------+--------------\n"
    for t, ft, fr in zip(auc_result['thresholds'], 
                         auc_result['fscore_transes'], 
                         auc_result['fscore_rots']):
        result += f"{t:9.3f} | {ft:16.4f} | {fr:12.4f}\n"
    result += "-----------+------------------+--------------\n"

    fscore_area_trans = auc_result['fscore_area_trans'].item() if hasattr(auc_result['fscore_area_trans'], 'item') else auc_result['fscore_area_trans']
    fscore_area_rot = auc_result['fscore_area_rot'].item() if hasattr(auc_result['fscore_area_rot'], 'item') else auc_result['fscore_area_rot']

    result += f"AUC (Trans): {fscore_area_trans:.4f}\n"
    result += f"AUC (Rot)  : {fscore_area_rot:.4f}\n"
    return result

def process_trajectory_pair(ref_file, est_file, config):
    max_diff = config.get_float('parameters.max_diff')
    trans_threshold = config.get_float('parameters.trans_threshold')
    rot_threshold = config.get_float('parameters.rot_threshold')
    threshold_start = config.get_float('parameters.threshold_start')
    threshold_end = config.get_float('parameters.threshold_end')
    threshold_interval = config.get_float('parameters.threshold_interval')

    traj_ref = file_interface.read_tum_trajectory_file(ref_file)
    full_len = len(traj_ref.positions_xyz)
    print("Ref Size (Complete length): ", full_len)

    traj_est = file_interface.read_tum_trajectory_file(est_file)
    offset_2 = abs(traj_est.timestamps[0] - traj_ref.timestamps[0])
    offset_2 *= -1 if traj_est.timestamps[0] > traj_ref.timestamps[0] else 1

    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est, max_diff, offset_2)
    print("Est Size: ", len(traj_est.positions_xyz))

    result_trans = main_rpe.rpe(traj_ref, traj_est, est_name='RPE translation',
                            pose_relation=PoseRelation.translation_part, 
                            delta=1.0, delta_unit=Unit.frames,
                            all_pairs=False, align=True, correct_scale=False,
                            support_loop=False)

    rpe_rot = metrics.RPE(PoseRelation.rotation_angle_deg, delta=1.0, delta_unit=Unit.frames, all_pairs=False)
    rpe_rot.process_data((traj_ref, traj_est))
    
    rpe_trans_result = result_trans.np_arrays["error_array"]
    rpe_rot_result = np.radians(rpe_rot.error)

    fscore_trans, fscore_rot = RobustnessMetric.calc_fscore(
        rpe_trans_result, rpe_rot_result, full_len, trans_threshold, rot_threshold
    )

    auc_result = RobustnessMetric.eval_robustness_batch(
        rpe_trans_result, rpe_rot_result, full_len, threshold_start, threshold_end, threshold_interval)

    return fscore_trans, fscore_rot, auc_result


def main(plot_mode):
    print(f"Current working directory: {os.getcwd()}")
    
    config = load_config()
    trajectory_pairs = config.get_list('trajectory_pairs')
    
    for i, pair in enumerate(trajectory_pairs):
        ref_file = pair.get_string('reference')
        est_file = pair.get_string('estimated')

        print(f"Reference: {ref_file}")
        print(f"Estimated: {est_file}")
        
        fscore_trans, fscore_rot, auc_result = process_trajectory_pair(ref_file, est_file, config)
        formatted_results = format_results(auc_result)
        
        print("---------------------------------")
        print("\nAUC Result:")
        print(formatted_results)
        
        RobustnessMetric.save_results(ref_file, est_file, fscore_trans, fscore_rot, auc_result)
        
        if plot_mode:
            RobustnessMetric.plot_robustness_metrics(auc_result, ref_file, est_file)

    print("\nScript execution completed. Results have been saved in the reference directories.")
    if plot_mode:
        print("Plots have been generated for each trajectory pair.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trajectory Robustness Analysis")
    parser.add_argument("--plot", action="store_true", help="Enable plotting of robustness metrics")
    args = parser.parse_args()
    main(plot_mode=args.plot)