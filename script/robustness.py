import torch
import os
import csv

class RobustnessMetric:
    def calc_fscore(trans_deriv1, rot_deriv1, trans_deriv2, rot_deriv2, trans_threshold, rot_threshold):
        """
        Calculate the F-score for velocity and angular velocity .

        Parameters:
        trans_deriv1 (list of torch.Tensor): estimate velocity.
        rot_deriv1 (list of torch.Tensor): estimated angular velocity.
        trans_deriv2 (list of torch.Tensor): GT velocity.
        rot_deriv2 (list of torch.Tensor): GT angular velocity.
        trans_threshold (float)
        rot_threshold (float)
        Returns:
        tuple: F-score for translational derivatives and F-score for rotational derivatives.
        """
        trans_num = 0
        rot_num = 0
        index = 0

        while index < len(trans_deriv1) and index < len(rot_deriv1):
            x1, y1, z1 = trans_deriv1[index]
            rx1, ry1, rz1 = rot_deriv1[index]
            x2, y2, z2 = trans_deriv2[index]
            rx2, ry2, rz2 = rot_deriv2[index]

            trans_val = torch.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
            rot_val = torch.sqrt((rx1-rx2)**2 + (ry1-ry2)**2 + (rz1-rz2)**2)

            if trans_val <= trans_threshold:
                trans_num += 1
            if rot_val <= rot_threshold:
                rot_num += 1

            index += 1

        precision_trans = trans_num / len(trans_deriv1)
        precision_rot = rot_num / len(rot_deriv1)
        recall_trans = trans_num / len(trans_deriv2)
        recall_rot = rot_num / len(rot_deriv2)

        fscore_trans = 0.0
        if precision_trans + recall_trans > 0:
            fscore_trans = (2 * precision_trans * recall_trans) / (precision_trans + recall_trans)

        fscore_rot = 0.0
        if precision_rot + recall_rot > 0:
            fscore_rot = (2 * precision_rot * recall_rot) / (precision_rot + recall_rot)

        return fscore_trans, fscore_rot
    
    def eval_robustness_batch(trans_deriv1, rot_deriv1, trans_deriv2, rot_deriv2, 
                              threshold_start, threshold_end, threshold_interval):
        """
        Evaluate robustness by calculating F-scores and AUC for a range of thresholds.

        Parameters:
        trans_deriv1 (list of torch.Tensor): Estimated velocity.
        rot_deriv1 (list of torch.Tensor): Estimated angular velocity.
        trans_deriv2 (list of torch.Tensor): Ground truth velocity.
        rot_deriv2 (list of torch.Tensor): Ground truth angular velocity.
        threshold_start (float): Starting threshold value.
        threshold_end (float): Ending threshold value.
        threshold_interval (float): Interval between thresholds.

        Returns:
        dict: Dictionary containing the F-scores and areas under the F-score curve for translational and rotational derivatives.
        """
        fscore_area_trans = 0.0
        fscore_area_rot = 0.0
        fscore_transes = []
        fscore_rots = []
        thresholds = []
        num = 0
        
        threshold = threshold_start
        while threshold <= threshold_end:
            fscore_trans, fscore_rot = RobustnessMetric.calc_fscore(
                trans_deriv1, rot_deriv1, trans_deriv2, rot_deriv2, threshold, threshold)
            x_axis_len = torch.exp(torch.tensor(-10.0 * (threshold - threshold_interval * 0.5))) - torch.exp(torch.tensor(-10.0 * (threshold + threshold_interval * 0.5)))
            fscore_area_trans += fscore_trans * x_axis_len
            fscore_area_rot += fscore_rot * x_axis_len
            fscore_transes.append(fscore_trans)
            fscore_rots.append(fscore_rot)
            thresholds.append(threshold)
            num += 1
            threshold += threshold_interval

        return {
            'fscore_transes': fscore_transes,
            'fscore_rots': fscore_rots,
            'thresholds': thresholds,
            'fscore_area_trans': fscore_area_trans,
            'fscore_area_rot': fscore_area_rot
        }

    def save_results(ref_file, est_file, fscore_trans, fscore_rot, auc_result):
        ref_dir = os.path.dirname(ref_file)
        result_file = os.path.join(ref_dir, 'robustness_results.csv')

        with open(result_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            writer.writerow(['Reference', ref_file])
            writer.writerow(['Estimated', est_file])
            writer.writerow(['F-score for translation', f'{fscore_trans:.4f}'])
            writer.writerow(['F-score for rotation', f'{fscore_rot:.4f}'])
            writer.writerow([])  
            
            writer.writerow(['AUC Result'])
            writer.writerow(['Thresholds', 'F-score (Trans)', 'F-score (Rot)'])
            
            for t, ft, fr in zip(auc_result['thresholds'], 
                                 auc_result['fscore_transes'], 
                                 auc_result['fscore_rots']):
                writer.writerow([f'{t:.2f}', f'{ft:.4f}', f'{fr:.4f}'])
            
            writer.writerow([])  
            writer.writerow(['AUC (Trans)', f"{auc_result['fscore_area_trans'].item():.4f}"])
            writer.writerow(['AUC (Rot)', f"{auc_result['fscore_area_rot'].item():.4f}"])

        print(f"Results saved to: {result_file}")