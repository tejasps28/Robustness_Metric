# Robustness_Metric

This project evaluates the robustness of trajectories by comparing them using Robustness Metrics (details in [paper](https://arxiv.org/pdf/2307.07607)). The evaluations are performed using Python scripts that interface with underlying C++ implementations via the `RobustMetricLib` Python module. 


### Description

The `eval_robustness.py` script compares two trajectories to evaluate their robustness using translation and rotation metrics.

The `eval_robustness_batch.py` script evaluates the robustness of trajectories over a range of thresholds, providing a detailed analysis of robustness metrics.


#### Usage

```bash
python3 script/eval_robustness.py pose_file1 pose_file2 keep_freq trans_threshold rot_threshold

python3 script/eval_robustness_batch.py pose_file1 pose_file2 keep_freq interv_ns threshold_start threshold_end threshold_interval
