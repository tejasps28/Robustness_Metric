# Robustness Metric

This project evaluates the robustness of trajectories by comparing them using Robustness Metrics based on [paper](https://arxiv.org/pdf/2307.07607). Traditional ATE and RPE primarily focus on the accuracy of trajectory and does not consider the completeness of trajectory (recall). To address these gaps, we introduce Robustness Metric based on estimated RPE that consider both precision and completeness.

https://github.com/user-attachments/assets/70e52b50-3a5a-4fec-955b-8e2c80cb6fee



## Quick Start

1. Configure your trajectory pairs in `config.conf`:
```hocon
parameters {
    max_diff = 0.01           # Maximum time difference for trajectory association
    trans_threshold = 0.1     # Translation threshold (meters)
    rot_threshold = 0.1       # Rotation threshold (radians)
    threshold_start = 0.01    # Starting threshold for AUC calculation
    threshold_end = 0.5       # Ending threshold for AUC calculation
    threshold_interval = 0.01 # Interval between thresholds
}

trajectory_pairs = [
    {
        reference = "/path/to/reference.txt"
        estimated = "/path/to/estimated.txt"
    }
]
```

2. Run the evaluation:
```bash
# Basic evaluation
python3 eval_robustness.py --config ./config.conf

# With visualization plots
python3 eval_robustness.py --config ./config.conf --plot
```

## Input Format

The script expects trajectories in TUM format:
```
timestamp tx ty tz qx qy qz qw
```
For more details on the format, see the [evo documentation](https://github.com/MichaelGrupp/evo/wiki/Formats).

## Reference

If you use this metrics in your research, consider cite our paper:
```
@InProceedings{Zhao2024CVPR,
    author    = {Zhao, Shibo and Gao, Yuanjun and Wu, Tianhao and Singh, Damanpreet and Jiang, Rushan and Sun, Haoxiang and Sarawata, Mansi and Qiu, Yuheng and Whittaker, Warren and Higgins, Ian and Du, Yi and Su, Shaoshu and Xu, Can and Keller, John and Karhade, Jay and Nogueira, Lucas and Saha, Sourojit and Zhang, Ji and Wang, Wenshan and Wang, Chen and Scherer, Sebastian},
    title     = {{SubT-MRS} Dataset: Pushing SLAM Towards All-weather Environments},
    booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    month     = {June},
    year      = {2024},
    pages     = {22647-22657}
}
```

