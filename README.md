# Robustness_Metric

This project evaluates the robustness of trajectories by comparing them using Robustness Metrics (details in [paper](https://arxiv.org/pdf/2307.07607)). The evaluations are performed using Python scripts that interface with underlying C++ implementations via the `RobustMetricLib` Python module. 


### Description

The `eval_robustness.py` calculates translational and rotation robustness metrics 

Add your process trajectory into `config.conf`

#### Usage
```bash
python3 eval_robustness.py 


