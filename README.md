# Robustness Metric

This project evaluates the robustness of trajectories by comparing them using Robustness Metrics (details in [paper](https://arxiv.org/pdf/2307.07607)). The evaluations are performed using Python scripts that interface with underlying C++ implementations via the `RobustMetricLib` Python module. 


### Description

The `eval_robustness.py` calculates translational and rotation robustness metrics 

Add your process trajectory into `config.conf`

The input reference and estimated trajectories must follow the TUM data format. For more information on this format, please refer to the [evo documentation](https://github.com/MichaelGrupp/evo/wiki/Formats).

#### Usage
```bash
python3 eval_robustness.py 
```

To run the script and generate plots:
```bash
python3 eval_robustness.py --plot
```


