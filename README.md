# GraphSem: Robust Multi-Agent Reinforcement Learning via Semantic-Graph Communication

## Overview
This repository contains the official implementation of **GraphSem**, a semantic-graph communication framework for robust multi-agent reinforcement learning (MARL).  
GraphSem is designed to improve coordination under **stochastic partial observability** by combining:
- **Transformer-based semantic encoding** to extract task-relevant features from noisy observations.  
- **Dynamic communication weighting** to selectively transmit critical messages among agents.  
- **Graph convolution with attention** to fuse distributed information into an expressive global representation.  

To evaluate robustness, we introduce controlled perturbations such as **observation noise** and **randomized initial states**, enabling reproducible experiments under uncertainty.  
Experiments on **SMAC** and **Traffic Junction** benchmarks show that GraphSem achieves **up to 30.4% higher win rates** than state-of-the-art baselines, with superior **sample efficiency** and **coordination stability**.

---
# Instructions

This code is implemented based on [PyMARL](https://github.com/oxwhirl/pymarl), 
and the running instructions are similar to that in the original project. 

We incorporate `Traffic Junction` and `SMAC` in this code, 
which you can experiment with config instructions like `env-config=messy_sc2/messy_traffic_junction/messy_traffic_junction_hard`. 
Furthermore, the environmental noise parameters correspond to the two parameters, failure_obs_prob and randomize_initial_state, in files such as `src\config\envs\messy_sc2.yaml`. 
We can modify them to verify the performance of the algorithm under different combinations of noise parameters. 
It should be noted that the perturbation variance parameter of the GraphSem algorithm needs to modify the obs_tamper parameter in the `src\config\algs\Graphsem.yaml` file.

For example, you can run `GraphSem` on traffic_junction (medium) by using:
```sh
python3 src/main.py --config=GraphSem --env-config=messy_traffic_junction with t_max=3005000
```
and you can run GraphSem in the 25m map of SMAC:
```sh
python3 src/main.py --config=GraphSem --env-config=messy_sc2 with env_args.map_name=25m t_max=5005000
```
If you want to test the ablation effect of the algorithm, you can find the Yaml files of several ablation algorithms of GraphSem under the configs folder.

This code will use tensorboard and save model by default, which will be saved in `./results/`

---
# Prerequisites

This project has been tested with the following environment setup.  
The Python environment includes **all dependencies required to run GraphSem as well as all baseline comparison algorithms** (e.g., QMIX, AERIAL, GACG, CAMA, DFAC, SIDE).  

- **Python:** 3.8+  
- **Core Libraries:**  
  - `torch==2.4.1+cu118`, `torchvision==0.19.1+cu118`, `torchaudio==2.4.1+cu118`  
  - `tensorboard==2.14.0`, `tensorboardX==2.6.2.2`  
  - `scipy==1.10.1`, `numpy==1.23.1`, `pandas==2.0.3`, `matplotlib==3.7.5`, `seaborn==0.13.2`  
  - `scikit-learn==1.3.2`  
- **Multi-agent RL / Environments:**  
  - `SMAC==1.0.0`, `PySC2==4.0.0`, `s2clientprotocol==5.0.14.93333.0`, `s2protocol==5.0.14.93333.0`  
  - `gym==0.23.1`, `gymnasium==1.0.0`, `atari-py==0.2.9`, `ale-py==0.10.1`  
  - `gfootball==2.10.2`, `pybullet==3.2.7`  
- **Communication / Graph Libraries:**  
  - `torch-scatter==2.1.2+pt24cu118`, `graphviz==0.20.3`, `networkx==3.0`  
- **Experiment Management:**  
  - `sacred==0.8.7`, `tensorboard-logger==0.1.0`  
- **Other Useful Tools:**  
  - `easydict==1.13`, `pyyaml==6.0.2`, `tqdm==4.67.1`, `rich==14.0.0`, `tabulate==0.9.0`

> ⚠️ **Note:** This environment has been verified to support not only **GraphSem** but also **all baseline algorithms** used in the experiments, ensuring reproducibility and consistency across comparisons.

---
# Acknowledgements

Special thanks to **Sitong Shen** for developing and sharing the open-source code that served as the foundation of this research.  Her dedication and contributions were essential to the successful implementation of GraphSem.  Without her support, this work would not have been possible.  
