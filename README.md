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
