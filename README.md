# MaxInfoRL: Boosting exploration in RL through information gain maximization

A jax implementation of [MaxInfoRL][paper], a simple, flexible, and scalable class of reinforcement learning algorithms that enhance exploration in RL by automatically combining intrinsic and extrinsic rewards. For a Pytorch implementation, visit this [Pytorch repository][torchrepo].

To learn more:

- [Research paper][paper]
- [Project website][website]
- [Twitter summary][tweet]

## MaxInfoRL

MaxInfoRL boosts exploration in RL by combining extrinsic rewards with intrinsic 
exploration bonuses derived from information gain of the underlying MDP.
MaxInfoRL naturally trades off maximization of the value function with that of the entropy over states, rewards,
and actions. MaxInfoRL is very general and can be combined with a variety
of off-policy model-free RL methods for continuous state-action spaces. We provide implementations of 
**MaxInfoSac, MaxInfoREDQ, MaxInfoDrQ, MaxInfoDrQv2**.

# Instructions

## Installation

```sh
pip install -e .
```

### Remark: 
The above command does not install the GPU version of [JAX][jax]. Please manually install the GPU version if needed.
For instance using 

```sh
pip install -U "jax[cuda12]"
```

To run HumanoidBench experiments, please install the benchmark dependencies following the instructions in the [original repo](https://github.com/carlosferrazza/humanoid-bench).

## Training

Training script:

1. State-based

```sh
python examples/state_based/experiment.py \
  --project_name maxinforl \
  --entity_name wandb_entity_name \
  --alg_name maxinfosac \
  --env_name cartpole-swingup_sparse \
  --wandb_log 1
```

For the state based experiments you can run sac, redq, maxinfosac or maxinforedq by specifying the alg_name flag.


2. Vision-based

```sh
python examples/vision_based/experiment.py \
  --project_name maxinforl \
  --entity_name wandb_entity_name \
  --alg_name maxinfodrq \
  --env_name cartpole-swingup_sparse \
  --wandb_log 1
```
For the vision based experiments you can run drq, drqv2, maxinfodrq or maxinfodrqv2 by specifying the alg_name flag.

All hyperparameters are listed in the `examples/state_based//configs.yaml` and `examples/vision_based//configs.yaml` 
files. You can override them if needed.

[jax]: https://github.com/google/jax#pip-installation-gpu-cuda
[paper]: https://arxiv.org/abs/2412.12098
[website]: https://sukhijab.github.io/projects/maxinforl/
[tweet]: https://sukhijab.github.io/
[torchrepo]: https://github.com/sukhijab/maxinforl_torch

## Custom environments

This repo relies on jaxrl to load environments, natively supporting Gym and DM Control environments. If your environment is registered in Gym, you can directly use it (just adjust the configs.yaml file accordingly). 

# Citation
If you find MaxInfoRL useful for your research, please cite this work:
```
@article{sukhija2024maxinforl,
  title={MaxInfoRL: Boosting exploration in reinforcement learning through information gain maximization},
  author={Sukhija, Bhavya and Coros, Stelian and Krause, Andreas and Abbeel, Pieter and Sferrazza, Carmelo},
  journal={arXiv preprint arXiv:2412.12098},
  year={2024}
}
```

# References
This codebase contains some files adapted from other sources:
* jaxrl (original repo): https://github.com/ikostrikov/jaxrl/tree/main
* jaxrl (fork): https://github.com/sukhijab/jaxrl