import numpy as np
import argparse
import yaml
from pathlib import Path
from typing import List
import os

def experiment(
        project_name: str,
        entity_name: str,
        alg_name: str,
        env_name: str,
        seed: int = 0,
        wandb_log: bool = True,
        logs_dir: str = './logs',
        save_video: bool = False,
        use_tqdm: bool = True,
        action_cost: float = -1,
        exp_hash: str = '',
):
    from maxinforl_jax.utils.train_utils import train
    ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(ROOT_PATH, 'configs.yaml')
    conf = yaml.safe_load(Path(config_path).read_text())
    # get class of environment: dmc_small, dmc_large, humanoid_bench
    env_class = conf[env_name]['env_class']
    max_steps, eval_interval = conf[env_name]['max_steps'], conf[env_name]['eval_interval']
    # select config for the agent
    conf = conf[env_class]
    conf_alg = conf[alg_name]
    conf_alg = {k: tuple(v) if isinstance(v, List) else v for k, v in conf_alg.items()}
    env_kwargs = conf['env_kwargs']
    if action_cost > 0:
        env_kwargs['action_cost'] = action_cost
    train_kwargs = conf['train_kwargs']
    train_kwargs['replay_buffer_size'] = min(train_kwargs['replay_buffer_size'], max_steps)

    train(
        project_name=project_name,
        entity_name=entity_name,
        alg_name=alg_name,
        env_name=env_name,
        seed=seed,
        alg_kwargs=conf_alg,
        env_kwargs=env_kwargs,
        wandb_log=wandb_log,
        log_config=conf_alg | env_kwargs | train_kwargs | {'alg_name': alg_name, 'env_name': env_name,
                                                           'seed': seed},
        logs_dir=logs_dir,
        save_video=save_video,
        use_tqdm=use_tqdm,
        exp_hash=exp_hash,
        max_steps=max_steps,
        eval_interval=eval_interval,
        **train_kwargs,
    )


def main(args):
    """"""
    from pprint import pprint
    print(args)
    pprint(args.__dict__)
    print('\n ------------------------------------ \n')

    """ Experiment core """
    np.random.seed(args.seed)

    experiment(
        project_name=args.project_name,
        entity_name=args.entity_name,
        alg_name=args.alg_name,
        env_name=args.env_name,
        wandb_log=bool(args.wandb_log),
        logs_dir=args.logs_dir,
        save_video=bool(args.save_video),
        use_tqdm=bool(args.use_tqdm),
        seed=args.seed,
        action_cost=args.action_cost,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MTTest')

    # general experiment args
    parser.add_argument('--project_name', type=str, default='MCTest')
    parser.add_argument('--entity_name', type=str, required=True)
    parser.add_argument('--alg_name', type=str, default='drq')
    parser.add_argument('--env_name', type=str, default='cartpole-swingup_sparse')
    parser.add_argument('--wandb_log', type=int, default=1)
    parser.add_argument('--logs_dir', type=str, default='./logs/')
    parser.add_argument('--save_video', type=int, default=0)
    parser.add_argument('--use_tqdm', type=int, default=1)
    parser.add_argument('--action_cost', type=float, default=-1)

    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    main(args)
