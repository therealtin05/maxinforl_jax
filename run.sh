#!/bin/bash

export MUJOCO_GL=egl

# for i in {0..9}; do
#   python examples/state_based/experiment.py \
#       --project_name maxinforl \
#       --entity_name therealtin-uit \
#       --alg_name sac \
#       --env_name walker-run \
#       --wandb_log 1 \
#       --seed $i \
#       --save_video 1
# done

# for i in {9..9}; do
#   python examples/state_based/experiment.py \
#       --project_name maxinforl \
#       --entity_name therealtin-uit \
#       --alg_name maxinfosac \
#       --env_name walker-run \
#       --wandb_log 1 \
#       --seed $i \
#       --save_video 1
# done

python examples/state_based/experiment.py \
    --project_name maxinforl \
    --entity_name therealtin-uit \
    --alg_name maxinfosac \
    --env_name humanoid_bench/h1-run-v0 \
    --wandb_log 1 \
    --seed 5 \
    --save_video 1

# python examples/vision_based/experiment.py \
#   --project_name maxinforl \
#   --entity_name therealtin-uit \
#   --alg_name drq \
#   --env_name cartpole-swingup_sparse \
#   --wandb_log 1 \
#   --seed 1