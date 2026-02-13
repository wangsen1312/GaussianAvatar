#!/bin/bash
#SBATCH --account=def-xinxin
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=8         # CPU cores or threads
#SBATCH --mem=32000M              # memory per node
#SBATCH --time=0-00:15
module load cuda/12.2 gcc/12.3 opencv/4.10
source /project/6098542/xinxin/envs/gs/bin/activate
cd /project/6098542/xinxin/code/GaussianAvatar/alliancecanenv/
# Debug prints (keep these until it’s stable)
python dgr_test.py