#!/bin/bash
#SBATCH --account=def-xinxin
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8         # CPU cores or threads
#SBATCH --mem=120000M              # memory per node
#SBATCH --time=0-00:30
module load cuda/12.2 gcc/12.3 opencv/4.10
source /home/xinxin/projects/def-xinxin/xinxin/envs/gs/bin/activate

python train_profile.py -s gs_data/dynvideo_male -m output/dynvideo_male_prof --train_stage 1
