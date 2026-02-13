#!/bin/bash
#SBATCH --account=def-xinxin
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8         # CPU cores or threads
#SBATCH --mem=60000M              # memory per node
#SBATCH --time=0-03:30
module load cuda/12.2 gcc/12.3 opencv/4.10
source /home/xinxin/projects/def-xinxin/xinxin/envs/gs/bin/activate

python train.py -s gs_data/dynvideo_male -m output/dynvideo_male --train_stage 1
python eval.py -s gs_data/dynvideo_male -m output/dynvideo_male --epoch 180
python render_novel_pose.py -s gs_data/dynvideo_male -m output/dynvideo_male --epoch 180
