#!/bin/bash
#SBATCH --account=def-xinxin
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --cpus-per-task=8        
#SBATCH --mem=32000M             
#SBATCH --time=0-04:00
module load cuda/12.2 gcc/12.3 opencv/4.10
source /project/6098542/xinxin/envs/gs/bin/activate
#python train.py -s gs_data/dynvideo_female -m output/dynvideo_female --train_stage 1
python eval.py -s gs_data/dynvideo_male -m output/dynvideo_male --epoch 180
python render_novel_pose.py -s gs_data/dynvideo_male -m output/dynvideo_male --epoch 180

python eval.py -s gs_data/m4c_processed -m output/m4c_processed --epoch 180
python render_novel_pose.py -s gs_data/m4c_processed -m output/m4c_processed --epoch 180