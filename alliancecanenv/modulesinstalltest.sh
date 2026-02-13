#!/bin/bash
#SBATCH --account=def-xinxin
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=8         # CPU cores or threads
#SBATCH --mem=32000M              # memory per node
#SBATCH --time=0-00:30
module purge
module load StdEnv/2023
module load cuda/12.2 gcc/12.3 opencv/4.10
source /project/6098542/xinxin/envs/gs/bin/activate
pip install pytorch3d==0.7.8 tqdm==4.60.0 lpips==0.1.4 torchmetrics==1.8.2 trimesh==4.11.2 PyOpenGL==3.1.9  
pip install tensorboard==2.20.0 protobuf==3.20.3
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
export TORCH_CUDA_ARCH_LIST="9.0"
export MAX_JOBS=8
cd /project/6098542/xinxin/code/gaussian-splatting/submodules/simple-knn/
python -m pip uninstall -y simple_knn simple-knn || true
python -m pip install -e . --no-build-isolation --force-reinstall -v
cd /project/6098542/xinxin/code/gaussian-splatting/submodules/diff-gaussian-rasterization/
python -m pip uninstall -y diff_gaussian_rasterization || true
python -m pip install -e . --no-build-isolation --force-reinstall -v
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
cd /project/6098542/xinxin/code/GaussianAvatar/alliancecanenv/
python dgrknn_test.py