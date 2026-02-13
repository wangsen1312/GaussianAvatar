cd ../envs/
module load python/3.11  
virtualenv --no-download gs
source gs/bin/activate

# download the gaussian repo
cd ../code
git clone https://github.com/graphdeco-inria/gaussian-splatting.git
cd gaussian-splatting
git submodule update --init --recursive

# for simple-knn needs change on setup.py, replace setup code part
# https://github.com/graphdeco-inria/gaussian-splatting/issues/880
setup(
    name="simple_knn",
    version='0.2',
    description='simple_knn',
    author='simple_knn',
    packages=['simple_knn'],
    ext_modules=[
        CUDAExtension(
            name="simple_knn._C",
            sources=[
            "spatial.cu", 
            "simple_knn.cu",
            "ext.cpp"],
            extra_compile_args={"nvcc": [], "cxx": cxx_compiler_flags})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

# Under should work on GPU env, so use sbatch submit a job to build
sbatch modulesinstalltest.sh

# pre-download lpips model
curl -L -o /project/6098542/xinxin/envs/gs/lib/python3.11/site-packages/lpips/weights/v0.1/alex.pth https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/master/lpips/weights/v0.1/alex.pth


# three patches
1. render part
for the gaussian_renderer/__init__.py
add dgr(..antialiasing=True) for newer dgr require this

rendered_image, _, _ = rasterizer()  # return 3 parts

2. dataloader part
under scene/dataset_mono.py
# color_img = image * mask + (1 - mask) * 255
# image = Image.fromarray(np.array(color_img, dtype=np.byte), "RGB")
img_np = np.array(image, dtype=np.uint8)          # <-- add this
color_img = img_np * mask + (1 - mask) * 255
image = Image.fromarray(color_img.astype(np.uint8), "RGB")  # <-- change dtype

3. for the train.py part 
line 4 change open3d to import trimesh
line 108 -111
change from 
# for i in range(save_poitns.shape[0]):
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(save_poitns[i])
#     o3d.io.write_point_cloud(os.path.join(model.model_path, 'log',"pred_%d.ply" % i) , pcd)
for i in range(save_points.shape[0]):
    pc = trimesh.points.PointCloud(save_points[i].astype(np.float32))
    pc.export(os.path.join(out_dir, f"pred_{i}.ply"))



# Also can use interactive mode allocate GPU to build modules
salloc --account=def-xinxin --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1 --cpus-per-task=12 --mem=120gb --time=00:20:00
# module load
module load cuda/12.2 gcc/12.3 opencv/4.10
# open3d include torch==2.6.0  use >= 2.5.1 to meet the H100 needs
pip install pytorch3d==0.7.8
pip install tqdm==4.60.0 lpips==0.1.4 torchmetrics==1.8.2 trimesh==4.11.2 PyOpenGL==3.1.9  
pip install tensorboard==2.20.0 protobuf==3.20.3
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
export TORCH_CUDA_ARCH_LIST="9.0"
export MAX_JOBS=8
# install diff-gaussian
cd submodules/diff-gaussian-rasterization
python -m pip install -e . --no-build-isolation --force-reinstall -v
# install diff-gaussian
cd submodules/simple-knn
python -m pip install -e . --no-build-isolation --force-reinstall -v





