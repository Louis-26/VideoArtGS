current progress: finish step 1-5
# step 1: create environment
```bash
cd $(git rev-parse --show-toplevel)
# bash ./install.sh
conda create -n videoartgs python=3.10 -y
conda activate videoartgs
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 "xformers>=0.0.27" --index-url https://download.pytorch.org/whl/cu124
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.1+cu124.html
pip install -r requirements.txt

# add this line
pip install "setuptools<70" wheel

# continue
pip install git+https://github.com/facebookresearch/pytorch3d.git@stable --no-build-isolation
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch --no-build-isolation

# build pointnet_lib for nearest farthest point sampling
cd utils/pointnet_lib
python setup.py install
cd ../..

# compile pointops2
cd third_party/TAPIP3D/third_party/pointops2
LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH python setup.py install
cd ../../../..


# notice: use gcc>10, cuda 12.5
# gslpat
pip install git+https://github.com/nerfstudio-project/gsplat.git --no-build-isolation

# simple-knn
pip install git+https://gitlab.inria.fr/bkerbl/simple-knn.git --no-build-isolation

conda install -c conda-forge ffmpeg -y

```

# step 2: data preparation
```bash
cd $(git rev-parse --show-toplevel)
pip install -U "huggingface_hub[cli]"
if [[ ":$PATH:" != *":$CONDA_PREFIX/bin:"* ]]; then
    export PATH="$CONDA_PREFIX/bin:$PATH"
fi
## download the dataset
hf download YuLiu/VideoArtGS-Data --repo-type dataset --local-dir ./data 

## reconstruct the dataset folder
cd "$(git rev-parse --show-toplevel)/data"
unzip -q VideoArtGS-20.zip
unzip -q realscan.zip

mkdir -p videoartgs
mv realscan videoartgs/
mv VideoArtGS-20 videoartgs/sapien
mkdir sapien
mv *_joint_*_bg_view_* sapien/

cd "$(git rev-parse --show-toplevel)/data"
rm -rf *.zip
### alternatively
cd SLURM_execution/SLURM_script
sbatch data_prepare.sh
```

# step 3: train
```bash
cd "$(git rev-parse --show-toplevel)"
echo "Running init_cano.sh"
bash scripts/init_cano.sh
echo "Running init_deform.sh"
bash scripts/init_deform.sh
echo "Running train.sh"
bash scripts/train.sh
```
alternatively, use SLURM script to run the training
```bash
cd "$(git rev-parse --show-toplevel)/SLURM_execution/SLURM_script"
sbatch train.sh
```

## notice
Do the following switch before running the scripts for `scripts/init_cano.sh`, `scripts/init_deform.sh`, and `scripts/train.sh`:
- VideoArtGS-20: select `videoartgs` dataset, and comment `v2a` part
- Video2Articulation-S: select `v2a` dataset, and comment `videoartgs` part


# step 4: render and evaluate
```bash
cd "$(git rev-parse --show-toplevel)"
echo "Running render.sh"
bash scripts/render.sh
echo "Running eval.sh"
bash scripts/eval.sh
```
alternatively, use SLURM script to run the training
```bash
cd "$(git rev-parse --show-toplevel)/SLURM_execution/SLURM_script"
JOB1_ID=$(sbatch --parsable render.sh)
sbatch --parsable --dependency=afterok:$JOB1_ID eval.sh
```

# step 5: visualization
## download checkpoints for TAPIP3D
```bash
cd "$(git rev-parse --show-toplevel)"
mkdir -p third_party/TAPIP3D/checkpoints
cd third_party/TAPIP3D/checkpoints
wget https://huggingface.co/zbww/tapip3d/resolve/main/tapip3d_final.pth
```


```bash
cd $(git rev-parse --show-toplevel)

mkdir -p logs
for d in data/videoartgs/realscan/*/; do
    obj=$(basename "$d")
    if [[ -f "$d/filtered_vis.npz" ]]; then
        echo "[skip] $obj already done"
        continue
    fi
    echo "=== $obj ===" | tee -a logs/extract_realscan.log
    python data_tools/extract_tapip3d_track.py \
        --data_dir ./data/videoartgs/realscan \
        --video_name "$obj" \
        --reprocess 2>&1 | tee -a logs/extract_realscan.log
done
```

alternatively, use SLURM script to run the visualization
```bash
cd "$(git rev-parse --show-toplevel)/SLURM_execution/SLURM_script"
sbatch visualization.sh
```
## then get the visualization results
```bash
cd "$(git rev-parse --show-toplevel)/vis_utils"
python vis_init.py
jupyter nbconvert --to notebook --execute --inplace vis_videoartgs.ipynb
```

# step 6: export URDF and USD files
```bash
cd "$(git rev-parse --show-toplevel)"
python vis_utils/json2urdf.py

```


install blender
```bash
wget https://download.blender.org/release/Blender3.6/blender-3.6.5-linux-x64.tar.xz
tar -xvf blender-3.6.5-linux-x64.tar.xz
```

```bash
cd "$(git rev-parse --show-toplevel)"
../blender-3.6.5-linux-x64/blender -b -P vis_utils/ply2glb.py
```