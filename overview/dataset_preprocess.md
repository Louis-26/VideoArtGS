# Dataset Preprocess
## videoartgs sapien 
These steps are already finished before packed into the dataset on HF. However, it is useful to rerun all of these steps to better understand the data preprocessing pipeline.

Just for illustration purpose, create a new folder called `new_data` and copy the first scene of dataset into this folder

### sub-step 1
- Input
    - multiview monocular video frames ${I_t}_{t=1}^T$ for each scene, as `images`
    - depth images ${D_t}_{t=1}^T$ for each scene, as `depth`
    - camera poses ${P_t}_{t=1}^T$ for each scene, as `camera.json`
- Output 
    - fused point cloud for each scene, as `point_cloud.ply`
    - detailed transformation matrix frame-by-frame for each scene, as `transforms.json`

- Purpose
Derive `point_cloud.ply`, `transforms.json` and `vis_depth` for each scene given images, depth images and camera poses

- Scripts
```bash
cd "$(git rev-parse --show-toplevel)"
conda activate videoartgs
python data_tools/process_sapien.py --data_path ./new_data/videoartgs/sapien
```

### sub-step 2
- Input
    - multiview monocular video frames ${I_t}_{t=1}^T$ for each scene, as `images`
- Output
    - joint information for each scene, as `joint_infos_vlm.json`   

- Purpose
Derive `joint_infos_vlm.json` from the multiview monocular video frames for each scene using OpenAI API, gpt-4o

- Scripts
Firstly, set up the OpenAI API key in `.env` file (a line `OPENAI_API_KEY=sk-...` at repo root)
Then run the commands following(costing 2¢ per scene)
```bash
# export every variable defined in .env into the current shell
set -a; source .env; set +a

for scene in new_data/videoartgs/sapien/*/; do
    obj=$(basename "$scene")
    python data_tools/vlm_process.py \
        --data_dir ./new_data --dataset videoartgs --subset sapien \
        --video_name "$obj" --mode video \
        --api_key "$OPENAI_API_KEY" --base_url https://api.openai.com/v1
done
```

### sub-step 3
- Input
    - multiview monocular video frames ${I_t}_{t=1}^T$ for each scene, as `images`
    - depth images ${D_t}_{t=1}^T$ for each scene, as `depth`
    - transformation matrices frame-by-frame for each scene, as `transforms.json`
    - joint information from VLM for each scene, as `joint_infos_vlm.json`
- Output
    - 3D tracking trajectories(7700 coordinates+visibility) frame-by-frame for each scene, as `filtered.npz`
    - joint information for each scene, as `joint_infos.json`

- Purpose
Derive `filtered.npz` and `joint_infos.json` for each scene


- Scripts
install TAPIP3D and download the TAPIP3D checkpoint model
```bash
mkdir -p third_party/TAPIP3D/checkpoints
wget -O third_party/TAPIP3D/checkpoints/tapip3d_final.pth \
    https://huggingface.co/zbww/tapip3d/resolve/main/tapip3d_final.pth
```

```bash
for scene in new_data/videoartgs/sapien/*/; do
    obj=$(basename "$scene")
    python data_tools/extract_tapip3d_track.py \
        --data_dir ./new_data/videoartgs/sapien \
        --video_name "$obj" \
        --tapip3d_dir ./third_party/TAPIP3D \
        --reprocess
    rm -f "${scene}${obj}".n*.npz "${scene}filtered_vis.npz"
done

```

After all sub steps, use the following to verify.
```bash
python debug/compare_files_vag_sp.py
```

## videoartgs realscan
### sub-step 0
This step is usually omitted as the `images` folder is already provided
- Input
    - raw scene video(end with .mp4)
- Output
    - multiview monocular video frames ${I_t}_{t=1}^T$ for each scene, as `images`
- Purpose
    Extract multiview monocular video frames from the raw scene video.
- Scripts
```bash
# extract frames from the raw scene video, swap subset into `sapien` or `realscan`, and swap 168 into the actual scene name
for video_path in ./new_data/videos/*.mp4; do
    video_name=$(basename "$video_path" .mp4)
    python data_tools/extract_frames.py \
        --video_dir ./new_data/videos \
        --video_name "$video_name" \
        --data_dir ./new_data/videoartgs/realscan \
        --interval 2 \
        --resize 2
done
```

### sub-step 1
- Input
    - multiview monocular video frames ${I_t}_{t=1}^T$ for each scene, as `images`
- Output
    - st2_result.npz for each scene, as `st2_result.npz`
- Purpose
Derive `st2_result.npz` for each scene given images using SpatialTrackerV2, storing the information of depth, intrinsics, extrinsics, and depth pixel confidence
- Scripts
Firstly, create the conda environment and install the dependencies
```bash
conda create -n st2 python=3.11 -y
conda activate st2
python -m pip install \
    torch==2.4.1 \
    torchvision==0.19.1 \
    torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu124
pip install -r third_party/SpatialTrackerV2/requirements.txt
```
Then run the spatial tracker for each scene
```bash
# run the spatial tracker
conda activate st2
for scene in new_data/videoartgs/realscan/*/; do
    obj=$(basename "$scene")
    python third_party/SpatialTrackerV2/infer_st2.py \
        --data_dir ./new_data/videoartgs/realscan \
        --video_name "$obj" \
        --reprocess
done

# visualization
python third_party/SpatialTrackerV2/tapip3d_viz.py new_data/videoartgs/realscan/microwave_ego/st2_result.npz
```


### sub-step 2
- Input
    - multiview monocular video frames ${I_t}_{t=1}^T$ for each scene, as `images`
- Output
    - masks for each scene, as `masks.npy`
    - consolidated dataset as `data.npz`
    - initialized point cloud for each scene, as `point_cloud.ply`
- Purpose
    - Derive masks and consolidated dataset for each scene from SAM2
- Scripts
Configure the sam2 environment
```bash
# download the SAM2 checkpoint for sam2.1_hiera_large.pt
cd "$(git rev-parse --show-toplevel)/third_party"
git clone --depth 1 https://github.com/facebookresearch/sam2.git
rm -rf sam2/.git
cd sam2
conda create -n sam2 python=3.11 -y
conda activate sam2

python -m pip install --upgrade pip

python -m pip install \
    torch==2.5.1 \
    torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124

python -m pip install numpy==1.26.4 tqdm hydra-core iopath pillow opencv-python
cd "$(git rev-parse --show-toplevel)/third_party/sam2"
python -m pip install -e . --no-deps --no-build-isolation

cd "$(git rev-parse --show-toplevel)/third_party/sam2/checkpoints"
bash ./download_ckpts.sh
find . -maxdepth 1 ! -name "sam2.1_hiera_large.pt" ! -name "download_ckpts.sh" ! -name "." -exec rm -rf {} +
```

Complete foreground mask segmentation and run the spatial tracker for each scene
```bash
conda activate sam2
# foreground mask segmentation
for scene in new_data/videoartgs/realscan/*/; do
    obj=$(basename "$scene")
    python data_tools/gen_sam2_masks.py \
        --image_dir "./new_data/videoartgs/realscan/$obj/images" \
        --output_dir "./new_data/videoartgs/realscan/$obj/masks" \
        --checkpoint ./third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
        --config configs/sam2.1/sam2.1_hiera_l.yaml \
        --frame_idx 0 \
        --bbox_padding 0 \
        --save_vis
done

conda activate videoartgs
# run the spatial tracker for 
for scene in new_data/videoartgs/realscan/*/; do
    obj=$(basename "$scene")
    python data_tools/process_vggt.py \
        --data_dir ./new_data/videoartgs/realscan \
        --video_name "$obj" \
        --model st2 \
        --reprocess
done
```

### sub-step 3
- Input
    - multiview monocular video frames ${I_t}_{t=1}^T$ for each scene, as `images`
- Output
    - coarse joint information from VLM for each scene, as `joint_infos_vlm.json`
- Purpose
Derive `joint_infos_vlm.json` from the multiview monocular video frames for each scene using OpenAI API, gpt-4o
- Scripts
```bash
# export every variable defined in .env into the current shell
set -a; source .env; set +a

for scene in new_data/videoartgs/realscan/*/; do
    obj=$(basename "$scene")
    python data_tools/vlm_process.py \
        --data_dir ./new_data --dataset videoartgs --subset realscan \
        --video_name "$obj" --mode video \
        --api_key "$OPENAI_API_KEY" --base_url https://api.openai.com/v1
done
```

### sub-step 4
- Input
    - multiview monocular video frames ${I_t}_{t=1}^T$ for each scene, as `images`
    - depth images ${D_t}_{t=1}^T$ for each scene, as `depth`
    - joint information from VLM for each scene, as `joint_infos_vlm.json`

- Output
    - 3D tracking trajectories(7700 coordinates+visibility) frame-by-frame for each scene, as `filtered.npz`
    - joint information for each scene, as `joint_infos.json`
- Purpose
Derive `filtered.npz` and `joint_infos.json` for each scene
- Scripts
Generate the 3D tracking trajectories and joint information for each scene using TAPIP3D
```bash
for scene in new_data/videoartgs/realscan/*/; do
    obj=$(basename "$scene")
    python data_tools/extract_tapip3d_track.py \
        --data_dir ./new_data/videoartgs/realscan \
        --video_name "$obj" \
        --tapip3d_dir ./third_party/TAPIP3D \
        --reprocess
    # rm -f "${scene}${obj}".n*.npz "${scene}filtered_vis.npz"
done
```
Visualization
```bash
python third_party/TAPIP3D/visualize.py ./new_data/videoartgs/realscan/microwave_ego/filtered_vis.npz
```

After all sub steps, use the following to verify.
```bash
python debug/compare_files_vag_rs.py
```

## v2a sapien
### sub-step 0
Download the video2articulation dataset from HF
```bash
mkdir -p raw_data/video2articulation
hf auth login
hf download \
    3dlg-hcvc/video2articulation \
    --repo-type dataset \
    --include "sim_data/origin_data.tar.gz.part*" \
    --include "partnet_mobility_data_split.yaml" \
    --include "new_partnet_mobility_dataset_correct_intr_meta.json" \
    --local-dir ./raw_data/video2articulation

cd raw_data/video2articulation/sim_data

cat origin_data.tar.gz.part* > origin_data.tar.gz
pv origin_data.tar.gz | tar -xz
```

Download PartNet-Mobility dataset from HF
```bash
cd "$(git rev-parse --show-toplevel)"

mkdir -p raw_data/partnet-mobility

hf download \
    sapien-sim/PartNetMobility \
    --repo-type dataset \
    --local-dir ./raw_data/partnet-mobility
```
pending for approval

### sub-step 1



