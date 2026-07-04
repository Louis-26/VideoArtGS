export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST="8.6"

# dataset=artgs
# subset=sapien
# scenes=('101908' '101917' '10211' '102255' '103111' '103706_eevee' '103706_rotate' '103776_eevee' '10537' '10537_rotate' '10905' '10905_bg' '25493' '31249' '45503' '47648')
source $(git rev-parse --show-toplevel)/scripts/scene_set.sh

MODE=${1:-1}

# Parse user input
case "$MODE" in
    1)
        dataset="videoartgs"
        subset="sapien"
        scenes=("${videoartgs_sapien_scenes[@]}")
        echo "=> Running mode 1: dataset=${dataset}, subset=${subset}"
        ;;
    2)
        dataset="videoartgs"
        subset="realscan"
        scenes=("${videoartgs_realscan_scenes[@]}")
        echo "=> Running mode 2: dataset=${dataset}, subset=${subset}"
        ;;
    3)
        dataset="v2a"
        subset="sapien"
        scenes=("${v2a_sapien_scenes[@]}")
        echo "=> Running mode 3: dataset=${dataset}, subset=${subset}"
        ;;
    *)
        echo "Error: Invalid input '$MODE'. Please enter 1, 2, or 3."
        exit 1
        ;;
esac


seed=0
model_name=base_tl0.5
res=1
iter=10000

for scene in ${scenes[@]};do
    # model_path=outputs/best/${dataset}/${scene}
    model_path=outputs/${dataset}/${subset}/${scene}/${model_name}
    python render_mask.py \
        --dataset ${dataset} \
        --subset ${subset} \
        --scene_name ${scene} \
        --model_path ${model_path} \
        --resolution ${res} \
        --iteration ${iter} \
        --visualize \

done
