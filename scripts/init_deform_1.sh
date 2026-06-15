export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST="8.6"

# dataset=videoartgs
# subset=sapien
# scenes=('100481' '101284' '101287' '101808' '101908' '103015' '103811' '10489' '10655' '168' '25493' '30666' '31249' '45194' '45503' '45612' '47648' '8961' '9016' '1280')


# subset=realscan
# scenes=('cab1' 'chair_1r' 'mac_1r' 'microwave_ego' 'cab_1r_1p' 'coffeemachine_2r' 'microwave_1r')

dataset=v2a
subset=sapien
scenes=('100068_joint_0_bg_view_0' '100071_joint_0_bg_view_1' '100072_joint_0_bg_view_0' '100087_joint_0_bg_view_0' '100092_joint_0_bg_view_0' '100106_joint_0_bg_view_1' '100128_joint_0_bg_view_1' '100133_joint_0_bg_view_0' '10040_joint_1_bg_view_0' '100664_joint_0_bg_view_1' '19179_joint_1_bg_view_1' '19855_joint_0_bg_view_1' '19898_joint_1_bg_view_1' '19898_joint_3_bg_view_0' '19898_joint_4_bg_view_1')

seed=0
model_name=init
for scene in ${scenes[@]};do
    model_path=outputs/${dataset}/${subset}/${scene}/${model_name}
    python init_deform.py \
        --dataset ${dataset} \
        --subset ${subset} \
        --scene_name ${scene} \
        --model_path ${model_path} \
        --iterations 10000 \
        --seed ${seed} \

done
