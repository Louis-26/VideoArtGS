export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST="8.6"

# dataset=videoartgs
# subset=sapien
# scenes=('100481' '101284' '101287' '101808' '101908' '103015' '103811' '10489' '10655' '168' '25493' '30666' '31249' '45194' '45503' '45612' '47648' '8961' '9016' '1280')


# subset=realscan
# scenes=('cab1' 'chair_1r' 'mac_1r' 'microwave_ego' 'cab_1r_1p' 'coffeemachine_2r' 'microwave_1r')

dataset=v2a
subset=sapien
scenes=('20745_joint_0_bg_view_1' '20985_joint_1_bg_view_0' '22241_joint_0_bg_view_1' '22339_joint_0_bg_view_1' '22367_joint_0_bg_view_0' '22367_joint_2_bg_view_0' '22433_joint_0_bg_view_0' '22433_joint_0_bg_view_1' '22433_joint_1_bg_view_0' '23372_joint_1_bg_view_1' '23724_joint_0_bg_view_1' '23724_joint_2_bg_view_0' '23807_joint_1_bg_view_1' '26525_joint_0_bg_view_1' '26608_joint_0_bg_view_1')
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
