export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST="8.6"
# dataset=videoartgs
# subset=sapien
# scenes=('100481' '101284' '101287' '101808' '101908' '103015' '103811' '10489' '10655' '168' '25493' '30666' '31249' '45194' '45503' '45612' '47648' '8961' '9016' '1280')

# dataset=videoartgs
# subset=realscan
# scenes=('cab1' 'chair_1r' 'mac_1r' 'microwave_ego' 'cab_1r_1p' 'coffeemachine_2r' 'microwave_1r')

dataset=v2a
subset=sapien
scenes=('26657_joint_1_bg_view_0' '27267_joint_0_bg_view_1' '35059_joint_0_bg_view_0' '40453_joint_1_bg_view_1' '41083_joint_3_bg_view_1' '41510_joint_1_bg_view_1' '44781_joint_0_bg_view_1' '44817_joint_1_bg_view_1' '44962_joint_2_bg_view_1' '45001_joint_1_bg_view_0' '45132_joint_2_bg_view_0' '45146_joint_0_bg_view_0' '7265_joint_0_bg_view_0' '7265_joint_0_bg_view_1' '9987_joint_0_bg_view_1')

model_name=init
res=1
for scene in ${scenes[@]};do
    model_path=outputs/${dataset}/${subset}/${scene}/${model_name}
    python init_cano.py \
        --dataset ${dataset} \
        --subset ${subset} \
        --scene_name ${scene} \
        --model_path ${model_path} \
        --resolution ${res} \
        --iterations 20000 \
        --metric_depth_loss_weight 1.0 \
        --densify_grad_threshold 0.0004 \
        --random_bg_color
done
