python -m torch.distributed.launch \
    --nproc_per_node=3 \
    tools/train_net.py \
    --master_port=$((RANDOM + 10000)) \
    --config-file configs/MEGA/vid_R_101_C4_MEGA_1x_UAVTOD.yaml \
    OUTPUT_DIR training_dir/vid_R_101_C4_MEGA_1x_UAVTOD

python tools/train_net.py \
    --config-file configs/FGFA/vid_R_101_C4_FGFA_1x_gaode_4.yaml \
    OUTPUT_DIR training_dir/vid_R_101_C4_FGFA_1x_gaode_4


python tools/test_prediction.py \
        --config-file configs/vid_R_101_DiffusionVID_UAVTOD.yaml \
        --prediction training_dir/vid_R_101_DiffusionVID_UAVTOD/model_final.pth


python tools/test_net.py \
        --config-file  configs/vid_R_101_DiffusionVID_UAVTOD.yaml \
        MODEL.WEIGHT training_dir/vid_R_101_DiffusionVID_UAVTOD/model_final.pth \
        DTYPE float16