python -m torch.distributed.launch \
    --nproc_per_node=3 \
    tools/train_net.py \
    --master_port=$((RANDOM + 10000)) \
    --config-file configs/vid_R_101_DiffusionVID_gaode_4.yaml \
    OUTPUT_DIR training_dir/vid_R_101_DiffusionVID_gaode_4

python tools/train_net.py \
    --config-file configs/vid_R_101_DiffusionVID_gaode_4.yaml \
    OUTPUT_DIR training_dir/vid_R_101_DiffusionVID_gaode_4

OMP_NUM_THREADS=1 python -m torch.distributed.run --use_env --nproc_per_node=2 train_net.py --master_port=12505 --config-file configs/vid_R_101_DiffusionVID_visdrone.yaml OUTPUT_DIR training_dir/vid_R_101_DiffusionVID_visdrone


CUDA_VISIBLE_DEVICES=0,1,2 PORT=20544 ./tools/dist_train.sh configs/vid_R_101_DiffusionVID_gaode_4.yaml training_dir/vid_R_101_DiffusionVID_gaode_4 3


python demo/demo.py configs/vid_R_101_DiffusionVID.yaml \
    pth/DiffusionVID_R101.pth \
    --suffix ".jpg" \
    --visualize-path /data1/jiahaoguo/dataset/gaode_4_all/images/Scene1 \
    --output-folder visualization --output-video

