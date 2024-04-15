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


CUDA_DEVICE_ORDER="PCI_BUS_ID" CUDA_VISIBLE_DEVICES=0 python -c "import torch;print(torch.cuda.is_available());"






# MEGA 1

DONE (t=5.62s).                                                                                                                                                                 
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.078                                                                                                 
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.187                                                                                                 
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.051                                                                                                 
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.005                                                                                                 
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.100                                                                                                 
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.535                                                                                                 
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.017                                                                                                 
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.095                                                                                                 
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.128                                                                                                 
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.052                                                                                                 
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.123                                                                                                 
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.536                                                                                                 
2024-04-15 11:03:07,305 mega_core.inference INFO:                                                                                                                               
Task: bbox                                                                                                                                                                      
AP, AP50, AP75, APs, APm, APl                                                                                                                                                   
0.0785, 0.1872, 0.0513, 0.0046, 0.1000, 0.5354                                                                                                                                  
                                                                                                                                                                                
INFO:mega_core.inference:                                                                                                                                                       
Task: bbox                                                                                                                                                                      
AP, AP50, AP75, APs, APm, APl                                                                                                                                                   
0.0785, 0.1872, 0.0513, 0.0046, 0.1000, 0.5354 