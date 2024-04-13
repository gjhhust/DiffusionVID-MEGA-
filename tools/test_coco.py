from pycocotools.coco import COCO  # noqa
# from pycocotools.cocoeval import COCOeval  # noqa
from coco_eval.cocoeval import COCOeval  # noqa

anno_json = "/data1/jiahaoguo/dataset/UAVTOD_1/all/coco/test.json"
pred_json = "/data1/jiahaoguo/DiffusionVID/training_dir/vid_R_101_DiffusionVID_UAVTOD/inference/VID_UAVTOD_val/bbox.json"
anno = COCO(str(anno_json))  # init annotations api
pred = anno.loadRes(str(pred_json))  # init predictions api (must pass string, not Path)
eval = COCOeval(anno, pred, 'bbox')
# if self.is_coco:
#     eval.params.imgIds = [int(Path(x).stem) for x in self.dataloader.dataset.im_files]  # images to eval
eval.evaluate()
eval.accumulate()
eval.summarize()
print(eval.stats)