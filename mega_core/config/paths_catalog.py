# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Centralized catalog of paths."""

import os
from copy import deepcopy

class DatasetCatalog(object):
    DATA_DIR = "datasets"
    DATASETS = {
        "coco_2017_train": {
            "img_dir": "coco/train2017",
            "ann_file": "coco/annotations/instances_train2017.json"
        },
        "coco_2017_val": {
            "img_dir": "coco/val2017",
            "ann_file": "coco/annotations/instances_val2017.json"
        },
        "coco_2014_train": {
            "img_dir": "coco/train2014",
            "ann_file": "coco/annotations/instances_train2014.json"
        },
        "coco_2014_val": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/instances_val2014.json"
        },
        "coco_2014_minival": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/instances_minival2014.json"
        },
        "coco_2014_valminusminival": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/instances_valminusminival2014.json"
        },
        "keypoints_coco_2014_train": {
            "img_dir": "coco/train2014",
            "ann_file": "coco/annotations/person_keypoints_train2014.json",
        },
        "keypoints_coco_2014_val": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/person_keypoints_val2014.json"
        },
        "keypoints_coco_2014_minival": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/person_keypoints_minival2014.json",
        },
        "keypoints_coco_2014_valminusminival": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/person_keypoints_valminusminival2014.json",
        },
        "voc_2007_train": {
            "data_dir": "voc/VOC2007",
            "split": "train"
        },
        "voc_2007_train_cocostyle": {
            "img_dir": "voc/VOC2007/JPEGImages",
            "ann_file": "voc/VOC2007/Annotations/pascal_train2007.json"
        },
        "voc_2007_val": {
            "data_dir": "voc/VOC2007",
            "split": "val"
        },
        "voc_2007_val_cocostyle": {
            "img_dir": "voc/VOC2007/JPEGImages",
            "ann_file": "voc/VOC2007/Annotations/pascal_val2007.json"
        },
        "voc_2007_test": {
            "data_dir": "voc/VOC2007",
            "split": "test"
        },
        "voc_2007_test_cocostyle": {
            "img_dir": "voc/VOC2007/JPEGImages",
            "ann_file": "voc/VOC2007/Annotations/pascal_test2007.json"
        },
        "voc_2012_train": {
            "data_dir": "voc/VOC2012",
            "split": "train"
        },
        "voc_2012_train_cocostyle": {
            "img_dir": "voc/VOC2012/JPEGImages",
            "ann_file": "voc/VOC2012/Annotations/pascal_train2012.json"
        },
        "voc_2012_val": {
            "data_dir": "voc/VOC2012",
            "split": "val"
        },
        "voc_2012_val_cocostyle": {
            "img_dir": "voc/VOC2012/JPEGImages",
            "ann_file": "voc/VOC2012/Annotations/pascal_val2012.json"
        },
        "voc_2012_test": {
            "data_dir": "voc/VOC2012",
            "split": "test"
            # PASCAL VOC2012 doesn't made the test annotations available, so there's no json annotation
        },

        ##############################################
        # These ones are deprecated, should be removed
        "cityscapes_fine_instanceonly_seg_train_cocostyle": {
            "img_dir": "cityscapes/images",
            "ann_file": "cityscapes/annotations/instancesonly_filtered_gtFine_train.json"
        },
        "cityscapes_fine_instanceonly_seg_val_cocostyle": {
            "img_dir": "cityscapes/images",
            "ann_file": "cityscapes/annotations/instancesonly_filtered_gtFine_val.json"
        },
        "cityscapes_fine_instanceonly_seg_test_cocostyle": {
            "img_dir": "cityscapes/images",
            "ann_file": "cityscapes/annotations/instancesonly_filtered_gtFine_test.json"
        },
        ##############################################

        "cityscapes_poly_instance_train": {
            "img_dir": "cityscapes/leftImg8bit/",
            "ann_dir": "cityscapes/gtFine/",
            "split": "train",
            "mode": "poly",
        },
        "cityscapes_poly_instance_val": {
            "img_dir": "cityscapes/leftImg8bit",
            "ann_dir": "cityscapes/gtFine",
            "split": "val",
            "mode": "poly",
        },
        "cityscapes_poly_instance_minival": {
            "img_dir": "cityscapes/leftImg8bit",
            "ann_dir": "cityscapes/gtFine",
            "split": "val",
            "mode": "poly",
            "mini": 10,
        },
        "cityscapes_mask_instance_train": {
            "img_dir": "cityscapes/leftImg8bit/",
            "ann_dir": "cityscapes/gtFine/",
            "split": "train",
            "mode": "mask",
        },
        "cityscapes_mask_instance_val": {
            "img_dir": "cityscapes/leftImg8bit",
            "ann_dir": "cityscapes/gtFine",
            "split": "val",
            "mode": "mask",
        },
        "cityscapes_mask_instance_minival": {
            "img_dir": "cityscapes/leftImg8bit",
            "ann_dir": "cityscapes/gtFine",
            "split": "val",
            "mode": "mask",
            "mini": 10,
        },

        ##############################################
        "VID_visdrone_train_2": {
            "img_dir": "images/train",
            "anno_path": "annotations/ilsvrc2015_Lables/Annotations/train",
            "img_index": "annotations/ilsvrc2015_Lables/ImageSets/train_2.txt"
        },
        "VID_visdrone_train": {
            "img_dir": "images/train",
            "anno_path": "annotations/ilsvrc2015_Lables/Annotations/train",
            "img_index": "annotations/ilsvrc2015_Lables/ImageSets/train.txt"
        },
        "VID_visdrone_val": {
            "img_dir": "images/val",
            "anno_path": "annotations/ilsvrc2015_Lables/Annotations/val",
            "img_index": "annotations/ilsvrc2015_Lables/ImageSets/val.txt",
            "coco_json":"/data1/jiahaoguo/dataset/VisDrone2019-VID/annotations/val_video.json"
        },
        
        "VID_gaode_4_train_2": {
            "dataset_dir": "/data1/jiahaoguo/dataset/gaode_4_all",
            "img_dir": "images",
            "anno_path": "gaode_4_vid/Annotations/train",
            "img_index": "gaode_4_vid/ImageSets/train_2.txt",
            "image_pattern_mode" : "gaode_4",
            "all_train_txt": "/data1/jiahaoguo/dataset/gaode_4_all/gaode_4_vid/ImageSets/train.txt"
        },
        "VID_gaode_4_val": {
            "dataset_dir": "/data1/jiahaoguo/dataset/gaode_4_all",
            "img_dir": "images",
            "anno_path": "gaode_4_vid/Annotations/test",
            "img_index": "gaode_4_vid/ImageSets/test.txt",
            "image_pattern_mode" : "gaode_4",
            "coco_json":"/data1/jiahaoguo/dataset/gaode_4_all/annotations/test_half.json"
        }, 

        "DET_train_30classes": {
            "img_dir": "ILSVRC2015/Data/DET",
            "anno_path": "ILSVRC2015/Annotations/DET",
            "img_index": "ILSVRC2015/ImageSets/DET_train_30classes.txt"
        },
        "VID_train_15frames": {
            "img_dir": "ILSVRC2015/Data/VID",
            "anno_path": "ILSVRC2015/Annotations/VID",
            "img_index": "ILSVRC2015/ImageSets/VID_train_15frames.txt"
        },
        "VID_train_every10frames": {
            "img_dir": "ILSVRC2015/Data/VID",
            "anno_path": "ILSVRC2015/Annotations/VID",
            "img_index": "ILSVRC2015/ImageSets/VID_train_every10frames.txt"
        },
        "VID_val_frames": {
            "img_dir": "ILSVRC2015/Data/VID",
            "anno_path": "ILSVRC2015/Annotations/VID",
            "img_index": "ILSVRC2015/ImageSets/VID_val_frames.txt"
        },
        "VID_val_videos": {
            "img_dir": "ILSVRC2015/Data/VID",
            "anno_path": "ILSVRC2015/Annotations/VID",
            "img_index": "ILSVRC2015/ImageSets/VID_val_videos.txt"
        },
        "VID_val_videos_miniset": {
            "img_dir": "ILSVRC2015/Data/VID",
            "anno_path": "ILSVRC2015/Annotations/VID",
            "img_index": "ILSVRC2015/ImageSets/VID_val_videos_miniset.txt"
        },
        "VID_val_videos_custom": {
            "img_dir": "ILSVRC2015/Data/VID",
            "anno_path": "ILSVRC2015/Annotations/VID",
            "img_index": "ILSVRC2015/ImageSets/VID_val_videos_custom.txt"
        },
        "YouTube_Objects": {
            "img_dir": "YTO_v2.2/Images",
            "anno_path": "YTO_v2.2/GroundTruth",
            "img_index": "YTO_v2.2/Ranges"
        }
    }
    '''
        "YouTube_Objects": {
            "img_dir": "YouTubeObjects/vo-release/categories",
            "anno_path": "YouTubeObjects/vo-release/categories",
            "img_index": "YouTubeObjects/YOT_val_videos.txt"
        }
    '''

    @staticmethod
    def get(name, method="base"):
        if "coco" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="COCODataset",
                args=args,
            )
        elif "voc" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(data_dir, attrs["data_dir"]),
                split=attrs["split"],
            )
            return dict(
                factory="PascalVOCDataset",
                args=args,
            )
        elif "cityscapes" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = deepcopy(DatasetCatalog.DATASETS[name])
            attrs["img_dir"] = os.path.join(data_dir, attrs["img_dir"])
            attrs["ann_dir"] = os.path.join(data_dir, attrs["ann_dir"])
            return dict(factory="CityScapesDataset", args=attrs)
        else:
            dataset_dict = {
                "base": "VIDDataset",
                "rdn": "VIDRDNDataset",
                "mega": "VIDMEGADataset",
                "dafa": "VIDMEGADataset",
                "diffusion": "VIDMEGADataset",
                "fgfa": "VIDFGFADataset",
                "dff": "VIDDFFDataset",
                "yot": "YOTMEGADataset"
            }
            if ("visdrone" in name):
                data_dir = "/data1/jiahaoguo/dataset/VisDrone2019-VID"
                attrs = DatasetCatalog.DATASETS[name]
                args = dict(
                    image_set=name,
                    data_dir=data_dir,
                    img_dir=os.path.join(data_dir, attrs["img_dir"]),
                    anno_path=os.path.join(data_dir, attrs["anno_path"]),
                    img_index=os.path.join(data_dir, attrs["img_index"])
                )
                if "coco_json" in attrs:
                    args["coco_json"] = attrs["coco_json"]
                return dict(
                    factory=dataset_dict[method],
                    args=args,
                )
            if "dataset_dir" in DatasetCatalog.DATASETS[name] or "all_train_txt" in DatasetCatalog.DATASETS[name] :
                attrs = DatasetCatalog.DATASETS[name]
                data_dir = attrs["dataset_dir"]
                args = dict(
                    image_set=name,
                    data_dir=data_dir,
                    img_dir=os.path.join(data_dir, attrs["img_dir"]),
                    anno_path=os.path.join(data_dir, attrs["anno_path"]),
                    img_index=os.path.join(data_dir, attrs["img_index"]),
                    all_train_txt = attrs["all_train_txt"] if "all_train_txt" in attrs else None,
                    image_pattern_mode = attrs["image_pattern_mode"]
                )
                if "coco_json" in attrs:
                    args["coco_json"] = attrs["coco_json"]
                return dict(
                    factory=dataset_dict[method],
                    args=args,
                )
            if ("DET" in name) or ("VID" in name):
                data_dir = DatasetCatalog.DATA_DIR
                attrs = DatasetCatalog.DATASETS[name]
                args = dict(
                    image_set=name,
                    data_dir=data_dir,
                    img_dir=os.path.join(data_dir, attrs["img_dir"]),
                    anno_path=os.path.join(data_dir, attrs["anno_path"]),
                    img_index=os.path.join(data_dir, attrs["img_index"])
                )
                return dict(
                    factory=dataset_dict[method],
                    args=args,
                )
            if "YouTube" in name:
                data_dir = DatasetCatalog.DATA_DIR
                attrs = DatasetCatalog.DATASETS[name]
                args = dict(
                    image_set=name,
                    data_dir=data_dir,
                    img_dir=os.path.join(data_dir, attrs["img_dir"]),
                    anno_path=os.path.join(data_dir, attrs["anno_path"]),
                    img_index=os.path.join(data_dir, attrs["img_index"])
                )
                #assert method == "mega"
                return dict(
                    factory=dataset_dict["yot"],
                    args=args,
                )
        raise RuntimeError("Dataset not available: {}".format(name))


class ModelCatalog(object):
    S3_C2_DETECTRON_URL = "https://dl.fbaipublicfiles.com/detectron"
    C2_IMAGENET_MODELS = {
        "MSRA/R-50": "ImageNetPretrained/MSRA/R-50.pkl",
        "MSRA/R-50-GN": "ImageNetPretrained/47261647/R-50-GN.pkl",
        "MSRA/R-101": "ImageNetPretrained/MSRA/R-101.pkl",
        "MSRA/R-101-GN": "ImageNetPretrained/47592356/R-101-GN.pkl",
        "FAIR/20171220/X-101-32x8d": "ImageNetPretrained/20171220/X-101-32x8d.pkl",
        "FAIR/X-101-64x4d": "ImageNetPretrained/FBResNeXt/X-101-64x4d.pkl",
    }

    C2_DETECTRON_SUFFIX = "output/train/{}coco_2014_train%3A{}coco_2014_valminusminival/generalized_rcnn/model_final.pkl"
    C2_DETECTRON_MODELS = {
        "35857197/e2e_faster_rcnn_R-50-C4_1x": "01_33_49.iAX0mXvW",
        "35857345/e2e_faster_rcnn_R-50-FPN_1x": "01_36_30.cUF7QR7I",
        "35857890/e2e_faster_rcnn_R-101-FPN_1x": "01_38_50.sNxI7sX7",
        "36761737/e2e_faster_rcnn_X-101-32x8d-FPN_1x": "06_31_39.5MIHi1fZ",
        "35858791/e2e_mask_rcnn_R-50-C4_1x": "01_45_57.ZgkA7hPB",
        "35858933/e2e_mask_rcnn_R-50-FPN_1x": "01_48_14.DzEQe4wC",
        "35861795/e2e_mask_rcnn_R-101-FPN_1x": "02_31_37.KqyEK4tT",
        "36761843/e2e_mask_rcnn_X-101-32x8d-FPN_1x": "06_35_59.RZotkLKI",
        "37129812/e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x": "09_35_36.8pzTQKYK",
        # keypoints
        "37697547/e2e_keypoint_rcnn_R-50-FPN_1x": "08_42_54.kdzV35ao"
    }

    @staticmethod
    def get(name):
        if name.startswith("Caffe2Detectron/COCO"):
            return ModelCatalog.get_c2_detectron_12_2017_baselines(name)
        if name.startswith("ImageNetPretrained"):
            return ModelCatalog.get_c2_imagenet_pretrained(name)
        raise RuntimeError("model not present in the catalog {}".format(name))

    @staticmethod
    def get_c2_imagenet_pretrained(name):
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        name = name[len("ImageNetPretrained/"):]
        name = ModelCatalog.C2_IMAGENET_MODELS[name]
        url = "/".join([prefix, name])
        return url

    @staticmethod
    def get_c2_detectron_12_2017_baselines(name):
        # Detectron C2 models are stored following the structure
        # prefix/<model_id>/2012_2017_baselines/<model_name>.yaml.<signature>/suffix
        # we use as identifiers in the catalog Caffe2Detectron/COCO/<model_id>/<model_name>
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        dataset_tag = "keypoints_" if "keypoint" in name else ""
        suffix = ModelCatalog.C2_DETECTRON_SUFFIX.format(dataset_tag, dataset_tag)
        # remove identification prefix
        name = name[len("Caffe2Detectron/COCO/"):]
        # split in <model_id> and <model_name>
        model_id, model_name = name.split("/")
        # parsing to make it match the url address from the Caffe2 models
        model_name = "{}.yaml".format(model_name)
        signature = ModelCatalog.C2_DETECTRON_MODELS[name]
        unique_name = ".".join([model_name, signature])
        url = "/".join([prefix, model_id, "12_2017_baselines", unique_name, suffix])
        return url
