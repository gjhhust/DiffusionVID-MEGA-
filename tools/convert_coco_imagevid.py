import os
import json
import xmltodict
import cv2,tqdm
# vid的frame从1开始
#classes分类也是1开始
#标注文件xml和图片文件名一致即可
def coco_to_imagevid(coco_json_path, coco_image_dir, output_dir):

    prefix = "train"
    image_set_filename = f"{prefix}.txt"
    
    # 读取COCO格式的JSON文件
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # 创建ImageSets文件夹
    image_sets_dir = os.path.join(output_dir, 'ImageSets')
    os.makedirs(image_sets_dir, exist_ok=True)

    # 创建Annotations文件夹
    annotations_dir = os.path.join(output_dir, 'Annotations', prefix)
    os.makedirs(annotations_dir, exist_ok=True)

    # 创建Data文件夹
    data_dir = os.path.join(output_dir, 'Data')
    os.makedirs(data_dir, exist_ok=True)

    # 创建视频的ImageSet文件
    image_set_path = os.path.join(image_sets_dir, image_set_filename)
    
    videos_dict = {}
    for image in coco_data['images']:
        video_name = os.path.dirname(image['file_name']) 
        frame_number = int(os.path.splitext(os.path.basename(os.path.basename(image['file_name']) ))[0].split("_")[-1])
        if frame_number!=0:
            assert frame_number == len(videos_dict[video_name])
        if video_name not in videos_dict:
            videos_dict[video_name] = []
        videos_dict[video_name].append(image)

    # 遍历每个视频
    with open(image_set_path, 'w') as image_set_file:
        for video_name, video_list in tqdm.tqdm(videos_dict.items()):
            total_frames = len(video_list)
            track_id_cnt = 0
            # 遍历每一帧
            for frame_number, image_info in enumerate(video_list):
                image_set_file.write(f"{video_name} 1 {frame_number+1} {total_frames}\n")

                image_filename = f"{video_name}_{frame_number:04d}.jpg"
                image_target_filename = f"{video_name}_{frame_number+1:04d}.jpg"
                annotation_filename = f"{video_name}_{frame_number+1:04d}.xml"
                annotation_path = os.path.join(annotations_dir, video_name, annotation_filename)
                
                
                image_path = os.path.join(coco_image_dir, video_name, image_filename)
                image = cv2.imread(image_path)
                h,w,_ = image.shape

            
                annotation_data = {
                    'annotation': {
                        'folder': video_name,
                        'filename': f"{frame_number+1:07d}",  # 保持和Visdrone数据集的格式一致
                        'source': {
                            'database': 'gaode_4',
                            'annotation': 'gaode_4',
                            'image': 'gaode_4',
                        },
                        'size': {
                            'width': w,  # 你需要根据实际情况提供图像的宽度
                            'height': h,  # 你需要根据实际情况提供图像的高度
                        },
                        'object': [],  # 这里存放每个目标的注释信息
                    }
                }

                # 添加每个目标的注释信息
                for object_info in coco_data['annotations']:
                    if object_info['image_id'] == image_info["id"]:
                        object_annotation = {
                            'trackid': track_id_cnt,
                            'name': object_info["category_id"]+1, 
                            'bndbox': {
                                'xmin': object_info['bbox'][0],
                                'ymin': object_info['bbox'][1],
                                'xmax': object_info['bbox'][0]+object_info['bbox'][2],
                                'ymax': object_info['bbox'][1]+object_info['bbox'][3],
                            },
                            'occluded': 0,
                            'generated': 0,
                        }
                        track_id_cnt += 1
                        annotation_data['annotation']['object'].append(object_annotation)

                # 保存XML注释文件
                os.makedirs(os.path.dirname(annotation_path),exist_ok=True)
                with open(annotation_path, 'w') as annotation_file:
                    xmltodict.unparse(annotation_data, output=annotation_file, pretty=True)

                # 复制图像到Data文件夹
                # data_image_path = os.path.join(data_dir, video_name, image_target_filename)
                # os.makedirs(os.path.dirname(data_image_path),exist_ok=True)
                # if not os.path.exists(data_image_path):
                #     os.symlink(image_path, data_image_path)

if __name__ == "__main__":
    # 输入COCO格式的JSON文件路径和COCO图片保存的目录
    coco_json_path = "/data1/jiahaoguo/dataset/gaode_4_all/annotations/train_half.json"
    coco_image_dir = "/data1/jiahaoguo/dataset/gaode_4_all/images"
 
    # 输出ImageVID格式的目录
    output_dir = "/data1/jiahaoguo/dataset/gaode_4_all/gaode_4_vid"

    coco_to_imagevid(coco_json_path, coco_image_dir, output_dir)
