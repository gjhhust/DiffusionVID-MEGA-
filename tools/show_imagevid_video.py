import os
import cv2
import xml.etree.ElementTree as ET
from tqdm import tqdm
def visualize_annotations(image_dir, annotation_dir, output_video_path):
    # 获取视频的宽度和高度，这里默认使用第一张图片的尺寸
    sample_image_file = os.listdir(image_dir)[0]
    video_name = os.path.basename(image_dir)
    sample_image_path = os.path.join(image_dir, sample_image_file)
    sample_image = cv2.imread(sample_image_path)
    height, width, _ = sample_image.shape

    # 创建VideoWriter对象
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height))

    # 遍历每个XML文件
    for xml_file in tqdm(sorted(os.listdir(annotation_dir))):
        if xml_file.endswith('.xml'):
            xml_path = os.path.join(annotation_dir, xml_file)
            frame_id = int(os.path.splitext(xml_file)[0].split("_")[-1])
            image_path = os.path.join(image_dir, f"{video_name}_{frame_id-1:04d}.jpg")

            if os.path.exists(image_path):
                image = cv2.imread(image_path)
                tree = ET.parse(xml_path)
                root = tree.getroot()

                # 绘制矩形框
                for obj in root.findall('object'):
                    bndbox = obj.find('bndbox')
                    xmin = int(bndbox.find('xmin').text)
                    ymin = int(bndbox.find('ymin').text)
                    xmax = int(bndbox.find('xmax').text)
                    ymax = int(bndbox.find('ymax').text)
                    name = obj.find('name').text

                    # 根据名称字段设置矩形框颜色和右上角标签
                    color = (255, 0, 0)  # 默认为蓝色
                    label = "1"  # 默认显示1
                    if name == "2":
                        color = (0, 0, 255)  # 红色
                        label = "2"
                    
                    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 1)
                    cv2.putText(image, label, (xmax, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

                # 写入视频帧
                out.write(image)

    # 释放资源
    out.release()

if __name__ == "__main__":
    image_dir = "/data1/jiahaoguo/dataset/gaode_4_all/images/Scene22"
    annotation_dir = "/data1/jiahaoguo/dataset/gaode_4_all/gaode_4_vid/Annotations/train/Scene22"
    output_video_path = "output_video.avi"

    visualize_annotations(image_dir, annotation_dir, output_video_path)
