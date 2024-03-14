import os,cv2
import xml.etree.ElementTree as ET
from tqdm import tqdm

import xml.etree.ElementTree as ET
from xml.dom import minidom
import cv2

def prettify(elem):
    """Return a pretty-printed XML string for the Element."""
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def create_annotations(visdrone_path, ilsvrc_path,model):
    annotations_dir = os.path.join(ilsvrc_path, "Annotations",model)
    image_sets_dir = os.path.join(ilsvrc_path, "ImageSets")
    videos_image_dir = os.path.join(visdrone_path, "images", model)

    if not os.path.exists(annotations_dir):
        os.makedirs(annotations_dir)

    if not os.path.exists(image_sets_dir):
        os.makedirs(image_sets_dir)

    # Read video names from the ImageSets file
    video_names = [name for name in os.listdir(videos_image_dir) if os.path.isdir(os.path.join(videos_image_dir, name))]

    with open(os.path.join(image_sets_dir, f"{model}_10.txt"), "w") as file:
        for video_name in tqdm(video_names):
            video_dir = os.path.join(videos_image_dir, video_name)
            annotation_file = os.path.join(visdrone_path, f"{model}-VID", "annotations", f"{video_name}.txt")
            total_length = len(os.listdir(video_dir))

            # Create a new folder for each video in Annotations
            video_annotation_dir = os.path.join(annotations_dir, video_name)
            if not os.path.exists(video_annotation_dir):
                os.makedirs(video_annotation_dir)

            # Dictionary to store annotations for each frame
            frame_annotations = {}

            with open(annotation_file, "r") as annot_file:
                for line in annot_file:
                    # Parse Visdrone annotation
                    frame_index, trackid, bbox_left, bbox_top, bbox_width, bbox_height, _, object_category, _, occluded = map(int, line.strip().split(','))
                    # if object_category==0:
                    #     continue
                    # else:
                    #     object_category = object_category-1
                    # Create or update dictionary for the frame
                    if frame_index not in frame_annotations:
                        frame_annotations[frame_index] = []

                    frame_annotations[frame_index].append({
                        'trackid': trackid,
                        'bbox_left': bbox_left,
                        'bbox_top': bbox_top,
                        'bbox_width': bbox_width,
                        'bbox_height': bbox_height,
                        'object_category': object_category
                    })

            # Create XML annotation for each frame
            for frame_index, annotations in frame_annotations.items():
                # if (frame_index)%10 == 0:
                #     continue

                image_path = os.path.join(video_dir, f"{frame_index:07d}.jpg")
                img = cv2.imread(image_path)
                img_height, img_width, _ = img.shape

                xml_root = ET.Element("annotation")
                ET.SubElement(xml_root, "folder").text = video_name
                ET.SubElement(xml_root, "filename").text = f"{frame_index:07d}"
                source = ET.SubElement(xml_root, "source")
                ET.SubElement(source, "database").text = "Visdrone"
                ET.SubElement(source, "annotation").text = "Visdrone"
                ET.SubElement(source, "image").text = "Visdrone"
                size = ET.SubElement(xml_root, "size")
                ET.SubElement(size, "width").text = str(img_width)
                ET.SubElement(size, "height").text = str(img_height)
                for annotation in annotations:
                    obj = ET.SubElement(xml_root, "object")
                    ET.SubElement(obj, "trackid").text = str(annotation['trackid'])
                    ET.SubElement(obj, "name").text = str(annotation['object_category'])
                    bndbox = ET.SubElement(obj, "bndbox")
                    ET.SubElement(bndbox, "xmin").text = str(annotation['bbox_left'])
                    ET.SubElement(bndbox, "ymin").text = str(annotation['bbox_top'])
                    ET.SubElement(bndbox, "xmax").text = str(annotation['bbox_left'] + annotation['bbox_width'])
                    ET.SubElement(bndbox, "ymax").text = str(annotation['bbox_top'] + annotation['bbox_height'])
                    ET.SubElement(obj, "occluded").text = "0"  # Assuming no occlusion
                    ET.SubElement(obj, "generated").text = "0"  # Assuming no truncation

                # Save XML annotation with indentation
                xml_string = prettify(xml_root)
                # with open(os.path.join(video_annotation_dir, f"{frame_index:07d}.xml"), "w") as xml_file:
                #     xml_file.write(xml_string)

                # Write to train.txt: video_path 1 frame_index img_width img_height
                file.write(f"{video_name} 1 {frame_index} {total_length}\n")

if __name__ == "__main__":
    visdrone_path = "/data1/jiahaoguo/dataset/VisDrone2019-VID"
    ilsvrc_path = "/data1/jiahaoguo/dataset/VisDrone2019-VID/annotations/ilsvrc2015_Lables"

    create_annotations(visdrone_path, ilsvrc_path, "train")
