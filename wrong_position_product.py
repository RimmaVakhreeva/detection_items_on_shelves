import json
from pathlib import Path
from typing import List, Dict

import cv2
import numpy as np
import tqdm
from sklearn.cluster import DBSCAN

from classes import Bbox, Shelve, Point, generate_color_by_text, Item
from gfl_r50 import Detector
from modeling import GoodsReid

root = Path('/Users/rimmavahreeva/Desktop/detection_pet_project/')
img_filename = root / 'SKU110K/test/test_7.jpg'
annotations_path = root / 'test_7.json'
detector_weights_path = root / 'epoch_12.pth'
reid_weights_path = root / 'checkpoints/resnet18_model_40_weights.pth'
threshold = 0.6


def _load_data():
    image = cv2.imread(str(img_filename))
    with open(annotations_path, 'r') as read_file:
        annotations = json.load(read_file)
    return image, annotations


def _create_models():
    detector = Detector(weights_path=detector_weights_path)
    reid_model = GoodsReid(num_classes=50030, last_stride=1,
                           neck='bnneck', neck_feat='after',
                           model_name='resnet18', weights_path=reid_weights_path)
    return detector, reid_model


def _process(image, annotations, detector, reid_model):
    result = detector.get_bboxes(img_filename)

    bboxes, crops = [], []
    for row in range(len(result)):
        if result[row][4] < 0.3:
            continue
        x1, y1, x2, y2 = map(int, result[row][:4])
        bbox = Bbox(x1=x1, y1=y1, x2=x2, y2=y2)
        bbox.clip(max_w=image.shape[1], max_h=image.shape[0])
        bboxes.append(bbox)
        crops.append(image[bbox.y1:bbox.y2, bbox.x1:bbox.x2, :])

    embeddings = reid_model.get_embeddings(crops)

    shelves_per_images: Dict[str, List[Shelve]] = {}

    for filename, shelves in annotations.items():
        shelves_per_images[filename] = []

        for shelve_idx, shelve in tqdm.tqdm(enumerate(shelves), total=len(shelves)):
            shelve_name, shelve_points = shelve

            shelve = Shelve(name='shelve_{}'.format(shelve_idx),
                            points=[Point(x, y) for x, y in shelve_points])
            shelves_per_images[filename].append(shelve)

            crops = []
            for bbox, embedding in zip(bboxes, embeddings):
                area = bbox.intersection_area(shelve.points)
                if area <= 0:
                    continue

                intersection_area = area / bbox.area
                if intersection_area <= threshold:
                    continue

                item = Item(bbox, embedding=embedding)
                shelve.items.add(item)
                x1 = bbox.x1
                y1 = bbox.y1
                x2 = bbox.x2
                y2 = bbox.y2
                crops.append(image[y1:y2, x1:x2, :])

    for filename, shelves in shelves_per_images.items():
        for shelve in shelves:
            embeddings = np.array([item.embedding for item in shelve.items])
            if len(embeddings) == 0:
                continue
            clustering = DBSCAN(eps=0.2, min_samples=2, metric='cosine').fit(embeddings)
            for item, label in zip(shelve.items, clustering.labels_):
                item.label = label

    return shelves_per_images


def _visualize(image, shelves_per_images):
    for shelve in shelves_per_images[img_filename.name]:
        image = shelve.render_img(image)

    for filename, shelves in shelves_per_images.items():
        for shelve in shelves:
            for item in shelve.items:
                color = generate_color_by_text(str(item.label))
                cv2.rectangle(image,
                              (item.bbox.x1, item.bbox.y1),
                              (item.bbox.x1 + item.bbox.width, item.bbox.y1 + item.bbox.height),
                              color, 4)

    cv2.imwrite(str(root / 'result_wrong_position_product.jpg'), image)
    cv2.imshow('test', image)
    cv2.waitKey(0)


def main():
    print('Loading data')
    image, annotations = _load_data()
    print('Creating models')
    detector, reid_model = _create_models()
    print('Processing shelves')
    shelves_per_images = _process(image, annotations, detector, reid_model)
    _visualize(image, shelves_per_images)


if __name__ == '__main__':
    main()
