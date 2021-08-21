import json
from pathlib import Path

import cv2

from sklearn.cluster import DBSCAN

from classes import Bbox, generate_color_by_text
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

def _process(image, detector, reid_model):
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
    clustering = DBSCAN(eps=0.2, min_samples=3, metric='cosine').fit(embeddings)

    for bbox, label in zip(bboxes, clustering.labels_):
        color = generate_color_by_text(str(label))
        cv2.rectangle(image,
                      (bbox.x1, bbox.y1),
                      (bbox.x1 + bbox.width, bbox.y1 + bbox.height),
                      color, 4)

    cv2.imwrite(str(root / 'test_7.jpg'), image)
    cv2.imshow('test', image)
    cv2.waitKey(0)


def main():
    print('Loading data')
    image, annotations = _load_data()
    print('Creating models')
    detector, reid_model = _create_models()
    print('Processing')
    _process(image, detector, reid_model)


if __name__ == '__main__':
    main()
