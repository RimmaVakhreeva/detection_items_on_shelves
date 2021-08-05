import json

from gfl_r50 import Detector
from modeling import GetEmbeddings

from dataclasses import dataclass

import cv2
import numpy as np
import hashlib

from sklearn.cluster import DBSCAN

img = '/Users/rimmavahreeva/Desktop/detection_pet_project/SKU110K/test/test_7.jpg'
pretrain_path = '/Users/rimmavahreeva/Desktop/detection_pet_project/checkpoints/resnet18_model_40_weights.pth'
#config_file = '/Users/rimmavahreeva/Desktop/detection_pet_project/gfl_r50_fpn_1x_coco.py'
#checkpoint_file = '/Users/rimmavahreeva/Desktop/detection_pet_project/epoch_12.pth'

detector = Detector()
result = detector.model(img)
#show_result = detector.show_result(img)

@dataclass
class Bbox:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1

    def clip(self, max_w, max_h):
        # self.x1 = max(min(self.x1, max_w), 0)
        self.x1 = np.clip(self.x1, 0, max_w)
        self.y1 = np.clip(self.y1, 0, max_h)
        self.x2 = np.clip(self.x2, 0, max_w)
        self.y2 = np.clip(self.y2, 0, max_h)
        return self.x1, self.y1, self.x2, self.y2

def generate_color_by_text(text):
    hash_code = int(hashlib.sha256(text.encode('utf-8')).hexdigest(), 16)
    r = int((hash_code / 255) % 255)
    g = int((hash_code / 65025) % 255)
    b = int((hash_code / 16581375) % 255)
    return b, g, r, 100

with open('/Users/rimmavahreeva/Desktop/detection_pet_project/test_7.json', 'r') as read_file:
    data = json.load(read_file)

image = cv2.imread(img)
bboxes = []
for row in range(len(result)):
    if result[row][4] < 0.3:
        continue
    x1 = int(result[row][0])
    y1 = int(result[row][1])
    x2 = int(result[row][2])
    y2 = int(result[row][3])
    bbox = Bbox(x1=x1, y1=y1, x2=x2, y2=y2)
    bbox.clip(max_w=image.shape[1], max_h=image.shape[0])
    bboxes.append(bbox)

crops = []
for idx, row in enumerate(bboxes):
    x1 = bboxes[idx].x1
    y1 = bboxes[idx].y1
    x2 = bboxes[idx].x2
    y2 = bboxes[idx].y2
    crops.append(image[y1:y2, x1:x2, :])

baseline = GetEmbeddings(num_classes=50030, last_stride=1,
                         neck='bnneck', neck_feat='after',
                         model_name='resnet18', weights_path=pretrain_path)

embeddings = baseline.get_embeddings(crops)
print(embeddings.shape)

clustering = DBSCAN(eps=0.2, min_samples=3, metric='cosine').fit(embeddings)

alpha = 0.4
for bbox, label in zip(bboxes, clustering.labels_):
    color = generate_color_by_text(str(label))
    cv2.rectangle(image,
                  (bbox.x1, bbox.y1),
                  (bbox.x1 + bbox.width, bbox.y1 + bbox.height),
                  color, 4)

cv2.imshow('test', image)
cv2.waitKey(0)


