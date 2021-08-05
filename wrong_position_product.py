from gfl_r50 import Detector
from modeling import GetEmbeddings

from shapely.geometry import Polygon, box

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from dataclasses import field
from blist import sortedlist

import random
import json
import numpy as np
import cv2
from pathlib import Path
import hashlib
import tqdm

from sklearn.cluster import DBSCAN

img = '/Users/rimmavahreeva/Desktop/detection_pet_project/SKU110K/test/test_7.jpg'
root_dir = Path('/media/svakhreev/fast/rimma_work/detection_pet_project/SKU110K/test')
image_name = 'test_7.jpg'
pretrain_path = '/Users/rimmavahreeva/Desktop/detection_pet_project/checkpoints/resnet18_model_40_weights.pth'
#config_file = '/Users/rimmavahreeva/Desktop/detection_pet_project/gfl_r50_fpn_1x_coco.py'
#checkpoint_file = '/Users/rimmavahreeva/Desktop/detection_pet_project/epoch_12.pth'

detector = Detector()
result = detector.model(img)
#show_result = detector.show_result(img)

@dataclass
class Point:
    x: int
    y: int

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

    @property
    def area(self):
        return self.width * self.height

    def clip(self, max_w, max_h):
        # self.x1 = max(min(self.x1, max_w), 0)
        self.x1 = np.clip(self.x1, 0, max_w)
        self.y1 = np.clip(self.y1, 0, max_h)
        self.x2 = np.clip(self.x2, 0, max_w)
        self.y2 = np.clip(self.y2, 0, max_h)
        return self.x1, self.y1, self.x2, self.y2

    def intersection_area(self, polygon: List[Point]):
        bbox_coords = box(self.x1, self.y1, self.x2, self.y2)
        polygon_coords = Polygon([[point.x, point.y] for point in polygon])
        intersection_figure = polygon_coords.intersection(bbox_coords)
        return intersection_figure.area

def _generate_color():
    r = int(random.random() * 256)
    g = int(random.random() * 256)
    b = int(random.random() * 256)
    return r, g, b

def generate_color_by_text(text):
    hash_code = int(hashlib.sha256(text.encode('utf-8')).hexdigest(), 16)
    r = int((hash_code / 255) % 255)
    g = int((hash_code / 65025) % 255)
    b = int((hash_code / 16581375) % 255)
    return b, g, r, 100

@dataclass
class Item:
    bbox: Bbox
    label: Optional[int] = None
    embedding: Optional[np.ndarray] = None
    distance_to_mean_embedding: Optional[float] = None

@dataclass
class Shelve:
    name: str
    points: List[Point]
    mean_width: Optional[int] = None
    items: List[Item] = field(default_factory=lambda: sortedlist([], key=lambda i: i.bbox.x1))
    mean_embedding: Optional[np.ndarray] = None
    color: Tuple[int, int, int] = field(default_factory=_generate_color)

    @property
    def polygon_2_bbox(self) -> Bbox:
        np_points = np.array([[point.x, point.y] for point in self.points])
        x1 = np.min(np_points[:, 0])
        y1 = np.min(np_points[:, 1])
        x2 = np.max(np_points[:, 0])
        y2 = np.max(np_points[:, 1])
        return Bbox(x1, y1, x2, y2)

    def render_img(self, image, alpha=0.4):
        image = image.copy()
        color = _generate_color()
        cv2.polylines(image,
                      pts=np.array([[[point.x, point.y] for point in self.points]], dtype=np.int32),
                      isClosed=True,
                      color=self.color,
                      thickness=6)

        overlay = image.copy()
        output = image.copy()

        for item in self.items:
            bbox = item.bbox
            cv2.rectangle(overlay,
                          (bbox.x1, bbox.y1),
                          (bbox.x1 + bbox.width, bbox.y1 + bbox.height),
                          self.color, -1)
            # cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, output)
        return output

    def mean_width_bbox(self):
        width_list = []
        mean_width = 0
        for item in self.items:
            bbox = item.bbox
            width_list.append(bbox.width)
        if len(self.items) != 0:
            mean_width = sum(width_list) / len(self.items)
        return mean_width

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

baseline = GetEmbeddings(num_classes=50030, last_stride=1,
                         neck='bnneck', neck_feat='after',
                         model_name='resnet18', weights_path=pretrain_path)

threshold = 0.6
shelves_per_images: Dict[str, List[Shelve]] = {}

for filename, shelves in data.items():
    shelves_per_images[filename] = []

    for shelve_idx, shelve in tqdm.tqdm(enumerate(shelves), total=len(shelves)):
        shelve_name, shelve_points = shelve

        shelve = Shelve(name='shelve_{}'.format(shelve_idx),
                        points=[Point(x, y) for x, y in shelve_points])
        shelves_per_images[filename].append(shelve)

        crops = []
        for bbox in bboxes:
            area = bbox.intersection_area(shelve.points)
            if area <= 0:
                continue

            intersection_area = area / bbox.area
            if intersection_area <= threshold:
                continue

            item = Item(bbox)
            shelve.items.add(item)
            x1 = bbox.x1
            y1 = bbox.y1
            x2 = bbox.x2
            y2 = bbox.y2
            crops.append(image[y1:y2, x1:x2, :])
            #item.embedding = model(input_crop[None,]).cpu().numpy()
            item.embedding = baseline.get_embeddings(crops)

for filename, shelves in shelves_per_images.items():
    for shelve in shelves:
        embeddings = np.array([item.embedding[0] for item in shelve.items])
        if len(embeddings) == 0:
            continue
        clustering = DBSCAN(eps=0.3, min_samples=1, metric='cosine').fit(embeddings)
        print(clustering.labels_)
        for item, label in zip(shelve.items, clustering.labels_):
            item.label = label

for shelve in shelves_per_images[image_name]:
    image = shelve.render_img(image)

for filename, shelves in shelves_per_images.items():
    for shelve in shelves:
        for item in shelve.items:
            color = generate_color_by_text(str(item.label))
            cv2.rectangle(image,
                          (item.bbox.x1, item.bbox.y1),
                          (item.bbox.x1 + item.bbox.width, item.bbox.y1 + item.bbox.height),
                          color, 4)

cv2.imshow('test', image)
cv2.waitKey(0)

