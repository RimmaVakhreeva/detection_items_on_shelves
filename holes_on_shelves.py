from gfl_r50 import Detector

from classes import Bbox, Shelve, Point, Hole

from typing import List, Dict
import json
import cv2

from pathlib import Path

img = '/Users/rimmavahreeva/Desktop/detection_pet_project/SKU110K/test/test_7.jpg'
root_dir = Path('/Users/rimmavahreeva/Desktop/detection_pet_project/SKU110K/test')
image_name = 'test_7.jpg'
#config_file = '/Users/rimmavahreeva/Desktop/detection_pet_project/gfl_r50_fpn_1x_coco.py'
#checkpoint_file = '/Users/rimmavahreeva/Desktop/detection_pet_project/epoch_12.pth'

detector = Detector()
result = detector.model(img)
show_result = detector.show_result(img)

with open('/Users/rimmavahreeva/Desktop/detection_pet_project/test_7.json', 'r') as read_file:
    data = json.load(read_file)

bboxes = []
for row in range(len(result)):
    if result[row][4] < 0.3:
        continue
    x1 = int(result[row][0])
    y1 = int(result[row][1])
    x2 = int(result[row][2])
    y2 = int(result[row][3])
    bbox = Bbox(x1=x1, y1=y1, x2=x2, y2=y2)
    bboxes.append(bbox)

threshold = 0.6
shelves_per_images: Dict[str, List[Shelve]] = {}

for filename, shelves in data.items():
    shelves_per_images[filename] = []

    for shelve_idx, shelve in enumerate(shelves):
        shelve_name, shelve_points = shelve

        shelve = Shelve(name='shelve_{}'.format(shelve_idx),
                        points=[Point(x, y) for x, y in shelve_points])
        shelves_per_images[filename].append(shelve)

        for bbox in bboxes:
            area = bbox.intersection_area(shelve.points)
            if area <= 0:
                continue

            intersection_area = area / bbox.area
            if intersection_area <= threshold:
                continue

            shelve.bboxes.add(bbox)

image = cv2.imread(str(root_dir / image_name))

for shelve in shelves_per_images[image_name]:
    image = shelve.render_img(image)

cv2.imshow('test', image)
cv2.waitKey(0)

for filename, shelves in shelves_per_images.items():
    for shelve in shelves:
        shelve.mean_width = shelve.mean_width_bbox()

for filename, shelves in shelves_per_images.items():
    for shelve in shelves:
        if len(shelve.bboxes) == 0:
            continue
        shelve.holes.clear()
        shelve_polygon_bbox = shelve.polygon_2_bbox

        shelve.holes.append(Hole(
            top_left=Point(x=shelve_polygon_bbox.x1, y=shelve_polygon_bbox.y1),
            top_right=Point(x=shelve.bboxes[0].x1, y=shelve.bboxes[0].y1),
            bottom_left=Point(x=shelve_polygon_bbox.x1, y=shelve_polygon_bbox.y2),
            bottom_right=Point(x=shelve.bboxes[0].x1, y=shelve.bboxes[0].y2)
        ))

        for last_bbox, next_bbox in zip(shelve.bboxes, shelve.bboxes[1:]):
            shelve.holes.append(Hole(
                top_left=Point(x=last_bbox.x1 + last_bbox.width, y=last_bbox.y1),
                top_right=Point(x=next_bbox.x1, y=next_bbox.y1),
                bottom_left=Point(x=last_bbox.x1 + last_bbox.width, y=last_bbox.y2),
                bottom_right=Point(x=next_bbox.x1, y=next_bbox.y2)
            ))

        shelve.holes.append(Hole(
            top_left=Point(x=shelve.bboxes[-1].x2, y=shelve.bboxes[-1].y1),
            top_right=Point(x=shelve_polygon_bbox.x2, y=shelve_polygon_bbox.y1),
            bottom_left=Point(x=shelve.bboxes[-1].x2, y=shelve.bboxes[-1].y2),
            bottom_right=Point(x=shelve_polygon_bbox.x2, y=shelve_polygon_bbox.y2)
        ))

for filename, shelves in shelves_per_images.items():
    for shelve in shelves:
        shelve.width_holes = []
        for hole in shelve.holes:
            shelve.width_holes.append(abs(hole.width))

image = cv2.imread(str(root_dir / image_name))
image = image.copy()

for filename, shelves in shelves_per_images.items():
    for shelve in shelves:
        count_empties = 0
        shelve_hole = []
        for hole in shelve.holes:
            if hole.width > shelve.mean_std_holes(shelve.width_holes):
                count_empties += 1
                shelve_hole.append(count_empties)
                shelve_hole_bbox = hole.hole_2_bbox
                cv2.rectangle(image,
                              (shelve_hole_bbox.x1, shelve_hole_bbox.y1),
                              (shelve_hole_bbox.x2, shelve_hole_bbox.y2),
                              color=(0, 0, 255), thickness=6)
        shelve.count_holes = shelve_hole

cv2.imshow('test1', image)
cv2.waitKey(0)

