from gfl_r50 import Detector

from classes import Bbox, Shelve, Point, Hole, Item

from typing import List, Dict
import json
import cv2

from pathlib import Path

root = Path('/Users/rimmavahreeva/Desktop/detection_pet_project/')
img_filename = root / 'SKU110K/test/test_28.jpg'
annotations_path = root / 'test_28.json'
detector_weights_path = root / 'epoch_12.pth'
threshold = 0.6


def _load_data():
    image = cv2.imread(str(img_filename))
    with open(annotations_path, 'r') as read_file:
        annotations = json.load(read_file)
    return image, annotations


def _create_models():
    detector = Detector(weights_path=detector_weights_path)
    return detector


def _process(image, annotations, detector):
    result = detector.get_bboxes(img_filename)

    bboxes = []
    for row in range(len(result)):
        if result[row][4] < 0.3:
            continue
        x1, y1, x2, y2 = map(int, result[row][:4])
        bbox = Bbox(x1=x1, y1=y1, x2=x2, y2=y2)
        bbox.clip(max_w=image.shape[1], max_h=image.shape[0])
        bboxes.append(bbox)

    shelves_per_images: Dict[str, List[Shelve]] = {}

    for filename, shelves in annotations.items():
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

                item = Item(bbox)
                shelve.items.add(item)

    for filename, shelves in shelves_per_images.items():
        for shelve in shelves:
            shelve.mean_width = shelve.mean_width_bbox()

    for filename, shelves in shelves_per_images.items():
        for shelve in shelves:
            if len(shelve.items) == 0:
                continue
            shelve.holes.clear()
            shelve_polygon_bbox = shelve.polygon_2_bbox

            shelve.holes.append(Hole(
                top_left=Point(x=shelve_polygon_bbox.x1, y=shelve_polygon_bbox.y1),
                top_right=Point(x=shelve.items[0].bbox.x1, y=shelve.items[0].bbox.y1),
                bottom_left=Point(x=shelve_polygon_bbox.x1, y=shelve_polygon_bbox.y2),
                bottom_right=Point(x=shelve.items[0].bbox.x1, y=shelve.items[0].bbox.y2)
            ))

            for last_bbox, next_bbox in zip(shelve.items, shelve.items[1:]):
                shelve.holes.append(Hole(
                    top_left=Point(x=last_bbox.bbox.x1 + last_bbox.bbox.width, y=last_bbox.bbox.y1),
                    top_right=Point(x=next_bbox.bbox.x1, y=next_bbox.bbox.y1),
                    bottom_left=Point(x=last_bbox.bbox.x1 + last_bbox.bbox.width, y=last_bbox.bbox.y2),
                    bottom_right=Point(x=next_bbox.bbox.x1, y=next_bbox.bbox.y2)
                ))

            shelve.holes.append(Hole(
                top_left=Point(x=shelve.items[-1].bbox.x2, y=shelve.items[-1].bbox.y1),
                top_right=Point(x=shelve_polygon_bbox.x2, y=shelve_polygon_bbox.y1),
                bottom_left=Point(x=shelve.items[-1].bbox.x2, y=shelve.items[-1].bbox.y2),
                bottom_right=Point(x=shelve_polygon_bbox.x2, y=shelve_polygon_bbox.y2)
            ))

    for filename, shelves in shelves_per_images.items():
        for shelve in shelves:
            shelve.width_holes = []
            for hole in shelve.holes:
                shelve.width_holes.append(abs(hole.width))

    for filename, shelves in shelves_per_images.items():
        for shelve in shelves:
            shelve.width_holes = []
            for hole in shelve.holes:
                shelve.width_holes.append(abs(hole.width))

    return shelves_per_images


def _visualize(image, shelves_per_images):
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

    cv2.imwrite(str(root / 'result_28.jpg'), image)
    cv2.imshow('test', image)
    cv2.waitKey(0)


def main():
    print('Loading data')
    image, annotations = _load_data()
    print('Creating models')
    detector = _create_models()
    print('Processing shelves')
    shelves_per_images = _process(image, annotations, detector)
    _visualize(image, shelves_per_images)

if __name__ == '__main__':
    main()
