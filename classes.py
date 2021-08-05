from shapely.geometry import Polygon, box

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from dataclasses import field
from blist import sortedlist

import random

import cv2
import numpy as np

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

    def intersection_area(self, polygon: List[Point]):
        bbox_coords = box(self.x1, self.y1, self.x2, self.y2)
        polygon_coords = Polygon([[point.x, point.y] for point in polygon])
        intersection_figure = polygon_coords.intersection(bbox_coords)
        return intersection_figure.area

@dataclass
class Hole:
    top_left: Point
    top_right: Point
    bottom_left: Point
    bottom_right: Point

    @property
    def width(self):
        return self.bottom_right.x - self.bottom_left.x

    @property
    def hole_2_bbox(self) -> Bbox:
        np_holes = []
        np_holes.append([self.top_left.x, self.top_left.y])
        np_holes.append([self.top_right.x, self.top_right.y])
        np_holes.append([self.bottom_left.x, self.bottom_left.y])
        np_holes.append([self.bottom_right.x, self.bottom_right.y])
        np_holes = np.array(np_holes)
        x1 = np.min(np_holes[:, 0])
        y1 = np.min(np_holes[:, 1])
        x2 = np.max(np_holes[:, 0])
        y2 = np.max(np_holes[:, 1])
        return Bbox(x1, y1, x2, y2)

def _generate_color():
    r = int(random.random() * 256)
    g = int(random.random() * 256)
    b = int(random.random() * 256)
    return r, g, b

@dataclass
class Shelve:
    name: str
    points: List[Point]
    mean_width: Optional[int] = None
    max_width_hole: Optional[int] = None
    bboxes: List[Bbox] = field(default_factory=lambda: sortedlist([], key=lambda bbox: bbox.x1))
    color: Tuple[int, int, int] = field(default_factory=_generate_color)
    count_holes: List[int] = field(default_factory=list)
    holes: List[Hole] = field(default_factory=list)
    width_holes: List[int] = field(default_factory=list)

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

        for bbox in self.bboxes:
            cv2.rectangle(overlay,
                          (bbox.x1, bbox.y1),
                          (bbox.x1 + bbox.width, bbox.y1 + bbox.height),
                          self.color, -1)
            cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, output)
        return output

    def mean_width_bbox(self):
        width_list = []
        mean_width = 0
        for bbox in self.bboxes:
            width_list.append(bbox.width)
        if len(self.bboxes) != 0:
            mean_width = sum(width_list) / len(self.bboxes)
        return mean_width

    def mean_std_holes(self, width_holes, threshold=10):
        mean_width_holes = np.mean(self.width_holes)
        std_width_holes = np.std(self.width_holes)
        return mean_width_holes + std_width_holes + threshold