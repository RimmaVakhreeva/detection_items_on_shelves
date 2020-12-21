import os
import json

import numpy as np
import cv2

root_dir = '/Users/rimmavahreeva/Desktop/detection_pet_project/SKU110K/test'
image_name = 'test_22.jpg'
image_path = os.path.join(root_dir, image_name)
#image = cv2.imread(image_path)[..., ::-1].copy()
image = cv2.imread(image_path)
#plt.figure(figsize=(15, 15))
#plt.imshow(image)

ix = -1
iy = -1
polygons = []
points = []

def draw_img(image):
    global ix, iy, points, polygons

    for pol in polygons:
        if len(pol) > 1:
            cv2.polylines(image, pts=np.array([pol], dtype=np.int32),
                          isClosed=True,
                          color=(255, 0, 0),
                          thickness=6)

    if len(points) > 1:
        cv2.polylines(image, pts=np.array([points], dtype=np.int32),
                      isClosed=True,
                      color=(0, 0, 255),
                      thickness=6)
    return image

def mouse_callback(event, x, y, *args):
    global ix, iy, image, points

    if event == cv2.EVENT_LBUTTONDOWN:  # indicates that the left mouse button is pressed
        ix = x
        iy = y
        points.append((ix, iy))

cv2.namedWindow('image', cv2.WINDOW_KEEPRATIO)
cv2.setMouseCallback('image', mouse_callback) # Sets mouse handler for the specified window

while True:
    image_copy = draw_img(image.copy())
    cv2.imshow('image', image_copy)
    key = cv2.waitKey(5)
    if key & 0xFF == 27: # is a keyboard binding function # wait for ESC key to exit
        break
    elif key == 13:
        polygons.append(points)
        points = []

output_data = {image_name: []}
for idx, polygon in enumerate(polygons):
    output_data[image_name].append([
        f'shelve_{idx}', np.array(polygon).tolist()
    ])

with open('/Users/rimmavahreeva/Desktop/detection_pet_project/test_100.json', 'w') as f:
    json.dump(output_data, f)

cv2.destroyAllWindows() # destroys all the windows we created