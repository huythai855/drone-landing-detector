# This file will use to score your implementations.
# The score would run with a different test_images and labels.csv
# You should not change this file

# Run:
# python test.py input_folder labels.csv
# python test.py ./test_images ./labels.csv


import os
import pandas as pd
import time
import sys
import cv2
import numpy as np

from landing_detector import LandingDetector


# TODO: Implementation IOU
def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def calc_precision(iou, t):
    return 1.0*len(iou[iou>t])/len(iou)


if __name__ == "__main__":

    input_folder = sys.argv[1]
    label_file = sys.argv[2]

    start_time = time.time()
    detector = LandingDetector()
    init_time = time.time() - start_time
    print("Run time in: %.2f s" % init_time)

    list_files = os.listdir(input_folder)
    print("Total test images: ", len(list_files))
    iou = []

    df = pd.read_csv(label_file)
    output_dict = {}

    start_time = time.time()
    total = 0
    for filename in list_files:
        if not ('jpg' in filename or 'jpeg' in filename):
            continue

        total += 1
        img = cv2.imread(os.path.join(input_folder, filename))
        target = (df[df['filename']==filename].x1.values[0],
                  df[df['filename']== filename].y1.values[0],
                  df[df['filename']== filename].x2.values[0],
                  df[df['filename']== filename].y2.values[0])

        t = {'x1': target[0], 'y1': target[1],
             'x2': target[2], 'y2': target[3]}

        output_x1, output_y1, output_x2, output_y2 = detector.detect(img)
        o = {'x1': output_x1, 'y1': output_y1,
             'x2': output_x2, 'y2': output_y2}
        print(t)
        iou.append(get_iou(o, t))

    print(iou)

    iou = np.array(iou)
    map_scores = []
    for i in range(50, 100, 5):
        map_scores += [calc_precision(iou, 0.01*i)]

    run_time = time.time() - start_time
    print("Map score: %.6f" % np.mean(map_scores))
    print("Run time: ", run_time)

