# This file will use to score your implementations.
# You should not change this file

import os
import sys
import time

import cv2
import numpy as np

from landing_detector import LandingDetector


def read_label(label_file):
    with open(label_file, "r") as f:
        line = f.readline()

    tmp = line.strip().split(' ')

    w, h = img.shape[1], img.shape[0]
    x = [(float)(w.strip()) for w in tmp]

    x1 = int(x[1] * w)
    width = int(x[3] * w)

    y1 = int(x[2] * h)
    height = int(x[4] * h)

    return x1 - width // 2, y1 - height // 2, x1 + width // 2, y1 + height // 2


# TODO: Implementation IOU
def iou_score(output, target):
    return 0.5


def calc_precision(iou, t):
    return 1.0 * len(iou[iou > t]) / len(iou)


if __name__ == "__main__":

    input_folder = sys.argv[1]
    label_folder = sys.argv[2]

    start_time = time.time()
    detector = LandingDetector()
    init_time = time.time() - start_time
    print("Run time in: %.2f s" % init_time)

    list_files = os.listdir(input_folder)
    print("Total test images: ", len(list_files))
    iou = []

    start_time = time.time()
    total = 0
    for filename in list_files:
        if not ('jpg' in filename or 'jpeg' in filename):
            continue

        total += 1
        img = cv2.imread(os.path.join(input_folder, filename))
        target = read_label(os.path.join(label_folder, filename[:-4] + "txt"))
        print(img.shape)

        output_x1, output_y1, output_x2, output_y2 = detector.detect(img)
        iou.append(iou_score((output_x1, output_y1, output_x2, output_y2),
                             target))

    map_scores = []
    for i in range(50, 100, 5):
        map_scores += calc_precision(iou, 0.01 * i)

    run_time = time.time() - start_time
    print("Map score: %.6f" % np.mean(map_scores))
    print("Run time: ", run_time)
