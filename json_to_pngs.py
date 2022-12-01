import argparse
import cv2
import json
import os
import numpy as np
'''
This script combine VGG.json and img files to png images.
'''
# Input management
ap = argparse.ArgumentParser()
ap.add_argument("-fj", "--file_json", type=str, required=False, default=r"labels_VGG.json")
ap.add_argument("-o", "--output", type=str, required=False, default=r"C:\Users\Susi\Desktop")
ap.add_argument("-iw", "--width", type=int, required=False, default=1175)
ap.add_argument("-ih", "--height", type=int, required=False, default=780)
def get_fill_convex(img, points):
    cv2.fillConvexPoly(img, points, 1)
    return img
def get_points(d, h, w, out):
    with open(d) as f:
        data = json.load(f)
    # image
    for key in data.keys():
        img_c0 = np.zeros((h, w))
        img_c1 = np.zeros((h, w))
        images = data[key]["filename"]
        # detection
        for key_det in data[key]["regions"].keys():
            points = []
            p_x = data[key]["regions"][key_det]["shape_attributes"]["all_points_x"]
            p_y = data[key]["regions"][key_det]["shape_attributes"]["all_points_y"]
            tag = data[key]["regions"][key_det]["region_attributes"]["label"]
            # merge p_x and p_y
            for p in range(len(p_x)):
                points.append([p_x[p], p_y[p]])
            points = np.array(points, np.int32)
            if tag == "licence_plate":
                img_c0 = cv2.bitwise_or(img_c0, get_fill_convex(img_c0, points), img_c0)
            else:
                img_c1 = cv2.bitwise_or(img_c1, get_fill_convex(img_c1, points), img_c1)
        # save images
        img_c0 = cv2.normalize(img_c0, None, 0, 255, cv2.NORM_MINMAX)
        img_c1 = cv2.normalize(img_c1, None, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(os.path.join(out, images.split('.')[0] + "_c0.png"), img_c0)
        cv2.imwrite(os.path.join(out, images.split('.')[0] + "_c1.png"), img_c1)
if __name__ == "__main__":
    args = vars(ap.parse_args())
    json_path = args['file_json']
    output_path = args['output']
    width = args['width']
    height = args['height']
    get_points(json_path, height, width, output_path)