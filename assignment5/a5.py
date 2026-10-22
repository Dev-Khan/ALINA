
# Hack to get local alina files in here
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Doing relative imports so I can break ALINA for this assignmnet
from alina.config import PipelineConfig, EvaluationConfig
from alina.histogram import calc_histogram
from alina.roi import select_roi, roi_points_to_quad
from alina.traversal import circular_threshold_pixel_discovery_and_traversal
from eval.cbem import create_cbem
from eval.superimpose import overlay_coords
from eval.evaluate import run_evaluation


import argparse

import cv2 

from dataclasses import dataclass
from datetime import datetime


import matplotlib.pyplot as plt

import numpy as np

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

import time
from tqdm import tqdm



# https://sightline.us/images/pdf/FAAAC_150-5370-10HP620.pdf
# Both are valid FED standard colors
# Effectively together they are a threshold, but true color not camera color
YELLOW_HSV = [22, 255, 242] # [19, 226, 239] 
YELLOW_RGB = [242, 176, 0] # [239, 160, 27]

# You'd think that white is just all maxed out values
# You'd be wrong
WHITE_RGB = [239, 241, 230]

SEED = 123456789

# Rip these from assignment 4 
# we could import them from assignmnet 4 but then we get all the import from 
# assignmnet 4 and I really dont want to deal with the model importing since it is heavy and time consuming

""" Need to have local functions here because since part of the Alina doc was to
install the Alina repository I cant make local changes in the thing without breaking
things. Realistically I could just uninstall it, but Im already this far in. 
What I should do is re-write alina to work better for this project, but that is 
a future me problem not a now me problem."""
def local_run_batch(config, rois):
    image_filenames = sorted(
        f for f in config.input_dir.iterdir() if f.suffix.lower() == ".jpg"
    )
    total = len(image_filenames)
    
        
    # Yoinked from run batch
    log_file = open(config.log_file, "w") if config.log_file else None

    try:
        for i, path in enumerate(image_filenames):
            start = time.perf_counter()
            img = cv2.imread(str(path))
            if img is None:
                continue

            if len(rois) > 1:
                annotated, coords, had_lines = process_image(img, rois[i], config)
            else: # there is only one becuase manual ROI
                annotated, coords, had_lines = process_image(img, rois[0], config)

            elapsed = time.perf_counter() - start

            text_filename = path.stem + ".txt"
            np.savetxt(config.output_coords_dir / text_filename, coords, fmt="%6d")
            cv2.imwrite(str(config.output_images_dir / path.name), annotated)

            status = "Labeled" if had_lines else "No lines found"
            log_line = (
                f"[{i}/{total}] [{status}] {path.name}: {len(coords)} px, "
                f"elapsed {elapsed:.3f}s ({datetime.now().strftime('%H:%M:%S')})"
            )
            print(log_line)

            if log_file:
                log_file.write(log_line + "\n")
                log_file.flush()

    finally:
        if log_file:
            log_file.close()


def process_image(img: np.ndarray, roi_points: np.ndarray, config: PipelineConfig) -> tuple[np.ndarray, np.ndarray, bool]:
    final_img = img.copy()
    no_lines_img = img.copy()

    # 3.3 perspective transformation -> bird's eye view of the ROI
    roi_quad = roi_points_to_quad(roi_points, dtype=np.float32)
    dst = np.array(
        [
            config.roi.dst_bottom_left,
            config.roi.dst_top_left,
            config.roi.dst_top_right,
            config.roi.dst_bottom_right,
        ],
        dtype=np.float32,
    ).reshape(1, 4, 2)

    M = cv2.getPerspectiveTransform(roi_quad, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))

    # 3.4 color feature normalization (HSV, per-channel min-max)
    def normalize_color_features(image: np.ndarray, save_path = None) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        h = cv2.normalize(h, None, 0, 179, cv2.NORM_MINMAX)
        s = cv2.normalize(s, None, 0, 255, cv2.NORM_MINMAX)
        v = cv2.normalize(v, None, 0, 255, cv2.NORM_MINMAX)
        color_features = np.stack((h, s, v), axis=-1)

        if save_path is not None:
            cv2.imwrite(str(save_path), color_features)

        return color_features
    
    color_features = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    color_features = normalize_color_features(warped)

    # 3.5 HSV-based color thresholding -> binary mask of candidate line pixels
    lowers = np.array(config.yellow_lower)
    uppers = np.array(config.yellow_upper)
    print(lowers, uppers)

    mask = []
    for lower, upper in zip(lowers, uppers):
        _mask = cv2.inRange(color_features, lower, upper)
        if config.mask_ignore_left_columns > 0:
            _mask[:, :config.mask_ignore_left_columns] = 0

        mask.append(_mask)

    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)
    ax[0].imshow(warped)
    ax[1].imshow(mask[0])
    ax[2].imshow(mask[1])
    plt.show()

    # merge the mask back together using the OR operator such
    # that the resulting mask is where any pixel is within
    # any threshold passed to the configuration system
    mask = (mask[0] | mask[1]).astype(np.uint8)

    # 3.6 histogram analysis -> vertical projection, peak column
    # min_white_pixels is actually a misnomer because it actually min MASKED pixels since it is looking at the binary image not specifically white colored pixels in the original image.
    avg_pixel, peak_value = calc_histogram(mask, min_white_pixels=config.min_white_pixels)

    # 3.7 threshold check -> is this peak an actual line marking or noise
    if not avg_pixel or peak_value <= config.peak_pixel_threshold:
        return no_lines_img, np.empty((0, 2), dtype=np.int32), False

    # 3.8 CIRCLEDAT -> traverse from the peak to collect all connected line pixels
    binary = mask.copy()
    binary[avg_pixel[1]][avg_pixel[0]] = 255
    line_marking_pixels = circular_threshold_pixel_discovery_and_traversal(
        binary, avg_pixel[0], avg_pixel[1], config.circular_threshold
    )

    height, width = binary.shape[:2]
    black_img = np.zeros((height, width), dtype=np.uint8)
    for x, y in line_marking_pixels:
        cv2.circle(black_img, (x, y), 1, 255, -1)

    # 3.9 frame unwarping -> map detected pixels back to the original perspective
    Minv = cv2.getPerspectiveTransform(dst, roi_quad)
    unwarped = cv2.warpPerspective(black_img, Minv, (img.shape[1], img.shape[0]))

    line_pixels_only = np.where(unwarped > 0)
    x_coords, y_coords = line_pixels_only[1], line_pixels_only[0]
    coords = np.column_stack((x_coords, y_coords))

    # 3.9 annotate -> mark line marking pixels in red on the original frame
    final_img[line_pixels_only] = (0, 0, 255)
    return final_img, coords, True


def roi_via_claude(path, plotter=False):
    
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    w, h = img.shape[:2]

    # Because the images are not contigious what I need to do
    # is rip the image naame out of the file path 
    # then point it to the proper txt file
    
    # We need to know which vidd and which image so we gotta decompose the path
    components = str(path).split("/")

    #vidd is the second to last and image is last
    vid_set = components[-2]
    img_num = components[-1]
    # need to rip out the file extension tho
    img_num = img_num[:img_num.index(".")]


    roi_path = f"../assignment4/ClaudeAttempt/roi_labels_{vid_set}/{img_num}.txt"
    roi = []
    with open(roi_path, "r") as f:
        for line in f:
            x,y = line.split(",")

            x = int(x)
            y = int(y)
            roi.append([x,y])

    
    if plotter:
        _roi = np.asarray(roi, dtype=np.int32)
        img_cpy = np.copy(img)
        img_cpy = cv2.drawContours(img_cpy, [_roi], -1, (255, 0, 0), thickness=5)
        fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
        ax[0].imshow(img)
        ax[1].imshow(img_cpy)
        plt.show()


    # Now to fix the ROI because it is broken in ALINA
    roi.insert(2, [-1, -1])
    roi.insert(4, [-1, -1])

    # also needs some bubble wrap for downstraem ALINA functions
    return [roi]

# This was in the a4.py however 
# We are updating it so it can do yellow and white
def kmean_color_threshold(path, target_color):
    
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert to HSV since that is what ALINA is expecting


    pixels = np.reshape(img, (-1, img.shape[-1]))

    K = 200
    kmeans = KMeans(n_clusters=K, init="k-means++", random_state=SEED)
    Y = kmeans.fit_predict(pixels)

    y_hat = kmeans.predict([target_color])[0]

    cluster = pixels[Y == y_hat] # get the points belonging to the predicted cluster

    # Get the min and max of the cluster to build the threshold bounds (but not the exact min and max since its just going to return black and white)
    # take from the 5% to 95% range of the cluster
    min_value = np.percentile(cluster, 5, axis=0).astype(np.uint8)
    max_value = np.percentile(cluster, 95, axis=0).astype(np.uint8)
    return min_value, max_value



def hsi_color_threshold(path, target_color):

    # Built the color matrix based off of the xls spread sheet 
    # All available color spectrum from 380nm to 730 nm
    R = np.array([
    [0.055,  0.058,   0.061,   0.062,   0.062,   0.062,   0.062,   0.062,   0.062,   0.062,   0.062,   0.063,   0.065,   0.070,   0.076,   0.079,   0.081,   0.084,   0.091,   0.103,   0.119,   0.134,   0.143,   0.147,   0.151,   0.158,   0.168,   0.179,   0.188,   0.190,   0.186,   0.181,   0.182,   0.187,   0.196,   0.209],
    [0.117,  0.143,   0.175,   0.191,   0.196,   0.199,   0.204,   0.213,   0.228,   0.251,   0.280,   0.309,   0.329,   0.333,   0.315,   0.286,   0.273,   0.276,   0.277,   0.289,   0.339,   0.420,   0.488,   0.525,   0.546,   0.562,   0.578,   0.595,   0.612,   0.625,   0.638,   0.656,   0.678,   0.700,   0.717,   0.734],
    [0.130,  0.177,   0.251,   0.306,   0.324,   0.330,   0.333,   0.331,   0.323,   0.311,   0.298,   0.285,   0.269,   0.250,   0.231,   0.214,   0.199,   0.185,   0.169,   0.157,   0.149,   0.145,   0.142,   0.141,   0.141,   0.141,   0.143,   0.147,   0.152,   0.154,   0.150,   0.144,   0.136,   0.132,   0.135,   0.147],
    [0.051,  0.054,   0.056,   0.057,   0.058,   0.059,   0.060,   0.061,   0.062,   0.063,   0.065,   0.067,   0.075,   0.101,   0.145,   0.178,   0.184,   0.170,   0.149,   0.133,   0.122,   0.115,   0.109,   0.105,   0.104,   0.106,   0.109,   0.112,   0.114,   0.114,   0.112,   0.112,   0.115,   0.120,   0.125,   0.130],
    [0.144,  0.198,   0.294,   0.375,   0.408,   0.421,   0.426,   0.426,   0.419,   0.403,   0.379,   0.346,   0.311,   0.281,   0.254,   0.229,   0.214,   0.208,   0.202,   0.194,   0.193,   0.200,   0.214,   0.230,   0.241,   0.254,   0.279,   0.313,   0.348,   0.366,   0.366,   0.359,   0.358,   0.365,   0.377,   0.398],
    [0.136,  0.179,   0.247,   0.297,   0.320,   0.337,   0.355,   0.381,   0.419,   0.466,   0.510,   0.546,   0.567,   0.574,   0.569,   0.551,   0.524,   0.488,   0.445,   0.400,   0.350,   0.299,   0.252,   0.221,   0.204,   0.196,   0.191,   0.188,   0.191,   0.199,   0.212,   0.223,   0.232,   0.233,   0.229,   0.229],
    [0.054,  0.054,   0.053,   0.054,   0.054,   0.055,   0.055,   0.055,   0.056,   0.057,   0.058,   0.061,   0.068,   0.089,   0.125,   0.154,   0.174,   0.199,   0.248,   0.335,   0.444,   0.538,   0.587,   0.595,   0.591,   0.587,   0.584,   0.584,   0.590,   0.603,   0.620,   0.639,   0.655,   0.663,   0.663,   0.667],
    [0.122,  0.164,   0.229,   0.286,   0.327,   0.361,   0.388,   0.400,   0.392,   0.362,   0.316,   0.260,   0.209,   0.168,   0.138,   0.117,   0.104,   0.096,   0.090,   0.086,   0.084,   0.084,   0.084,   0.084,   0.084,   0.085,   0.090,   0.098,   0.109,   0.123,   0.143,   0.169,   0.205,   0.244,   0.287,   0.332],
    [0.096,  0.115,   0.131,   0.135,   0.133,   0.132,   0.130,   0.128,   0.125,   0.120,   0.115,   0.110,   0.105,   0.100,   0.095,   0.093,   0.092,   0.093,   0.096,   0.108,   0.156,   0.265,   0.399,   0.500,   0.556,   0.579,   0.588,   0.591,   0.593,   0.594,   0.598,   0.602,   0.607,   0.609,   0.609,   0.610],
    [0.092,  0.116,   0.146,   0.169,   0.178,   0.173,   0.158,   0.139,   0.119,   0.101,   0.087,   0.075,   0.066,   0.060,   0.056,   0.053,   0.051,   0.051,   0.052,   0.052,   0.051,   0.052,   0.058,   0.073,   0.096,   0.119,   0.141,   0.166,   0.194,   0.227,   0.265,   0.309,   0.355,   0.396,   0.436,   0.478],
    [0.061,  0.061,   0.062,   0.063,   0.064,   0.066,   0.069,   0.075,   0.085,   0.105,   0.139,   0.192,   0.271,   0.376,   0.476,   0.531,   0.549,   0.546,   0.528,   0.504,   0.471,   0.428,   0.381,   0.347,   0.327,   0.318,   0.312,   0.310,   0.314,   0.327,   0.345,   0.363,   0.376,   0.381,   0.378,   0.379],
    [0.063,  0.063,   0.063,   0.064,   0.064,   0.064,   0.065,   0.066,   0.067,   0.068,   0.071,   0.076,   0.087,   0.125,   0.206,   0.305,   0.383,   0.431,   0.469,   0.518,   0.568,   0.607,   0.628,   0.637,   0.640,   0.642,   0.645,   0.648,   0.651,   0.653,   0.657,   0.664,   0.673,   0.680,   0.684,   0.688],
    [0.066,  0.079,   0.102,   0.146,   0.200,   0.244,   0.282,   0.309,   0.308,   0.278,   0.231,   0.178,   0.130,   0.094,   0.070,   0.054,   0.046,   0.042,   0.039,   0.038,   0.038,   0.038,   0.038,   0.039,   0.039,   0.040,   0.041,   0.042,   0.044,   0.045,   0.046,   0.046,   0.048,   0.052,   0.057,   0.065],
    [0.052,  0.053,   0.054,   0.055,   0.057,   0.059,   0.061,   0.066,   0.075,   0.093,   0.125,   0.178,   0.246,   0.307,   0.337,   0.334,   0.317,   0.293,   0.262,   0.230,   0.198,   0.165,   0.135,   0.115,   0.104,   0.098,   0.094,   0.092,   0.093,   0.097,   0.102,   0.108,   0.113,   0.115,   0.114,   0.114],
    [0.050,  0.049,   0.048,   0.047,   0.047,   0.047,   0.047,   0.047,   0.046,   0.045,   0.044,   0.044,   0.045,   0.046,   0.047,   0.048,   0.049,   0.050,   0.054,   0.060,   0.072,   0.104,   0.178,   0.312,   0.467,   0.581,   0.644,   0.675,   0.690,   0.698,   0.706,   0.715,   0.724,   0.730,   0.734,   0.738],
    [0.058,  0.054,   0.052,   0.052,   0.053,   0.054,   0.056,   0.059,   0.067,   0.081,   0.107,   0.152,   0.225,   0.336,   0.462,   0.559,   0.616,   0.650,   0.672,   0.694,   0.710,   0.723,   0.731,   0.739,   0.746,   0.752,   0.758,   0.764,   0.769,   0.771,   0.776,   0.782,   0.790,   0.796,   0.799,   0.804],
    [0.145,  0.195,   0.283,   0.346,   0.362,   0.354,   0.334,   0.306,   0.276,   0.248,   0.218,   0.190,   0.168,   0.149,   0.127,   0.107,   0.100,   0.102,   0.104,   0.109,   0.137,   0.200,   0.290,   0.400,   0.516,   0.615,   0.687,   0.732,   0.760,   0.774,   0.783,   0.793,   0.803,   0.812,   0.817,   0.825],
    [0.108,  0.141,   0.192,   0.236,   0.261,   0.286,   0.317,   0.353,   0.390,   0.426,   0.446,   0.444,   0.423,   0.385,   0.337,   0.283,   0.231,   0.185,   0.146,   0.118,   0.101,   0.090,   0.082,   0.076,   0.074,   0.073,   0.073,   0.074,   0.076,   0.077,   0.076,   0.075,   0.073,   0.072,   0.074,   0.079],
    [0.189,  0.255,   0.423,   0.660,   0.811,   0.862,   0.877,   0.884,   0.891,   0.896,   0.899,   0.904,   0.907,   0.909,   0.911,   0.910,   0.911,   0.914,   0.913,   0.916,   0.915,   0.916,   0.914,   0.915,   0.918,   0.919,   0.921,   0.923,   0.924,   0.922,   0.922,   0.925,   0.927,   0.930,   0.930,   0.933],
    [0.171,  0.232,   0.365,   0.507,   0.567,   0.583,   0.588,   0.590,   0.591,   0.590,   0.588,   0.588,   0.589,   0.589,   0.591,   0.590,   0.590,   0.590,   0.589,   0.591,   0.590,   0.590,   0.587,   0.585,   0.583,   0.580,   0.578,   0.576,   0.574,   0.572,   0.571,   0.569,   0.568,   0.568,   0.566,   0.566],
    [0.144,  0.192,   0.272,   0.331,   0.350,   0.357,   0.361,   0.363,   0.363,   0.361,   0.359,   0.358,   0.358,   0.359,   0.360,   0.360,   0.361,   0.361,   0.360,   0.362,   0.362,   0.361,   0.359,   0.358,   0.355,   0.352,   0.350,   0.348,   0.345,   0.343,   0.340,   0.338,   0.335,   0.334,   0.332,   0.331],
    [0.105,  0.131,   0.163,   0.180,   0.186,   0.190,   0.193,   0.194,   0.194,   0.192,   0.191,   0.191,   0.191,   0.192,   0.192,   0.192,   0.192,   0.192,   0.192,   0.193,   0.192,   0.192,   0.191,   0.189,   0.188,   0.186,   0.184,   0.182,   0.181,   0.179,   0.178,   0.176,   0.174,   0.173,   0.172,   0.171],
    [0.068,  0.077,   0.084,   0.087,   0.089,   0.090,   0.092,   0.092,   0.091,   0.090,   0.090,   0.090,   0.090,   0.090,   0.090,   0.090,   0.090,   0.090,   0.090,   0.090,   0.090,   0.089,   0.089,   0.088,   0.087,   0.086,   0.086,   0.085,   0.084,   0.084,   0.083,   0.083,   0.082,   0.081,   0.081,   0.081],
    [0.031,  0.032,   0.032,   0.033,   0.033,   0.033,   0.033,   0.033,   0.032,   0.032,   0.032,   0.032,   0.032,   0.032,   0.032,   0.032,   0.032,   0.032,   0.032,   0.032,   0.032,   0.032,   0.032,   0.032,   0.032,   0.032,   0.032,   0.032,   0.032,   0.032,   0.032,   0.032,   0.032,   0.032,   0.032,   0.033]
])

    # Build the C matrix
    # This is based on the sRGB which is the closest color space to a GoPro (which is what I am assuming was used based off of comments in class)
    C = np.array([
    [116,  81,  67],
    [199, 147, 129],
    [ 91, 122, 156],
    [ 90, 108,  64],
    [130, 128, 176],
    [ 92, 190, 172],
    [224, 124,  47],
    [ 68,  91, 170],
    [198,  82,  97],
    [ 94,  58, 106],
    [159, 189,  63],
    [230, 162,  39],
    [ 35,  63, 147],
    [ 67, 149,  74],
    [180,  49,  57],
    [238, 198,  20],
    [193,  84, 151],
    [  0, 136, 170],
    [245, 245, 243],
    [200, 202, 202],
    [161, 163, 163],
    [121, 121, 122],
    [ 82,  84,  86],
    [ 49,  49,  51]
    ])

    #print(R.shape, C.shape)

    # Compute M
    M = np.linalg.pinv(C) @ R


    # read the image 
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize the image for the HSI conversion
    img = img.astype(np.float32) / 255.0

    H, W, c = img.shape
    pixels = img.reshape(-1, 3) # flatten the image for the matrix conversion
    hsi_pixels = pixels @ M

    hsi_cube = hsi_pixels.reshape(H, W, 36) # we get 31 wavelenghts out of the conversion

    #hsi_cube = np.clip(hsi_cube, 0.0, 1.0) # clip so we get real refletance values



    wvln = [380 + x for x in range(0, 360, 10)] # [380,730] nm range
    #print(wvln, len(wvln))
    #print(hsi_cube.shape)

    # HSI Plot
    if False:
        fig, ax = plt.subplots(6,6,sharex=True, sharey=True)
        for i in range(0, 6):
            for j in range(0, 6):

                idx = i * 6 + j

                if idx >= len(wvln):
                    continue
                else:
                    ax[i, j].imshow(hsi_cube[..., idx], cmap="gray")
                    ax[i,j].set_title(f"{wvln[idx]} nm")

        plt.show()

    # Get the target color spectra
    tc_spec = np.asarray(target_color) @ M


    def cosine_sim(A, B):
        cs = np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))
        return np.nanmin([cs, 1])


    # Cosine Plotting
    if False:
        fig, ax = plt.subplots(3, 1)
        pbar = tqdm(total=(H//10)*(W//10))
        i = 0
        for h in range(0, H, 10):
            for w in range(0, W, 10):
                ax[1].plot(hsi_cube[h, w])

                ax[2].scatter(i, cosine_sim(tc_spec, hsi_cube[h, w]))

                pbar.update(1)
        pbar.close()
        ax[1].set_title("Image Pixel Spectras")
        ax[2].set_title("Cosine sim between target color and image pixel")
        ax[0].plot(tc_spec)
        ax[0].set_title("Target Color Spectra")


        plt.show()


    K = 10
    pbar = tqdm(total=(H//K)*(W//K))
    sim = []
    i = 0
    for h in range(0, H, K):
        _w = []
        for w in range(0, W, K):
            _w.append(cosine_sim(tc_spec, hsi_cube[h, w]))
            pbar.update(1)
        sim.append(_w)

    pbar.close()


    print(np.mean(sim), np.std(sim))

    # Get the min and max based on the min and max pixel values for those pixels within the range of mean+/- std
    # Can't do anything else fancy since there isnt really enough data and the other option is clustering and that is kinda been done

    pthresh = np.mean(sim) + np.std(sim)
    nthresh = np.mean(sim) - np.std(sim)

    idx = np.where((nthresh <= sim) & (sim <= pthresh))
    Y, X = idx[0], idx[1]
    Y *= K
    X *= K

    # return the min and max for that range

    # WARNING THIS DATA IS NORMALIZED
    img = img * 255
    cluster = img[Y, X, :]
    min_value = np.percentile(cluster, 5, axis=0).astype(np.uint8)
    max_value = np.percentile(cluster, 95, axis=0).astype(np.uint8)

    return min_value, max_value
    








#----------------------------------------------
#
#            M A I N   L O O P
#
#---------------------------------------------
# Similar to that of a4.py
# but we dont need a lot of the other things now
# So while a good MLE process would keep and refine, 
# I have to throw it away or I will waste all my time managing functions that 
# dont matter here
def label_piepeline(args, custom_roi=True, custom_thresh=False):
    """
    recreate the pipeline of the label command so 
    tests can be done a lot easier since the alina pipeline
    is already compiled and cant change the "base" code
    """

    # set the configurations
    config = PipelineConfig(
        input_dir=args['input_dir'] or ".",
        output_images_dir=args['output_images_dir'],
        output_coords_dir=args['output_coords_dir'],
        log_file=args['log_file'],
        peak_pixel_threshold=args['peak_threshold'],
        min_white_pixels=args['min_white_pixels'],
        circular_threshold=args['circular_threshold'],
        yellow_lower=[tuple(args['yellow_lower'])],
        yellow_upper=[tuple(args['yellow_upper'])],
        mask_ignore_left_columns=args['mask_ignore_left_columns'],
    )

    # Get image ROI via some fancy algorithms 
    image_filenames = sorted(
        f for f in config.input_dir.iterdir() if f.suffix.lower() == ".jpg"
    )
    total = len(image_filenames)


    rois = []
    if args["custom_roi"]:
        for i, path, in enumerate(image_filenames, start=1):

            # We leave the claude one left since it did the best 
            # And I really dont want to deal with:
            # a) manual labeling
            # b) dependecy issues between keras, torch, transformers again
            match args["custom_roi"]:
                case "claude":
                    img_roi = roi_via_claude(path, False) # Don't plot the BBox because that was a4 not needed here

            rois.append(img_roi)
            #print(img_roi)
    else:
        # Yoinked from the cli.py
        reference_image = sorted(config.input_dir.glob("*.jpg"))[0]
        print("Draw the ROI: click Bottom-Left, Top-Left, Top-Right, Bottom-Right, then press any key.")
        roi_points = select_roi(str(reference_image))
        rois.append(roi_points)


    # This has been moved to assignment 5, but it will live here for now.
    if custom_thresh:
        # Lets do some fancy color theory here :)

        # yellow lower upper
        yls, yus = [], []
        # white lower upper
        wls, wus = [], []
        for i, path in (enumerate(tqdm(image_filenames), start=1)):

            if args["custom_thresh"] == "kmeans":
                # Beacause the Kmeans is every pixel gets touched, we are going to say every ten frames
                # which for this assignmnet is fine because the frame changes significantly about every ten or so anyway
                if i % 10 == 0: 
                    yl, yu = kmean_color_threshold(path, YELLOW_RGB)
                    wl, wu = kmean_color_threshold(path, WHITE_RGB)

                    yls.append(yl)
                    yus.append(yu)
                    wls.append(wl)
                    wus.append(wu)

            if args["custom_thresh"] == "hsi":
                    yl, yu = hsi_color_threshold(path, YELLOW_RGB)
                    wl, wu = hsi_color_threshold(path, WHITE_RGB)

                    yls.append(yl)
                    yus.append(yu)
                    wls.append(wl)
                    wus.append(wu)

        # Have to average because the pipeline only accepts a single threshold
        # realistically the entire ALINA pipeline needs to be broken at this point
        # because it is not designed for per-image labeling but rather "batches"
        # which works great when the entire video is of the same thing, but within each example
        # the view is all over the place :(
        yellow_lower = np.uint8(np.min(yls, axis=0))
        yellow_upper = np.uint8(np.max(yus, axis=0))

        white_lower = np.uint8(np.min(wls, axis=0))
        white_upper = np.uint8(np.max(wus, axis=0))

        fig, ax = plt.subplots(3,1)    
        ax[0].imshow(np.asarray([[YELLOW_RGB]]))
        ax[0].set_yticklabels("REF")
        ax[1].imshow(np.asarray([[yellow_lower]]))
        ax[1].set_yticklabels("LOWER")
        ax[2].imshow(np.asarray([[yellow_upper]]))
        ax[2].set_yticklabels("UPPER")
        for a in ax:
            a.set_yticks([])
            a.set_xticks([])

        plt.subplots_adjust(wspace=0)
        plt.show()

        # CONVERT THE RGB TO HSV 

        yellow_lower = cv2.cvtColor(np.asarray([[yellow_lower]]), cv2.COLOR_RGB2HSV)
        yellow_upper = cv2.cvtColor(np.asarray([[yellow_upper]]), cv2.COLOR_RGB2HSV)
        # Normalize them because for some reason
        # ALINA normalizes INCORRECTLY since cv is [0, 180) [0, 255), [0, 255)
        # but it expects the range for all to be within 255
        lh, ls, lv = yellow_lower[0, 0, :]
        uh, us, uv = yellow_upper[0, 0, :]

        # fix some issues where the staturations are not lower->higher (same with value)
        yl = (min(lh, uh), min(ls, us), min(lv, uv))
        yu = (max(lh, uh), max(ls, us), max(lv, uv))

        # Do the same for white
        white_lower = cv2.cvtColor(np.asarray([[white_lower]]), cv2.COLOR_RGB2HSV)
        white_upper = cv2.cvtColor(np.asarray([[white_upper]]), cv2.COLOR_RGB2HSV)
        # Normalize them because for some reason
        # ALINA normalizes INCORRECTLY since cv is [0, 180) [0, 255), [0, 255)
        # but it expects the range for all to be within 255
        lh, ls, lv = white_lower[0, 0, :]
        uh, us, uv = white_upper[0, 0, :]

        # fix some issues where the staturations are not lower->higher (same with value)
        wl = (min(lh, uh), min(ls, us), min(lv, uv))
        wu = (max(lh, uh), max(ls, us), max(lv, uv))

        config.yellow_lower = [yl, wl]
        config.yellow_upper = [yu, wu]
    else:
        config.yellow_lower = [config.yellow_lower[0][0], config.yellow_lower[0][1]]
        config.yellow_upper = [config.yellow_upper[0][0], config.yellow_upper[0][1]]

    print(config)


    # run the batch (bad because it doesnt do ROI per image)
    #run_batch(config, rois)
    # Do the localized version since it just works :)
    local_run_batch(config, rois)


if __name__ == "__main__":

    args = {
        "input_dir": "./../data/Raw_Data/vidd_1",
        "output_images_dir": "./../outputs/tmp/annotated",
        "output_coords_dir": "./../outputs/tmp/coords", 
        "log_file": "./../outputs/tmp/timing.log",
        "peak_threshold": 5,
        "min_white_pixels": 20,
        "circular_threshold": 15,
        "yellow_lower": [(0, 70, 170), (125,125,125)], # HSV
        "yellow_upper": [(255, 255, 255), (255,255,255)], # HSV
        "mask_ignore_left_columns": 300,
        "custom_roi": "claude", # unet, segnet, segformer, kmeans, claude (None, if manual ALINA pipeline)
        "custom_thresh": "hsi" # kmeans, hsi, etc
    }

    label_piepeline(args, custom_roi=False, custom_thresh=False)