
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
    color_features = normalize_color_features(warped)
    #color_features = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)

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
    print(components)

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

    print(roi)

    # also needs some bubble wrap for downstraem ALINA functions
    return [roi]

# This was in the a4.py however 
# We are updating it so it can do yellow and white
def get_color_threshold(path, target_color):
    
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
            print(img_roi)
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
                    yl, yu = get_color_threshold(path, YELLOW_RGB)
                    wl, wu = get_color_threshold(path, WHITE_RGB)

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
        "yellow_lower": [0, 70, 170], # HSV
        "yellow_upper": [255, 255, 255], # HSV
        "mask_ignore_left_columns": 300,
        "custom_roi": "claude", # unet, segnet, segformer, kmeans, claude (None, if manual ALINA pipeline)
        "custom_thresh": "kmeans" # kmeans, etc
    }

    label_piepeline(args, custom_roi=False, custom_thresh=True)