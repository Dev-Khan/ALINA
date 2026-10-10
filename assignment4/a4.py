

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

from huggingface_hub import hf_hub_download, snapshot_download



import matplotlib.pyplot as plt

import numpy as np

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

import torch

print("Before load of transformer")
from transformers import pipeline
print("transformers loaded")

# THIS HAS TO BE AFTER TORCH WHYYYYYYYYYY
# This is horrific, why
# C figured out how to do this decades ago
# Why does import order matter like this....
# Torch is nothing but issues
import tensorflow as tf
import keras

from PIL import Image

import time

from tqdm import tqdm


SEED = 123456789

keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()

# https://sightline.us/images/pdf/FAAAC_150-5370-10HP620.pdf
# Both are valid FED standard colors
YELLOW_HSV = [22, 255, 242] 
# [19, 226, 239] 

YELLOW_RGB = [242, 176, 0]
# [239, 160, 27]

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
    lower = np.array(config.yellow_lower)
    upper = np.array(config.yellow_upper)
    mask = cv2.inRange(color_features, lower, upper)
    if config.mask_ignore_left_columns > 0:
        mask[:, :config.mask_ignore_left_columns] = 0


    # 3.6 histogram analysis -> vertical projection, peak column
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



def roi_via_unet(path, model, plotter=False):
    
    # Concept: Using an image segemenation model we can 
    # identify "road" that can be used as the bounding box for the ROI
    # should be applied to each frame


    # Per the model instructions we need to reduce the image size
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    w, h = img.shape[:2]

    X = cv2.resize(img, (256,256))
    X = X / 255.0
    X = np.expand_dims(X, axis=0)

    pred = model.predict(X)
    mask = np.argmax(pred[0], axis=-1)

    # Class maps:
    # 0 = background | 1 = road | 2 = lane
    # We only care about 1 = road (though lane will be intresting)

    # Expand the mask back into the original image dims
    mask_full = cv2.resize(mask.astype(np.uint8),
                           (h, w),
                           interpolation=cv2.INTER_NEAREST
    )
    print(mask_full.shape, img.shape)

    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[mask_full == 1] = (255, 0, 0) # road
    color_mask[mask_full == 2] = (0, 255, 0) # lane

    if plotter:
        # plot the two images side by side for some verification
        fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
        ax[0].imshow(img)
        ax[1].imshow(color_mask)
        plt.show()

    # For naive ROI since the area is plotchy is 
    # Take the limits of any "road" area 

    # IMPORTANT: take the points in the order ALINA expect
    # Which is: bottom_left, top_left, top_right, bottom_right
    ys, xs = np.where(mask_full == 1) # any pixel that is a road
    x_min, x_max = np.min(xs), np.max(xs)
    y_min, y_max = np.min(ys), np.max(ys)
    bottom_left = [x_min, y_max]
    top_left = [x_min, y_min]
    top_right = [x_max, y_min]
    bottom_right = [x_max, y_max]

    # Have to inlucde dummy data because for some reason the ALINA program
    # uses ALL points and not just unique set...
    # Needs ot be wrapped as well because extra idk why 
    roi = [[bottom_left, top_left, [-1,-1], top_right, [-1,-1], bottom_right]]

    return roi



def roi_via_segnet(path, model, plotter=False):

    # Per the model instructions we need to reduce the image size
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    w, h = img.shape[:2]

    X = cv2.resize(img, (2048, 2048))
    X = X / 255.0
    X = torch.from_numpy(X).permute(2, 0, 1) # Channels first in PyTorch (for some reason idk why) 
    X = X.unsqueeze(0).float()
    print(X.shape)

    with torch.no_grad():
        pred = model(X)
    pred = torch.sigmoid(pred)
    print(pred.min(), pred.max())
    # Look what PyTorch needs to mimic a fraction of Keras' power smh my head
    mask = pred.squeeze().cpu().numpy()
    mask = (mask > 0.05).astype(np.uint8) * 255

    mask_full = cv2.resize(mask.astype(np.uint8),
                        (h, w),
                        interpolation=cv2.INTER_NEAREST
    )
    print(mask_full.shape, img.shape)

    if plotter or True:
        # plot the two images side by side for some verification
        fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
        ax[0].imshow(img)
        ax[1].imshow(mask_full)
        plt.show()

    # For naive ROI since the area is plotchy is 
    # Take the limits of any "road" area 

    # IMPORTANT: take the points in the order ALINA expect
    # Which is: bottom_left, top_left, top_right, bottom_right
    ys, xs = np.where(mask_full == 1) # any pixel that is a road
    x_min, x_max = np.min(xs), np.max(xs)
    y_min, y_max = np.min(ys), np.max(ys)
    bottom_left = [x_min, y_max]
    top_left = [x_min, y_min]
    top_right = [x_max, y_min]
    bottom_right = [x_max, y_max]

    # Have to inlucde dummy data because for some reason the ALINA program
    # uses ALL points and not just unique set...
    # Needs ot be wrapped as well because extra idk why 
    roi = [[bottom_left, top_left, [-1,-1], top_right, [-1,-1], bottom_right]]

    return roi


def roi_via_segformer(path, model, processor, plotter=False):
    # https://huggingface.co/nvidia/segformer-b0-finetuned-cityscapes-640-1280
    
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    w, h = img.shape[:2]

    # Doesnt like open-cv and that makes me sad
    _img = Image.fromarray(img)

    #inputs = processor(imgaes=img, return_tensors="pt")
    outputs = model(_img)
    road_label = outputs[0] # road is the first label
    mask = road_label["mask"]
    print(mask)

    # convert the mask back to cv2
    numpy_image = np.array(mask)
    mask = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

    # Expand the mask back into the original image dims because accorind to the docs it does quater compression within the model itself
    mask_full = cv2.resize(mask.astype(np.uint8),
                           (h, w),
                           interpolation=cv2.INTER_NEAREST
    )
    print(mask_full.shape, img.shape)

    if plotter:
        # plot the two images side by side for some verification
        fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
        ax[0].imshow(img)
        ax[1].imshow(mask_full)
        plt.show()

    ys, xs = np.where(mask_full[..., 0] == 255) # any pixel that is a road
    x_min, x_max = np.min(xs), np.max(xs)
    y_min, y_max = np.min(ys), np.max(ys)
    bottom_left = [x_min, y_max]
    top_left = [x_min, y_min]
    top_right = [x_max, y_min]
    bottom_right = [x_max, y_max]

    # Have to inlucde dummy data because for some reason the ALINA program
    # uses ALL points and not just unique set...
    # Needs ot be wrapped as well because extra idk why 
    roi = [[bottom_left, top_left, [-1,-1], top_right, [-1,-1], bottom_right]]

    return roi



def roi_via_horizon(path, plotter=False):

    # Via Kmeans clustering determine if pixel is "sky" or "ground" and use that terminator to select the ROI anything below horizon
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    w, h = img.shape[:2]
    
    h, w, _ = img.shape
    blurred = cv2.GaussianBlur(img, (7, 7), 0)

    # Reshape image pixels to a 2D array of RGB values
    pixel_values = blurred.reshape((-1, 3)).astype(np.float32)
    
    # Cluster into K=2 groups (ideally Sky vs. Ground/Sea)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_values, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Convert back to an 8-bit mask separating the two dominant zones
    segmented_mask = labels.reshape((h, w)).astype(np.uint8) * 255

    mask_full = segmented_mask
    if plotter:
        # plot the two images side by side for some verification
        fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
        ax[0].imshow(img)
        ax[1].imshow(mask_full)
        plt.show()

    # Get the ROI by finding where the average value of ground sky transition occurs
    transition_height = []
    for y in range(mask_full.shape[1]):
        pixels = mask_full[:, y]
        ground_idx = np.where(pixels == 255)[0]
        if ground_idx.size > 0: # actually make sure the transition occurs
            # the ground ends at the first index where it is 255 (since an image starts with y=0 at top)
            transition_height.append(ground_idx[0])

    y_min = np.mean(transition_height)
    ys, xs = np.where(mask_full == 255) # any pixel that is a road
    x_min, x_max = np.min(xs), np.max(xs)
    y_max = np.max(ys)
    bottom_left = [x_min, y_max]
    top_left = [x_min, y_min]
    top_right = [x_max, y_min]
    bottom_right = [x_max, y_max]

    # Have to inlucde dummy data because for some reason the ALINA program
    # uses ALL points and not just unique set...
    # Needs ot be wrapped as well because extra idk why 
    roi = [[bottom_left, top_left, [-1,-1], top_right, [-1,-1], bottom_right]]

    return roi


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


    roi_path = f"./ClaudeAttempt/roi_labels_{vid_set}/{img_num}.txt"
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




    



"""
# Does not work because of unsupported numpy versions. 
# Pretrained model: https://huggingface.co/leeyunjai/yolo11-road-seg
model_path = hf_hub_download(
    repo_id="leeyunjai/yolo11-road-seg",
    filename="yolo11m-road-seg.pt",
    local_dir="./yolo_road_cache_dir",
    local_dir_use_symlinks=False
)
MODEL = YOLO(model_path)
def roi_via_yolo11_finetuned(path, model, plotter=False):

    # According to the document I just pass it the file path and it does its magic
    # Per the model instructions we need to reduce the image size
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    w, h = img.shape[:2]

    X = cv2.resize(img, (256,256))
    plt.imshow(X)
    plt.show()
    result = model.predict(source=X)

    # Just show it to see if it actuall works
    print(result)
    result.show()

    assert True == False
"""

def get_yellow_threshold(path):
    
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert to HSV since that is what ALINA is expecting


    pixels = np.reshape(img, (-1, img.shape[-1]))

    K = 200
    kmeans = KMeans(n_clusters=K, init="k-means++", random_state=SEED)
    Y = kmeans.fit_predict(pixels)

    y_hat = kmeans.predict([YELLOW_RGB])[0]

    cluster = pixels[Y == y_hat] # get the points belonging to the predicted cluster

    # Get the min and max of the cluster to build the threshold bounds (but not the exact min and max since its just going to return black and white)
    # take from the 5% to 95% range of the cluster
    min_value = np.percentile(cluster, 5, axis=0).astype(np.uint8)
    max_value = np.percentile(cluster, 95, axis=0).astype(np.uint8)


    """
    # Keeping this here for posterity sake atm
    # Just check to see what color everything is actually
    ref = [
        [YELLOW_RGB],
        [min_value],
        [max_value]
    ]
    ref = np.asarray(ref, dtype=np.uint8)
    print(ref)

    #ref = cv2.cvtColor(ref, cv2.COLOR_HSV2RGB)

    plt.imshow(ref, aspect="auto")
    plt.show()

    assert True == False
    """

    return min_value, max_value



def label_piepeline(args, custom_roi=True, custom_thresh=False):
    """
    recreate the pipeline of the label command so 
    tests can be done a lot easier since the alina pipeline
    is already compiled and cant change the "base" code
    """

    if args["custom_roi"] == "unet":
        # Pretrained model: # https://huggingface.co/yaraa11/road-lane-semantic-segmentation-unet-resnet50
        model_path = hf_hub_download(
            repo_id="yaraa11/road-lane-semantic-segmentation-unet-resnet50",
            filename="road_segmentation.keras",
            local_dir="./road_lane_cache_dir",
            local_dir_use_symlinks=False
        )
        MODEL = keras.saving.load_model(model_path, compile=False)
        MODEL.summary()
    elif args["custom_roi"] == "segnet":
        model_path = hf_hub_download(
            repo_id="manish2607/road_extrcation_model_BAH",
            filename="road_model_Deployment.pt",
            local_dir="./segnet_road_cache_dir",
            local_dir_use_symlinks=False
        )
        MODEL = torch.jit.load(model_path)
        MODEL.eval()
    elif args["custom_roi"] == "segformer":
        print("HERE")
        MODEL = pipeline("image-segmentation", model="nvidia/segformer-b0-finetuned-cityscapes-640-1280")
        PROCESSOR = None
        print(PROCESSOR)
        print(MODEL)






    # set the configurations
    config = PipelineConfig(
        input_dir=args['input_dir'] or ".",
        output_images_dir=args['output_images_dir'],
        output_coords_dir=args['output_coords_dir'],
        log_file=args['log_file'],
        peak_pixel_threshold=args['peak_threshold'],
        min_white_pixels=args['min_white_pixels'],
        circular_threshold=args['circular_threshold'],
        yellow_lower=tuple(args['yellow_lower']),
        yellow_upper=tuple(args['yellow_upper']),
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
            # Plot every 10th image just for some sanity checks
            do_plot = False
            if i % 10 == 0:
                do_plot = True

            match args["custom_roi"]:
                case "unet":
                    img_roi = roi_via_unet(path, MODEL, do_plot)
                case "segnet":
                    img_roi = roi_via_segnet(path, MODEL, do_plot)
                case "segformer":
                    img_roi = roi_via_segformer(path, MODEL, PROCESSOR, do_plot)
                case "kmeans":
                    img_roi = roi_via_horizon(path, do_plot
                                              )
                case "claude":
                    img_roi = roi_via_claude(path, do_plot)

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
        lowers, uppers = [], [] 
        for i, path in (enumerate(tqdm(image_filenames), start=1)):
            if i % 10 == 0 or True: # just gonna leave this here iwth a short cucurit because itll be handy later
                lower, upper = get_yellow_threshold(path)
                lowers.append(lower)
                uppers.append(upper)

        # Have to average because the pipeline only accepts a single threshold
        # realistically the entire ALINA pipeline needs to be broken at this point
        # because it is not designed for per-image labeling but rather "batches"
        # which works great when the entire video is of the same thing, but within each example
        # the view is all over the place :(
        yellow_lower = np.uint8(np.min(lowers, axis=0))
        yellow_upper = np.uint8(np.max(uppers, axis=0))

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
        l = (min(lh, uh), min(ls, us), min(lv, uv))
        u = (max(lh, uh), max(ls, us), max(lv, uv))

        config.yellow_lower = l
        config.yellow_upper = u

    print(config)

    # run the batch (bad because it doesnt do ROI per image)
    #run_batch(config, rois)
    # Do the localized version since it just works :)
    local_run_batch(config, rois)







if __name__ == "__main__":

    args = {
        "input_dir": "./../data/Raw_Data/vidd_3",
        "output_images_dir": "./../outputs/tmp/annotated",
        "output_coords_dir": "./../outputs/tmp/coords", 
        "log_file": "./../outputs/tmp/timing.log",
        "peak_threshold": 5,
        "min_white_pixels": 20,
        "circular_threshold": 15,
        "yellow_lower": [0, 70, 170], # HSV
        "yellow_upper": [255, 255, 255], # HSV
        "mask_ignore_left_columns": 300,
        "custom_roi": "claude" # unet, segnet, segformer, kmeans, claude (None, if manual ALINA pipeline)
    }

    label_piepeline(args, custom_roi=False, custom_thresh=False)