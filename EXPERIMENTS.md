# Assignment 5

## Running experiments

All code required is within the `assignment5` folder. The only command to run the pipeline needed is the following:
```python a5.py```

However, to conduct various experiments the bottom of the code configuration will need to be updated by the user. This could have been command line arguments, but for rapid prototyping on the assignmnet timeline it is manual configurations. 

```    args = {
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
        "custom_roi": "claude", # We only do the cluade version in a5
        "custom_thresh": "kmeans" # This controls the algorithms for color thresholding detection
}
```



# Color Thresholding in ALINA

By default ALINA does not offer any way to detect anything other than the yellow lines within its program. In order to do this we must change a few processes to allow yellow and white line detection. First we must change the `PipelineConfig` dataclass to redefine the type-hinting from a static tuple that accepts three values (h,s,v) in two a dynamic 2D-array of size [N, 3] where N is the number of thresholds to pass and 3 denoting the values h, s, v. 

The result of that update is:
```python
    yellow_lower: np.ndarray = field(
        default_factory=lambda: np.array([[0, 70, 170]], dtype=np.int32)
    )
    yellow_upper: np.ndarray = field(
        default_factory=lambda: np.array([[255,255,255]], dtype=np.int32)
    )
```

Note: we do not change the naming scheme here instead hijack the variable for time constrains. This variable should be updated to `color_thresholds_lower` and `color_thresholds_upper` but that requires a more deep dive tear down within ALINA to refactor. 

As pointed out in the assignment 4 debrief the function `normalize_color_features` does not correctly normalize the Hue component of HSV color space in CSV. Therefore, we corect it so that it conforms to the HSV OpenCV color space such that hue is constrained to [0, 179]. 

From there inside of the process image function we must update the process to generate a mask for each threshold which can be done via a simple for loop over each of the lower and upper pairs. 

```python
    mask = []
    for lower, upper in zip(lowers, uppers):
        _mask = cv2.inRange(color_features, lower, upper)
        if config.mask_ignore_left_columns > 0:
            _mask[:, :config.mask_ignore_left_columns] = 0

        mask.append(_mask)
```

In the case of this assignment mask will now be a 3D arary of shape [T, w, h] where T is the number of thresholds passed and the image width (w) and height (h). For this case T is constrianed to 2 since we only pass in a threshold for yellow and white. 

To prevent unessissary computation we merge the masks back together via a binary OR operation. This operation was selected becasue the CIRCLEDAT process does not care about color or type of pixels, rather only the intentisty of them and how many of them there are. So the reuslting mask that CIRCLEDAT cares about is a mask where high pixels belong to any threshold. 

Note: This could be changed so the process returns two seperated independent values for different line colors. However the way the ALINA process is built is a mentality that "a line is a line regardless of color". In a real application for intelligent aircraft mobility, the line colors do carry semantic meaning and should be segemented out approriately. While, that could be done for this assignmnet it would require significant overhaul of ALINA and generating a new EVAL pipeline to produce segemented labels for metric verification. 

Note: Inside of the process_image the calcualtion for the historgarm (`calc_histogram` function) consumes a variable from the configuration called `min_white_pixels`. However this variable name is a misnomer and should be changed. The histogram is not consuming pixels of any color, instead the image it consumes is a binary image mask with high/low pixels. Recommend to refactor to `min_mask_pixels` to keep the program readable. 


# Data considerations

There is a signficiant lack of data to do color thresholding on both yellow and/or white lines. 

Only 20 images within all of ALINA actually contain both white or yellow lines (note: use of OR is CS based or that it can be yellow, white or both at the same time). 
Those 20 images are images `00001.jpg` - `00020.jpg` in the `vidd_1` videoset. Therefore, there is not enough data for any meaningful process to take place. The lack of data results in highly specific models that work only on this one view at this one part of the runway; no meaning statistics can be derived as the sample size is not large or varied enough to have any form of train/test/validation or any direvitive of data spliting for metrics (i.e. k-fold cross validation). 

Additionally, the camera is not defined within the ALINA paper itself. However, in the AssistTaxi dataset paper it was stated that the cameras were GoPro Hero 3 models. However, the camera(s) themselves were not color calibrated via an industry standard (such as the Calibrite Color Checker). Without the true color response of the camera there is no objective way to reduce the problem into a simple computer vision based task or provide insight into simplisitic models for color identification. 


As for the color(s) themselves the assignment describes them as high abstract concepts "yellow" and "white". However, FAA governed airports are strictly controlled areas and therefore every color has an offical governmnet assignment. 

For our case we only care about two of them the offical yellow and white.
It took a while to find the offical government document that outlines these requirements as most of them allude to a "reference" document however that document is `Item P-620 Runway and Taxiway Markings` [Source](https://sightline.us/images/pdf/FAAAC_150-5370-10HP620.pdf)

Within this document there is an unlabled tabled that lists all the colors and the offical Fed Standard No. 595 Color Number. This table is reproduced below:

| Paint Color | Fed Std. No. 595 Color Number |
| :--- | ---: |
| White | 37925 |
| Red | 31136 |
| Yellow | 33538 or 33655 |
| Black | 37038 |
| Pink | 1 part 31136 to 2 parts 37925 |
| Green | 34108 |

From here we can define the target colors as the Color number 33538 or 33655 for yellow and 37925 for white. These end up being RGB values of [244, 159, 33], [247, 173, 16], and [239, 241, 230] respectively. 

Because of the lack of data there is an exploit to make the problem simpler. Under the table in the document above is a key statement about the standardization of the color implementation that reads:
```
Waterborne or solvent base black paint should be used to outline a border at least 6 inches (150 mm) wide around markings on all light-colored pavements. Performed thermoplastic markings shall have a non-reflectorized black border integral to the marking.
```

Because the airport used for the data provided is considered "light-colored pavement" we can solve the any color line problem by simply identifying the black borders around any line marking within the ROI. The only consideration to have is that the propeller of the aircraft is nearly identidical to that of the black border. However, that is where the width measurments come into play. In an reprojected space the propeller would be much larger than the 150mm since we would effectively be looking for a sequence of B*B where B is the color black the wildcard * denoting any color then another Black. Then using the size of the wildcard we can infer the size of the black border and determine if it is within the theortical sizing threshold. 


# Experiments


## KMeans Unsupervised

The inital concept for application of dynamic color thresholding is a K-Means unsuprivised learning. The reason that this works unlike in assignment 4 for horizon detection is the fact that we have a known ground truth "probe" color. In this case it is the Fed standard yellow and white colors. 

By taking every pixel's RGB value and clustering them by likeness we can find the cluster of pixels closest to each of the probe colors. The boundary of the cluster define the thresholding for that color. However, to prevent a pure black or white washout we take the 5th to 95th percintile boundary so the resulting cluster is in real color space (i.e. returns colors that we actually care about). 
This method was selected over taking the mean cluster value and adding +/- the standard deviation to get the threshold because the problem could then effecitvely be solved by taking the probe color and applying a defined standard deviation to it to get the approximate threshold. Therefore, the precentile based system still leverages a more ML based approach than the naive application approach. 

As seen in Figure 1 when we plot the target color with the upper and lower bound found from the image it looks way off. However, in Figure 2 we see that the line detection is actually quite good for both the yellow and white lines. 


![Colors Found](./assignment5/Figures/KM_CTUL.png)
Figure 1: The target color (top) with the lower threshold color (middle) and the upper threshold color (bottom) found by the K means algorithm. 

![Colors Found](./assignment5/Figures/KM_Mask.png)
Figure 2: The masks generated by the ALINA (modified) program. Warpped image perspective (left) with the yellow (middle) and white (right) masks gerneated.


## Hyperspectral Imaging

Another option outside of leaving the data within the RBG/HSV data space is to convert the image to a hyperspectral image. There are numerous ways to convert a camera to Hyperspectral, however as mentioned above this method requires knowing the color response of the camera to accurately identifiy colors. Though, for this applicaiton it was determined that the sRGB color space is good enough of a comparision for the GoPro line of cameras. 

The resulting Hyperspectral image as seen in Figure 3 shows that the rough approximation of the wavelengths does follow the expected reality. The white and yellow lines have more response in the red and green wavelengths compared to the blue ones. This means that we can do further analysis on an automatic way of determining which pixels belong to the group closest to the target color. 

![HSI](./assignment5/Figures/Hyperspec.png)
Figure 3: Hyperspectral image of the video frame where each wavelength is its own grayscale image with pixel correlating to the intensity response at that wavelength.

When we compare the pixels to the target color spectra as seen in Figure 4, we can easily identify which should belong to that. In order to id reliabily we deploy a cosine similarity metric to measure the angle between the pixel spectra and the target spectra where 0 is the same and 1 is completely opposite. In the bottom of Figure 4, we see that there is a decent spread of the cosine similarity between the pixels and the target spectra, but more importantly that there are some that are within the 0 tolerance.


![CSim](./assignment5/Figures/CosineSim.png)
Figure 4: Target spectra of the yellow color (top) spectral responses of the pixels within the image (middle) and the cosine similarity between the target and each pixel (bottom). 


From there we deployed mean and standard deviation thresholding to dynamically and intelligently select pixels that belong to the target grouping. Then like K-means the min and max of the precentiles to build the boundaries with the threshold sample seen in Figure 5. Unlike the K-means algorithm, this does not return a neat boundary. Instead the color range is effectively black to white (i.e. all colors). This is because there is no higher order intelligence like K-means. The only way to improve it is to do some form of unsuerpvised clustering (e.g. K-means) and therefore becomes a reductive problem solved above. Addtionally, the side effect is that the masks generated are completely off; masks generate blobs instead of masking the lines cleanly (Figure 6). 

![HSI_CT](./assignment5/Figures/HSI_CTUL.png)
Figure 5: The target color (top) with the lower threshold color (middle) and the upper threshold color (bottom) found by the hyperspectral algorithm. 

![HSI_Mask](./assignment5/Figures/HSI_Mask.png)
Figure 6: Mask genreated by the hyperspectral algorithm. Warpped image perspective (left) with the yellow (middle) and white (right) masks gerneated.



## Results

All around, not the best. No proposed solution offered a significant improvement over the base line system. No solution can quite beat the base line solution as the inital baseline guess is based on a priori knowledge of the dataset that no amount of unsupervised machine learning can apporach. Additionally, because of the limited amount of white lines within the dataset, there is no real way to measure the performance on that as only about 20 images actually are valid samples. 


| Experiement | Video Set | Avg Recall (%) | Avg Precision (%) | Avg F1 (%) |
| :--- | :---: | :---: | :---: | ---: |
| Default| Vid 1 | 84.30 | 7.34 | 13.44 |
|        | Vid 2 | 65.32 | 67.12 | 65.94 |
|        | Vid 3 | 0.00  | 0.00 | 0.00 |
| Kmeans | Vid 1 | 0.00 | 0.00  | 0.00 |
|        | Vid 2 | 55.30 | 68.08  | 60.81 |
|        | Vid 3 | 0.00 | 0.00  | 0.00  |
| Hyperspectral | Vid 1 | 0.00 | 0.00  | 0.00  |
|               | Vid 2 | 17.98 | 12.50 | 6.59 |
|               | Vid 3 | 0.00 | 0.00  | 0.00  |




# Assignment 5 Debrief

There were numerous issues during this project. 

One such issue was a warp black-bar issue. Figure 7 shows the warp projection specifically for video set 1 having a black bar that ends up getting across the image. This is present within the other video sets, just not as drastic as found in image set 1. It is unclear why this is occuring and no amount of trace back can pinpoint the exact cause only that the image is fine before being warpped and after warp before normalization is not correct. More concerningly is that is not exclusive to the assignment 5, but rather something within the ALINA pipeline that is causing this and therefore impacted the results of assignment 4 as well but was not documented due to not needed to delve into this portion of the code until now. 

![BarIssue](./assignment5/Figures/Vidd1_BlackbarIssue.png)
Figure 7: The black bar on the warpped video frame. The bar is nearly centered on the entire frame, while not blocking directly view of the line. It does mess with the color thresholding aglorithm. 

Addtionally, the color thresholding is not something that can be easily updated within the code. The best way to solve this problem is through more traditional SE and Computer Vision (CV) techqunies that are non-ML or learning based approaches. While some solutions like the hyperspectral processing offer a better view into the higher order data, the analysis can and should be statisical in nature, but I do not agree with that a process that is statistical makes the solution ML. Filtering is a standard process in traditional software and while it is run based on statistics from the data itself, there is an expected input and output that is a deterministic process (a filter response is a known artefact). 


Furthermore, the way ALINA is configured by identify a color threshold based on the data in of itself is identifying all pixels that belong to a target class. Therefore, the problem can be solved without the use of ALINA at all as CIRCLEDAT becomes a redudent process. The underlying concept is that for unsupervised learning we want to find the color threshold intelligently based on some data. To do that we must group all alike pixels together to understand their distribution. That distribution is all pixels that belong to that color class, by extending to the location in the image we now have genreated the image mask itself with a higher fidelity than the circling method. The only issue is a methodology to close blobs (but that can be done via morphological closing). The concept is worse for ALINA in a supervised setting as both the ROI and the color thresholding goes away as the problem becomes a semantic segmentation problem; however the only favor for ALINA in this case is that the required labeled data does not exist and ALINA's purpose is to provide that labeled data. Therefore any supervised ML system becomes a catch-22 for ALINA. 

# Alina Evaluate (Reference command)
There is no content to read, this command is here for ease of access for collecting results needed for the report.

```
alina evaluate \
  --canny-dirs data/gt_alina_labels/canny_textfiles/canny_textfiles_1 data/gt_alina_labels/canny_textfiles/canny_textfiles_2 data/gt_alina_labels/canny_textfiles/canny_textfiles_3 \
  --alina-dirs outputs/KM/KM_V1/coords/ outputs/KM/KM_V1/coords/ outputs/KM/KM_V1/coords/
```


