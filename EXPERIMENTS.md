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

One such issue was a warp black-bar issue. Figure 7 shows the warp projection specifically for video set 1 having a black bar that ends up getting across the image. This is present within the other video sets, just not as drastic as found in image set 1. It is unclear why this is occuring and no amount of trace back can pinpoint the exact cause only that the image is fine before being warpped and after warp before normalization is not correct. 

![BarIssue](./assignment5/Figures/Vidd1_BlackbarIssue.png)
Figure 7: The black bar on the warpped video frame. The bar is nearly centered on the entire frame, while not blocking directly view of the line. It does mess with the color thresholding aglorithm. 

Addtionally, the color thresholding is not something that can be easily updated within the code. The best way to solve this problem is through more traditional SE and Computer Vision (CV) techqunies that are non-ML or learning based approaches. While some solutions like the hyperspectral processing offer a better view into the higher order data, the analysis can and should be statisical in nature, but I do not agree with that a process that is statistical makes the solution ML. Filtering is a standard process in traditional software and while it is run based on statistics from the data itself, there is an expected input and output that is a deterministic process (a filter response is a known artefact). 


Furthermore, the way ALINA is configured by identify a color threshold based on the data in of itself is identifying all pixels that belong to a target class. Therefore, the problem can be solved without the use of ALINA at all as CIRCLEDAT becomes a redudent process. The underlying concept is that for unsupervised learning we want to find the color threshold intelligently based on some data. To do that we must group all alike pixels together to understand their distribution. That distribution is all pixels that belong to that color class, by extending to the location in the image we now have genreated the image mask itself with a higher fidelity than the circling method. The only issue is a methodology to close blobs (but that can be done via morphological closing). The concept is worse for ALINA in a supervised setting as both the ROI and the color thresholding goes away as the problem becomes a semantic segmentation problem; however the only favor for ALINA in this case is that the required labeled data does not exist and ALINA's purpose is to provide that labeled data. Therefore any supervised ML system becomes a catch-22 for ALINA. 






# Assignment 4

Everything below here is exactly the same as was submitted for assignment 4. No new information can stop here for grading purposes.

## Running experiments

All code required is within the `assignment4` folder. The only command to run the pipeline needed is the following:
```python a4.py```

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
        "custom_roi": None # unet, segnet, segformer, kmeans, claude (None, if manual ALINA pipeline)
    }
```

The arg dictionary is a placeholder for the lack of commandline interfacing for this assignmnet. End users will need to change the "custom_roi" key to the ones commented there: unet, segnet, segformer, kmeans, claude for each of the methods. If the None object is passed in then the system defaults to manual ROI detection. 

# ALINA Manual ROI


Command to run the manual ROI Alinia program

```
alina label   --input-dir data/Raw_Data/vidd_3 \
--output-images-dir outputs/tmp/annotated   \
--output-coords-dir outputs/tmp/coords  \
--log-file outputs/tmp/timing.log
```


Experiments conducted with each video set:
- Large ROI (From Plane nose to Horizon and edge to edge of the frame)
- Small ROI (From Plane nose to Horison as wide as the plane is)
- Trapezoid (small side of traepzoid as wide as plane nose to as wide as the taxiway on horizon)
- Trapezoid (large size of trapeizode wider than plane nose to smaller than horizon size of taxiway)

| Experiement | Avg Recall (%) | Avg Precision (%) | Avg F1 (%) |
| :--- | :---: | :---: | ---: |
| Large ROI | 45.43 | 39.02 | 41.97 |
| Small ROI | 57.14 | 53.12 | 55.08 |
| Trapezoid (Away from plane) | 0.00 | 0.00 | 0.00 |
| Trapezoid (Near plane) | 0.00 | 0.00 | 0.00 | 0.00 |

Table 1: Results of the experiements of manual ROI.

![Large ROI](./Figures/ManualLargeROI.png)
![Small ROI](./Figures/ManualSmallROI.png)
![Trap Small ROI](./Figures/ManualTrapSmallROI.png)
![Trap Large ROI](./Figures/ManualTrapLargeROI.png)

Figure 1: Manual ROI selection of the first image in video set 3. The Large ROI (top), Small ROI (second from top), Trapezoid Small (second from bottom), and the trapezoid large (bottom) are shown. 


The results of the manual ROI experiements show that with all the default settings, the best ROI is when it is very small and focus on what is directly infront of the plane. A side note is that the frames where the line curves to one side aggressively (see Figure 1) the small ROI struggles while the larger ROI captures the wider taxiway. 


This leads to the conclusion that the ROI needs to have full coverage of the taxiway at any given time. This mean *only* the paved taxiway should be boxed in for consideration. 

Another note: The default line coloring threshold does cause a lot of issues as the reflection in the windscreen is often time found as the "valid" solution when it is indeed, not the valid solution. 



# ROI Experimenation

Before the disscussion on the ML concept, its application, and experimentations. There is a brief need to mention that models were selected due to various constraints that lead to not realistic or to results that would withstand serious review. 

**First constraint**: time, the timeline for the assignment limits the breadth of models that can be trained as having more data points to compare to increases understanding of how and why the data is interacting through the pipeline. However, there is not enough time to fully test heavy state of the art models. 

**Second constrain**: resources, a lot of state of the art VLMs, ViTs, LLMs, etc are dense that requires significant computational resources to store, load, and predict. Additionally there are not many models with out of the box performance that aligns with the identified problem that will be discussed below. This leads into the third constraint. 

**Third constraint**: Data. The data included within the repositorty is a very small sub-set of the entire AssitTaxi dataset. While that dataset could be downloaded there is significant disk storage space constarains that prevent a dataset of that size from being downloaded two of them being time: downloading takes time; resources: need to store and train a model on the data. 

With these three main constrains and their intertwined connections in mind, the goal was to proritze the balance between what the actual problem demands with as state-of-the-art as system requiremetns allow. 

## ROI ML Concept
Based on the results of the manual ROI, the problem is currently defined simply as:
```given an image identify any pixel that belongs to the taxiway```

With this in mind the probelm now falls into the category of "image segmentation". Therefore, the basis of the ML system will leverage a image segemenation model to initally classify which pixels belong to the taxiway or not. 

### ROI Methodology 1


To do this we leverage a pre-trained model for road segementation (for cars). [GITHUB REPO USED](https://github.com/yaraeslamm/lane-detection)
While the model linked here is designed for lane detection systems, it offers three classifications:
- Background
- Road
- Lane Marker

Four our purposes only the classification of `Road` is what we want. The justification for this is that the lane markers are different in numerous ways between taxiways and highways. Mainly size and color, but also their meaning. However, the "concept" of road is the same between the two. Additionally, while not exactly the same materials, they are close approximate of the same materials and therefore have similar visual features between the two. 

![UNET Example](./Figures/UNet_Segmentation_Example.png)

Figure 2: Example of the UNET segemnation (right) on the input image (left). The red pixels in the segmentation image denote road pixels while green denotes road lines. 

When we pass the frames through the segmentation model (see Figure 2), the road does clearly get segemented and the line classification does pick up on some lines, but is not accurate as it views the horizon as a form of line. However, the larger concern is that the road segementation is not a continious feature and in some frames wraps around the plane body itself. 

From here we will analyze experiments on sub-AI or ML based systems to ingest the segemenation masks and generate a ROI for the ALINA pipeline. 

The naive approach is to take the "bounding box" of the road segemented areas. This means that gaps and incontinuities are ignored -- effectively making all road segementations on blob. From there we can extract the four bounding box corners that are then fed into the pipeline.

### ROI Methodology 2

This is a quick section because I did not read properly.

The high level idea was to use a Segmentation Network (SegNet). These are Autoencoder based methods similar to the UNet from the previous section. However, they do not have the "large scale" skip connnections. Instead they have a backbone like VGG or ResNet with the internal residual skip connections. The theory behind their performance is the "Deep" aspect that they have unlike UNets are capable of learning more abstract feature rich representations that can be converted into the segmentation level. 

![SegNet](./Figures/SegNet_Ex.png)

Figure 3: SegNet on the input image (left) with the resulting mask (right). The model was trained on satellite imagery and therefore segements items that would look like a road from space. 

There are no results for this because all the models I found are road segmentation... from satellite view. In Figure 3 we see that the mask result is just identify small skinny lines in all directions. This is expected as the view from space roads would appear small thin and criss-crossing in all direction. Unfortunatley for what we need, this is not acceptable. Therefore, no result table as any more experimentation was deemed not worth the time.

## ROI Methodology 3

Building on the previous two we reach the state of the art with SegFormers (**Seg**ment Trans**formers**). This applies the novel technology of transformer attention mechanism for image segmentation. While highly studied in recent times, we are looking at it for the sole purpose of image segementation with better resoltuion. This is because transformers generally display better small scale spatial resolution for segmentation than convolutions (this is due to transformers having a single "downsampling" mode at the top in the patchify step comapred to convolutions that downsample frequently.)

![SegFormer](./SegFormer_Ex.png)

Figure 4: SegFormer mask (right) based on input image (left). White pixels in the mask denote the road labeled data. 

In the results we see that Figure 4 display simalr outcomes as the first attempted methodology. However, it does a better job of ignoring the support pillars of the airframe and the propeller. While both still struggle with ignoring the aircrafts airframe between the windscreen and the propeller causing confusing and non-optimal ROI based on the naive approach outlined in the previous section. 

However, for some reason the results for video set 2 actually returned valid results. The theory behind why is that the road in view in that data set is much tigher and in center view compared to sets 1 and 3 where the taxiway expands and shrinks throughout the video and is not within the center at all times. 


## ROI Methodolgy 4

Through the experimentation the key problem is really where is the horizon. The reason is that because the taxiway expands horizontalally it becomes harder for the road segemnation models to work because roadways for cars are usually a fixed width. But here the horizon is a relatively fixed point in space and the segemenation models effectively just take eveything below the horizon. So can we do that unsupervised learning by identifying the horizon. 

Now this is a straighforward process for tradtional CV as it can be defined as roughly the halfway point in any given frame, but that has issues as the aircraft rocks and shifts as it goes down the taxiway. Additionaly CV methods for identifying lines and finding the horizontal line since the horizon is the mostly horizontal. However there are too many other noise factors since the cockpit is visible in most if not all frames. 

![Kmeans IMG1](./Figures/KMeans_Ex1.png)
![Kmeans IMG2](./Figures/Kmenas_Ex2.png)

Figure 5: Kmeans results (right column) based on the input images (left columns). The resulting masks clearly segement sky and ground pixels, however the lack of supervision does not mean they are garunteed to share the same label between images.

To solve this we implement a K-means system to identify "ground" and "sky" pixels. Then where the transition occurs we just take everything from that to the bounds of all "ground" pixels similar to the system for the segmentation. While in some cases as Figure 5 (top) shows the average transition point is rougnly the horizon. However, in high cloud coverage this system fails as seen in Figure 5 (bottom) What would help this method is having a set of training data to help provide labels as the issue with performance is mainly the fact that ground and sky can flip for any given frame since there is nothing to anchor them to a specific label throughout all frames. 

## ROI Methodology 5

There is nothing more "state-of-the-art" than mixture of expert that have various subcomponents such as LLMs, VLMs, ViTs, etc. In order to understand how these can perform compared to more standard image segemnation tasks, we deploy Anthropic's Cluade Sonnet 5.5 Medium to do ROI detection.

The ROIs are conducted by zipping the raw data for each of the three videos and passed to the claude web page. The model is then prompted along side the zipped folder with the following prompt:
```
Here is a zip of a bunch of images what I need is a zip folder that contains a text file for each image in which it contains a points of an ROI bounding box that identifies the runway in the image. The points need to be in units of the image pixels and provided in the order of:
bottom_left, top_left, top_right, bottom_right
```

![Cluade Ex](./Figures/Claude_Ex.png)

Figure 6: The ROI defined as a red bounding box identified by Claude Sonnet 5.5 Medium (right) based on the input image (left). 


From there each image is loaded and the resulting claude determined ROI is loaded and passed to the ALINA pipeline. Figure 6 shows that the LLM does indeed offer a decent ROI similar in selection to the trapezoid based system in the manual ROI section. Howver, it is a lot more foucsed in its ROI selection specifically going for a shape that is smaller than the width of the runway often times capturing just the center line. While that increases performance, the semantic implications of it need to be examined further because the model was prompted with context that this is a runway and therefore the lines are center of the image (for the most part). 


### ROI Methodology Issues

One of the methodologies I wished to test was the You Only Look Once (YOLO) family of models. The reason is that these are specifically defined for ROI bounding box problems. Numerous examples of pretrained models sepcifical on road and urban environments exist on Huggingfacce. 

However, while these models are a good option they are techincal liminations that are not easily worked around within the constraints of the project time limits. 

Specifically, the pretrained models all use a python package called ```ultralytics``` the issue is that this pacakge requires numpy<2.0.0. The Numpy 1.x reached End of Life (EOL) Septemebr 2025. All modern packages required to run ML inferencing have migrated to numpy2.x. While a solution work around does exist it requires breaking code into multiple sub-scripts each with their own python venv to support the variety of package conflicts. 

While this is a solution, it does not allow for easy interopertable verification as RNG seeds are not garunteed to be stable throught massive version migrations like that. 




### Results & Discussion

All of the above methodologies were ran for each of the video sets provided in the ALINA github. Additionally it should be noted that all of these models were run per frame (question 3b). The reason for the per image ROI detection is that the image sets are not continious nor do they appear to be from the same "run"; instead they are sudden jumps and changes in perspective that doing one per set does not caputre. Even the manual ROI struggles as the change in perspective and width of taxiway lead to the decreased performance, but for time reasons per frame manual was not done. 

| Experiement | Video Set | Avg Recall (%) | Avg Precision (%) | Avg F1 (%) |
| :--- | :---: | :---: | :---: | ---: |
| UNet   | Vid 1 | 0.00 | 0.00 | 0.00 |
|        | Vid 2 | 0.00 | 0.00 | 0.00 |
|        | Vid 3 | 0.00 | 0.00 | 0.00 |
| SegNet | Vid 1 | N/A | N/A | N/A |
|        | Vid 2 | N/A | N/A | N/A |
|        | Vid 3 | N/A | N/A | N/A |
| SegFormer | Vid 1 | 0.00 | 0.00 | 0.00 |
|           | Vid 2 | 68.30 | 63.61 | 65.40 |
|           | Vid 3 | 0.00 | 0.00 | 0.00 |
| Kmeans | Vid 1 | 50.0 | 2.73 | 5.17 |
|        | Vid 2 | 65.53 | 37.23 | 39.42 |
|        | Vid 3 | 0.00 | 0.00 | 0.00 |
| Claude | Vid 1 | 84.30 | 7.34 | 13.44 |
|        | Vid 2 | 65.32 | 67.12 | 65.94 |
|        | Vid 3 | 0.00 | 0.00 | 0.00 |

Table 2: Results of the ALINA evaluation script for each of the methodologies and video sets. SegNet was the only one without a reported value due to not being run. 


The overall results show a mix of performance with nothing reaching the claimed performance in the ALINA paper. For all of the models except KMEANS and Claude, the lack of results stems from the inability to generalize well to the problem of airport taxiways. Each of those models were pretrained on road datasets (with the exception of SegNet which was trained on satellite imagery). The road datasets did lead to some decent identification of taxiway, however the naivie approach to bounding them lead to issues where most of the airframe ended up in the image or sections outside of the runway was also included. The biggest issue stems from the fact that the inside of the aircraft is a similar color to that of the runway itself. 

The SegFormer performed the best out of the three segemenation networks having been trained on more urban environments and therefore implicitly better at identify the boundaries of the road and cars. 

An intresting note that is unquantifiable in the time of this assignmnet is the fact that video set 2 (Vidd_2) yieled the best results across the board for all metrics (except Claude video set 1 accuracy). There is something about either that dataset or the ground truth comparisions that lead to higher performance. 

Doing unsuperivised learning worked well enough. Results were impacted by the issue from unsupervised learning not labeling consistently between images. Some times the ground was labeled as sky and vice versa. To fix this known issue creating a supervision layer that either has a human say which binary label belongs to the sky/groung or a true ML supervised system. The ML supervision requires data that is note accessible for this project timeline. In order to do that we would need to generate a dataset of input image with mask denoting where the horizon is. At that level a better solution would be to just build a better labeled image set for full semantic segemnation. 

Claude impressed me the most out of all methodologies tried. The unerlying mixture of experts does a good job of building decent ROIs for each image that help constrain the resulting search space. However, becasue of the black box nature of web/cloud based LLMs there is not much deconstruction and analysis as to why and how this happens. In the future I would like to try to run the model locally so I can strip out activations and layers as needed, however that is not fesible for class structure as a workstation/datacenter level GPU is required to run these models. 


# Color Thresholding (Assignmnet 5 but here for now)

The issue with color thresholding is that the default bounds the values to a fixed lower and upper threshold. While this works, the varying camera settings and environmental factors complicate this. 

While a throw the newest thing could solve this problem. Specifically VLMs to given an input image and return the threshold value for the yellow color we want or using another segemenation type model to classify colors into their high level colors (i.e. all shades of yellow map to "yellow"). 

The simpilest solution (and one that might be the most naive) is to use an unsupervised clustering algorithm to group pixels into self-segemeneted groups. From there the "known yellow" value is fed into the classifier and assigned its nearest neighbor grouping. With that grouping it is possible to extract the exact dynamic range of the groups color. 

The justification for leveraging the self-supervised learning is really a lack of labeled training data to leverage any more complex "deep learning" approach to solving the problem. As well as the fact that airports are highly controlled areas and as such the actual yellow lines on the taxiways are an offically designated color. So building up a set of clusters allows us to find a set of pixels that closely matches the expected color. 

Some limitations may occure in the night, however there is a lack of data within this assignmnet to better quantify the impacts of non-daylight running environments. 


## Results

To test the impact of this, the ALINA pipeline was run in manual ROI mode in two ways: Large ROI and Small ROI as defined in the section above. This was done to prevent any impact from automatic ROI inducing any downstream artefacts or issues. 

| Experiement | Avg Recall (%) | Avg Precision (%) | Avg F1 (%) |
| :--- | :---: | :---: | ---: |
| Large ROI | 0.00 | 0.00 | 0.00 |
| Small ROI | 0.00 | 0.00 | 0.00 |



# ALINA Debrief

## Issues
There are numerous issues that lead to downstream impacts on the assignment. Here we outline the ones that have come up and the severity of them.

- Eval [**Medium**]: The evaluation script relies on the canny directory. However it is not clear what data actually relates to where and when. So when running the evaluate based on the github README all ground truth files are compared however while it does skip ones that do not exist, the question remains why it compares multiple examples. If the examples are only one to one then they should be compared only once, if they are different then they need to be homogonized. Comparing multiple examples at a time ends up where the reported statistics have a foggy meaning since the relation cannot be broken down. 

- Color thresholding [**CRITICAL**]: The code as it stands does not except anything more than a single color threshold, trying to have a single threshold identify multiple colors looses resulting data meaning. Especially in the case of white color since white is all colors. Recommended fix is implement a thresholding list where ALINA generates a color mask per threshold and merges them together. 

- Color Normalization [**CRITICAL**]: The normalization of the HSV values within alina normalize all components to range [0, 255], however the open-cv standard limits the hue component (H) to [0, 179] isntead of the full 360 degree color space. This fix has been submitted as a pull requiest. Until this is fixed color threshold will not work as the inputed HSV values will be off by approximate 76 units of remapped space. 

- Per Image ROI [**LOW**]: This does not impact performance of the ALINA software, however it is an underlying concern in that the only accepted way to do ROI injection per frame is via command line arguments. This is not ideal for dynamic software needs. Recommended fix is to allow the process batch to accept a list of ROIs. 



## Perspective from a software developer

While leveraging an existing packages is very common and there is nothing wrong with that. From what this assignment was asking to do, leveraging a pre-existing package like ALINA got in the way more than helpping. There are numerous ways to solve any given problem, and throughout the course of this project there were numerous road blocks that occured because the software is being forced to do things that it is not designed to do. 

One of the biggest questions I could not answer is why is a reporjection of the image needed? While it certainly gives more pixels to do a detection, they are fake interpolated pixels because it is rescaled. The problem could be simplifed much more compactly by identify the ROI then doing computer-vision with ML within viewed space. The added advantage is that spatial context is much better perserved for color thresholding and line detection where as the reprojection distorts the meaning of how the lines relate. Another benefit is accidential color matching, in a few tests of the color thresholding (before it was moved to assignment 5) I noticed that the yellow in the taxiway signage was being identified. The reproejcted space flattens that and the lines together and makes it harder to determine correct or incorrect (with the cavet that a better ROI should have excluded that anyway). 






# Alina Evaluate (Reference command)
There is no content to read, this command is here for ease of access for collecting results needed for the report.

```
alina evaluate \
  --canny-dirs data/gt_alina_labels/canny_textfiles/canny_textfiles_1 data/gt_alina_labels/canny_textfiles/canny_textfiles_2 data/gt_alina_labels/canny_textfiles/canny_textfiles_3 \
  --alina-dirs outputs/KM/KM_V1/coords/ outputs/KM/KM_V1/coords/ outputs/KM/KM_V1/coords/
```


