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


The results of the manual ROI experiements show that with all the default settings, the best ROI is when it is very small and focus on what is directly infront of the plane. A side note is that the frames where the line curves to one side aggressively (***SEE FIGURE***) the small ROI struggles while the larger ROI captures the wider taxiway. 


This leads to the conclusion that the ROI needs to have full coverage of the taxiway at any given time. This mean *only* the paved taxiway should be boxed in for consideration. 

Another note: The default line coloring threshold does cause a lot of issues as the reflection in the windscreen is often time found as the "valid" solution when it is indeed, not the valid solution. 



# ROI Experimenation

## ML Concept
Based on the results of the manual ROI, the problem is currently defined simply as:
```given an image identify any pixel that belongs to the taxiway```

With this in mind the probelm now falls into the category of "image segmentation". Therefore, the basis of the ML system will leverage a image segemenation model to initally classify which pixels belong to the taxiway or not. 

To do this we leverage a pre-trained model for road segementation (for cars). [GITHUB REPO USED](https://github.com/yaraeslamm/lane-detection)
While the model linked here is designed for lane detection systems, it offers three classifications:
- Background
- Road
- Lane Marker

Four our purposes only the classification of `Road` is what we want. The justification for this is that the lane markers are different in numerous ways between taxiways and highways. Mainly size and color, but also their meaning. However, the "concept" of road is the same between the two. Additionally, while not exactly the same materials, they are close approximate of the same materials and therefore have similar visual features between the two. 


<<TODO: Insert figure of one of the segemented frames>>

When we pass the frames through the segmentation model (**FIGURE XXX**), the road does clearly get segemented and the line classification does pick up on some lines, but is not accurate as it views the horizon as a form of line. However, the larger concern is that the road segementation is not a continious feature and in some frames wraps around the plane body itself. 

From here we will analyze experiments on sub-AI or ML based systems to ingest the segemenation masks and generate a ROI for the ALINA pipeline. 

### ROI Methodology 1

#### Implementation
The naive approach is to take the "bounding box" of the road segemented areas. This means that gaps and incontinuities are ignored -- effectively making all road segementations on blob. From there we can extract the four bounding box corners that are then fed into the pipeline.

#### Results




# Alina Evaluate (Reference command)
```
alina evaluate \
  --canny-dirs data/gt_alina_labels/canny_textfiles/canny_textfiles_1 data/gt_alina_labels/canny_textfiles/canny_textfiles_2 data/gt_alina_labels/canny_textfiles/canny_textfiles_3 \
  --alina-dirs outputs/VideoSet1/Trapezoid_LargeByPlane/coords/ outputs/Trapezoid_LargeByPlane/coords/ outputs/VideoSet3/Trapezoid_LargeByPlane/coords/
```


