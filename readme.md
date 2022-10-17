# Synapsica Assignment
Prepared By: Jay Pratap \\
Date of Submission: September 23, 2022


## Content


*   Problem statement
*   Data
*   Possible Solutions
*   Applied Method 
*   Implementation
      *  Reference Generation
      *  Testing
*   possible Improvements



##Problem Statement
Estimation of length of CANAL in vertebrae cross section in images

## Data
Images


*   Original Images: 11 grayscale 
*   Tagged images: 11 with a line along the length of canal
*   Image Size: 230 X 320


## Possible Solutions
Since objective is to estimate the length of the canal as described. We can estimate the position of 2 extreme points and then calculate the length between them.
Now problem is reduced to indentification of 2 extreme points. \\
There are might be several methods for the same, \\
1. Statistical and hand-crafted feature based methods
2. Feature extraction (especially Deep learning) Based methods 

### Statistical and hand-crafted feature based methods
There are lot of methods to extract hand crafted features like edges, corners etc. 

* Advantages
    * When data has low variance these can work very well
    * Less data points are required to create as compare to Machine/deep learning based methods.
    * Lesser parameters to be fine tuned as compared to deep learning based methods.
* Disadvantage
    * These may not as flexible as other methods.
    * Small noise can affect the performances

### Feature extraction (especially Deep learning) Based methods
There are lot of methods for object detection and image segmentation, keyoint detection are available. 

* Advantage
    * Flexibility and robustness for small changes in data and noise

* Disadvantage
    * Need of lot of data for training
    * Lot of experimentation are required to finetune the model
    * These are more costly in development and deployment


## Solution used in assignment
For given situation
* Less datapoints (Images): Not enough data to train any ML/DL model
* Cost: 
    * For estimation of position of only 2 points in an image of size 230 X 320, using complex Machine/deep learning model will not be appropriate.
    * If scope would have broader then it would be logical to use image segmentaion, bounding box detection, or keypoints detection using deep learning. 

Keeping all in mind I have chosen method of Hand-crafted features with some innovations.



## Applied Method
SIFT is a hand crafted feature extractor which is very successful in stitching photographs. It estimate ciritcal points using Laplacian of Gaussian (LoG) at multiple scale and resolution. And estimates point descriptors using slope histograms. \\
Sift features are translation and scale invarient.

###Steps
1. Aggregation of reference features

    1. Using tagged images find extreme points for each image.
    2. Estimate keypoints and corresponding SIFT Descriptors (features) for image
    3. Get Sift descriptors for nearest keypoints to extreme points on canal in image
    4. A threshold is used for distance of nearest keypoint to canal points to select better representation and reject others. 
    5. Aggregate the features for extreme keypoints (2 different set of features aggregated for 2 extreme points of canal)

2. Application
    1. Estimate SIFT keypoints and descriptors for an image (We need to estimate around 200 points so that we do not lose canal points)
    2. Comparing descriptors with reference descriptors and using threshold, find potential extreme points of canal (2 sets, one for each extreme point)
    3. Pair points from 2 sets and selet potential pairs of extreme points based on threshold of maximum distance between them. 
    4. Filter out the pair based on there relative orientation 


# testing
Instructions:
* To create reference files (npy) (already saved files can also be used)
    * Create a zip file with named "assigment_dataset" and keep "original_images" and "tagged_images" in corresponding folders.
    * Now Edit paths of zip file, working directory(codes will run in this directory), main directory(file will be saved in this directory) in YAML configuration file 
    * Edit other parameters if needed
    * Run the create_reference_files
* Testing (To create output csv file with canal length estimations)
    * Create a zip file with named "assigment_dataset" and keep "original_images" and "tagged_images" in corresponding folders.
    * Now Edit following paths in YAML configuration file
        * DATA_ZIP_PATH: path of data zip
        * WORKING_DIR: Dataset zip will be unpacked in this directory
        * REFERENCE_FEATURE_PATH: reference features will be saved in this directory
        * REFERENCE_START_POINT_FEATURE_PATH: path of already saved reference file for start points
        * REFERENCE_END_POINT_FEATURE_PATH: path of already saved reference file for end points
        * OUTPUT_FILE_PATH: Output csv will be generated with this path and file name

    * Run test_model
    * created test.py to run both create_reference_files and test_model


## Possible Improvements
There are 2 major parts for imrovements
1. Aggregation of reference descriptor
2. Better Criterion for descriptor comparision 
3. This model is sensitive to rotation which can be solved by rearranging sift descriptors