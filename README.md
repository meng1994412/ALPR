# Automatic License Plate Recognition
## Project Objectives


## Software/Packages Used
* Python 3.5
* [OpenCV](https://docs.opencv.org/3.4.1/) 3.4
* [Scikit-Learn](http://scikit-learn.org/stable/)
* [Scikit-Image](http://scikit-image.org/docs/0.13.x/)
* [Imutils](https://github.com/jrosebr1/imutils)

## Algorithms & Methods Used
* License plate localization
  * Apply morphological operations to reveal possible license plate region.
    * Blackhat operation
    * Sobel gradient
    * Otsu automatic thresholding
    * Erosion & dilation
  * Utilize contour properties to prune license plate candidates.
* Characters segmentation
  * Apply perspective transform to extract license plate region from car, obtaining a top-down, bird’s eye view more suitable for character segmentation.
    * 4-point transform
    * Adaptive thresholding
  * Perform a connected component analysis on the license plate region to find character-like sections of the image.
  * Utilize contour properties to segment the foreground license plate characters from the background of the license plate.

## Approaches
* The dataset is obtained from [Medialab group](http://www.medialab.ntua.gr/research/LPRdatabase.html), National Technical University of Athens in Greece.
* Assumptions made in this project:
  * The license plate text is always darker than the license plate background
  * The license plate itself is in a rectangular shape
  * Width of the license plate is longer than length of it.

## Results
### License Plate Localization
Figure 1 & Figure 2 demonstrate the two sample process of localizing the license plate.

<img src="https://github.com/meng1994412/ALPR/blob/master/automatic_license_number_recognition/output/milestone_demo/localization_1.png" width="500">

Figure 1: Process of localizing the license plate for sample # 1.

<img src="https://github.com/meng1994412/ALPR/blob/master/automatic_license_number_recognition/output/milestone_demo/localization_2.png" width="500">

Figure 2: Process of localizing the license plate for sample # 2.

In the process, the top left image is blackhat operation to reveal the dark text of license plate characters against the light background. the top middle image is thresholding to reveal light region. The top right image computes the gradient along the x-axis of blackhat image. Comparing to to the top left image, the top right image has highlighted regions that contain strong vertical edge, eg. characters on license plate.

The bottom left image applys rectangular closing operation to close gaps between the license plate characters. Then Otsu automatic thresholding is used on the closed image to obtain binary representation.The bottom middle image uses a series of erosions and dilations to clean up the binary image and remove the regions that do not interest us. Bottom right image has been further cleaned up by applying a bitwise and on the threshold image, keeping on the light regions of the image.

Finally, the bottom far right image shows the results after series operations.  

Though there are some false positive cases, such as Figure 3 shown below, this problem will be solved when segmenting characters from the license plate.

<img src="https://github.com/meng1994412/ALPR/blob/master/automatic_license_number_recognition/output/milestone_demo/localization_false_01.png" width="400">

Figure 3: False positive case for license plate localization.

### Characters Segmentation
Figure 4 & Figure 5 illustrate the process of segmenting the characters on license plate .

<img src="https://github.com/meng1994412/ALPR/blob/master/automatic_license_number_recognition/output/milestone_demo/character_segmentation_1.png" width="500">

Figure 4: Process of segmenting the characters in license plate for sample # 1.

<img src="https://github.com/meng1994412/ALPR/blob/master/automatic_license_number_recognition/output/milestone_demo/character_segmentation_2.png" width="500">

Figure 5: Process of segmenting the characters in license plate for sample # 2.

Since the original license plate the dataset could be distorted or skewed, which could hurt the performance of character classification later in the pipeline, a perspective transform is applied to obtain a top-down, 90-degree viewing angle of the license plate, as top part of right image shown.

By applying adaptive thresholding to the license plate image, the gap between characters and other things (eg. bolt, license plate frame, special symbols) can be clearly visible, as middle part of right image shown.

By applying connected component analysis and computing convex hull to the thresholded image, character-like regions can be found, as bottom part of right image shown.

### Character Scissoring

 
