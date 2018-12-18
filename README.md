# Automatic License Plate Recognition
## Project Objectives
* Developed a program to recognize automatic license plates based on limited dataset.
* Localized license plate by applying morphological operations and utilizing contour properties.
* Segmented character-like regions of the license plates by applying perspective transforms, performing a connected component analysis, and utilizing contour properties.
* Scissored the true characters from previous step by pruning extraneous license plate character candidates, and extracting each character from binary image to create a training set for building classifiers.
* Extracted BBPS features from the training set and Built two SVM classifiers for recognizing the letters and numbers of the license plate.  

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
    * 8-connectivity component analysis
    * Convex hull
  * Utilize contour properties to segment the foreground license plate characters from the background of the license plate.
* Character Scissoring
  * Develop and implement a heuristic to prune extraneous license plate character candidates, leaving with only the real characters.
  * Define a method to extract each of the license plate characters from the binary image.
* Character Classification
  * Extract and label license character examples from license plate dataset.
  * Extract block-binary-pixel-sum (BBPS) features from real-world license plate character examples.
    * Block-binary-pixel-sum descriptor
  * Train two classifiers on the BBPS features: one classifier for letter recognition and a second classifier for digit recognition.
    * Support vector machine

## Approaches
* The dataset is obtained from [Medialab group](http://www.medialab.ntua.gr/research/LPRdatabase.html), National Technical University of Athens in Greece.
* Assumptions made in this project:
  * The license plate text is always darker than the license plate background
  * The license plate itself is in a rectangular shape
  * Width of the license plate is longer than length of it.

## Results
### License Plate Localization
Figure 1 & Figure 2 demonstrate the two sample process of localizing the license plate.

<img src="https://github.com/meng1994412/ALPR/blob/master/automatic_license_number_recognition/output/milestone_demo/localization_1.png" width="1000">

Figure 1: Process of localizing the license plate for sample # 1.

<img src="https://github.com/meng1994412/ALPR/blob/master/automatic_license_number_recognition/output/milestone_demo/localization_2.png" width="1000">

Figure 2: Process of localizing the license plate for sample # 2.

The process of morphological operations:
1. Blackhat operation (top-left): reveal dark text of license plate characters against light background.
2. Thresholding operation (top-middle): reval light region.
3. Solbel gradient along x-axis
    * license plate characters become more predominant as comapre to blackhat image (top-left).
   - reduce some of the "noise" in the image.
4. Gaussian blur: smooth details & noises.
5. Closing operation: close gaps between the license plate characters.
6. OTSU's thresholded: obtain binary repsentation
7. Erosion & Dilation: clean up the binary image and remove regions that do not interest us.
8. Bitwise AND & Erosion, Dilation:
   - keep only thresholded regions of the image that are also brighter than rest of image
   - further clean up the image

Finally, the bottom far right image shows the results after series operations.  

Though there are some false positive cases, such as Figure 3 shown below, this problem will be solved when segmenting characters from the license plate.

<img src="https://github.com/meng1994412/ALPR/blob/master/automatic_license_number_recognition/output/milestone_demo/localization_false_01.png" width="400">

Figure 3: False positive case for license plate localization.

### Characters Segmentation
Figure 4 & Figure 5 illustrate the process of segmenting the characters on license plate.

<img src="https://github.com/meng1994412/ALPR/blob/master/automatic_license_number_recognition/output/milestone_demo/character_segmentation_1.png" width="800">

Figure 4: Process of segmenting the characters in license plate for sample # 1.

<img src="https://github.com/meng1994412/ALPR/blob/master/automatic_license_number_recognition/output/milestone_demo/character_segmentation_2.png" width="800">

Figure 5: Process of segmenting the characters in license plate for sample # 2.

Since the original license plate the dataset could be distorted or skewed, which could hurt the performance of character classification later in the pipeline, a perspective transform is applied to obtain a top-down, 90-degree viewing angle of the license plate, as top part of right image shown.

By applying adaptive thresholding to the license plate image, the gap between characters and other things (e.g. bolt, license plate frame, special symbols) can be clearly visible, as middle part of right image shown.

By applying connected component analysis and computing convex hull to the thresholded image, character-like regions can be found, as bottom part of right image shown.

### Character Scissoring
Figure 6 & Figure 7 show the processing of scissoring the character on the license plate.

<img src="https://github.com/meng1994412/ALPR/blob/master/automatic_license_number_recognition/output/milestone_demo/character_scissoring_1.png" width="900">

Figure 6: Process of scissoring the character on licensen plate for sample # 1.

<img src="https://github.com/meng1994412/ALPR/blob/master/automatic_license_number_recognition/output/milestone_demo/character_scissoring_2.png" width="900">

Figure 7: Process of scissoring the character on licensen plate for sample # 2.

The top 2 images in the middle part of both figures are from previous step. After applying a character-like region pruning method, true character regions are found, as shown in bottom two images in the middle part.

After applying a scissoring method, each character can be segmented, which will be useful for building the SVM model later.

### Character classification
Extracted character examples can be found [here](https://github.com/meng1994412/ALPR/tree/master/automatic_license_number_recognition/output/examples). Letter and number classifier can be found [here](https://github.com/meng1994412/ALPR/tree/master/automatic_license_number_recognition/output), which is named [char.cpickle](https://github.com/meng1994412/ALPR/blob/master/automatic_license_number_recognition/output/char.cpickle) and [digit.cpickle](https://github.com/meng1994412/ALPR/blob/master/automatic_license_number_recognition/output/digit.cpickle).

Figure 8 & Figure 9 shows the final results of two samples recognizing the license plate.

<img src="https://github.com/meng1994412/ALPR/blob/master/automatic_license_number_recognition/output/milestone_demo/license_plate_recognition_1.png" width="400">

Figure 8: Final results for recognizing the license plate for sample # 1.

<img src="https://github.com/meng1994412/ALPR/blob/master/automatic_license_number_recognition/output/milestone_demo/license_plate_recognition_2.png" width="400">

Figure 9: Final results for recognizing the license plate for sample # 2.
