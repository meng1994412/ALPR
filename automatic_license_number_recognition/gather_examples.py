# import packages
from __future__ import print_function
from pipeline.license_plate import LicensePlateDetector
from imutils import paths
import traceback
import argparse
import imutils
import numpy as np
import random
import cv2
import os

# construct the argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required = True, help = "Path to the images to be classified")
ap.add_argument("-e", "--examples", required = True, help = "Path to the output examples directory")
args = vars(ap.parse_args())

# randomly select a portion of the images and initialize the dictionary of character counts
imagePaths = list(paths.list_images(args["images"]))
random.shuffle(imagePaths)
imagePaths = imagePaths[:int(len(imagePaths) * 0.5)]
counts = {}

# loop over the images
for imagePath in imagePaths:
    # show the image path
    print("[EXAMINING] {}".format(imagePath))

    try:
        # load the image
        image = cv2.imread(imagePath)

        # if the width is greater than 640 pixels, then resize the image
        if image.shape[1] > 640:
            image = imutils.resize(image, width = 640)

        # initialize the license plate detector and detect character on the license plate
        lpd = LicensePlateDetector(image, numChars = 7)
        plates = lpd.detect()

        # loop over the license plates
        for (lpBox, chars) in plates:
            # restructure lpBox
            # print(lpBox)
            # print(lpBox.shape)
            lpBox = np.array(lpBox).reshape((-1, 1, 2)).astype(np.int32)

			# draw the bounding box surrounding the license plate and display it for
			# reference purposes
            plate = image.copy()
            cv2.drawContours(plate, [lpBox], -1, (0, 255, 0), 2)
            cv2.imshow("License Plate", plate)

            # loop over thge characters
            for char in chars:
                # displaye the character and wait for a keypress
                cv2.imshow("Char", char)
                key = cv2.waitKey(0)

                # if the '`' was pressed then ignore the character
                if key == ord("`"):
                    print("[IGNORING] {}".format(imagePath))
                    continue

                # grab the key that was pressed and construct the path to
                # the output directory
                key = chr(key).upper()
                dirPath = "{}/{}".format(args["examples"], key)

                # if the output directory does not exist, create it
                if not os.path.exists(dirPath):
                    os.makedirs(dirPath)

                # write the labeled character to file
                count = counts.get(key, 1)
                path = "{}/{}.png".format(dirPath, str(count).zfill(5))
                cv2.imwrite(path, char)

                # increment the count for the current key
                counts[key] = count + 1

    # we are trying to control-c out of the script, so break from the loop
    except KeyboardInterrupt:
        break

    # an unknown error occured for this particular image, so do not process it
    # and display a traceback for debugging purposes
    except:
        print(traceback.format_exc())
        print("[ERROR] {}".format(imagePath))
