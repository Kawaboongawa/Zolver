import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

import streamlit as st

from Img.GreenScreen import remove_background
from Img.filters import export_contours


PREPROCESS_DEBUG_MODE = 0


def show_image(img, ind=None, name="image", show=True):
    """Helper used for matplotlib image display"""
    plt.axis("off")
    plt.imshow(img)
    if show:
        plt.show()


def show_contours(contours, imgRef):
    """Helper used for matplotlib contours display"""
    whiteImg = np.zeros(imgRef.shape)
    cv2.drawContours(whiteImg, contours, -1, (255, 0, 0), 1, maxLevel=1)
    show_image(whiteImg)
    cv2.imwrite(os.path.join(os.environ["ZOLVER_TEMP_DIR"], "cont.png"), whiteImg)


class Extractor:
    """
    Class used for preprocessing and pieces extraction
    """

    def __init__(self, path, green_screen=False, factor=0.84):
        self.path = path
        self.img = cv2.imread(self.path, cv2.IMREAD_COLOR)
        if green_screen:
            self.img = cv2.medianBlur(self.img, 5)
            divFactor = 1 / (self.img.shape[1] / 640)
            print(self.img.shape)
            print("Resizing with factor", divFactor)
            self.img = cv2.resize(self.img, (0, 0), fx=divFactor, fy=divFactor)
            cv2.imwrite(os.path.join(os.environ["ZOLVER_TEMP_DIR"], "resized.png"), self.img)
            remove_background(os.path.join(os.environ["ZOLVER_TEMP_DIR"], "resized.png"), factor=factor)
            self.img_bw = cv2.imread(
                os.path.join(os.environ["ZOLVER_TEMP_DIR"], "green_background_removed.png"), cv2.IMREAD_GRAYSCALE
            )
            # rescale self.img and self.img_bw to 640
        else:
            self.img_bw = cv2.imread(self.path, cv2.IMREAD_GRAYSCALE)
        self.green_ = green_screen
        self.kernel_ = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    def log(self, *args):
        """Helper function to log informations to the GUI"""
        print(" ".join(map(str, args)))

    def extract(self):
        """
        Perform the preprocessing of the image and call functions to extract
        informations of the pieces.
        """

        kernel = np.ones((3, 3), np.uint8)

        bw_path = os.path.join(os.environ["ZOLVER_TEMP_DIR"], "binarized.png")
        cv2.imwrite(bw_path, self.img_bw)
        st.image(bw_path, caption="Binarized")

        ### Implementation of random functions, actual preprocessing is down below
        def fill_holes():
            """filling contours found (and thus potentially holes in pieces)"""

            contour, _ = cv2.findContours(
                self.img_bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
            )
            for cnt in contour:
                cv2.drawContours(self.img_bw, [cnt], 0, 255, -1)

        def generated_preprocesing():
            ret, self.img_bw = cv2.threshold(
                self.img_bw, 254, 255, cv2.THRESH_BINARY_INV
            )
            otsu_bin_path = os.path.join(os.environ["ZOLVER_TEMP_DIR"], "otsu_binarized.png")
            cv2.imwrite(otsu_bin_path, self.img_bw)
            st.image(otsu_bin_path, caption="Otsu Binarized")

            self.img_bw = cv2.morphologyEx(self.img_bw, cv2.MORPH_CLOSE, kernel)
            self.img_bw = cv2.morphologyEx(self.img_bw, cv2.MORPH_OPEN, kernel)

        def real_preprocessing():
            """Apply morphological operations on base image."""
            self.img_bw = cv2.morphologyEx(self.img_bw, cv2.MORPH_CLOSE, kernel)
            self.img_bw = cv2.morphologyEx(self.img_bw, cv2.MORPH_OPEN, kernel)

        ### PREPROCESSING: starts there

        # With this we apply morphologic operations (CLOSE, OPEN and GRADIENT)
        if not self.green_:
            generated_preprocesing()
        else:
            real_preprocessing()
        # These prints are activated only if the PREPROCESS_DEBUG_MODE variable at the top is set to 1
        if PREPROCESS_DEBUG_MODE == 1:
            show_image(self.img_bw)

        # With this we fill the holes in every contours, to make sure there is no fragments inside the pieces
        if not self.green_:
            fill_holes()

        if PREPROCESS_DEBUG_MODE == 1:
            show_image(self.img_bw)

        bin_thres_path = os.path.join(os.environ["ZOLVER_TEMP_DIR"], "binarized_threshold_filled.png")
        cv2.imwrite(bin_thres_path, self.img_bw)
        st.image(bin_thres_path, caption="Binarized Threshold Filled")

        contours, hier = cv2.findContours(
            self.img_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        self.log("Found nb pieces: " + str(len(contours)))

        # With this we can manually set the maximum number of pieces manually, or we try to guess their number
        # to guess it, we only keep the contours big enough
        nb_pieces = None

        # TEMPORARY TO AVOID DEBUG ORGINAL:
        if len(sys.argv) < 0:
            # Number of pieces specified by user
            nb_pieces = int(sys.argv[2])
            contours = sorted(
                np.array(contours), key=lambda x: x.shape[0], reverse=True
            )[:nb_pieces]
            self.log("Found nb pieces after manual setting: " + str(len(contours)))
        else:
            # Try to remove useless contours
            contours = sorted(contours, key=lambda x: x.shape[0], reverse=True)
            max = contours[1].shape[0]
            contours = [elt for elt in contours if elt.shape[0] > max / 3]
            self.log("Found nb pieces after removing bad ones: " + str(len(contours)))

        if PREPROCESS_DEBUG_MODE == 1:
            show_contours(contours, self.img_bw)  # final contours

        ### PREPROCESSING: the end

        # In case with fail to find the pieces, we fill some holes and then try again
        # while True: # TODO Add this at the end of the project, it is a fallback tactic

        self.log(">>> START contour/corner detection")
        puzzle_pieces = export_contours(
            self.img,
            self.img_bw,
            contours,
            os.path.join(os.environ["ZOLVER_TEMP_DIR"], "contours.png"),
            5,
            green=self.green_,
        )
        if puzzle_pieces is None:
            # Export contours error
            return None
        return puzzle_pieces
