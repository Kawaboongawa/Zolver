from scipy import ndimage

from cv2 import cv2
import sys
import numpy as np
from Img.GreenScreen import *
from Img.filters import *

PREPROCESS_DEBUG_MODE = 0

def show_image(img, ind=None, name='image', show=True):
    """ Helper used for matplotlib image display """

    plt.axis("off")
    plt.imshow(img)
    if show:
        plt.show()

def show_contours(contours, imgRef):
    """ Helper used for matplotlib contours display """

    whiteImg = np.zeros(imgRef.shape)
    cv2.drawContours(whiteImg, contours, -1, (255, 0, 0), 1, maxLevel=1)
    show_image(whiteImg)
    cv2.imwrite("/tmp/cont.png", whiteImg)

class Extractor():
    """
        Class used for preprocessing and pieces extraction
    """

    def __init__(self, path, viewer=None, green_screen=False, factor=0.84):
        self.path = path
        self.img = cv2.imread(self.path, cv2.IMREAD_COLOR)
        if green_screen:
            self.img = cv2.medianBlur(self.img, 5)
            divFactor = 1 / (self.img.shape[1] / 640)
            print(self.img.shape)
            print('Resizing with factor', divFactor)
            self.img = cv2.resize(self.img, (0, 0), fx=divFactor, fy=divFactor)
            cv2.imwrite("/tmp/resized.png", self.img)
            remove_background("/tmp/resized.png", factor=factor)
            self.img_bw = cv2.imread("/tmp/green_background_removed.png", cv2.IMREAD_GRAYSCALE)
            # rescale self.img and self.img_bw to 640
        else:
            self.img_bw = cv2.imread(self.path, cv2.IMREAD_GRAYSCALE)
        self.viewer = viewer
        self.green_ = green_screen
        self.kernel_ = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    def log(self, *args):
        """ Helper function to log informations to the GUI """

        print(' '.join(map(str, args)))
        if self.viewer:
            self.viewer.addLog(args)

    def extract(self):
        """
            Perform the preprocessing of the image and call functions to extract
            informations of the pieces.
        """

        kernel = np.ones((3, 3), np.uint8)

        cv2.imwrite("/tmp/binarized.png", self.img_bw)
        if self.viewer is not None:
            self.viewer.addImage("Binarized", "/tmp/binarized.png")

        ### Implementation of random functions, actual preprocessing is down below

        def fill_holes():
            """ filling contours found (and thus potentially holes in pieces) """

            _, contour, _ = cv2.findContours(self.img_bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contour:
                cv2.drawContours(self.img_bw, [cnt], 0, 255, -1)

        def generated_preprocesing():
            ret, self.img_bw = cv2.threshold(self.img_bw, 254, 255, cv2.THRESH_BINARY_INV)
            cv2.imwrite("/tmp/otsu_binarized.png", self.img_bw)
            self.img_bw = cv2.morphologyEx(self.img_bw, cv2.MORPH_CLOSE, kernel)                
            self.img_bw = cv2.morphologyEx(self.img_bw, cv2.MORPH_OPEN, kernel)
            
           
                

        def real_preprocessing():
            """ Apply morphological operations on base image. """
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

        cv2.imwrite("/tmp/binarized_treshold_filled.png", self.img_bw)
        if self.viewer is not None:
            self.viewer.addImage("Binarized treshold", "/tmp/binarized_treshold_filled.png")

        self.img_bw, contours, hier = cv2.findContours(self.img_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        self.log('Found nb pieces: ' + str(len(contours)))

        # With this we can manually set the maximum number of pieces manually, or we try to guess their number
        # to guess it, we only keep the contours big enough
        nb_pieces = None

        #TEMPORARY TO AVOID DEBUG ORGINAL:
        if len(sys.argv) < 0:
            # Number of pieces specified by user
            nb_pieces = int(sys.argv[2])
            contours = sorted(np.array(contours), key=lambda x: x.shape[0], reverse=True)[:nb_pieces]
            self.log('Found nb pieces after manual setting: ' + str(len(contours)))
        else:
            # Try to remove useless contours
            contours = sorted(np.array(contours), key=lambda x: x.shape[0], reverse=True)
            max = contours[1].shape[0]
            contours = np.array([elt for elt in contours if elt.shape[0] > max / 3])
            self.log('Found nb pieces after removing bad ones: ' + str(len(contours)))

        if PREPROCESS_DEBUG_MODE == 1:
            show_contours(contours, self.img_bw) # final contours


        ### PREPROCESSING: the end

        # In case with fail to find the pieces, we fill some holes and then try again
        # while True: # TODO Add this at the end of the project, it is a fallback tactic
       
        self.log('>>> START contour/corner detection')
        puzzle_pieces = export_contours(self.img, self.img_bw, contours, "/tmp/contours.png", 5, viewer=self.viewer, green=self.green_)
        if puzzle_pieces is None:
            # Export contours error
            return None

        return puzzle_pieces
