from scipy import ndimage

from cv2 import cv2
import sys
import numpy as np
from Img.GreenScreen import *
from Img.filters import *

PREPROCESS_DEBUG_MODE = 0

def show_image(img, ind=None, name='image', show=True):
    plt.axis("off")
    plt.imshow(img)
    if show:
        plt.show()

def show_multiple_images(imgs):
    fig = plt.figure("Images")
    for i, img in enumerate(imgs):
        ax = fig.add_subplot(len(imgs), 1, i + 1)
        ax.set_title(str(i))
        show_image(img, show=False)
    plt.show()

def show_contours(contours, imgRef):
    whiteImg = np.zeros(imgRef.shape)
    cv2.drawContours(whiteImg, contours, -1, (255, 0, 0), 1, maxLevel=1)
    show_image(whiteImg)
    cv2.imwrite("/tmp/cont.png", whiteImg)

class Extractor():
    def __init__(self, path, viewer=None, green_screen=False):
        self.path = path
        self.img = cv2.imread(self.path, cv2.IMREAD_COLOR)
        if green_screen:
            print(green_screen)
            divFactor = 1 / (self.img.shape[1] / 640)
            print(self.img.shape)
            print('Resizing with factor', divFactor)
            self.img = cv2.resize(self.img, (0, 0), fx=divFactor, fy=divFactor)
            cv2.imwrite("/tmp/resized.png", self.img)
            remove_background("/tmp/resized.png")
            self.img_bw = cv2.imread("/tmp/green_background_removed.png", cv2.IMREAD_GRAYSCALE)
            # rescale self.img and self.img_bw to 640
        else:
            self.img_bw = cv2.imread(self.path, cv2.IMREAD_GRAYSCALE)
        self.viewer = viewer
        self.green_ = green_screen

    def log(self, *args):
        print(' '.join(map(str, args)))
        if self.viewer:
            self.viewer.addLog(args)

    def extract(self):
        kernel = np.ones((3, 3), np.uint8)
        # img = cv2.resize(initial_img, None, fx=0.5, fy=0.5)

        cv2.imwrite("/tmp/binarized.png", self.img_bw)
        if self.viewer is not None:
            self.viewer.addImage("Binarized", "/tmp/binarized.png")

        # show_image(self.img_bw)
        # self.img_bw = cv2.cvtColor(self.img_bw, cv2.COLOR_RGB2GRAY)
        # show_image(self.img_bw)

        ### Implementation of random functions, actual preprocessing is down below

        def fill_holes():
            # filling contours found (and thus potentially holes in pieces)
            _, contour, _ = cv2.findContours(self.img_bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contour:
                cv2.drawContours(self.img_bw, [cnt], 0, 255, -1)

        # ret, self.img_bw = cv2.threshold(self.img_bw, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # ret, self.img_bw = cv2.threshold(self.img_bw, 240, 255, cv2.THRESH_BINARY_INV)
        # show_image(self.img_bw)


        def apply_morpho():
            morph = self.img.copy()
            # nbMorpho is updated with empiric values, they can obviously be changed
            nbMorph = 3
            if self.img.shape[0] * self.img.shape[1] < 1000 * 2000:
                nbMorph = 1
            if self.img.shape[0] * self.img.shape[1] > 3000 * 3000:
                nbMorph = 5
            for r in range(nbMorph):
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
                morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
                morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

            # show_image(morph)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mgrad = cv2.morphologyEx(morph, cv2.MORPH_GRADIENT, kernel)
            # print('Morphology gradient')
            # show_image(mgrad)
            self.img_bw = np.max(mgrad, axis=2)  # BGR 2 GRAY

            def f(x):
                if x < 5:
                    return 0
                else:
                    return 255
            f = np.vectorize(f)
            for i, tab in enumerate(self.img_bw):
                self.img_bw[i] = f(self.img_bw[i])
            return

        def real_preprocessing():
            element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            self.img_bw = cv2.morphologyEx(self.img_bw, cv2.MORPH_CLOSE, element)                
            self.img_bw = cv2.morphologyEx(self.img_bw, cv2.MORPH_OPEN, element)

        ### PREPROCESSING: starts there

        # With this we apply morphologih operations (CLOSE, OPEN and GRADIENT)
        if not self.green_:
            apply_morpho()
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
        #if len(sys.argv) > 2: 
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
        #     try:
        self.log('>>> START contour/corner detection')
        puzzle_pieces = export_contours(self.img, self.img_bw, contours, "/tmp/contours.png", 5, viewer=self.viewer, green=self.green_)
        # break
        # except (IndexError):
        #     fill_holes()
        #     nb_error_max -= 1
        #     if nb_error_max <= 0:
        #         print('Could not find the pieces, exiting the app')
        #         sys.exit(1)
        #     print('Error while trying to find the pieces, trying again after filling some holes')
        # if self.viewer is not None:
        #     self.viewer.addImage("Contours", "/tmp/contours.png")

        # fshift, magnitude = get_fourier(self.img_bw)
        # cv2.imwrite("/tmp/yolo.png", magnitude)
        # if self.viewer is not None:
        #     self.viewer.add_image_widget("/tmp/yolo.png", 1, 0)
        # rows, cols = self.img_bw.shape
        # crow, ccol = int(rows / 2), int(cols / 2)
        # fshift[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0
        # f_ishift = np.fft.ifftshift(fshift)
        # img_back = np.fft.ifft2(f_ishift)
        # img_back = np.abs(img_back)
        #
        # cv2.imwrite("/tmp/yolo.png", img_back)
        return puzzle_pieces
