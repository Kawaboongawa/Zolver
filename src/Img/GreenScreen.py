from PIL import Image
import cv2
import numpy as np

GREEN_RANGE_MIN_HSV = (100, 200, 70)
GREEN_RANGE_MAX_HSV = (185, 255, 255)
MAGIC_SAT_FACTOR = 0.16

def rgb_to_hsv(r, g, b):
    maxc = max(r, g, b)
    minc = min(r, g, b)
    v = maxc
    if minc == maxc:
        return 0.0, 0.0, v
    s = (maxc-minc) / maxc
    rc = (maxc-r) / (maxc-minc)
    gc = (maxc-g) / (maxc-minc)
    bc = (maxc-b) / (maxc-minc)
    if r == maxc:
        h = bc-gc
    elif g == maxc:
        h = 2.0+rc-bc
    else:
        h = 4.0+gc-rc
    h = (h/6.0) % 1.0
    return h, s, v

def remove_background(path):
    # Load image and convert it to RGBA, so it contains alpha channel

    im = Image.open(path)
    im = im.convert('RGBA')

    # Go through all pixels and turn each 'green' pixel to transparent
    pix = im.load()
    width, height = im.size
    mean_s = []
    for x in range(width):
        for y in range(height):
            r, g, b, a = pix[x, y]
            
            h_ratio, s_ratio, v_ratio = rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
            h, s, v = (h_ratio * 360, s_ratio * 255, v_ratio * 255)

            min_h, min_s, min_v = GREEN_RANGE_MIN_HSV
            max_h, max_s, max_v = GREEN_RANGE_MAX_HSV
            if min_h <= h <= max_h:
                mean_s.append(s)

    mean_s = sum(mean_s) / len(mean_s)

    for x in range(width):
        for y in range(height):
            r, g, b, a = pix[x, y]
            
            h_ratio, s_ratio, v_ratio = rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
            h, s, v = (h_ratio * 360, s_ratio * 255, v_ratio * 255)

            min_h, min_s, min_v = GREEN_RANGE_MIN_HSV
            max_h, max_s, max_v = GREEN_RANGE_MAX_HSV
            if min_h <= h <= max_h and mean_s - mean_s * MAGIC_SAT_FACTOR <= s and min_v <= v <= max_v:
                pix[x, y] = (0, 0, 0)
            else:
                pix[x, y] = (255, 255, 255)

    im.save('/tmp/green_background_removed.png')