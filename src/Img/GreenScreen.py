from PIL import Image
import cv2
import numpy as np

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
    hsv = []
    for x in range(10):
        for y in range(10):
            r, g, b, a = pix[x, y]
            h_ratio, s_ratio, v_ratio = rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
            hsv.append((h_ratio * 360, s_ratio * 255, v_ratio * 255))

            r, g, b, a = pix[width-x-1, y]
            h_ratio, s_ratio, v_ratio = rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
            hsv.append((h_ratio * 360, s_ratio * 255, v_ratio * 255))

            r, g, b, a = pix[x, height-y-1]
            h_ratio, s_ratio, v_ratio = rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
            hsv.append((h_ratio * 360, s_ratio * 255, v_ratio * 255))

            r, g, b, a = pix[width-x-1, height-y-1]
            h_ratio, s_ratio, v_ratio = rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
            hsv.append((h_ratio * 360, s_ratio * 255, v_ratio * 255))
    h = sum([x[0] for x in hsv]) / 400
    s = sum([x[1] for x in hsv]) / 400
    v = sum([x[2] for x in hsv]) / 400
    print(h,s,v)

    factor = 0.90
    GREEN_RANGE_MIN_HSV = (100, max(int(s - s*factor), 0)  , 70)
    GREEN_RANGE_MAX_HSV = (185, min(int(s + s*factor), 255), 255)
    print(GREEN_RANGE_MIN_HSV)
    print(GREEN_RANGE_MAX_HSV)

    for x in range(width):
        for y in range(height):
            r, g, b, a = pix[x, y]
            
            h_ratio, s_ratio, v_ratio = rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
            h, s, v = (h_ratio * 360, s_ratio * 255, v_ratio * 255)

            min_h, min_s, min_v = GREEN_RANGE_MIN_HSV
            max_h, max_s, max_v = GREEN_RANGE_MAX_HSV
            if min_h <= h <= max_h and min_s <= s <= max_s and min_v <= v <= max_v:
                pix[x, y] = (0, 0, 0)
            else:
                pix[x, y] = (255, 255, 255)

    im.save('/tmp/green_background_removed.png')