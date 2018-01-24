import cv2
import sys

if len(sys.argv) != 2:
    print("Wrong number of arguments, leaving")
    sys.exit(1)

image = cv2.imread(sys.argv[1])

# Resize image
print("Resizing image")
h, w = image.shape[0], image.shape[1]
image = cv2.resize(image, (min(h, w), min(h, w)))

print("Changing histogram of image")
# Change histogram of color
h, w = image.shape[0], image.shape[1]
for y in range(0, h):
    for x in range(0, w):
        # image[y, x] = image[y, x]
        image[y, x] = image[y, x] * (200 / 255)

cv2.imwrite("out.png", image)
