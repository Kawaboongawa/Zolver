import subprocess
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import os

if len(sys.argv) < 2:
    images=[('resources/parpaing6.png', 'resources/parpaing6.pngRef')]
else:
    images=[(sys.argv[1], sys.argv[1] + 'Ref')]

for img, ref in images:
  print('Solving puzzle')
  subprocess.run(["python", "src/main_no_gui.py", img])
  print('Trimming output')
  subprocess.run(["convert", "-trim", "/tmp/colored.png", "tmp.png"])
  print('Diff between solved puzzle and ref')
  out = subprocess.run(["compare", "-metric", "ae", ref, "tmp.png", "null:"], stderr=subprocess.PIPE)

  #diff = int(out.stderr)
  #cvImage = cv2.imread(img)
  #w, h, channels = cvImage.shape
  #ratio = (diff / (w * h))
  #print('ratio:', ratio)
  #if (ratio > 0.001):
  #  print('Estimation : FAILED')
  #else:
  #  print('Estimation : OK')


  fig = plt.figure("Images")

  ax = fig.add_subplot(1, 2, 1)
  ax.set_title("Ref image")
  plt.imshow(mpimg.imread(ref))
  plt.axis("off")

  ax = fig.add_subplot(1, 2, 2)
  ax.set_title("Reconstitution")
  plt.imshow(mpimg.imread("tmp.png"))
  plt.axis("off")
  plt.show()

