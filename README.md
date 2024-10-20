<p align="center">
<img src="resources/logo/logo.png" width="300">
</p>

# Zolver

Zolver is a jigsaw puzzle solver written in python. This end-of-studies project was
developed by 4 EPITA students. It solves computer generated and real jigsaw puzzles.

A short video presentation is available at [https://www.youtube.com/watch?v=Oq36FtMg0-k](https://www.youtube.com/watch?v=Oq36FtMg0-k)

A jigsaw puzzle
<p align="center">
<img src="resources/jigsaw-samples/van-gogh.png" width="600">
</p>
Zolver reconstruction
<p align="center">
<img src="resources/jigsaw-solved/van-gogh.png" width="500">
</p>

## Requirements

This projects was developed on Linux distribution. Windows compatibility is absolutely not guaranteed.

To avoid compatibility issues, we recommend setting up a virtual environment and then using the command :

`pip install -r requirements.txt`

## Usage
### With GUI
Use `python3 main.py`


### Without GUI
Use `python3 main_no_gui.py {-p} {-g} path/to/image`
Options:
- `-g`: enable green background
- `-p`: enable profiling

## Zolver overview

### Preprocessing

First, we apply a binary threshold to the processed image. The threshold is different if we process
a real puzzle or a real one. We then apply some mathematical morphologies to link edges
in case some edges have holes. Finally, we apply OpenCV contour detection.

### Edge classification

We now need to find four edges of each puzzle piece. To do so, we compute the relative angle
between adjacent points and we get local extrema. Once our edges have been splitted, we classify
our edges into three categories: frame edges, indent padding edges and outdent padding edges by
recognizing patterns with relative angle curves.

### Edge matching

We use three different ways to match edges :
- Comparing the edge size to find edges with similar length.
- Using color to match edges. For every pixel on the edge, we take the median color of
its neighboring pixels and we then compare the distance to the other edge pixel to pixel.
- Comparing shapes using a Euclidian distance along the edges.

### Puzzle Solving

We first solve the puzzle edge because once we have its frame, every puzzle piece can be matched
using at least two edges. The more edges we have to check how likely is the piece to match, the less
Zolver is likely to make an error. Zolver will always try to fill the empty space with the highest
number of edges already filled.

## Authors
SCIA 2018 - EPITA

* Cyril Cetre
* Denis Castéran
* Jérémy Lugand
* Hugo Rybinski

![Screenshot](resources/jigsaw-solved/craies_32.png)
![Screenshot](resources/jigsaw-solved/link.png)
