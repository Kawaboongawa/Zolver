import sys
from Puzzle.Puzzle import Puzzle
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Solve Puzzles!')
parser.add_argument('-g', '--green_screen', help='enable green background removing', action="store_true")
parser.add_argument("file", type=str, help="input_file")
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = fig_size[0] * 2
plt.rcParams["figure.figsize"] = fig_size

args = parser.parse_args()
Puzzle(args.file, green_screen=args.green_screen)
