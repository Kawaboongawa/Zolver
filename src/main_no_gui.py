import sys
from Puzzle.Puzzle import Puzzle
import matplotlib.pyplot as plt

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = fig_size[0] * 2
plt.rcParams["figure.figsize"] = fig_size

# Puzzle(sys.argv[1])
path = '/home/hugo/ing3/pfe/zolver/resources/'
# Puzzle(path + 'tomatoes.png')
# Puzzle(path + 'moogly.png')
# Puzzle(path + 'parpaing6.png')
# Puzzle(path + 'statue.png')
Puzzle(path + 'colorfull.png')
