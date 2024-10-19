import argparse
import multiprocessing as mp

import matplotlib.pyplot as plt

from Puzzle.Puzzle import Puzzle


parser = argparse.ArgumentParser(description="Solve Puzzles!")
parser.add_argument(
    "-g", "--green_screen", help="enable green background removing", action="store_true"
)
parser.add_argument(
    "-p", "--profile", help="enable profiling", action="store_true"
)
parser.add_argument("file", type=str, help="input_file")
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = fig_size[0] * 2
plt.rcParams["figure.figsize"] = fig_size

args = parser.parse_args()

mp.set_start_method('fork')

if args.profile:
    import cProfile, pstats, io
    from pstats import SortKey
    with cProfile.Profile() as pr:
        Puzzle(args.file, green_screen=args.green_screen)
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats(SortKey.CUMULATIVE)
        ps.print_stats(50)
        print(s.getvalue())
else:
    Puzzle(args.file, green_screen=args.green_screen)
