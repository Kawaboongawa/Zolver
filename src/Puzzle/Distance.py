import numpy as np
import math

def normalize_vect_len(e1, e2):
    longest = e1 if len(e1) > len(e2) else e2
    shortest = e2 if len(e1) > len(e2) else e1
    # indexes = np.array(range(len(longest)))
    # np.random.shuffle(indexes)
    # indexes = indexes[:len(shortest)]
    # longest = longest[sorted(indexes)]
    return shortest, longest


def diff_match_edges2(e1, e2, reverse=True):
    shortest, longest = normalize_vect_len(e1, e2)
    diff = 0
    for i, p in enumerate(shortest):
        ratio = i / len(shortest)
        j = int(len(longest) * ratio)
        x1, y1 = longest[j]
        x2, y2 = shortest[i]
        diff += (x2 - x1) ** 2 + (y2 - y1) ** 2
    return diff / len(shortest)



# Match edges by performing a simple norm on each points
def diff_match_edges(e1, e2, reverse=True):
    diff = 0
    # print(e1[0], e2[0])
    for i, p in enumerate(e1):
        if i < len(e2):
            if reverse:
                diff += np.linalg.norm(p - e2[len(e2) - i - 1])
            else:
                diff += np.linalg.norm(p - e2[len(e2) - i - 1])
        else:
            break
    # print(diff)
    return diff
    # if len(e2) < len(e1) * 0.9 or len(e2) > len(e1) * 1.1:
    #     return float('inf')
    # shortest, longest = normalize_vect_len(e1, e2)
    # diff = 0
    # for i, p in enumerate(shortest):
    #     diff += np.linalg.norm(p[0] - longest[len(longest) - i - 1][0]) ** 2
    # return math.sqrt(diff / len(shortest))  # RMSE

def diff_full_compute(e1, e2):
    return 1 * diff_match_edges(e1.shape, e2.shape)