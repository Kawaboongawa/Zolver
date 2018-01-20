"""
    Module with list of function to help with operations on tuples
"""

def add_tuple(a, b):
    return a[0] + b[0], a[1] + b[1]

def sub_tuple(a, b):
    return a[0] - b[0], a[1] - b[1]

def equals_tuple(a, b):
    return a[0] == b[0] and a[1] == b[1]

def is_neigbhor(c1, c2, dir):
    for c in dir:
        if equals_tuple(c[0], c1):
            return False
    if c1[0] == c2[0]:
        return c1[1] == c2[1] + 1 or c1[1] == c2[1] - 1
    elif c1[1] == c2[1]:
        return c1[0] == c2[0] + 1 or c1[0] == c2[0] - 1
    else:
        return False

def corner_puzzle_alignement(c, p, l):
    for c2, p2 in l:
        if c2[0] == c[0] or c2[1] == c[1]:
            return True  # Add more orientation check
    return False

def display_dim(dims):
    l = []
    for x, y in dims:
        l.append((x + 1, y + 1))
    return l