from Puzzle.Distance import diff_match_edges, diff_match_edges2, diff_full_compute
from Puzzle.PuzzlePiece import *

from Puzzle.Extractor import Extractor
from Puzzle.Mover import *
from Img.filters import load_signatures
from cv2 import cv2

from Puzzle.Enums import *
import sys
import scipy

from Puzzle.tuple_helper import equals_tuple, add_tuple, sub_tuple, is_neigbhor, corner_puzzle_alignement, display_dim


class Puzzle():
    def log(self, *args):
        print(' '.join(map(str, args)))
        if self.viewer:
            self.viewer.addLog(args)

    def __init__(self, path, viewer=None):
        self.extract = Extractor(path, viewer)
        self.pieces_ = self.extract.extract()
        self.viewer = viewer


        self.connected_directions = []
        self.conrner_pos = [(0, 0)]  # we start with a corner
        self.diff = {}
        self.edge_to_piece = {}

        for p in self.pieces_:
            for e in p.edges_:
                self.edge_to_piece[e] = p


        self.extremum = (-1, -1, 1, 1)
        self.log('>>> START solving puzzle')

        border_pieces = []
        non_border_pieces = []
        connected_pieces = []
        # Separate border pieces from the other
        for piece in self.pieces_:
            if piece.number_of_border():
                border_pieces.append(piece)
            else:
                non_border_pieces.append(piece)

        self.possible_dim = self.compute_possible_size(len(self.pieces_), len(border_pieces))

            # Start by a corner piece
        for piece in border_pieces:
            if piece.number_of_border() > 1:
                connected_pieces = [piece]
                border_pieces.remove(piece)
                break
        self.log("Number of border pieces: ", len(border_pieces) + 1)

        self.export_pieces('/tmp/stick{0:03d}'.format(1) + ".png",
                           '/tmp/colored{0:03d}'.format(1) + ".png",
                           'Border types'.format(1),
                           'Step {0:03d}'.format(1), display_boder=True)

        self.log('>>> START solve border')
        start_piece = connected_pieces[0]
        self.corner_pos = [((0, 0), start_piece)]  # we start with a corner

        for i in range(4):
            if start_piece.edge_in_direction(Directions.S).connected and start_piece.edge_in_direction(Directions.W).connected:
                break
            start_piece.rotate_edges(1)

        # coeff = [1, 1, 1, 1]
        # for i, d in enumerate(directions):
        #     if start_piece.edge_in_direction(d).connected:
        #         coeff[i] = 0
        # print(coeff)
        # self.extremum = (- coeff[3], - coeff[2], coeff[1], coeff[0])
        self.extremum = (0, 0, 1, 1)

        self.strategy = Strategy.BORDER
        connected_pieces = self.solve(connected_pieces, border_pieces)
        self.log('>>> START solve middle')
        self.strategy = Strategy.FILL
        self.solve(connected_pieces, non_border_pieces)


        # Simple fill
        # self.strategy = Strategy.FILL
        # connected_pieces = [self.pieces_[0]]
        # left_pieces = self.pieces_[1:]
        # self.solve(connected_pieces, left_pieces)


        self.log('>>> SAVING result...')
        self.translate_puzzle()
        self.export_pieces("/tmp/stick.png", "/tmp/colored.png", display=False)


        # Two sets of pieces: Already connected ones and pieces remaining to connect to the others
        # The first piece has an orientation like that:
        #         N          edges:    0
        #      W     E              3     1
        #         S                    2
        #
        # Pieces are placed on a grid like that (X is the first piece at position (0, 0)):
        # +--+--+--+
        # |  |  |  |
        # +--+--+--+
        # |  | X|  |
        # +--+--+--+
        # |  |  |  |
        # +--+--+--+
        #
        # Then if we test the NORTH edge:
        # +--+--+--+
        # |  | X|  |
        # +--+--+--+
        # |  | X|  |
        # +--+--+--+
        # |  |  |  |
        # +--+--+--+
        # Etc until the puzzle is complete i.e. there is no pieces left on left_pieces.

    def solve(self, connected_pieces, left_pieces, border=False):
        # angles = filter(lambda x: x.type == TypePiece.ANGLE, left_pieces)
        # borders = filter(lambda x: x.type == TypePiece.BORDER, left_pieces)
        # centers = filter(lambda x: x.type == TypePiece.CENTER, left_pieces)
        if len(self.connected_directions) == 0:
            self.connected_directions = [((0, 0), connected_pieces[0])] # ((x, y), p), x & y relative to the first piece, init with 1st piece
            self.diff = self.compute_diffs(left_pieces, self.diff, connected_pieces[0]) # edge on the border of the block -> edge on a left piece -> diff between edges
        else:
            self.diff = self.add_to_diffs(left_pieces)

        while len(left_pieces) > 0:
            self.log("<--- New match ---> pieces left: ", len(left_pieces), 'extremum:', self.extremum, 'puzzle dimension:', display_dim(self.possible_dim))
            block_best_e, best_e = self.best_diff(self.diff, self.connected_directions, left_pieces)
            block_best_p, best_p = self.edge_to_piece[block_best_e], self.edge_to_piece[best_e]

            stick_pieces(block_best_p, block_best_e, best_p, best_e, final_stick=True)

            self.update_direction(block_best_e, best_p, best_e)
            self.connect_piece(self.connected_directions, block_best_p, block_best_e.direction, best_p)

            connected_pieces.append(best_p)
            del left_pieces[left_pieces.index(best_p)]

            self.diff = self.compute_diffs(left_pieces, self.diff, best_p, edge_connected=block_best_e)
            self.export_pieces('/tmp/stick{0:03d}'.format(len(self.connected_directions)) + ".png",
                               '/tmp/colored{0:03d}'.format(len(self.connected_directions)) + ".png",
                               name_colored='Step {0:03d}'.format(len(self.connected_directions)))

        return connected_pieces


    def compute_diffs(self, left_pieces, diff, new_connected, edge_connected=None):
        # Remove former edge from the bloc border
        if edge_connected is not None:
            del diff[edge_connected]

        # build the list of edge to test
        edges_to_test = []
        for piece in left_pieces:
            for edge in piece.edges_:
                if not edge.connected:
                    edges_to_test.append((piece, edge))

        # Remove the edge of the new piece from the bloc border diffs
        for e in new_connected.edges_:
            for _, v in diff.items():
                if e in v:
                    del v[e]

            if e.connected:
                continue

            diff_e = {}
            for piece, edge in edges_to_test:
                if not e.is_compatible(edge):
                    continue
                for e2 in piece.edges_:
                    e2.backup_shape()
                stick_pieces(new_connected, e, piece, edge)
                diff_e[edge] = diff_full_compute(edge, e)

                for e2 in piece.edges_:
                    e2.restore_backup_shape()

            diff[e] = diff_e
        return diff


    def fallback(self, diff, connected_direction, left_piece, strat=Strategy.NAIVE):
        self.log('Fail to solve the puzzle with', self.strategy, 'falling back to', strat)
        old_strat = self.strategy
        self.strategy = Strategy.NAIVE
        best_bloc_e, best_e = self.best_diff(diff, connected_direction, left_piece)
        self.strategy = old_strat
        return best_bloc_e, best_e

    def best_diff(self, diff, connected_direction, left_piece):
        best_bloc_e, best_e, best_p, min_diff = None, None, None, float('inf')
        minX, minY, maxX, maxY = self.extremum

        if self.strategy == Strategy.FILL:
            best_coords = []

            # this is ugly
            for i in range(4, -1, -1): # 4 to 0
                best_coord = []
                for x in range(minX, maxX + 1):
                    for y in range(minY, maxY + 1):
                        neighbor = list(filter(lambda e: is_neigbhor((x, y), e[0], connected_direction), connected_direction))
                        if len(neighbor) == i:
                            best_coord.append(((x, y), neighbor))
                best_coords.append(best_coord)

            for best_coord in best_coords:
                for c, neighbor in best_coord:
                    for p in left_piece:
                        for rotation in range(4):
                            diff_score = 0
                            p.rotate_edges(1)
                            last_test = None, None
                            for block_c, block_p in neighbor:
                                direction_exposed = Directions(sub_tuple(c, block_c))
                                edge_exposed = block_p.edge_in_direction(direction_exposed)
                                edge = p.edge_in_direction(get_opposite_direction(direction_exposed))
                                # print(edge_exposed.connected, edge.connected)
                                if edge_exposed.connected or edge.connected or not edge.is_compatible(edge_exposed):
                                    diff_score = float('inf')
                                    break
                                else:
                                    diff_score += diff[edge_exposed][edge]
                                    last_test = edge_exposed, edge
                            if diff_score < min_diff:
                                best_bloc_e, best_e, min_diff = last_test[0], last_test[1], diff_score
                if best_e is not None:
                    break
                elif len(best_coord):
                    self.log('Fall back to a worst', self.strategy)
            if best_e is None:
                best_bloc_e, best_e = self.fallback(diff, connected_direction, left_piece)
            return best_bloc_e, best_e


        elif self.strategy == Strategy.BORDER:
            best_coord = []
            for x in range(minX, maxX + 1):
                for y in range(minY, maxY + 1):
                    neighbor = list(
                        filter(lambda e: is_neigbhor((x, y), e[0], connected_direction), connected_direction))
                    if len(neighbor) == 1 or (len(neighbor) == 2 and len(left_piece) == 1):
                        best_coord.append(((x, y), neighbor[0]))

            for c, neighbor in best_coord:
                for p in left_piece:
                    for rotation in range(4):
                        diff_score = 0
                        p.rotate_edges(1)
                        block_c, block_p = neighbor

                        direction_exposed = Directions(sub_tuple(c, block_c))
                        edge_exposed = block_p.edge_in_direction(direction_exposed)
                        edge = p.edge_in_direction(get_opposite_direction(direction_exposed))

                        if p.type == TypePiece.ANGLE:
                            if not corner_puzzle_alignement(c, p, self.corner_pos):
                                diff_score = float('inf')
                            elif not self.corner_place_fit_size(c):
                                diff_score = float('inf')
                        if p.type == TypePiece.BORDER and self.is_edge_at_corner_place(c):
                            diff_score = float('inf')
                        if diff_score != 0 or edge_exposed.connected or edge.connected \
                                or not edge.is_compatible(edge_exposed) or not p.is_border_aligned(block_p):
                            diff_score = float('inf')
                        else:
                            diff_score = diff[edge_exposed][edge]
                        
                        if diff_score < min_diff:
                            best_bloc_e, best_e, min_diff = edge_exposed, edge, diff_score
            if best_e is None:
                best_bloc_e, best_e = self.fallback(diff, connected_direction, left_piece, strat=Strategy.FILL)
            return best_bloc_e, best_e


        elif self.strategy == Strategy.NAIVE:
            for block_e, block_e_diff in diff.items():
                for e, diff_score in block_e_diff.items():
                    if diff_score < min_diff:
                        best_bloc_e, best_e, min_diff = block_e, e, diff_score
            return best_bloc_e, best_e
        else:
            return None, None



    def add_to_diffs(self, left_pieces):
        # build the list of edge to test
        edges_to_test = []
        for piece in left_pieces:
            for edge in piece.edges_:
                if not edge.connected:
                    edges_to_test.append((piece, edge))

        for e, diff_e in self.diff.items():
            for piece, edge in edges_to_test:
                if not e.is_compatible(edge):
                    continue
                for e2 in piece.edges_:
                    e2.backup_shape()
                stick_pieces(self.edge_to_piece[e], e, piece, edge)
                diff_e[edge] = diff_full_compute(edge, e)

                for e2 in piece.edges_:
                    e2.restore_backup_shape()

        return self.diff


    def update_direction(self, e, best_p, best_e):
        opp = get_opposite_direction(e.direction)
        step = step_direction(opp, best_e.direction)
        for edge in best_p.edges_:
            edge.direction = rotate_direction(edge.direction, step)

    def connect_piece(self, connected_directions, curr_p, dir, best_p):

        # Then we need to search the other pieces already in the puzzle that are going to be also connected:
        # +--+--+--+
        # |  | X| O|
        # +--+--+--+
        # |  | X| X|
        # +--+--+--+
        # |  |  |  |
        # +--+--+--+
        #
        # For example if I am going to put a piece at the marker 'O' only one edge will be connected to the piece
        # therefore we need to search the adjacent pieces and connect them properly

        old_coord = list(filter(lambda x: x[1] == curr_p, connected_directions))[0][0]
        new_coord = add_tuple(old_coord, dir.value)

        for (coord, p) in connected_directions:
            for d in directions:
                if equals_tuple(coord, add_tuple(new_coord, d.value)):
                    for edge in best_p.edges_:
                        if edge.direction == d:
                            edge.connected = True
                            break
                    for edge in p.edges_:
                        if edge.direction == get_opposite_direction(d):
                            edge.connected = True
                            break
        connected_directions.append((new_coord, best_p))



        minX, minY, maxX, maxY = self.extremum
        coeff = [1, 1, 1, 1]
        for i, d in enumerate(directions):
            if best_p.edge_in_direction(d).connected:
                coeff[i] = 0
        self.extremum = (min(minX, new_coord[0] - coeff[3]), min(minY, new_coord[1] - coeff[2]),
                         max(maxX, new_coord[0] + coeff[1]), max(maxY, new_coord[1] + coeff[0]))

        if best_p.type == TypePiece.ANGLE:
            self.corner_place_fit_size(new_coord, update_dim=True)
            self.corner_pos.append((new_coord, best_p))
        else:
            self.update_dimension()

        self.log('Placed:', best_p.type, 'at', new_coord)

    def stick_best(self, cur_piece, cur_edge, pieces, border=False):
        if cur_edge.connected:
            return

        def compatible_edges(e1, e2):
            return (e1.type == TypeEdge.HOLE and e2.type == TypeEdge.HEAD) or (e1.type == TypeEdge.HEAD and e2.type == TypeEdge.HOLE)

        edges_to_test = []
        for piece in pieces:
            if piece != cur_piece:
                for edge in piece.edges_:
                    # if border and piece.nBorders_ < 2 and piece.borders_[(index_edge + 2) % 4]: # FIXME ?
                    #     continue
                    if not compatible_edges(edge, cur_edge):
                        continue
                    if not edge.connected:
                        edges_to_test.append((piece, edge))

        diff = []
        for piece, edge in edges_to_test:
            # Stick pieces to test distance
            for e in piece.edges_:
                e.backup_shape()
            stick_pieces(cur_piece, cur_edge, piece, edge)
            diff.append(1 * diff_match_edges(edge.shape, cur_edge.shape))

            for e in piece.edges_:
                e.restore_backup_shape()


        # Stick the best piece found
        best_p, best_e = edges_to_test[np.argmin(diff)]
        stick_pieces(cur_piece, cur_edge, best_p, best_e, final_stick=True)

        return best_p, best_e

    def translate_puzzle(self):
        # Translate all pieces to the top left corner to be sure the puzzle is in the image
        minX = sys.maxsize
        minY = sys.maxsize
        for p in self.pieces_:
            for e in p.edges_:
                for pixel in e.shape:
                    if pixel[0] < minX:
                        minX = pixel[0]
                    if pixel[1] < minY:
                        minY = pixel[1]

        for p in self.pieces_:
            for e in p.edges_:
                for ip, _ in enumerate(e.shape):
                    e.shape[ip] += (-minX, -minY)

        for piece in self.pieces_:
            for p in piece.img_piece_:
                p.translate(minX, minY)

    def export_pieces(self, path_contour, path_colored, name_contour=None, name_colored=None, display=True, display_boder=False):
        minX, minY = float('inf'), float('inf')
        maxX, maxY = -float('inf'), -float('inf')
        for piece in self.pieces_:
            for p in piece.img_piece_:
                x, y = p.pos
                minX, minY = min(minX, x), min(minY, y)
                maxX, maxY = max(maxX, x), max(maxY, y)

        colored_img = np.zeros((maxX - minX, maxY - minY, 3))

        border_img = np.zeros((maxX - minX, maxY - minY, 3))

        for piece in self.pieces_:
            for p in piece.img_piece_:
                p.apply(colored_img, dx=-minX, dy=-minY)
            # Contours
            for e in piece.edges_:
                for y, x in e.shape:
                    y, x = y - minY, x - minX
                    if 0 <= y < border_img.shape[1] and 0 <= x < border_img.shape[0]:
                        rgb = (0, 0, 0)
                        if e.type == TypeEdge.HOLE:
                            rgb = (102, 178, 255)
                        if e.type == TypeEdge.HEAD:
                            rgb = (255, 255, 102)
                        if e.type == TypeEdge.UNDEFINED:
                            rgb = (255, 0, 0)
                        if e.connected:
                            rgb = (0, 255, 0)
                        border_img[x, y, 0] = rgb[2]
                        border_img[x, y, 1] = rgb[1]
                        border_img[x, y, 2] = rgb[0]

        # cv2.circle(tests_img, tuple((int(puzzle_pieces[1].edges_[0][0][0]), int(centerY))), 10, 255, -1)
        cv2.imwrite(path_contour, border_img)
        cv2.imwrite(path_colored, colored_img)
        if self.viewer and display:
            if display_boder:
                self.viewer.addImage(name_contour, path_contour, display=False)
            self.viewer.addImage(name_colored, path_colored)


    def compute_possible_size(self, nb_piece, nb_border):
        nb_edge_border = nb_border - 4
        nb_middle = nb_piece - nb_border
        possibilities = []
        for i in range(nb_edge_border // 2 + 1):
            w, h = i, (nb_edge_border // 2) - i
            if w * h == nb_middle:
                possibilities.append((w + 1, h + 1))
        self.log('Possible sizes: (', nb_piece, 'pieces with', nb_border, 'borders among them):', display_dim(possibilities))
        return possibilities


    def corner_place_fit_size(self, c,update_dim=False):
        def almost_equals(idx, target, val):
            return val[idx] == target or val[idx] == -target

        if len(self.possible_dim) == 1:
            # We have already picked a dimension
            return (c[0] == 0 or c[0] == self.possible_dim[0][0] or c[0] == -self.possible_dim[0][0]) and \
                   (c[1] == 0 or c[1] == self.possible_dim[0][1] or c[0] == -self.possible_dim[0][1])
        else:
            if c[0] == 0:
                filtered = list(filter(lambda x: almost_equals(1, c[1], x), self.possible_dim))
                if len(filtered):
                    if update_dim and len(filtered) != len(self.possible_dim):
                        self.log('Update possible dimensions with corner place:', display_dim(filtered))
                        self.possible_dim = filtered
                    return True
                else:
                    return False
            elif c[1] == 0:
                filtered = list(filter(lambda x: almost_equals(0, c[0], x), self.possible_dim))
                if len(filtered):
                    if update_dim and len(filtered) != len(self.possible_dim):
                        self.log('Update possible dimensions with corner place:', display_dim(filtered))
                        self.possible_dim = filtered
                    return True
                else:
                    return False
        return False

    def is_edge_at_corner_place(self, c):
        if len(self.possible_dim) == 1:
            # We have already picked a dimension
            return (c[0] == 0 or c[0] == self.possible_dim[0][0] or c[0] == -self.possible_dim[0][0]) and \
                   (c[1] == 0 or c[1] == self.possible_dim[0][1] or c[0] == -self.possible_dim[0][1])
        return False

    def update_dimension(self):
        if len(self.possible_dim) == 1:
            return
        dims = []
        _, _, maxX, maxY = self.extremum
        for x, y in self.possible_dim:
            if maxX <= x and maxY <= y:
                dims.append((x, y))
        if len(dims) != len(self.possible_dim):
            self.log('Update possible dimensions with extremum', self.extremum, ':', display_dim(dims))
            self.possible_dim = dims