from Puzzle.Distance import diff_match_edges, real_edge_compute, generated_edge_compute
from Puzzle.PuzzlePiece import *

from Puzzle.Extractor import Extractor
from Puzzle.Mover import *
from cv2 import cv2

from Puzzle.Enums import *
import sys
import scipy

from Puzzle.tuple_helper import equals_tuple, add_tuple, sub_tuple, is_neigbhor, corner_puzzle_alignement, display_dim


class Puzzle():
    """
        Class used to store all informations about the puzzle
    """

    def log(self, *args):
        """ Helper to log informations to the GUI """

        print(' '.join(map(str, args)))
        if self.viewer:
            self.viewer.addLog(args)

    def __init__(self, path, viewer=None, green_screen=False):
        """ Extract informations of pieces in the img at `path` and start computation of the solution """

        self.pieces_ = None
        factor = 0.40
        while self.pieces_ is None:
            factor += 0.01
            self.extract = Extractor(path, viewer, green_screen, factor)
            self.pieces_ = self.extract.extract()

        self.viewer = viewer
        self.green_ = green_screen
        self.connected_directions = []
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
                           'Step {0:03d}'.format(1), display_border=True)

        self.log('>>> START solve border')
        start_piece = connected_pieces[0]
        self.corner_pos = [((0, 0), start_piece)]  # we start with a corner

        for i in range(4):
            if start_piece.edge_in_direction(Directions.S).connected and start_piece.edge_in_direction(Directions.W).connected:
                break
            start_piece.rotate_edges(1)

        self.extremum = (0, 0, 1, 1)

        self.strategy = Strategy.BORDER
        connected_pieces = self.solve(connected_pieces, border_pieces)
        self.log('>>> START solve middle')
        self.strategy = Strategy.FILL
        self.solve(connected_pieces, non_border_pieces)

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
        """
            Solve the puzzle by finding the optimal piece in left_pieces matching the edges
            available in connected_pieces

            :param connected_pieces: pieces already connected to the puzzle
            :param left_pieces: remaining pieces to place in the puzzle
            :param border: Boolean to determine if the strategy is border
            :return: List of connected pieces
        """

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
        """
            Compute the diff between the left pieces edges and the new_connected piece edges
            by sticking them and compute the distance

            :param left_pieces: remaining pieces to place in the puzzle
            :param diff: pre computed diff between edges to speed up the process
            :param new_connected: Connected pieces to test for a match
            :return: updated diff matrix
        """

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
                if self.green_:
                    diff_e[edge] = real_edge_compute(edge, e)
                else:
                    diff_e[edge] = generated_edge_compute(edge, e)
                for e2 in piece.edges_:
                    e2.restore_backup_shape()

            diff[e] = diff_e
        return diff


    def fallback(self, diff, connected_direction, left_piece, strat=Strategy.NAIVE):
        """ If a strategy does not work fallback to another one """

        self.log('Fail to solve the puzzle with', self.strategy, 'falling back to', strat)
        old_strat = self.strategy
        self.strategy = Strategy.NAIVE
        best_bloc_e, best_e = self.best_diff(diff, connected_direction, left_piece)
        self.strategy = old_strat
        return best_bloc_e, best_e

    def best_diff(self, diff, connected_direction, left_piece):
        """
            Find the best matching edge for a piece edge

            :param diff: pre computed diff between edges to speed up the process
            :param connected_direction: Direction of the edge to connect
            :param left_piece: Piece to connect
            :return: the best edge found in the bloc
        """

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
        """ build the list of edge to test """

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
                if self.green_:
                    diff_e[edge] = real_edge_compute(edge, e)
                else:
                    diff_e[edge] = generated_edge_compute(edge, e)
                for e2 in piece.edges_:
                    e2.restore_backup_shape()

        return self.diff


    def update_direction(self, e, best_p, best_e):
        """ Update the direction of the edge after matching it """

        opp = get_opposite_direction(e.direction)
        step = step_direction(opp, best_e.direction)
        for edge in best_p.edges_:
            edge.direction = rotate_direction(edge.direction, step)

    def connect_piece(self, connected_directions, curr_p, dir, best_p):
        """
            Then we need to search the other pieces already in the puzzle that are going to be also connected:
            +--+--+--+
            |  | X| O|
            +--+--+--+
            |  | X| X|
            +--+--+--+
            |  |  |  |
            +--+--+--+

            For example if I am going to put a piece at the marker 'O' only one edge will be connected to the piece
            therefore we need to search the adjacent pieces and connect them properly
        """

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

    def translate_puzzle(self):
        """ Translate all pieces to the top left corner to be sure the puzzle is in the image """

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

    def export_pieces(self, path_contour, path_colored, name_contour=None, name_colored=None, display=True, display_border=False):
        """
            Export the contours and the colored image

            :param path_contour: Path used to export contours
            :param path_colored: Path used to export the colored image
            :return: the best edge found in the bloc
        """

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

        cv2.imwrite(path_contour, border_img)
        cv2.imwrite(path_colored, colored_img)
        if self.viewer and display:
            if display_border:
                self.viewer.addImage(name_contour, path_contour, display=False)
            self.viewer.addImage(name_colored, path_colored)


    def compute_possible_size(self, nb_piece, nb_border):
        """
            Compute all possible size of the puzzle based on the number
            of pieces and the number of border pieces
        """

        nb_edge_border = nb_border - 4
        nb_middle = nb_piece - nb_border
        possibilities = []
        for i in range(nb_edge_border // 2 + 1):
            w, h = i, (nb_edge_border // 2) - i
            if w * h == nb_middle:
                possibilities.append((w + 1, h + 1))
        self.log('Possible sizes: (', nb_piece, 'pieces with', nb_border, 'borders among them):', display_dim(possibilities))
        return possibilities


    def corner_place_fit_size(self, c, update_dim=False):
        """ Update the possible dimensions of the puzzle when a corner is placed """

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
        """ Determine of an edge is at a corner place """

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