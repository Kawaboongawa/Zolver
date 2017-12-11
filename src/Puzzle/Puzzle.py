from Puzzle.Distance import diff_match_edges, diff_match_edges2, diff_full_compute
from Puzzle.PuzzlePiece import *

from Puzzle.Extractor import Extractor
from Puzzle.Mover import *
from Img.filters import load_signatures
from cv2 import cv2

from Puzzle.Enums import *
import sys
import scipy

from Puzzle.tuple_helper import equals_tuple, add_tuple, sub_tuple, is_neigbhor


class Puzzle():
    def __init__(self, path, pixmapWidget=None):
        self.extract = Extractor(path, pixmapWidget)
        self.pieces_ = self.extract.extract()
        self.connected_directions = []
        self.diff = {}
        self.edge_to_piece = {}
        for p in self.pieces_:
            for e in p.edges_:
                self.edge_to_piece[e] = p
        # self.extremum = (0, 0, 0, 0)
        self.extremum = (-1, -1, 1, 1)
        print('>>> START solving puzzle')

        border_pieces = []
        non_border_pieces = []
        connected_pieces = []
        # Separate border pieces from the other
        for piece in self.pieces_:
            if piece.number_of_border():
                border_pieces.append(piece)
            else:
                non_border_pieces.append(piece)

        # Start by a corner piece
        for piece in border_pieces:
            if piece.number_of_border() > 1:
                connected_pieces = [piece]
                border_pieces.remove(piece)
                break
        print("Number of border pieces: ", len(border_pieces) + 1)

        print('>>> START solve border')
        connected_pieces = self.solve(connected_pieces, border_pieces)
        print('>>> START solve middle')
        self.solve(connected_pieces, non_border_pieces)

        # connected_pieces = [self.pieces_[0]]
        # left_pieces = self.pieces_[1:]
        # self.solve(connected_pieces, left_pieces)
        print('>>> SAVING result...')
        self.translate_puzzle()
        self.export_pieces("/tmp/stick.png", "/tmp/colored.png")


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
            print("<--- New match ---> (left: ", len(left_pieces), ')')
            block_best_e, best_e = self.best_diff(self.diff, self.connected_directions, left_pieces)
            block_best_p, best_p = self.edge_to_piece[block_best_e], self.edge_to_piece[best_e]

            stick_pieces(block_best_p, block_best_e, best_p, best_e, final_stick=True)

            self.update_direction(block_best_e, best_p, best_e)
            self.connect_piece(self.connected_directions, block_best_p, block_best_e.direction, best_p)

            connected_pieces.append(best_p)
            del left_pieces[left_pieces.index(best_p)]

            self.diff = self.compute_diffs(left_pieces, self.diff, best_p, edge_connected=block_best_e)
            self.export_pieces("/tmp/stick" + str(len(left_pieces)) + ".png",
                               "/tmp/colored" + str(len(left_pieces)) + ".png")

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
                for e2 in piece.edges_:
                    e2.backup_shape()
                stick_pieces(new_connected, e, piece, edge)
                diff_e[edge] = diff_full_compute(edge, e)

                for e2 in piece.edges_:
                    e2.restore_backup_shape()

            diff[e] = diff_e
        return diff


    def best_diff(self, diff, connected_direction, left_piece):
        best_bloc_e, best_e, min_diff = None, None, float('inf')

        minX, minY, maxX, maxY = self.extremum
        best_coord = []

        # this is ugly
        for i in range(4, -1, -1): # 4 to 0
            for x in range(minX, maxX + 1):
                for y in range(minY, maxY + 1):
                    neighbor = list(filter(lambda e: is_neigbhor((x, y), e[0], connected_direction), connected_direction))
                    if len(neighbor) == i:
                        best_coord.append(((x, y), neighbor))
            if len(best_coord):
                break

        for c, neighbor in best_coord:
            for p in left_piece:
                for rotation in range(4):
                    diff_score = float('inf')
                    p.rotate_edges(1)
                    last_test = None, None
                    for block_c, block_p in neighbor:
                        direction_exposed = Directions(sub_tuple(c, block_c))
                        edge_exposed = block_p.edge_in_direction(direction_exposed)
                        edge = p.edge_in_direction(get_opposite_direction(direction_exposed))
                        if edge_exposed.connected or edge.connected:
                            diff_score = float('inf')
                            break
                        else:
                            diff_score += diff[edge_exposed][edge]
                            last_test = edge_exposed, edge
                    if diff_score < min_diff:
                        best_bloc_e, best_e, min_diff = last_test[0], last_test[1], diff_score


                        # for block_e, block_e_diff in diff.items():
        #     for e, diff_score in block_e_diff.items():
        #         if diff_score < min_diff:
        #             best_bloc_e, best_e, min_diff = block_e, e, diff_score
        return best_bloc_e, best_e



    def add_to_diffs(self, left_pieces):
        # build the list of edge to test
        edges_to_test = []
        for piece in left_pieces:
            for edge in piece.edges_:
                if not edge.connected:
                    edges_to_test.append((piece, edge))

        for e, diff_e in self.diff.items():
            for piece, edge in edges_to_test:
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
        # self.extremum = (min(minX, new_coord[0]), min(minY, new_coord[1]), max(maxX, new_coord[0]), max(maxY, new_coord[1]))
        self.extremum = (min(minX, new_coord[0] - 1), min(minY, new_coord[1] - 1), max(maxX, new_coord[0] + 1), max(maxY, new_coord[1] + 1))
        print('matched:', list([e[0] for e in connected_directions]))




    def stick_best(self, cur_piece, cur_edge, pieces, border=False):
        if cur_edge.connected:
            return

        edges_to_test = []
        for piece in pieces:
            if piece != cur_piece:
                for edge in piece.edges_:
                    # if border and piece.nBorders_ < 2 and piece.borders_[(index_edge + 2) % 4]: # FIXME ?
                    #     continue
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

    def export_pieces(self, path_contour, path_colored):
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
                        rgb = (0, 255, 0) if e.connected else (255, 255, 255)
                        border_img[x, y, 0] = rgb[0]
                        border_img[x, y, 1] = rgb[1]
                        border_img[x, y, 2] = rgb[2]

        # cv2.circle(tests_img, tuple((int(puzzle_pieces[1].edges_[0][0][0]), int(centerY))), 10, 255, -1)
        cv2.imwrite(path_contour, border_img)
        cv2.imwrite(path_colored, colored_img)

