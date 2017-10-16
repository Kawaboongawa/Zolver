from Puzzle.PuzzlePiece import *

from Puzzle.Extractor import Extractor
from Puzzle.Mover import *
from cv2 import cv2

from Puzzle.Enums import Directions

def diff_match_edges(e1, e2):
    diff = 0
    for i, p in enumerate(e1):
        if i < len(e2):
            diff += np.linalg.norm(p[0] - e2[len(e2) - i - 1][0])
        else:
            break
    return diff

def neg_dir(dir):
    return Directions((-dir.value[0], -dir.value[1]))

def add_tuples(tuple1, tuple2):
    return tuple(map(lambda x, y: x + y, tuple1, tuple2))

class Puzzle():
    def __init__(self, path, pixmapWidget):
        self.extract = Extractor(path, pixmapWidget)
        self.pieces_ = self.extract.extract()

        connected_pieces = [self.pieces_[0]]
        left_pieces = self.pieces_[1:]

        while len(left_pieces) > 0:
            to_break = False
            for ip, p in enumerate(connected_pieces):
                for ie, e in enumerate(p.edges_):
                    if not p.connected_[ie]:
                        print("<--- New match --->")
                        print("connected: {}".format(p.connected_))
                        tmp_ip, tmp_ie = self.stick_best(connected_pieces[ip], ie, left_pieces)

                        print("orientation {}, position {}".format(p.orientation[ie].value, p.position))
                        left_pieces[tmp_ip].position = add_tuples(p.position, p.orientation[ie].value)
                        left_pieces[tmp_ip].orientation[(ie + 2) % 4] = neg_dir(connected_pieces[ip].orientation[ie])
                        print("orientation {}, position {}".format(left_pieces[tmp_ip].orientation[(ie + 2) % 4].value, left_pieces[tmp_ip].position))

                        for i_tmp in range(1, 3):
                            o_pos = (ie + 2 + i_tmp - 1) % 4
                            n_pos = (o_pos + 1) % 4
                            if left_pieces[tmp_ip].orientation[o_pos] == Directions.N:
                                left_pieces[tmp_ip].orientation[n_pos] = Directions.E
                            elif left_pieces[tmp_ip].orientation[o_pos] == Directions.E:
                                left_pieces[tmp_ip].orientation[n_pos] = Directions.S
                            elif left_pieces[tmp_ip].orientation[o_pos] == Directions.S:
                                left_pieces[tmp_ip].orientation[n_pos] = Directions.W
                            elif left_pieces[tmp_ip].orientation[o_pos] == Directions.W:
                                left_pieces[tmp_ip].orientation[n_pos] = Directions.N

                        for ip2, p2 in enumerate(connected_pieces):
                            if p2.position == add_tuples(left_pieces[tmp_ip].position, Directions.N.value):
                                connected_pieces[ip2].connected_[p2.orientation.index(Directions.S)] = True
                                left_pieces[tmp_ip].connected_[left_pieces[tmp_ip].orientation.index(Directions.N)] = True
                            elif p2.position == add_tuples(left_pieces[tmp_ip].position, Directions.S.value):
                                connected_pieces[ip2].connected_[p2.orientation.index(Directions.N)] = True
                                left_pieces[tmp_ip].connected_[left_pieces[tmp_ip].orientation.index(Directions.S)] = True
                            elif p2.position == add_tuples(left_pieces[tmp_ip].position, Directions.W.value):
                                connected_pieces[ip2].connected_[p2.orientation.index(Directions.E)] = True
                                left_pieces[tmp_ip].connected_[left_pieces[tmp_ip].orientation.index(Directions.W)] = True
                            elif p2.position == add_tuples(left_pieces[tmp_ip].position, Directions.E.value):
                                connected_pieces[ip2].connected_[p2.orientation.index(Directions.W)] = True
                                left_pieces[tmp_ip].connected_[left_pieces[tmp_ip].orientation.index(Directions.E)] = True

                        #connected_pieces[ip].connected_[ie] = True
                        #left_pieces[tmp_ip].connected_[tmp_ie] = True

                        connected_pieces.append(left_pieces[tmp_ip])
                        del left_pieces[tmp_ip]
                        to_break = True
                        break
                if to_break:
                    break

        self.pieces_ = connected_pieces

        # stick_pieces(self.pieces_[1], 3, self.pieces_[8], 1)
        #self.stick_best(self.pieces_[1], 2)
        #self.stick_best(self.pieces_[1], 3)
        #self.stick_best(self.pieces_[4], 3)
        #self.stick_best(self.pieces_[4], 2)
        #self.stick_best(self.pieces_[8], 3)
        #self.stick_best(self.pieces_[6], 3)

        #self.stick_best(self.pieces_[0], 0)
        #self.stick_best(self.pieces_[3], 3)
        self.export_pieces("/tmp/test_stick.png")

    def stick_best(self, cur_piece, edge_cur_piece, pieces):
        if cur_piece.connected_[edge_cur_piece]:
            return

        tests = []
        for index_piece, piece in enumerate(pieces):
            if piece != cur_piece:
                for index_edge, edge in enumerate(piece.edges_):
                    tests.append((index_piece, index_edge, piece.fourier_descriptors_[index_edge].match_descriptors(cur_piece.fourier_descriptors_[edge_cur_piece])))

        l = sorted(tests, key=lambda x: x[2])
        diff = []
        for i in range(len(l)):
            if not pieces[l[i][0]].connected_[l[i][1]]:
                # Save edges to restore them
                tmp = np.array(pieces[l[i][0]].edges_)
                for j in range(4):
                    tmp[j] = np.array(pieces[l[i][0]].edges_[j])

                # Stick pieces to test distance
                stick_pieces(cur_piece, edge_cur_piece, pieces[l[i][0]], l[i][1])
                diff.append(diff_match_edges(pieces[l[i][0]].edges_[l[i][1]], cur_piece.edges_[edge_cur_piece]))

                # Restore state of edges
                pieces[l[i][0]].edges_ = tmp
            else:
                diff.append(float('inf'))

        m = np.argmin(diff)
        print(l[m][0], l[m][1])

        # Stick the best piece found
        stick_pieces(cur_piece, edge_cur_piece, pieces[l[m][0]], l[m][1])
        return l[m][0], l[m][1]

    def export_pieces(self, path):
        tests_img = np.zeros_like(self.extract.img)

        for piece in self.pieces_:
            for i in range(4):
                for p in piece.edges_[i]:
                    # if not piece.border[i] and p[0][0] < self.extract.img.shape[1] and p[0][1] < self.extract.img.shape[0]:
                    if p[0][0] < self.extract.img.shape[1] and p[0][1] < self.extract.img.shape[0]:
                        tests_img[p[0][1], p[0][0]] = 255

        # cv2.circle(tests_img, tuple((int(puzzle_pieces[1].edges_[0][0][0]), int(centerY))), 10, 255, -1)
        cv2.imwrite(path, tests_img)
