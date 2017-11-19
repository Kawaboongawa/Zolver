import math

class Pixel:
    def __init__(self, pos, color):
        self.pos = pos
        self.color = color

    def apply(self, img):
        if self.pos[0] >= 0 and self.pos[1] >= 0 and self.pos[0] < img.shape[0] and self.pos[1] < img.shape[1]:
            img[self.pos] = self.color

    def translate(self, dx, dy):
        self.pos = (self.pos[0] + dx, self.pos[1] + dy)

    def rotate(self, origin, angle):
        ox, oy = origin
        py, px = self.pos

        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
        if qx != qx or qy != qy:
            print("NAN DETECTED: {} {} {} {} {}".format(ox, oy, px, py, qx, qy, angle))

        self.pos = (int(qy), int(qx))
