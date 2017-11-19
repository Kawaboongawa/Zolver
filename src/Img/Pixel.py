class Pixel:
    def __init__(self, pos, color):
        self.pos = pos
        self.color = color

    def apply(self, img):
        if self.pos[0] < img.shape[0] and self.pos[1] < img.shape[1]:
            img[self.pos] = self.color

    def translate(self, dx, dy):
        self.pos = (self.pos[0] + dx, self.pos[1] + dy)