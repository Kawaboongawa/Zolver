import math
import numpy as np

class FourierDescriptor():
    def __init__(self, shape, size):
        self.shape_ = shape
        self.size_ = size
        self.centroid_ = (0.0, 0.0)
        self.compute_centroid()
        self.descriptors_ = self.compute_centroid_distance()
        self.compute_descriptors()

    def compute_centroid(self):
        for i in range(0, self.size_):
            curr = self.shape_[i]
            self.centroid_ = (self.centroid_[0] + float(curr[0]), self.centroid_[1] + float(curr[1]))
        self.centroid_ = (self.centroid_[0] / float(self.size_), self.centroid_[1] / float(self.size_))

    def compute_centroid_distance(self):
        dst = []
        for i in range(0,  self.size_):
            dst.append((float(self.shape_[i][0]) - self.centroid_[0], float(self.shape_[i][1]) - self.centroid_[1]))
        return dst

    def compute_magnitude(self):
        for i, e in enumerate(self.descriptors_):
            self.descriptors_[i] = math.sqrt(e[0] * e[0] + e[1] * e[1])

    def compute_centroid_signature(self, centroid):
        for i in range(1, int(len(self.descriptors_) / 2)):
            self.descriptors_[i - 1] = self.descriptors_[i] / centroid
        self.descriptors_.pop()

    def compute_descriptors(self):
        np.fft.fft(self.descriptors_)
        self.compute_magnitude()
        self.compute_centroid_signature(self.descriptors_[0])

    def match_descriptors(self, d2):
        res = 0.0
        size = min(len(self.shape_), len(d2.descriptors_))
        for i in range(0, size):
            mod = math.fabs(self.descriptors_[i] - d2.descriptors_[i])
            res += mod * mod
        return math.sqrt(res)