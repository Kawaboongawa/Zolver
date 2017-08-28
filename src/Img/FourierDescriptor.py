import math
import numpy as np

class FourierDescriptor():
    def __init__(self, shape, size):
        self.shape_ = shape
        self.size_ = size
        self.centroid_ = (0.0, 0.0)
        self.compute_centroid()
        descriptors = self.compute_centroid_distance()
        print(descriptors)
        self.compute_descriptors(descriptors)
        print(descriptors)

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

    def compute_magnitude(self, src):
        for e in src:
            modulus = math.sqrt(e[0] * e[0] + e[1] * e[1])
            e = (modulus, 0.0)
        return src

    def compute_signature_centroid(self, src, centroid):
        for e in src:
            print(centroid)
            #e = (e[0] / centroid[0])
        return src

    def compute_descriptors(self, src):
        np.fft.fft(src)
        self.compute_magnitude(src)
        self.compute_signature_centroid(src[1:], src[0])