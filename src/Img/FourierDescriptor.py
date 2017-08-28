import math
import numpy as np

class FourierDescriptor():
    def __init__(self, shape, size):
        self.shape_ = shape
        self.size_ = size
        self.centroid_ = tuple(float(0), float(0))
        self.compute_centroid()
        descriptors = self.compute_centroid_distance()
        print(descriptors)
        self.compute_descriptors(descriptors)
        print(descriptors)

    def compute_centroid(self):
        for i in range(0, self.size_):
            curr = self.shape_[i]
            self.centroid_[0] += float(curr[0])
            self.centroid_[1] += float(curr[1])
        self.centroid_[0] /= float(self.size_)
        self.centroid_[1] /= float(self.size_)

    def compute_centroid_distance(self):
        dst = [float(0)] * self.size_
        for i in range(0,  self.size_):
            dst[i] = float(self.shape_[i]) - self.centroid_
        return dst

    def compute_magnitude(self, src):
        for e in src:
            modulus = math.sqrt(e[0] * e[0] + e[1] * e[1])
            e[0] = modulus
            e[1] = 0.0
        return src

    def compute_signature_centroid(self, src, centroid):
        for e in src:
            e[0] = e[0] / centroid
        return src

    def compute_descriptors(self, src):
        np.fft.fft(src)
        self.get_magnitude(src)
        self.compute_signature_centroid(src[1:], src[0])