import math
import numpy as np

class FourierDescriptor():
    def __init__(self, shape, size):
        self.size_ = size
        self.centroid_ = 0j
        self.compute_centroid(shape)
        self.descriptors_ = self.compute_centroid_distance(shape)
        self.compute_descriptors()

    def compute_centroid(self, shape): #valid
        for i in range(0, self.size_):
            self.centroid_ += shape[i][0] + shape[i][1] * 1j
        self.centroid_ /= self.size_

    def compute_centroid_distance(self, shape):
        dst = np.zeros(len(shape),).astype(complex)
        for i in range(0,  self.size_):
            r = float(shape[i][0]) - self.centroid_.real
            im = float(shape[i][1]) - self.centroid_.imag
            dst[i] = math.sqrt(r * r + im * im)
        return dst

    def compute_magnitude(self):
        for i, e in enumerate(self.descriptors_):
            self.descriptors_[i] = math.sqrt(e[0] * e[0] + e[1] * e[1])

    def compute_centroid_signature(self, mag, centroid):
        res = []
        for i in range(0, len(self.descriptors_) // 2):
            res.append(mag[i] / centroid)
        self.descriptors_ = res

    def compute_descriptors(self):
        fft = np.fft.fft(self.descriptors_)
        mag = np.absolute(fft)
        self.compute_centroid_signature(mag[1:], mag[0])

    def match_descriptors(self, d2):
        res = 0.0
        size = min(len(self.descriptors_), len(d2.descriptors_))
        for i in range(0, size):
            mod = math.fabs(self.descriptors_[i] - d2.descriptors_[i])
            res += mod * mod
        return math.sqrt(res)