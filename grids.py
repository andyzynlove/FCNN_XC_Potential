import numpy as np

class Grids(object):
    def __init__(self, coords, weights):
        self._coords = coords
        self.original_coords = coords.copy() 
        self.weights = weights
        self.a = 0.566915 # Cube length
        self.na = 7       # Cube point  
        self.coords = self._coords.copy() 
        self._generate_offset()
        self._extend_coords()

    def _generate_offset(self):
        na3 = self.na * self.na * self.na
        offset = np.empty([na3, 3])
        dd = 1. / (self.na - 1)
        p = 0
        for i in range(self.na):
            for j in range(self.na):
                for k in range(self.na):
                    offset[p][0] = -0.5 + dd * i
                    offset[p][1] = -0.5 + dd * j
                    offset[p][2] = -0.5 + dd * k
                    p += 1
        self.offset = offset * self.a

    def _extend_coords(self):
        na = self.na
        na3 = na * na * na
        extended_coords = np.empty([len(self.coords)*na3, 3])
        p = 0
        for i, c in enumerate(self.coords):
            extended_coords[p:p+na3] = c + self.offset
            p += na3
        self.extended_coords = extended_coords