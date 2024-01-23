import numba
from numba import int32, int64
from numba.experimental import jitclass
import numpy as np


@numba.jit((int64[:], int64[:]), nopython=True)
def is_equal(a1, a2):
    return a1[0] == a2[0] and a1[1] == a2[1]


spec = [
    ("xo", int32[:]),
    ("yo", int32[:]),
    ("zo", int32[:]),
    ("xmin", int32),
    ("ymin", int32),
    ("zmin", int32),
    ("ymax", int32),
    ("xmax", int32),
    ("zmax", int32),
]


@jitclass(spec)
class GridNghFinder:
    def __init__(self, xmin, ymin, zmin, xmax, ymax, zmax):
        self.xo = np.array([0, 0, -1, 0, 1, 0, 0], dtype=np.int32)
        self.yo = np.array([0, 1, 0, 0, 0, -1, 0], dtype=np.int32)
        self.zo = np.array([1, 0, 0, 0, 0, 0, -1], dtype=np.int32)
        self.xmin = xmin
        self.ymin = ymin
        self.zmin = zmin
        self.xmax = xmax
        self.ymax = ymax
        self.zmax = zmax

    def find(self, x, y, z):
        xs = self.xo + x
        ys = self.yo + y
        zs = self.zo + z

        xd = (xs >= self.xmin) & (xs <= self.xmax)
        xs = xs[xd]
        ys = ys[xd]
        zs = zs[xd]

        yd = (ys >= self.ymin) & (ys <= self.ymax)
        xs = xs[yd]
        ys = ys[yd]
        zs = zs[yd]

        zd = (zs >= self.zmin) & (zs <= self.zmax)
        xs = xs[zd]
        ys = ys[zd]
        zs = zs[zd]

        return np.stack((xs, ys, zs), axis=-1)
