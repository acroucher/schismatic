"""SCHISM vertical grids."""

import numpy as np

class lsc2(object):
    """LSC2 vertical grid object"""

    def __init__(self, filename = None):
        if filename is not None:
            self.read(filename)

    def read(self, filename):
        """Reads LSC2 grid from file."""

        with open(filename) as f: lines = f.readlines()

        self.ivcor = int(lines[0].strip().split()[0])
        if self.ivcor != 1:
            raise Exception('%s is not an LSC2 grid.' % filename)

        self.num_levels = int(lines[1].strip().split()[0])
        self.bottom_index = np.array(lines[2].split()).astype('int') - 1

        self.sigma = np.array([line.split()[1:]
                               for line in lines[3:]]).T.astype('float')
        nines = self.sigma < -1
        self.sigma[nines] = -1
