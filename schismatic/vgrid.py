"""Copyright 2023 University of Auckland.

This file is part of schismatic.

schismatic is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

schismatic is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with schismatic.  If not, see <http://www.gnu.org/licenses/>."""

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
