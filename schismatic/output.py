"""Copyright 2023 University of Auckland.

This file is part of schismatic.

schismatic is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

schismatic is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with schismatic.  If not, see <http://www.gnu.org/licenses/>."""

"""SCHISM output files"""

import os
from datetime import datetime, timedelta
import numpy as np
import xarray as xr

class output(object):
    """SCHISM output object"""

    def __init__(self, output_dir = 'outputs'):
        """Initialise SCHISM output."""

        def open_dataset(output_dir, filename):
            return xr.open_mfdataset(os.path.join(output_dir, filename),
                                     concat_dim = 'time',
                                     combine = 'nested', data_vars = 'minimal',
                                     coords = 'minimal', compat = 'override').sortby('time')

        def schism_start_datetime(ds):
            start_str = ds.time.base_date
            items = start_str.split()
            y = int(items[0])
            m = int(items[1])
            d = int(items[2])
            h = float(items[3])
            hr = int(h)
            mins = int((h - hr) * 60)
            return datetime(y, m, d, hr, mins)

        ds_2d = open_dataset(output_dir, 'out2d_*.nc')

        self.start_datetime = schism_start_datetime(ds_2d)
        self.datetime = self.start_datetime + np.array([timedelta(seconds = s)
                                                        for s in ds_2d['time'].values])
        self.end_datetime = self.datetime[-1]
        self.num_times = len(self.datetime)

        self.elevation = ds_2d['elevation']
        self.depthAverageVelX = ds_2d['depthAverageVelX']
        self.depthAverageVelY = ds_2d['depthAverageVelY']

        self.zCoordinates = open_dataset(output_dir, 'zCoordinates_*.nc')['zCoordinates']
        self.temperature = open_dataset(output_dir, 'temperature_*.nc')['temperature']
        self.salinity = open_dataset(output_dir, 'salinity_*.nc')['salinity']
        self.horizontalVelX = open_dataset(output_dir, 'horizontalVelX_*.nc')['horizontalVelX']
        self.horizontalVelY = open_dataset(output_dir, 'horizontalVelY_*.nc')['horizontalVelY']

    def close(self):
        """Closes SCHISM output datasets."""

        self.elevation.close()
        self.depthAverageVelX.close()
        self.depthAverageVelY.close()

        self.zCoordinates.close()
        self.temperature.close()
        self.salinity.close()
        self.horizontalVelX.close()
        self.horizontalVelY.close()

