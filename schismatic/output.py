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

    def __init__(self, output_dir = 'outputs', indices = None, oldio = False):
        """Initialise SCHISM output."""

        def open_dataset(output_dir, filename):
            if isinstance(filename, str):
                fnames = os.path.join(output_dir, filename)
            else:
                fnames = [os.path.join(output_dir, f) for f in filename]
            return xr.open_mfdataset(fnames,
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

        def filenames(base, indices):
            if indices is None:
                return base + '_*.nc'
            else:
                return ['%s_%d.nc' % (base, i) for i in indices]

        def init_oldio(output_dir, indices):
            # old SCHISM output (flag OLDIO in SCHISM 5.10)

            self.ds = open_dataset(output_dir, filenames('schout', indices))
            self.start_datetime = schism_start_datetime(self.ds)
            t = self.ds['time'].values
            secs = (t - t[0]) / np.timedelta64(1, 's')
            self.datetime = self.start_datetime + np.array([timedelta(seconds = s)
                                                            for s in secs])
            self.end_datetime = self.datetime[-1]
            self.num_times = len(self.datetime)

            self.elevation = self.ds['elev']
            self.depthAverageVelX = self.ds['dahv'][:,:,0]
            self.depthAverageVelY = self.ds['dahv'][:,:,1]

            self.zCoordinates = self.ds['zcor']
            self.temperature = self.ds['temp']
            self.salinity = self.ds['salt']
            self.horizontalVelX = self.ds['hvel'][:,:,:,0]
            self.horizontalVelY = self.ds['hvel'][:,:,:,1]

        def init_newio(output_dir, indices):
            # New SCHISM >= 5.10 scribed output

            self.ds_2d = open_dataset(output_dir, filenames('out2d', indices))
            self.start_datetime = schism_start_datetime(self.ds_2d)
            self.datetime = self.start_datetime + np.array([timedelta(seconds = s)
                                                            for s in self.ds_2d['time'].values])
            self.end_datetime = self.datetime[-1]
            self.num_times = len(self.datetime)

            self.elevation = self.ds_2d['elevation']
            self.depthAverageVelX = self.ds_2d['depthAverageVelX']
            self.depthAverageVelY = self.ds_2d['depthAverageVelY']

            self.ds_z = open_dataset(output_dir, filenames('zCoordinates', indices))
            self.zCoordinates = self.ds_z['zCoordinates']
            self.ds_t = open_dataset(output_dir, filenames('temperature', indices))
            self.temperature = self.ds_t['temperature']
            self.ds_s = open_dataset(output_dir, filenames('salinity', indices))
            self.salinity = self.ds_s['salinity']
            self.ds_vx = open_dataset(output_dir, filenames('horizontalVelX', indices))
            self.horizontalVelX = self.ds_vx['horizontalVelX']
            self.ds_vy = open_dataset(output_dir, filenames('horizontalVelY', indices))
            self.horizontalVelY = self.ds_vy['horizontalVelY']

        self.oldio = oldio
        if self.oldio:
            init_oldio(output_dir, indices)
        else:
            init_newio(output_dir, indices)

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

        if self.oldio:
            self.ds.close()
        else:
            self.ds_2d.close()
            self.ds_z.close()
            self.ds_t.close()
            self.ds_s.close()
            self.ds_vx.close()
            self.ds_vy.close()
