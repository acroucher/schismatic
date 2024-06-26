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
import calendar

def date_range_filenames(start, end, base_dir = './', filename = 'schout'):
    """Returns list of filenames, based on given filename, for specified
       date range.  Files are assumed to be daily results in
       year/month directories.
    """
    filenames = []
    delt = timedelta(days = 1)
    d = start
    while d < end:
        fname = os.path.join(base_dir, '%4d' % d.year, '%02d' % d.month,
                             '%s_%d.nc' % (filename, d.day))
        filenames.append(fname)
        d += delt
    return filenames

class output(object):
    """SCHISM output object"""

    def __init__(self, filenames, oldio = False):
        """Initialise SCHISM output."""

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

        def init_oldio(filenames):
            # old SCHISM output (flag OLDIO in SCHISM 5.10)

            self.ds = xr.open_mfdataset(filenames)
            self.start_datetime = schism_start_datetime(self.ds)
            t = self.ds['time'].values
            epoch = np.datetime64(0, 's')
            zone = float(self.ds.time.base_date.split()[-1])
            secs = (t - epoch) / np.timedelta64(1, 's')
            utc_datetimes = np.array([datetime.utcfromtimestamp(s) for s in secs])
            self.datetime = utc_datetimes - timedelta(hours = zone)
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

        def init_newio(filenames):
            # New SCHISM >= 5.10 scribed output

            def dataset_filenames(filenames, dataset_filename):
                return [f.replace('*', dataset_filename) for f in filenames]

            self.ds_2d = xr.open_mfdataset(dataset_filenames(filenames, 'out2d'))
            self.start_datetime = schism_start_datetime(self.ds_2d)
            self.datetime = self.start_datetime + np.array([timedelta(seconds = s)
                                                            for s in self.ds_2d['time'].values])
            self.end_datetime = self.datetime[-1]
            self.num_times = len(self.datetime)

            self.elevation = self.ds_2d['elevation']
            self.depthAverageVelX = self.ds_2d['depthAverageVelX']
            self.depthAverageVelY = self.ds_2d['depthAverageVelY']

            self.ds_z = xr.open_mfdataset(dataset_filenames(filenames, 'zCoordinates'))
            self.zCoordinates = self.ds_z['zCoordinates']
            self.ds_t = xr.open_mfdataset(dataset_filenames(filenames, 'temperature'))
            self.temperature = self.ds_t['temperature']
            self.ds_s = xr.open_mfdataset(dataset_filenames(filenames, 'salinity'))
            self.salinity = self.ds_s['salinity']
            self.ds_vx = xr.open_mfdataset(dataset_filenames(filenames, 'horizontalVelX'))
            self.horizontalVelX = self.ds_vx['horizontalVelX']
            self.ds_vy = xr.open_mfdataset(dataset_filenames(filenames, 'horizontalVelY'))
            self.horizontalVelY = self.ds_vy['horizontalVelY']

        self.oldio = oldio
        if self.oldio:
            init_oldio(filenames)
        else:
            init_newio(filenames)

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
