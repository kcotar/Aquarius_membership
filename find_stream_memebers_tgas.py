import os, glob
import numpy as np

from astropy.table import Table
from velocity_transformations import *

tgas_data = glob.glob('GaiaTgas/TgasSource_*.fits')

# stream coordinates
rv_stream = 200.
ra_stream = np.deg2rad(164.)  # alpha
de_stream = np.deg2rad(13.)  # delta

# velocity vector of stream
e = copute_xyz_vel(ra_stream, de_stream, 1.)
v = copute_xyz_vel(ra_stream, de_stream, rv_stream)

rv = np.sum(e * v)
print copute_rv(ra_stream, de_stream, v)
