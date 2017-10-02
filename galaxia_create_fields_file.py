import os
from astropy.table import Table, unique
import astropy.coordinates as coord
import astropy.units as un
import numpy as np

simulation_dir = '/home/klemen/GALAH_data/Galaxia_simulation/GALAH/'
fields_original = 'fields_galah.csv'
simulation_fields = fields_original.split('.')[0]+'.txt'

os.chdir(simulation_dir)
# cenra,cendec

fields = Table.read(fields_original)
print len(fields)
fields = fields[np.logical_and(fields['cenra'] != 0, fields['cendec'] != 0)]
print len(fields)
fields = fields.filled()
fields_uniq = unique(fields, keys=('cenra', 'cendec'))
print len(fields_uniq)

# convert to galactic coordinates
coords = coord.ICRS(ra=fields_uniq['cenra'].data*un.deg, dec=fields_uniq['cendec'].data*un.deg).transform_to(coord.Galactic)
f_l = coords.l.value
f_b = coords.b.value

n_fields = len(fields_uniq)
print 'Fields: '+str(n_fields)

txt_out = open(simulation_fields, 'w')
txt_out.write('<head>\n')
txt_out.write(str(n_fields)+' 3 -1 2 0 1\n')
txt_out.write('<head>\n')
for i_f in range(n_fields):
    txt_out.write(str(f_l[i_f])+' '+str(f_b[i_f])+' 0\n')
txt_out.close()
