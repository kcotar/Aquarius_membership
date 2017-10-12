import os
from astropy.table import Table, unique
import astropy.coordinates as coord
import astropy.units as un
import numpy as np
import matplotlib.pyplot as plt

RAVE = True
GALAH = False

if RAVE:
    simulation_dir = '/home/klemen/GALAH_data/Galaxia_simulation/RAVE/'
    rave_data = '/home/klemen/RAVE_data/RAVE_DR5.fits'
    simulation_fields = 'fields_rave.txt'

    obs_data = Table.read(rave_data)
    fields_uniq = np.unique(obs_data['FieldName'])
    # skip repited field with longer names
    # 1607m49, 1607m49a, 1607m49b etc
    fields_uniq = [f for f in fields_uniq if len(f)==7]
    field_ra = [float(val[:2])*15.+float(val[2:4])/60.*15. for val in fields_uniq]
    field_dec = [float(val[-2:]) if val[-3]=='p' else -1.*float(val[-2:]) for val in fields_uniq]

    print fields_uniq[-20:]
    print field_ra[-20:]
    print field_dec[-20:]

    print np.min(field_dec), np.max(field_dec)
    print np.min(field_ra), np.max(field_ra)

    coords = coord.ICRS(ra=field_ra * un.deg, dec=field_dec * un.deg).transform_to(coord.Galactic)
    f_l = coords.l.value
    f_b = coords.b.value

    # # plt.scatter(field_ra, field_dec)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # print f_l
    # print f_b
    # ax.scatter((f_l), (f_b))
    # plt.show()

if GALAH:
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

# create output file with coordinates of field centers
n_fields = len(f_l)
print 'Fields: '+str(n_fields)

txt_out = open(simulation_fields, 'w')
txt_out.write('<head>\n')
txt_out.write(str(n_fields)+' 3 -1 2 0 1\n')
txt_out.write('<head>\n')
for i_f in range(n_fields):
    txt_out.write(str(f_l[i_f])+' '+str(f_b[i_f])+' 0\n')
txt_out.close()
