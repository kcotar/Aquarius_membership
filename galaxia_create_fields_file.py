import os
from astropy.table import Table, unique
import astropy.coordinates as coord
import astropy.units as un
import numpy as np
import matplotlib.pyplot as plt

RAVE = False
GALAH = True

select_percent = 100.  # aka random selection function
select_n = None  # TODO: implement the number of selected objects per field

if select_percent < 100:
    set_seed = np.int32(np.random.rand(1)*1000)
    print 'Seed', set_seed
    np.random.seed(set_seed)

if RAVE:
    simulation_dir = '/home/klemen/GALAH_data_depricated/Galaxia_simulation/RAVE/'
    rave_data = '/home/klemen/RAVE_data/RAVE_DR5.fits'
    galaxia_complete_fits ='galaxy_rave_complete.fits'
    simulation_fields = 'fields_rave.txt'

    r_field = 3.  # radius in degrees

    obs_data = Table.read(rave_data)
    fields_uniq = np.unique(obs_data['FieldName'])
    # skip repited field with longer names
    # 1607m49, 1607m49a, 1607m49b etc
    fields_uniq = [f for f in fields_uniq if len(f)==7]
    field_ra = [float(val[:2])*15.+float(val[2:4])/60.*15. for val in fields_uniq]
    field_dec = [float(val[-2:]) if val[-3]=='p' else -1.*float(val[-2:]) for val in fields_uniq]

    coords = coord.ICRS(ra=field_ra * un.deg, dec=field_dec * un.deg).transform_to(coord.Galactic)
    f_l = coords.l.value
    f_b = coords.b.value

    # # plt.scatter(field_ra, field_dec)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='Mollweide')
    # print f_l
    # print f_b
    # ax.scatter((f_l), (f_b))
    # plt.show()

if GALAH:
    simulation_dir = '/home/klemen/GALAH_data_depricated/Galaxia_simulation/GALAH/'
    fields_original = 'fields_galah.csv'
    galaxia_complete_fits = 'galaxy_galah_complete_mag_10_16.fits'
    simulation_fields = fields_original.split('.')[0]+'.txt'

    r_field = 1.  # radius in degrees

    os.chdir(simulation_dir)
    # cenra,cendec

    fields = Table.read(fields_original)
    print len(fields)
    fields = fields[np.logical_and(fields['cenra'] != 0, fields['cendec'] != 0)]
    print len(fields)
    fields = fields.filled()
    fields = fields[np.abs(fields['cendec']) <= 90]
    fields_uniq = unique(fields, keys=('cenra', 'cendec'))

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

# cut out circles from complete galaxia simulation file
galaxia_data = Table.read(simulation_dir + galaxia_complete_fits)
id_subset = np.ndarray(len(galaxia_data), dtype=bool)
id_subset.fill(False)

galaxia_coords = coord.Galactic(l=galaxia_data['glon']*un.deg, b=galaxia_data['glat']*un.deg)

for i_f in range(n_fields):
    print 'Subset field '+str(i_f)
    field_c_coord = coord.Galactic(l=f_l[i_f]*un.deg, b=f_b[i_f]*un.deg)
    idx_in_field = np.where(galaxia_coords.separation(field_c_coord) < r_field*un.deg)[0]
    if select_percent < 100.:
        n_obj_field = len(idx_in_field)
        n_obj_field_sel = np.int32(n_obj_field*select_percent/100.)
        idx_subset = np.int32(np.random.rand(n_obj_field_sel)*n_obj_field)
        idx_in_field = idx_in_field[idx_subset]
    id_subset[idx_in_field] = True
galaxia_data_subset = galaxia_data[id_subset]

out_suffix = '_fields_r{:.1f}'.format(r_field)
if select_percent < 100:
    out_suffix += '_perc{:.1f}_seed{:.0f}'.format(select_percent, set_seed[0])
galaxia_data_subset.write(simulation_dir + galaxia_complete_fits[:-5] + out_suffix + '.fits')
