import numpy as np

from astropy.table import Table, unique, Column
from tk_window_class import *

# read GALAH and RAVE data - used for radial velocity data
print 'Reading data sets'
out_file_fits = 'RAVE_GALAH_TGAS_stack.fits'
tgas_data = Table.read(out_file_fits)

# perform some data cleaning and housekeeping
idx_ok = tgas_data['parallax'] > 0  # remove negative parallaxes - objects far away or problems in data reduction
idx_ok = np.logical_and(idx_ok,
                        np.isfinite(tgas_data['ra_gaia','dec_gaia','pmra','pmdec','RV','parallax'].to_pandas().values).all(axis=1))
# remove observations with large RV uncertainties
idx_ok = np.logical_and(idx_ok,
                        tgas_data['RV_error'] < 25.)
print 'Number of removed observations: '+str(len(tgas_data)-np.sum(idx_ok))
tgas_data = tgas_data[idx_ok]
print 'Number of valid observations: '+str(len(tgas_data))

# remove problems with masks
tgas_data = tgas_data.filled()
# remove duplicates
tgas_data = unique(tgas_data, keys=['ra_gaia', 'dec_gaia'], keep='first')
# convert parallax to parsec distance
tgas_data.add_column(Column(1e3/tgas_data['parallax'].data, name='parsec'))
# add systematic error to the parallax uncertainties as suggested for the TGAS dataset
tgas_data['parallax_error'] = np.sqrt(tgas_data['parallax_error'].data**2 + 0.3**2)
# limit data by parsec
tgas_data = tgas_data[np.logical_and(tgas_data['parsec'] < 250, tgas_data['parsec'] < 250)]
print 'Number of points after distance limits: ' + str(len(tgas_data))


window = TkWindow()
window.add_data(tgas_data)
window.start()



# f = Figure(figsize=(5, 4), dpi=100)
# a = f.add_subplot(111)
# t = arange(0.0, 3.0, 0.01)
# s = sin(2*pi*t)
#
# a.plot(t, s)
# a.set_xlabel('X axis label')
# a.set_ylabel('Y label')
#
#
# # a tk.DrawingArea
# canvas = FigureCanvasTkAgg(f, master=main_window)
# canvas.show()
# canvas._tkcanvas.place(x=10, y=20)
#
# button = Tk.Button(master=main_window, text='Quit', command=sys.exit)
# button.place(x=10, y=350)


