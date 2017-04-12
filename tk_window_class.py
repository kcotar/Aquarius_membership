import numpy as np
import uncertainties.unumpy as unumpy
import matplotlib
matplotlib.use('TkAgg')

from velocity_transformations import *
from find_streams_plots import *
from find_streams_analysis import *
from find_streams_selection import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import sys
if sys.version_info[0] < 3:
    import Tkinter as tk
else:
    import tkinter as tk


class TkWindow:
    def __init__(self, w=800, h=500, title=''):
        # stream parameters
        self.ra_stream = 0.
        self.dec_stream = 0.
        self.rv_stream = 25.
        self.std_pm = 3.
        self.std_rv = 1.
        self.std_control_visible = False

        # window
        self.main_window = tk.Tk()
        self.main_window.title(title)
        canvas = tk.Canvas(self.main_window, width=w, height=h)
        canvas.pack()

        # create first input rows
        x_off = 5
        y_line_w = 30
        l_title = tk.Label(self.main_window, text='Initial input stream parameters:')
        l_title.place(x=x_off, y=x_off)
        # RA
        l_ra = tk.Label(self.main_window, text='RA: ')
        l_ra.place(x=x_off, y=y_line_w+x_off)
        self.ra_entry = tk.Entry(self.main_window, width=8)
        self.ra_entry.place(x=x_off+30, y=y_line_w+x_off)
        ra_p = tk.Button(self.main_window, text='+', command=lambda: self.ra_set(sign='+'))
        ra_p.place(x=x_off+100, y=y_line_w+x_off, width=30, height=20)
        ra_m = tk.Button(self.main_window, text='-', command=lambda: self.ra_set(sign='-'))
        ra_m.place(x=x_off+130, y=y_line_w + x_off, width=30, height=20)
        # DEC
        l_dec = tk.Label(self.main_window, text='DEC: ')
        l_dec.place(x=x_off, y=2*y_line_w + x_off)
        self.dec_entry = tk.Entry(self.main_window, width=8)
        self.dec_entry.place(x=x_off + 30, y=2*y_line_w + x_off)
        dec_p = tk.Button(self.main_window, text='+', command=lambda: self.dec_set(sign='+'))
        dec_p.place(x=x_off + 100, y=2*y_line_w + x_off, width=30, height=20)
        dec_m = tk.Button(self.main_window, text='-', command=lambda: self.dec_set(sign='-'))
        dec_m.place(x=x_off + 130, y=2*y_line_w + x_off, width=30, height=20)
        # RV
        l_rv = tk.Label(self.main_window, text='RV: ')
        l_rv.place(x=x_off, y=3 * y_line_w + x_off)
        self.rv_entry = tk.Entry(self.main_window, width=8)
        self.rv_entry.place(x=x_off + 30, y=3 * y_line_w + x_off)
        rv_p = tk.Button(self.main_window, text='+', command=lambda: self.rv_set(sign='+'))
        rv_p.place(x=x_off + 100, y=3 * y_line_w + x_off, width=30, height=20)
        rv_m = tk.Button(self.main_window, text='-', command=lambda: self.rv_set(sign='-'))
        rv_m.place(x=x_off + 130, y=3 * y_line_w + x_off, width=30, height=20)
        # show values
        self.update_values()
        # large plot button
        plot_b = tk.Button(self.main_window, text='Begin stream\n\nanalysis', command=lambda: self.analysis_first_step())
        plot_b.place(x=200, y=35, width=120, height=80)

    def ra_set(self, sign='+'):
        if sign is '+':
            self.ra_stream += 5.
        elif sign is '-':
            self.ra_stream -= 5.
        self.update_values()

    def dec_set(self, sign='+'):
        if sign is '+':
            self.dec_stream += 5.
        elif sign is '-':
            self.dec_stream -= 5.
        self.update_values()

    def rv_set(self, sign='+'):
        if sign is '+':
            self.rv_stream += 5.
        elif sign is '-':
            self.rv_stream -= 5.
        self.update_values()

    def pm_std_set(self, sign='+'):
        if sign is '+':
            self.std_pm += 0.5
        elif sign is '-':
            self.std_pm -= 0.5
        self.update_values_std()

    def rv_std_set(self, sign='+'):
        if sign is '+':
            self.std_rv += 0.5
        elif sign is '-':
            self.std_rv -= 0.5
        self.update_values_std()

    def update_values(self):
        self.ra_entry.delete(0, tk.END)
        self.ra_entry.insert(0, str(self.ra_stream))
        self.dec_entry.delete(0, tk.END)
        self.dec_entry.insert(0, str(self.dec_stream))
        self.rv_entry.delete(0, tk.END)
        self.rv_entry.insert(0, str(self.rv_stream))

    def update_values_std(self, select_data=False):
        self.std_pm_entry.delete(0, tk.END)
        self.std_pm_entry.insert(0, str(self.std_pm))
        self.std_rv_entry.delete(0, tk.END)
        self.std_rv_entry.insert(0, str(self.std_rv))
        if select_data:
            self.analysis_first_step_selection()

    def analysis_first_step(self):
        # add selection control buttons:
        if not self.std_control_visible:
            x_off = 350
            y_line_w = 30
            l_title = tk.Label(self.main_window, text='Selection parameters:')
            l_title.place(x=x_off, y=5)
            # pm std buttons
            l_pm = tk.Label(self.main_window, text='PM std: ')
            l_pm.place(x=x_off, y=1 * y_line_w + 5)
            self.std_pm_entry = tk.Entry(self.main_window, width=8)
            self.std_pm_entry.place(x=x_off + 50, y=1 * y_line_w + 5)
            pm_p = tk.Button(self.main_window, text='+', command=lambda: self.pm_std_set(sign='+'))
            pm_p.place(x=x_off + 120, y=1 * y_line_w + 5, width=30, height=20)
            pm_m = tk.Button(self.main_window, text='-', command=lambda: self.pm_std_set(sign='-'))
            pm_m.place(x=x_off + 150, y=1 * y_line_w + 5, width=30, height=20)
            # rv std buttons
            l_rv = tk.Label(self.main_window, text='RV std: ')
            l_rv.place(x=x_off, y=2 * y_line_w + 5)
            self.std_rv_entry = tk.Entry(self.main_window, width=8)
            self.std_rv_entry.place(x=x_off + 50, y=2 * y_line_w + 5)
            rv_p = tk.Button(self.main_window, text='+', command=lambda: self.rv_std_set(sign='+'))
            rv_p.place(x=x_off + 120, y=2 * y_line_w + 5, width=30, height=20)
            rv_m = tk.Button(self.main_window, text='-', command=lambda: self.rv_std_set(sign='-'))
            rv_m.place(x=x_off + 150, y=2 * y_line_w + 5, width=30, height=20)
            self.std_control_visible = True
            self.update_values_std(select_data=False)
        # perform selection based on selected values

        # velocity vector of stream in xyz equatorial coordinate system with Earth in the center of it
        self.v_xyz_stream = compute_xyz_vel(np.deg2rad(self.ra_stream), np.deg2rad(self.dec_stream), self.rv_stream)

        # compute predicted stream pmra and pmdec, based on stars ra, dec and parsec distance
        self.rv_stream_predicted = compute_rv(np.deg2rad(self.input_data['ra_gaia']), np.deg2rad(self.input_data['dec_gaia']),
                                              self.v_xyz_stream)

        self.pmra_stream_predicted_u = compute_pmra(np.deg2rad(self.input_data['ra_gaia']), np.deg2rad(self.input_data['dec_gaia']),
                                                    self.parsec_u, self.v_xyz_stream)
        self.pmdec_stream_predicted_u = compute_pmdec(np.deg2rad(self.input_data['ra_gaia']), np.deg2rad(self.input_data['dec_gaia']),
                                                      self.parsec_u, self.v_xyz_stream)
        self.analysis_first_step_selection()

    def analysis_first_step_selection(self):
        # option 1 - match proper motion values in the same sense as described in the Gaia open clusters paper
        idx_match = np.logical_and(match_proper_motion_values(self.pmra_stream_predicted_u, self.pmra_u, dispersion=0.,
                                                              sigma=self.std_pm, prob_thr=None),
                                   match_proper_motion_values(self.pmdec_stream_predicted_u, self.pmdec_u, dispersion=0.,
                                                              sigma=self.std_pm, prob_thr=None))

        # selection based on RV observation
        idx_rv_match = match_rv_values(self.rv_stream_predicted, self.rv_u, sigma=self.std_rv, prob_thr=None)

        # first final selection
        self.idx_possible = np.logical_and(idx_match, idx_rv_match)

    def analysis_first_step_selection_plot(self):
        self.s=1

    def add_data(self, data):
        self.input_data = data
        # define parallax values with uncertainties using uncertainties library
        self.parallax_u = unumpy.uarray(self.input_data['parallax'], self.input_data['parallax_error'])
        self.parsec_u = 1e3 / self.parallax_u
        # define other parameters with uncertainties
        self.pmra_u = unumpy.uarray(self.input_data['pmra'], self.input_data['pmra_error'])
        self.pmdec_u = unumpy.uarray(self.input_data['pmdec'], self.input_data['pmdec_error'])
        self.rv_u = unumpy.uarray(self.input_data['RV'], self.input_data['RV_error'])

    def start(self):
        tk.mainloop()
