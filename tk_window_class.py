import time
import numpy as np
import matplotlib
matplotlib.use('Agg')

from velocity_transformations import *
# from find_streams_plots import *
from find_streams_analysis import *
from find_streams_selection import *
from find_streams_analysis_functions import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import sys
if sys.version_info[0] < 3:
    import Tkinter as tk
else:
    import tkinter as tk

# --------------------------------------------------------
# ---------------- Constants -----------------------------
# --------------------------------------------------------
QUIVER_SCALE = 200.
QUIVER_WIDTH = 0.001


class TkWindow:
    def __init__(self, w=1350, h=1130, title=''):
        # stream parameters
        self.ra_stream = 90.
        self.dec_stream = 5.
        self.rv_stream = 45.

        # first step analysis variables
        self.parallax_MC_n = 100
        self.parallax_MC = None
        self.std_pm = 3.
        self.std_rv = 2.
        self.set_mc_match_percent = 50.
        self.std_control_visible = False

        # second step analysis variables
        self.xyz_mc = 0
        self.xyz_control_visible = False

        # window
        self.main_window = tk.Tk()
        self.main_window.title(title)
        canvas = tk.Canvas(self.main_window, width=w, height=h)
        canvas.pack()

        # figures containers
        self.canvas_pm = None
        self.canvas_rv = None
        self.canvas_vel3d = None
        self.canvas_velxyz = None

        # create first input column
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
        self.mc_entry.delete(0, tk.END)
        self.mc_entry.insert(0, str(self.parallax_MC_n))
        if select_data:
            self.analysis_first_step_selection()

    def update_values_xyz(self, reanalyse=False):
        self.xyz_mc_entry.delete(0, tk.END)
        self.xyz_mc_entry.insert(0, str(self.xyz_mc))
        if reanalyse:
            self.analysis_second_step_plots()

    def analysis_first_step(self):
        # add selection control buttons:
        if not self.std_control_visible:
            # create second input column
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
            # MC button buttons
            l_rv = tk.Label(self.main_window, text='MC : ')
            l_rv.place(x=x_off, y=3 * y_line_w + 5)
            self.mc_entry = tk.Entry(self.main_window, width=8)
            self.mc_entry.place(x=x_off + 50, y=3 * y_line_w + 5)
            mc_p = tk.Button(self.main_window, text='+', command=lambda: self.mc_set(sign='+'))
            mc_p.place(x=x_off + 120, y=3 * y_line_w + 5, width=30, height=20)
            mc_m = tk.Button(self.main_window, text='-', command=lambda: self.mc_set(sign='-'))
            mc_m.place(x=x_off + 150, y=3 * y_line_w + 5, width=30, height=20)
            self.std_control_visible = True
            self.update_values_std(select_data=False)

        # velocity vector of stream in xyz equatorial coordinate system with Earth in the center of it
        self.v_xyz_stream = compute_xyz_vel(np.deg2rad(self.ra_stream), np.deg2rad(self.dec_stream), self.rv_stream)

        # # compute predicted stream pmra and pmdec, based on stars ra, dec and parsec distance
        self.rv_stream_predicted = compute_rv(np.deg2rad(self.input_data['ra_gaia']), np.deg2rad(self.input_data['dec_gaia']),
                                              self.v_xyz_stream)
        self.pmra_stream_predicted = compute_pmra(np.deg2rad(self.input_data['ra_gaia']), np.deg2rad(self.input_data['dec_gaia']),
                                                  self.input_data['parsec'], self.v_xyz_stream)
        self.pmdec_stream_predicted = compute_pmdec(np.deg2rad(self.input_data['ra_gaia']), np.deg2rad(self.input_data['dec_gaia']),
                                                    self.input_data['parsec'], self.v_xyz_stream)
        # perform first selection step
        self.analysis_first_step_selection()

    def analysis_first_step_selection(self):
        # create MC values of parallaxes
        if self.parallax_MC is None:
            # compute parallax distribution in the case that no MC was not performed yet
            print 'Computing input parallax distribution.'
            self.parallax_MC = MC_parallax(self.input_data['parallax'], self.input_data['parallax_error'], self.parallax_MC_n)
        elif len(self.parallax_MC[0]) != self.parallax_MC_n:
            # compute parallax distribution in the case that requested number of MC  samples has changed
            print 'Computing input parallax distribution as number of samples has changed.'
            self.parallax_MC = MC_parallax(self.input_data['parallax'], self.input_data['parallax_error'], self.parallax_MC_n)

        # start = time.time()
        # # option 1 - match proper motion values in the same sense as described in the Gaia open clusters paper
        idx_pm_match = proper_motion_match_mc(self.input_data['ra_gaia', 'dec_gaia', 'pmra', 'pmra_error', 'pmdec', 'pmdec_error'],
                                              self.parallax_MC, self.v_xyz_stream, std=self.std_pm, percent=self.set_mc_match_percent)
        # print time.time() - start

        # # selection based on RV observation
        idx_rv_match = match_values_within_std(self.input_data['RV'], self.input_data['RV_error'],
                                               self.rv_stream_predicted, std=self.std_rv)

        # first final selection
        self.idx_possible_1 = np.logical_and(idx_pm_match, idx_rv_match)
        # plot selection
        self.analysis_first_step_selection_plot()
        # go to te next analysis step
        self.analysis_second_step()

    def analysis_first_step_selection_plot(self):
        data_possible_1 = self.input_data[self.idx_possible_1]
        if self.canvas_pm is not None:
            self.canvas_pm.get_tk_widget().destroy()
            self.canvas_pm.draw()
        if self.canvas_rv is not None:
            self.canvas_rv.get_tk_widget().destroy()
            self.canvas_rv.draw()

        pm_fig, pm_ax = plt.subplots(1,1)
        pm_ax.set(xlim=(0, 360), ylim=(-90, 90))
        pm_ax.scatter(data_possible_1['ra_gaia'], data_possible_1['dec_gaia'], lw=0, c='black', s=5)
        pm_ax.scatter(self.ra_stream, self.dec_stream, lw=0, s=15, c='black', marker='*')
        pm_ax.quiver(data_possible_1['ra_gaia'], data_possible_1['dec_gaia'], data_possible_1['pmra'], data_possible_1['pmdec'],
                     pivot='tail', scale=QUIVER_SCALE, color='green', width=QUIVER_WIDTH)
        pm_ax.quiver(data_possible_1['ra_gaia'], data_possible_1['dec_gaia'],
                     self.pmra_stream_predicted[self.idx_possible_1], self.pmdec_stream_predicted[self.idx_possible_1],
                     pivot='tail', scale=QUIVER_SCALE, color='red', width=QUIVER_WIDTH)
        pm_fig.tight_layout()
        # add plot to canvas
        self.canvas_pm = FigureCanvasTkAgg(pm_fig, master=self.main_window)
        self.canvas_pm._tkcanvas.place(x=10, y=140)
        self.canvas_pm.draw()

        dec_offset = 0.2
        rv_fig, rv_ax = plt.subplots(1, 1)
        rv_ax.set(xlim=(0, 360), ylim=(-90, 90))
        rv_ax.scatter(data_possible_1['ra_gaia'], data_possible_1['dec_gaia'], lw=0, c='black', s=5)
        rv_ax.scatter(self.ra_stream, self.dec_stream, lw=0, s=15, c='black', marker='*')
        rv_ax.quiver(data_possible_1['ra_gaia'], data_possible_1['dec_gaia'] - dec_offset, data_possible_1['RV'], 0.,
                     pivot='tail', scale=QUIVER_SCALE, color='green', width=QUIVER_WIDTH)
        rv_ax.quiver(data_possible_1['ra_gaia'], data_possible_1['dec_gaia'] + dec_offset, self.rv_stream_predicted[self.idx_possible_1], 0.,
                     pivot='tail', scale=QUIVER_SCALE, color='red', width=QUIVER_WIDTH)
        rv_fig.tight_layout()
        # add plot to canvas
        self.canvas_rv = FigureCanvasTkAgg(rv_fig, master=self.main_window)
        self.canvas_rv._tkcanvas.place(x=670, y=140)
        self.canvas_rv.draw()

    def analysis_second_step(self):
        # add selection control buttons:
        if not self.xyz_control_visible:
            # create third input column
            x_off = 650
            y_line_w = 30
            l_title = tk.Label(self.main_window, text='XYZ plane parameters:')
            l_title.place(x=x_off, y=5)
            # pm std buttons
            l_mc = tk.Label(self.main_window, text='MC : ')
            l_mc.place(x=x_off, y=1 * y_line_w + 5)
            self.xyz_mc_entry = tk.Entry(self.main_window, width=8)
            self.xyz_mc_entry.place(x=x_off + 50, y=1 * y_line_w + 5)
            mc_p = tk.Button(self.main_window, text='+', command=lambda: self.mc_xyz_set(sign='+'))
            mc_p.place(x=x_off + 120, y=1 * y_line_w + 5, width=30, height=20)
            mc_m = tk.Button(self.main_window, text='-', command=lambda: self.mc_xyz_set(sign='-'))
            mc_m.place(x=x_off + 150, y=1 * y_line_w + 5, width=30, height=20)
            self.update_values_xyz(reanalyse=False)
            self.xyz_control_visible = True
        # create a stream object that is further used for the analysis
        self.strea_obj = STREAM(self.input_data[self.idx_possible_1], radiant=[self.ra_stream, self.dec_stream])
        # produce visualization or the second step
        self.analysis_second_step_plots()

    def analysis_second_step_plots(self):
        if self.canvas_vel3d is not None:
            self.canvas_vel3d.get_tk_widget().destroy()
            self.canvas_vel3d.draw()
        if self.canvas_velxyz is not None:
            self.canvas_velxyz.get_tk_widget().destroy()
            self.canvas_velxyz.draw()

        if self.xyz_mc > 0:
            self.strea_obj.monte_carlo_simulation(samples=self.xyz_mc, distribution='normal')
            mc_plot = True
        else:
            mc_plot = False

        # add its graphs to the tk gui
        self.canvas_vel3d = FigureCanvasTkAgg(self.strea_obj.estimate_stream_dimensions(path=None, MC=mc_plot, GUI=True), master=self.main_window)
        self.canvas_vel3d._tkcanvas.place(x=10, y=640)
        self.canvas_vel3d.draw()
        self.canvas_velxyz = FigureCanvasTkAgg(self.strea_obj.plot_velocities(xyz=True, xyz_stream=self.v_xyz_stream, path=None, MC=mc_plot, GUI=True), master=self.main_window)
        self.canvas_velxyz._tkcanvas.place(x=670, y=640)
        self.canvas_velxyz.draw()

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
        self.update_values_std(select_data=True)

    def rv_std_set(self, sign='+'):
        if sign is '+':
            self.std_rv += 0.5
        elif sign is '-':
            self.std_rv -= 0.5
        self.update_values_std(select_data=True)

    def mc_set(self, sign='+'):
        if sign is '+':
            self.parallax_MC_n += 25
        elif sign is '-':
            self.parallax_MC_n -= 25
        self.update_values_std(select_data=True)

    def mc_xyz_set(self, sign='+'):
        if sign is '+':
            self.xyz_mc += 25
        elif sign is '-':
            self.xyz_mc -= 25
        self.update_values_xyz(reanalyse=True)

    def add_data(self, data):
        self.input_data = data

    def start(self):
        tk.mainloop()
