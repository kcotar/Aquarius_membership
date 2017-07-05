import time
import numpy as np
import matplotlib
matplotlib.use('Agg')

from velocity_transformations import *
# from find_streams_plots import *
from find_streams_analysis import *
from find_streams_selection import *
from find_streams_analysis_functions import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.backend_bases import MouseEvent
from matplotlib.widgets import LassoSelector, Lasso
from matplotlib.path import Path

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
    def __init__(self, w=1750, h=1000, title=''):
        # stream parameters
        self.ra_stream = 90.
        self.dec_stream = 0.
        self.rv_stream = 45.

        # first step analysis variables
        self.parallax_MC_n = 500
        self.parallax_MC = None
        self.pmra_MC = None
        self.pmdec_MC = None
        self.std_pm = -3.
        self.std_rv = 1.5
        self.set_mc_match_percent = 50.
        self.std_control_visible = False

        # second step analysis variables
        self.xyz_mc = 0
        self.xyz_control_visible = False

        # third step analysis variables
        self.density_control_visible = False
        self.den_w = 30.
        self.dbscan_samp = 10.
        self.dbscan_eps = 10.

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
        self.canvas_density = None
        self.canvas_density_selected = None

        # toolbar constainers
        self.toolbar_pmpos = None
        self.toolbar_inter = None
        self.toolbar_density = None

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
        # method selector
        l_method = tk.Label(self.main_window, text='Method: ')
        l_method.place(x=x_off, y=4 * y_line_w + x_off)
        self.method_entry = tk.StringVar(self.main_window)
        self.method_entry.set("1")  # initial value
        method_dd = tk.OptionMenu(self.main_window, self.method_entry, "1", "2")
        method_dd.place(x=x_off + 60, y=4 * y_line_w + x_off, width=100, height=20)

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

    def update_values_density(self, reanalyse=False):
        self.den_w_entry.delete(0, tk.END)
        self.den_w_entry.insert(0, str(self.den_w))
        if reanalyse:
            self.analysis_third_step_proceed(density=True)

    def update_values_dbscan(self, reanalyse=False):
        self.samp_w_entry.delete(0, tk.END)
        self.samp_w_entry.insert(0, str(self.dbscan_samp))
        self.esp_w_entry.delete(0, tk.END)
        self.esp_w_entry.insert(0, str(self.dbscan_eps))
        if reanalyse:
            self.analysis_third_step_proceed(dbscan=True)

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
            self.parallax_MC = MC_values(self.input_data['parallax'], self.input_data['parallax_error'], self.parallax_MC_n)
            self.pmra_MC = MC_values(self.input_data['pmra'], self.input_data['pmra_error'], self.parallax_MC_n)
            self.pmdec_MC = MC_values(self.input_data['pmdec'], self.input_data['pmdec_error'], self.parallax_MC_n)
        elif len(self.parallax_MC[0]) != self.parallax_MC_n:
            # compute parallax distribution in the case that requested number of MC  samples has changed
            print 'Computing input parallax distribution as number of samples has changed.'
            self.parallax_MC = MC_values(self.input_data['parallax'], self.input_data['parallax_error'], self.parallax_MC_n)
            self.pmra_MC = MC_values(self.input_data['pmra'], self.input_data['pmra_error'], self.parallax_MC_n)
            self.pmdec_MC = MC_values(self.input_data['pmdec'], self.input_data['pmdec_error'], self.parallax_MC_n)

        # start = time.time()
        if self.method_entry.get() is '1':
            # METHOD 1 - match proper motion values in the same sense as described in the Gaia open clusters paper
            idx_pm_match = observations_match_mc(self.input_data['ra_gaia', 'dec_gaia', 'pmra', 'pmra_error', 'pmdec', 'pmdec_error'],
                                                 self.v_xyz_stream, parallax_mc=self.parallax_MC, std=self.std_pm, percent=self.set_mc_match_percent)
        elif self.method_entry.get() is '2':
            # METHOD 2
            idx_pm_match = observations_match_mc(self.input_data['ra_gaia', 'dec_gaia', 'parallax', 'parallax_error'], self.v_xyz_stream,
                                                 pmra_mc=self.pmra_MC, pmdec_mc=self.pmdec_MC, std=self.std_pm, percent=self.set_mc_match_percent)

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
        if self.toolbar_pmpos is not None:
            self.toolbar_pmpos.destroy()

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
        pm_fig.set_size_inches(5.6, 4, forward=True)
        # add plot to canvas
        self.canvas_pm = FigureCanvasTkAgg(pm_fig, master=self.main_window)
        self.toolbar_pmpos = tk.Frame()
        self.toolbar_pmpos.place(x=10, y=140)
        toolbar = NavigationToolbar2TkAgg(self.canvas_pm, self.toolbar_pmpos)
        toolbar.draw()
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
        rv_fig.set_size_inches(5.6, 4, forward=True)
        # add plot to canvas
        self.canvas_rv = FigureCanvasTkAgg(rv_fig, master=self.main_window)
        self.canvas_rv._tkcanvas.place(x=590, y=140)
        self.canvas_rv.draw()

    def analysis_second_step(self):
        # add selection control buttons:
        if not self.xyz_control_visible:
            # create third input column
            x_off = 650
            y_line_w = 30
            l_title = tk.Label(self.main_window, text='XYZ plane parameters:')
            l_title.place(x=x_off, y=5)
            # mc buttons
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
        self.stream_obj = STREAM(self.input_data[self.idx_possible_1], radiant=[self.ra_stream, self.dec_stream])
        # produce visualization or the second step
        self.analysis_second_step_plots()

    def analysis_second_step_plots(self):
        if self.canvas_vel3d is not None:
            self.canvas_vel3d.get_tk_widget().destroy()
            self.canvas_vel3d.draw()
        if self.canvas_velxyz is not None:
            self.canvas_velxyz.get_tk_widget().destroy()
            self.canvas_velxyz.draw()
        if self.toolbar_inter is not None:
            self.toolbar_inter.destroy()

        if self.xyz_mc > 0:
            self.stream_obj.monte_carlo_simulation(samples=self.xyz_mc, distribution='normal')
            self.mc_plot = True
        else:
            self.mc_plot = False

        # add its graphs to the tk gui
        # vel3d_fig = self.stream_obj.estimate_stream_dimensions(path=None, MC=self.mc_plot, GUI=True)
        vel3d_fig = self.stream_obj.plot_intersections(xyz_vel_stream=self.v_xyz_stream, path=None, MC=self.mc_plot, GUI=True)
        vel3d_fig.set_size_inches(5.6, 4, forward=True)
        self.canvas_vel3d = FigureCanvasTkAgg(vel3d_fig, master=self.main_window)
        self.toolbar_inter = tk.Frame()
        self.toolbar_inter.place(x=10, y=560)
        toolbartemp = NavigationToolbar2TkAgg(self.canvas_vel3d, self.toolbar_inter)
        toolbartemp.draw()
        self.canvas_vel3d._tkcanvas.place(x=10, y=560)
        self.canvas_vel3d.draw()
        velxyz_fig = self.stream_obj.plot_velocities(xyz=True, xyz_stream=self.v_xyz_stream, path=None, MC=self.mc_plot, GUI=True)
        velxyz_fig.set_size_inches(5.6, 4, forward=True)
        self.canvas_velxyz = FigureCanvasTkAgg(velxyz_fig, master=self.main_window)
        self.canvas_velxyz._tkcanvas.place(x=590, y=560)
        self.canvas_velxyz.draw()
        # go to te next analysis step
        self.analysis_third_step()

    def get_density_width_estimation(self):
        self.den_w = self.stream_obj.estimate_kernel_bandwidth_cv(MC=self.mc_plot, kernel=self.den_kernel_entry.get())
        self.update_values_density(reanalyse=False)

    def analysis_third_step(self):
        # add selection control buttons:
        if not self.density_control_visible:
            # create third input column
            x_off = 950
            y_line_w = 30
            l_title = tk.Label(self.main_window, text='Intersections density:')
            l_title.place(x=x_off, y=5)
            # pm std buttons
            l_wdth = tk.Label(self.main_window, text='Width : ')
            l_wdth.place(x=x_off, y=1 * y_line_w + 5)
            self.den_w_entry = tk.Entry(self.main_window, width=8)
            self.den_w_entry.place(x=x_off + 50, y=1 * y_line_w + 5)
            w_p = tk.Button(self.main_window, text='+', command=lambda: self.den_w_set(sign='+'))
            w_p.place(x=x_off + 120, y=1 * y_line_w + 5, width=30, height=20)
            w_m = tk.Button(self.main_window, text='-', command=lambda: self.den_w_set(sign='-'))
            w_m.place(x=x_off + 150, y=1 * y_line_w + 5, width=30, height=20)
            # automatic width estimation
            w_auto = tk.Button(self.main_window, text='Automatic width estimation',
                               command=lambda: self.get_density_width_estimation())
            w_auto.place(x=x_off, y=3 * y_line_w + 5, width=200, height=20)
            # method selector
            l_method = tk.Label(self.main_window, text='Kernel: ')
            l_method.place(x=x_off, y=2 * y_line_w + 5)
            self.den_kernel_entry = tk.StringVar(self.main_window)
            self.den_kernel_entry.set("cosine")  # initial value
            method_dd = tk.OptionMenu(self.main_window, self.den_kernel_entry,
                                      "gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine")
            method_dd.place(x=x_off + 60, y=2 * y_line_w + 5, width=130, height=20)
            # visualize results button
            w_show = tk.Button(self.main_window, text='Show density plot',
                               command=lambda: self.analysis_third_step_proceed(density=True))
            w_show.place(x=x_off, y=4 * y_line_w + 5, width=200, height=20)
            # set/update values
            self.update_values_density(reanalyse=False)

            # DBSCAN settings
            x_off = 1250
            l_title = tk.Label(self.main_window, text='Intersections DBSCAN:')
            l_title.place(x=x_off, y=5)
            # dbscan buttons
            l_smp = tk.Label(self.main_window, text='Samples : ')
            l_smp.place(x=x_off, y=1 * y_line_w + 5)
            self.samp_w_entry = tk.Entry(self.main_window, width=8)
            self.samp_w_entry.place(x=x_off + 50, y=1 * y_line_w + 5)
            w_p = tk.Button(self.main_window, text='+', command=lambda: self.dbscan_samp_set(sign='+'))
            w_p.place(x=x_off + 120, y=1 * y_line_w + 5, width=30, height=20)
            w_m = tk.Button(self.main_window, text='-', command=lambda: self.dbscan_samp_set(sign='-'))
            w_m.place(x=x_off + 150, y=1 * y_line_w + 5, width=30, height=20)
            l_esp = tk.Label(self.main_window, text='Esp : ')
            l_esp.place(x=x_off, y=2 * y_line_w + 5)
            self.esp_w_entry = tk.Entry(self.main_window, width=8)
            self.esp_w_entry.place(x=x_off + 50, y=2 * y_line_w + 5)
            w_p = tk.Button(self.main_window, text='+', command=lambda: self.dbscan_eps_set(sign='+'))
            w_p.place(x=x_off + 120, y=2 * y_line_w + 5, width=30, height=20)
            w_m = tk.Button(self.main_window, text='-', command=lambda: self.dbscan_eps_set(sign='-'))
            w_m.place(x=x_off + 150, y=2 * y_line_w + 5, width=30, height=20)
            self.update_values_dbscan(reanalyse=False)
            # visualize results button
            w_show = tk.Button(self.main_window, text='Show DBSCAN plot',
                               command=lambda: self.analysis_third_step_proceed(dbscan=True))
            w_show.place(x=x_off, y=3 * y_line_w + 5, width=200, height=20)

            self.density_control_visible = True

    def analysis_third_step_proceed(self, dbscan=False, density=False):
        # handle plots
        if self.canvas_density is not None:
            self.canvas_density.get_tk_widget().destroy()
            self.canvas_density.draw()
        if self.canvas_density_selected is not None:
            self.canvas_density_selected.get_tk_widget().destroy()
            self.canvas_density_selected.draw()
        if self.toolbar_density is not None:
            self.toolbar_density.destroy()

        if density:
            # acquire density image from stream object
            xyz_grid_range = 750
            xyz_grid_bins = 2000
            density_fig = self.stream_obj.show_density_field(bandwidth=self.den_w,
                                                             kernel=self.den_kernel_entry.get(),
                                                             MC=False, GUI=True, peaks=True,
                                                             # MC should always be disabled for this
                                                             grid_size=xyz_grid_range, grid_bins=xyz_grid_bins,
                                                             recompute=True)

            def callback(event):
                x_dash, y_dash = self.stream_obj.get_nearest_density_peak(x_img=event.xdata, y_img=event.ydata)
                print "Selected peak at X':{0} Y':{1}".format(x_dash, y_dash)
                # handle plots
                if self.canvas_density_selected is not None:
                    self.canvas_density_selected.get_tk_widget().destroy()
                    self.canvas_density_selected.draw()
                selected_density_fig = self.stream_obj.show_density_selection(x_img=event.xdata, y_img=event.ydata, xyz_stream=self.v_xyz_stream,
                                                                              MC=self.mc_plot, GUI=True)
                self.canvas_density_selected = FigureCanvasTkAgg(selected_density_fig, master=self.main_window)
                self.canvas_density_selected._tkcanvas.place(x=1170, y=560)
                self.canvas_density_selected.draw()

            # add its graphs to the tk gui
            self.canvas_density = FigureCanvasTkAgg(density_fig, master=self.main_window)
            self.canvas_density._tkcanvas.place(x=1170, y=140)
            self.canvas_density.mpl_connect('button_press_event', callback)
            self.canvas_density.draw()
            print ' Density plotted'
        elif dbscan:
            density_fig = self.stream_obj.show_dbscan_field(samples=self.dbscan_samp, eps=self.dbscan_eps,
                                                            GUI=True, peaks=True)
            # add its graphs to the tk gui
            self.canvas_density = FigureCanvasTkAgg(density_fig, master=self.main_window)
            self.toolbar_density = tk.Frame()
            self.toolbar_density.place(x=1170, y=140)
            toolbar = NavigationToolbar2TkAgg(self.canvas_density, self.toolbar_density)
            toolbar.draw()
            self.canvas_density._tkcanvas.place(x=1170, y=140)
            self.canvas_density.draw()
            self.stream_obj.evaluate_dbscan_field(MC=self.mc_plot)

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

    def den_w_set(self, sign='+'):
        if sign is '+':
            self.den_w += 10
        elif sign is '-':
            self.den_w -= 10
        self.update_values_density(reanalyse=False)

    def dbscan_samp_set(self, sign='+'):
        if sign is '+':
            self.dbscan_samp += 5
        elif sign is '-':
            self.dbscan_samp -= 5
        self.update_values_dbscan(reanalyse=False)

    def dbscan_eps_set(self, sign='+'):
        if sign is '+':
            self.dbscan_eps += 2
        elif sign is '-':
            self.dbscan_eps -= 2
        self.update_values_dbscan(reanalyse=False)

    def add_data(self, data):
        self.input_data = data

    def start(self):
        tk.mainloop()
