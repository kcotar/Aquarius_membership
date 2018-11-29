import imp
import astropy.units as un
import astropy.coordinates as coord
import matplotlib.pyplot as plt
import numpy as np

from galpy.orbit import Orbit
from galpy.potential import MWPotential2014
from astropy.table import Table, vstack, Column
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from skimage.feature import peak_local_max
from scipy.signal import argrelextrema, savgol_filter
from scipy.interpolate import splev, splrep
# from scipy.ndimage import watershed_ift
# from skimage.morphology import watershed
from lmfit import minimize, Parameters, report_fit
from vector_plane_calculations import *
import gala.coordinates as gal_coord

imp.load_source('veltrans', '../tSNE_test/velocity_transform.py')
from veltrans import *


class STREAM:
    def __init__(self, data, radiant=None, to_galactic=True):
        """

        :param data:
        :param radiant:
        """
        self.input_data = data
        # add unique id to input rows
        self.input_data.add_column(Column(data=['u_'+str(i_d) for i_d in range(len(data))], name='id_uniq', dtype='S32'))
        self.radiant = radiant
        # transform coordinates in cartesian coordinate system
        self.to_galactic = to_galactic
        input_coord = coord.SkyCoord(ra=self.input_data['ra'] * un.deg,
                                     dec=self.input_data['dec'] * un.deg,
                                     distance=self.input_data['parsec'] * un.pc)
        if self.to_galactic:
            # compute pml and pmb
            ra_dec_pm = np.vstack((self.input_data['pmra'], self.input_data['pmdec'])) * un.mas/un.yr
            l_b_pm = gal_coord.pm_icrs_to_gal(coord.ICRS(ra=self.input_data['ra'] * un.deg,
                                                         dec=self.input_data['dec'] * un.deg), ra_dec_pm)
            self.input_data['pml'] = l_b_pm[0].value
            self.input_data['pmb'] = l_b_pm[1].value
            # unset values
            ra_dec_pm = None
            l_b_pm = None
            # compute galactic coordinates/positions
            input_coord = input_coord.transform_to(coord.Galactic)
            self.input_data['l'] = input_coord.l.value
            self.input_data['b'] = input_coord.b.value

        self.cartesian = input_coord.cartesian
        if self.radiant is not None:
            if self.to_galactic:
                # TODO: temporary fix for galactic coordinates
                self.radiant = None
                self.radiant_cartesian = None
            else:
                self.radiant_cartesian = coord.SkyCoord(ra=radiant[0]*un.deg,
                                                        dec=radiant[1]*un.deg,
                                                        distance=3000).cartesian
        # prepare labels that will be used later on
        if radiant is not None:
            self._rotate_coordinate_system(MC=False)
        else:
            self.cartesian_rotated = None
        self.stream_params = None
        # store galactic and cartesian velocities
        self.uvw_vel = None
        if self.to_galactic:
            xyz_vel = motion_to_cartesic(np.array(self.input_data['l']), np.array(self.input_data['b']),
                                         np.array(self.input_data['pml']), np.array(self.input_data['pmb']),
                                         np.array(self.input_data['rv']), plx=np.array(self.input_data['parallax']))
        else:
            xyz_vel = motion_to_cartesic(np.array(self.input_data['ra']), np.array(self.input_data['dec']),
                                         np.array(self.input_data['pmra']), np.array(self.input_data['pmdec']),
                                         np.array(self.input_data['rv']), plx=np.array(self.input_data['parallax']))
        self.xyz_vel = np.transpose(xyz_vel)
        # store results of monte carlo simulation
        self.xyz_vel_MC = None
        self.data_MC = None
        self.n_samples_MC = None
        self.cartesian_MC = None
        self.cartesian_rotated_MC = None
        # clusters analysis
        self.cluster_labels = None
        self.cluster_ids = None
        self.n_clusters = None
        # density field analysis
        self.density_field = None
        self.density_peaks = None
        self.meaningful_peaks = list([])  # empty list of detected meaningful peaks in the density field

    def _rotate_coordinate_system(self, MC=False):
        """
        Rotate coordinate system about the x/y/z axis so that the rotation axis (radiant axis) lies in one of the
        axises of the new coordinate system.

        :return:
        """
        if self.radiant is None:
            raise ArithmeticError('Cannot rotate coordinate system as radiant is not defined.')
        # compute angles between new and old coordinate system
        radiat_orig = np.array([self.radiant_cartesian.x, self.radiant_cartesian.y, self.radiant_cartesian.z])
        x_rot_ang = np.arctan2(radiat_orig[1], radiat_orig[2])  # angle between  axis and new axis
        # rotation around x axis
        x_rot_matrix = np.array([[1., 0., 0.],
                                 [0., np.cos(x_rot_ang), np.sin(x_rot_ang)],
                                 [0., -np.sin(x_rot_ang), np.cos(x_rot_ang)]])
        # rotate around y axis
        rot_temp = radiat_orig.dot(x_rot_matrix)
        y_rot_ang = -np.arctan2(rot_temp[0], rot_temp[2])
        y_rot_matrix = np.array([[np.cos(y_rot_ang), 0, -np.sin(y_rot_ang)],
                                 [0., 1., 0.],
                                 [np.sin(y_rot_ang), 0, np.cos(y_rot_ang)]])

        # create rotation matrix
        rot_matrix = x_rot_matrix.dot(y_rot_matrix)
        # preform coordinate rotation
        if MC:
            old_coordinates = np.transpose(np.vstack((self.cartesian_MC.x, self.cartesian_MC.y, self.cartesian_MC.z)))
        else:
            old_coordinates = np.transpose(np.vstack((self.cartesian.x, self.cartesian.y, self.cartesian.z)))
        new_coordinates = old_coordinates.value.dot(rot_matrix)
        new_coordinates_cartesian = coord.SkyCoord(x=new_coordinates[:, 0]*un.pc, y=new_coordinates[:, 1]*un.pc, z=new_coordinates[:, 2]*un.pc,
                                                frame='icrs', representation='cartesian').cartesian
        if MC:
            self.cartesian_rotated_MC = new_coordinates_cartesian
        else:
            self.cartesian_rotated = new_coordinates_cartesian

    def _determine_stream_param(self, method='mass'):
        if method is 'mass':
            # compute stream "mass" center
            stream_center_x = np.mean(self.cartesian_rotated.x)
            stream_center_y = np.mean(self.cartesian_rotated.y)
        # elif method is 'minimum':
        #
        stream_radius = np.max(np.sqrt(np.power(self.cartesian_rotated.x - stream_center_x, 2) +
                                       np.power(self.cartesian_rotated.y - stream_center_y, 2)))
        stream_length = np.max(self.cartesian_rotated.z) - np.min(self.cartesian_rotated.z)
        return [stream_center_x, stream_center_y, stream_radius, stream_length]

    def monte_carlo_simulation(self, samples=10, distribution='normal'):
        """

        :param samples:
        :param distribution:
        :return:
        """
        # remove possible results from previous mc simulation
        self.data_MC = None
        self.n_samples_MC = samples
        # create new dataset based on original data considering measurement errors using monte carlo approach
        n_input_rows = len(self.input_data)
        # n_MC_rows = n_input_rows * samples
        cols_MC = ['rv', 'parallax', 'pmra', 'pmdec']
        cols_const = ['id_uniq', 'source_id', 'ra', 'dec']
        n_cols_MC = len(cols_MC)
        # create multiple instances of every row
        print 'Creating random observations from given error values'
        for i_r in range(n_input_rows):
            if i_r % 250 == 0:
                print ' MC on row '+str(i_r+1)+' out of '+str(n_input_rows)+'.'
            temp_table = Table(np.ndarray((samples, len(cols_MC)+len(cols_const))),
                               names=np.hstack((cols_const, cols_MC)).flatten(),
                               dtype=['S32', 'i8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8'])
            data_row = self.input_data[i_r]
            for i_c in range(n_cols_MC):
                col = cols_MC[i_c]
                # fill temp table with randomly generated values
                if data_row[col + '_error'] > 0:
                    # scale value can be set
                    if distribution is 'uniform':
                        temp_table[col] = np.random.uniform(data_row[col] - data_row[col + '_error'],
                                                            data_row[col] + data_row[col + '_error'], samples)
                    elif distribution in 'normal':
                        temp_table[col] = np.random.normal(data_row[col], data_row[col + '_error'], samples)
                else:
                    temp_table[col] = data_row[col]
                # fill temp table with values that are constant for every MC value
            for col in cols_const:
                temp_table[col] = data_row[col]
            # add created values to the final MC database
            if self.data_MC is None:
                self.data_MC = Table(temp_table)
            else:
                self.data_MC = vstack([self.data_MC, temp_table], join_type='inner')
        # TODO: investigate effect of negative parallax limit
        # # limit negative parallax values
        # idx_bad = self.data_MC['parallax'] <= 0
        # n_bad = np.sum(idx_bad)
        # if n_bad > 0:
        #     print ' Removing '+str(n_bad)+' or {:.1f}% of rows with negative parallax values'.format(100.*n_bad/n_MC_rows)
        #     self.data_MC = self.data_MC[np.logical_not(idx_bad)]
        # compote cartesian coordinates od simulated data
        MC_input_coord = coord.SkyCoord(ra=self.data_MC['ra'] * un.deg,
                                        dec=self.data_MC['dec'] * un.deg,
                                        distance=1./self.data_MC['parallax']*1e3 * un.pc)
        if self.to_galactic:
            # compute pml and pmb
            ra_dec_pm = np.vstack((self.data_MC['pmra'], self.data_MC['pmdec'])) * un.mas/un.yr
            l_b_pm = gal_coord.pm_icrs_to_gal(coord.ICRS(ra=self.data_MC['ra'] * un.deg,
                                                         dec=self.data_MC['dec'] * un.deg), ra_dec_pm)
            self.data_MC['pml'] = l_b_pm[0].value
            self.data_MC['pmb'] = l_b_pm[1].value
            # unset values
            ra_dec_pm = None
            l_b_pm = None
            # compute galactic coordinates/positions
            MC_input_coord = MC_input_coord.transform_to(coord.Galactic)
            self.data_MC['l'] = MC_input_coord.l.value
            self.data_MC['b'] = MC_input_coord.b.value

        self.cartesian_MC = MC_input_coord.cartesian

        # compute xyz velocities
        if self.to_galactic:
            xyz_vel = motion_to_cartesic(np.array(self.data_MC['l']), np.array(self.data_MC['b']),
                                         np.array(self.data_MC['pml']), np.array(self.data_MC['pmb']),
                                         np.array(self.data_MC['rv']), plx=np.array(self.data_MC['parallax']))
        else:
            xyz_vel = motion_to_cartesic(np.array(self.data_MC['ra']), np.array(self.data_MC['dec']),
                                         np.array(self.data_MC['pmra']), np.array(self.data_MC['pmdec']),
                                         np.array(self.data_MC['rv']), plx=np.array(self.data_MC['parallax']))
        self.xyz_vel_MC = np.transpose(xyz_vel)

    def stream_show(self, path=None, view_pos=None, MC=False):
        """

        :param path:
        :param view_pos:
        :return:
        """
        if MC:
            plot_dataset = self.cartesian_MC
        else:
            plot_dataset = self.cartesian
        plot_lim = (-2000, 2000)
        fig = plt.subplot(111, projection='3d')
        fig.scatter(0, 0, 0, c='black', marker='*', s=20)
        if MC:
            alpha_use = 0.3
        else:
            alpha_use = 1.
        fig.scatter(plot_dataset.x, plot_dataset.y, plot_dataset.z,
                    c='blue', depthshade=False, alpha=alpha_use, s=20, lw=0)
        # add line that connects point of radiant and anti-radiant
        if self.radiant is not None:
            # compute the cartesian coordinates of both points
            fig.plot([-self.radiant_cartesian.x, self.radiant_cartesian.x],
                     [-self.radiant_cartesian.y, self.radiant_cartesian.y],
                     [-self.radiant_cartesian.z, self.radiant_cartesian.z])
            # compute elevation and azimuth of the line that crosses trough Earth and radiant point
        if view_pos is not None:
            fig.view_init(elev=view_pos[1], azim=view_pos[0])
        fig.set(xlim=plot_lim, ylim=plot_lim, zlim=plot_lim, xlabel='X [pc]', ylabel='Y [pc]', zlabel='Z [pc]')
        if path is None:
            plt.show()
        else:
            plt.savefig(path)
        plt.close()

    def plot_velocities(self, uvw=False, xyz=False, uvw_stream=None, xyz_stream=None, path='vel.png', MC=False, GUI=False):
        plot_range = 10
        if xyz and self.xyz_vel is not None:
            if MC:
                plot_data = self.xyz_vel_MC
            else:
                plot_data = self.xyz_vel
            labels = ['X', 'Y', 'Z']
            stream_center = xyz_stream
        elif uvw and self.uvw_vel is not None:
            plot_data = self.uvw_vel
            labels = ['U', 'V', 'W']
            stream_center = uvw_stream
        # Create a plot
        plot_comb = [[0,1], [2,1], [0,2]]
        plot_pos = [[0,0], [0,1], [1,0]]
        fig, ax = plt.subplots(2,2)
        for i_c in range(len(plot_comb)):
            fig_pos = (plot_pos[i_c][0], plot_pos[i_c][1])
            i_x = plot_comb[i_c][0]
            i_y = plot_comb[i_c][1]
            if MC:
                alpha_use = 0.2
            else:
                alpha_use = 1.
            if stream_center is not None:
                ax[fig_pos].scatter(stream_center[i_x], stream_center[i_y], lw=0, c='black', s=10, marker='*')
            ax[fig_pos].scatter(plot_data[:,i_x], plot_data[:,i_y], lw=0, c='blue', s=2, alpha=alpha_use)
            ax[fig_pos].set(xlabel=labels[i_x], ylabel=labels[i_y],
                            xlim=[stream_center[i_x]-plot_range, stream_center[i_x]+plot_range],
                            ylim=[stream_center[i_y]-plot_range, stream_center[i_y]+plot_range])
        # fig.tight_layout()
        if GUI:
            return fig
        elif path is not None:
            plt.savefig(path, dpi=250)
        else:
            plt.show()
        plt.close()

    def estimate_stream_dimensions(self, path=None, color=None, MC=False, GUI=False):
        """

        :param path:
        :param color:
        :return:
        """
        if MC:
            # first transform coordinate system in that way that radian lies on a z axis
            if self.cartesian_rotated_MC is None:
                self._rotate_coordinate_system(MC=True)
            plot_data = self.cartesian_rotated_MC
        else:
            # first transform coordinate system in that way that radian lies on a z axis
            if self.cartesian_rotated is None:
                self._rotate_coordinate_system()
            plot_data = self.cartesian_rotated
        plot_lim = (-1500, 1500)
        fig, ax = plt.subplots(1, 1)
        # compute stream parameters
        # self.stream_params = self._determine_stream_param(method='mass')
        # plot results
        # plt.scatter(self.stream_params[0], self.stream_params[1], c='black', marker='+', s=15)
        # ax = plt.gca()
        # c1 = plt.Circle((self.stream_params[0].value, self.stream_params[1].value), self.stream_params[2].value, color='0.85', fill=False)
        # ax.add_artist(c1)
        # plot stream in xy plane
        if MC:
            alpha_use = 0.2
        else:
            alpha_use = 1.
        ax.scatter(0, 0, c='black', marker='*', s=15)
        if color is None:
            ax.scatter(plot_data.x, plot_data.y,
                        c='blue', alpha=alpha_use, s=5, lw=0)
        else:
            ax.scatter(plot_data.x, plot_data.y,
                        c=color, s=15, lw=0, cmap='jet')
            ax.colorbar()
        #plt.axis('equal')
        ax.set(xlim=plot_lim, ylim=plot_lim, xlabel='X\' [pc]', ylabel='Y\' [pc]')
        # plt.title('radius {:4.0f}   length {:4.0f}'.format(self.stream_params[2], self.stream_params[3]))
        plt.tight_layout()
        if GUI:
            return fig
        elif path is not None:
            plt.savefig(path, dpi=250)
        else:
            plt.show()
        # also return computed parameters
        plt.close()
        return self.stream_params

    def plot_intersections(self, xyz_vel_stream=None, path='intersect.png', MC=False, GUI=False):
        plot_lim = (-1500, 1500)
        if MC:
            star_vel = self.xyz_vel_MC
            star_pos = self.cartesian_MC
            alpha_use = 0.2
        else:
            star_vel = self.xyz_vel
            star_pos = self.cartesian
            alpha_use = 1.
        if xyz_vel_stream is None:
            xyz_vel_stream = np.nanmedian(star_vel, axis=0)
        # compute plane intersects of given dat
        plane_intersects = stream_plane_vector_intersect(star_pos, star_vel, xyz_vel_stream)
        self.plane_intersects_2d = intersects_to_2dplane(plane_intersects, xyz_vel_stream)
        # Create a plot
        fig, ax = plt.subplots(1, 1)
        ax.scatter(self.plane_intersects_2d[:, 0], self.plane_intersects_2d[:, 1], lw=0, c='blue', s=2, alpha=alpha_use)
        ax.scatter(0, 0, lw=0, c='black', s=10, marker='*')  # solar position
        ax.set(xlabel='X stream plane', ylabel='Y stream plane', xlim=plot_lim, ylim=plot_lim)
        fig.tight_layout()
        if GUI:
            return fig
        elif path is not None:
            plt.savefig(path, dpi=250)
        else:
            plt.show()
        plt.close()

    def find_overdensities(self, path='dbscan.png'):
        """

        :param path:
        :return:
        """
        # first prepare data if they are not available yet
        if self.cartesian_rotated is None:
            self._rotate_coordinate_system()
        m_s = 3
        e_v = 75
        dbscan_data = np.transpose(np.vstack((self.cartesian_rotated.x.value, self.cartesian_rotated.y.value)))
        db_fit = DBSCAN(min_samples=m_s, eps=e_v, algorithm='auto', metric='euclidean', n_jobs=16).fit(dbscan_data)
        self.cluster_labels = db_fit.labels_
        self.cluster_ids = set(self.cluster_labels)
        self.n_clusters = len(self.cluster_ids)
        if path is not None:
            self.estimate_stream_dimensions(path=path, color=self.cluster_labels)

    def analyse_overdensities(self, xyz_stream=None, path_prefix='dbscan'):
        """

        :return:
        """
        if self.cluster_ids is None:
            self.find_overdensities(path=path_prefix+'.png')
        if self.n_clusters <= 1:
            print 'Insufficient number of clusters for further cluster analysis'
        else:
            # further analyse selected groups of stars
            results_xyz_vector = list([])
            for c_id in self.cluster_ids:
                if c_id == -1:
                    # members with cluster id of -1 were not assigned to any of the dbscan determined clusters
                    continue
                idx_members = self.cluster_labels == c_id
                # analyse physical parameters (metalicity) of the selected cluster members
                mh = self.input_data['M_H'][idx_members]
                mh_range = np.nanmax(mh) - np.nanmin(mh)
                mh_std = np.nanstd(mh)
                if mh_std < 0.25:
                    # possible homogeneous cluster with uniform metalicity
                    xyz_vel_median = np.nanmedian(self.xyz_vel[idx_members], axis=0)
                    results_xyz_vector.append(xyz_vel_median)
                    fig, ax = plt.subplots(2, 2)
                    fig.suptitle('Median values X:{:3.2f}  Y:{:3.2f}  Z:{:3.2f}'.format(xyz_vel_median[0], xyz_vel_median[1], xyz_vel_median[2]))
                    ax[0, 0].hist(mh, bins=30, range=[-2, 1])
                    ax[0, 0].set_title('Metalicity')
                    ax[0, 1].hist(self.xyz_vel[:, 0][idx_members], bins=30, range=[xyz_stream[0]-7, xyz_stream[0]+7])
                    ax[0, 1].set_title('X velocity')
                    ax[1, 0].hist(self.xyz_vel[:, 1][idx_members], bins=30, range=[xyz_stream[1]-7, xyz_stream[1]+7])
                    ax[1, 0].set_title('Y velocity')
                    ax[1, 1].hist(self.xyz_vel[:, 2][idx_members], bins=30, range=[xyz_stream[2]-7, xyz_stream[2]+7])
                    ax[1, 1].set_title('Z velocity')
                    plt.savefig(path_prefix+'_{:02.0f}.png'.format(c_id), dpi=250)
                    plt.close()
            return results_xyz_vector

    def _get_density_estimate_data(self, MC=False, intersects=True):
        if intersects:
            # use intersections
            return self.plane_intersects_2d
        else:
            # compute density space of line-of-sight stars as seen from radiant
            if MC:
                stars_pos = self.cartesian_rotated_MC
            else:
                stars_pos = self.cartesian_rotated
            return np.vstack((stars_pos.x, stars_pos.y)).T

    def estimate_kernel_bandwidth_cv(self, MC=False, kernel='gaussian', verbose=False):
        # density bandwidth estimation
        cv_grid = GridSearchCV(KernelDensity(kernel=kernel), {'bandwidth': np.arange(0, 110, 10)}, cv=10)
        cv_grid.fit(self._get_density_estimate_data(MC=MC))
        if verbose:
            print cv_grid.cv_results_
        return cv_grid.best_params_.get('bandwidth')

    def _compute_density_field(self, bandwidth=1., kernel='gaussian', MC=False):
        stars_pos = self._get_density_estimate_data(MC=MC)
        print 'Computing density of stars'
        self.density_bandwith = bandwidth
        self.density_kernel = kernel
        stars_density = KernelDensity(bandwidth=self.density_bandwith, kernel=self.density_kernel).fit(stars_pos)
        grid_pos = np.linspace(-self.grid_density_size, self.grid_density_size, self.grid_density_bins)
        _x, _y = np.meshgrid(grid_pos, grid_pos)
        print 'Computing density field'
        density_field = stars_density.score_samples(np.vstack((_x.ravel(), _y.ravel())).T) + np.log(stars_pos.shape[0])
        self.density_field = np.exp(density_field).reshape(_x.shape) * 1e3

    def _compute_density_field_peaks(self):
        # some quick checks for data availability
        if self.density_field is None:
            # compute density field from given stream data
            self._compute_density_field()
        # start actual computation of peaks
        # _, max = ndimage_extream(self.density_field)
        print 'Computing local density peaks'
        self.density_peaks = peak_local_max(self.density_field, min_distance=int(2.*self.density_bandwith), num_peaks=20,
                                            threshold_abs=1., threshold_rel=None)

    def get_nearest_density_peak(self, x_img=None, y_img=None):
        if x_img is None or y_img is None:
            print 'Cannot compute nearest peak position'
            return None, None
        else:
            idx_peak = np.argmin(np.sqrt((self.density_peaks[:, 1] - x_img)**2 + (self.density_peaks[:, 0] - y_img)**2))
            x_peak = 1. * self.density_peaks[idx_peak, 1] / self.grid_density_bins * (2 * self.grid_density_size) - self.grid_density_size
            y_peak = 1. * self.density_peaks[idx_peak, 0] / self.grid_density_bins * (2 * self.grid_density_size) - self.grid_density_size
            return x_peak, y_peak

    def show_density_field(self, bandwidth=1., kernel='gaussian', MC=False, peaks=False, analyze_peaks=False,
                           GUI=False, path='density.png', txt_out=None, 
                           grid_size=750, grid_bins=2000, recompute=False, plot_orbits=False, galah=None):
        self.grid_density_size = grid_size
        self.grid_density_bins = grid_bins
        if self.density_field is None or recompute:
            # compute density field from given stream data
            self._compute_density_field(bandwidth=bandwidth, kernel=kernel, MC=MC)

        if peaks:
            # determine peaks in density field
            self._compute_density_field_peaks()
            if analyze_peaks:
                for i_pe in range(len(self.density_peaks[:, 1])):
                    source_selected = self.show_density_selection(x_img=self.density_peaks[i_pe, 1], y_img=self.density_peaks[i_pe, 0],
                                                                  xyz_stream=None, MC=MC, GUI=GUI, path=None, txt_out=txt_out)

                    if plot_orbits and len(source_selected) > 0:
                        data_plot = self.input_data[np.in1d(self.input_data['source_id'], source_selected)]
                        ts = np.linspace(0., 100, 1e4) * un.Myr
                        fig, ax = plt.subplots(2, 2)
                        for star_data in data_plot:
                            orbit = Orbit(vxvv=[np.float64(star_data['ra']) * un.deg,
                                                np.float64(star_data['dec']) * un.deg,
                                                1e3 / np.float64(star_data['parallax']) * un.pc,
                                                np.float64(star_data['pmra']) * un.mas / un.yr,
                                                np.float64(star_data['pmdec']) * un.mas / un.yr,
                                                np.float64(star_data['rv']) * un.km / un.s],
                                          radec=True,
                                          ro=8.2, vo=238., zo=0.025,
                                          solarmotion=[-11., 10., 7.25])  # as used by JBH in his paper on forced oscillations and phase mixing
                            orbit.turn_physical_on()
                            orbit.integrate(ts, MWPotential2014)
                            orbit_xyz = [orbit.x(ts) * 1e3, orbit.y(ts) * 1e3, orbit.z(ts) * 1e3, orbit.R(ts) * 1e3]

                            ax[0, 0].plot(orbit_xyz[0], orbit_xyz[1], lw=0.8)
                            ax[1, 0].plot(orbit_xyz[0], orbit_xyz[2], lw=0.8)
                            ax[0, 1].plot(orbit_xyz[2], orbit_xyz[1], lw=0.8)
                            ax[1, 1].plot(orbit_xyz[3], orbit_xyz[2], lw=0.8)
                        x_peak, y_peak = self.get_nearest_density_peak(self.density_peaks[i_pe, 1], self.density_peaks[i_pe, 0])
                        fig.suptitle("Peak location  X':"+str(x_peak)+"   Y':"+str(y_peak)+" \n")
                        ax[0, 0].set(xlabel='X', ylabel='Y')
                        ax[1, 0].set(xlabel='X', ylabel='Z')
                        ax[0, 1].set(xlabel='Z', ylabel='Y')
                        ax[1, 1].set(xlabel='R', ylabel='Z')
                        plt.savefig(path[:-4]+'_peak{:02.0f}.png'.format(i_pe+1), dpi=250)
                        plt.close()

                    # plot orbits if requested
                    if path is not None and len(source_selected) > 0:
                        txt_w = open(txt_out, 'a')
                        if galah is not None:
                            idx_galah = np.in1d(galah['source_id'], source_selected)
                            if np.sum(idx_galah) > 0:
                                txt_w.write('GALAH: ' + ','.join([str(g_s_i) for g_s_i in galah['source_id'][idx_galah]])+'\n')
                        txt_w.write('\n')
                        txt_w.close()

        fig, ax = plt.subplots(1, 1)
        im_ax = ax.imshow(self.density_field, interpolation=None, cmap='seismic',
                          origin='lower', vmin=0.)  # , vmax=4.)
        if peaks:
            ax.scatter(self.density_peaks[:, 1], self.density_peaks[:, 0], lw=0, s=8, c='#00ff00')
        fig.colorbar(im_ax)
        ax.set_axis_off()
        fig.tight_layout()
        if GUI:
            fig.set_size_inches(5.6, 4)
            return fig
        elif path is not None:
            plt.savefig(path, dpi=250)
        else:
            plt.show()
        plt.close()

    def show_density_selection(self, x_img=None, y_img=None, xyz_stream=None,
                               MC=False, GUI=False, path='density_selection.png', txt_out=None):
        # first get nearest peak coordinates from image to x' y' coordinates
        x_peak, y_peak = self.get_nearest_density_peak(x_img, y_img)

        # determine objects that lie in the vicinity of the peak
        vicinity_range = self.density_bandwith  # * 2.
        # correct vicinity range for the distance from the sun, tipple size at 1kp
        vicinity_range *= 1. + 2.*np.sqrt(x_peak**2 + y_peak**2)/1000.
        #
        idx_in_selection = np.sqrt((self.plane_intersects_2d[:, 0] - x_peak) ** 2 +
                                   (self.plane_intersects_2d[:, 1] - y_peak) ** 2) < vicinity_range
        if MC:
            id_uniq_val, id_uniq_count = np.unique(self.data_MC['id_uniq'][idx_in_selection], return_counts=True)
            # determine probability of selection for every id_uniq
            id_uniq_prob = 1. * id_uniq_count/self.n_samples_MC
            # probability that all MC created samples lie in vicinity
            # print id_uniq_prob
            possible_members = id_uniq_val[id_uniq_prob > 0.68]
            # print possible_members
            # plot_data = self.xyz_vel_MC[idx_in_selection]
            idx_in_selection = np.in1d(self.input_data['id_uniq'], possible_members)
            # print np.where(idx_in_selection)

        n_in_range = np.sum(idx_in_selection)
        print n_in_range
        if n_in_range <= 0:
            print 'No viable object in selection'
            return []

        input_data_selection = self.input_data['source_id','ra','dec', 'rv','pmra','pmdec','parallax'][idx_in_selection]
        min_objects = 5
        if n_in_range >= min_objects:
            # store result internally only when operating in batch mode
            if not GUI:
                self.meaningful_peaks.append([x_peak, y_peak])
            # output only significant congestions with at least two members
            if txt_out is not None:
                txt_w = open(txt_out, 'a')
                txt_w.write("Peak location  X':"+str(x_peak)+"   Y':"+str(y_peak)+" \n")
                txt_w.write(str(input_data_selection)+" \n")
                txt_w.write('Source_ids: '+','.join([str(sel_s_i) for sel_s_i in self.input_data['source_id'][idx_in_selection]])+"\n")
                txt_w.close()
            else:
                print input_data_selection
            print self.input_data['source_id'][idx_in_selection]

        if GUI is not True and path is None:
            if n_in_range >= min_objects:
                return self.input_data['source_id'][idx_in_selection]
            else:
                return []

        plot_data = self.xyz_vel[idx_in_selection]
        plot_range = 10
        stream_center = xyz_stream
        labels = ['X', 'Y', 'Z']
        # Create a plot
        plot_comb = [[0, 1], [2, 1], [0, 2]]
        plot_pos = [[0, 0], [0, 1], [1, 0]]
        fig, ax = plt.subplots(2, 2)
        for i_c in range(len(plot_comb)):
            fig_pos = (plot_pos[i_c][0], plot_pos[i_c][1])
            i_x = plot_comb[i_c][0]
            i_y = plot_comb[i_c][1]
            if MC:
                alpha_use = 1.
            else:
                alpha_use = 1.
            if stream_center is not None:
                ax[fig_pos].scatter(stream_center[i_x], stream_center[i_y], lw=0, c='black', s=10, marker='*')
            ax[fig_pos].scatter(plot_data[:, i_x], plot_data[:, i_y], lw=0, c='blue', s=2, alpha=alpha_use)
            ax[fig_pos].set(xlabel=labels[i_x], ylabel=labels[i_y],
                            xlim=[stream_center[i_x] - plot_range, stream_center[i_x] + plot_range],
                            ylim=[stream_center[i_y] - plot_range, stream_center[i_y] + plot_range])
        # fig.tight_layout()
        if GUI:
            fig.set_size_inches(5.6, 4)
            return fig
        elif path is not None:
            plt.savefig(path, dpi=250)
        else:
            plt.show()
        plt.close()

    def compare_with_simulation(self, sim_data, r_vel=10., xyz_stream=None, txt_out=None, img_path=None):
        n_m_peaks = len(self.meaningful_peaks)
        if n_m_peaks <= 0:
            return
        idx_close_vel= np.sqrt((sim_data['vx'] - xyz_stream[0]) ** 2 +
                               (sim_data['vy'] - xyz_stream[1]) ** 2 +
                               (sim_data['vz'] - xyz_stream[2]) ** 2) < r_vel
        sim_data_subset = sim_data[idx_close_vel]
        xyz_pos_stars = np.vstack((sim_data_subset['px'], sim_data_subset['py'], sim_data_subset['pz'])).T * 1000.  # conversion from kpc to pc
        xyz_vel_stars = np.vstack((sim_data_subset['vx'], sim_data_subset['vy'], sim_data_subset['vz'])).T
        print ' Computing intersections from simulated data'
        plane_intersects_3D = stream_plane_vector_intersect(xyz_pos_stars, xyz_vel_stars, xyz_stream)
        plane_intersects_2D = intersects_to_2dplane(plane_intersects_3D, xyz_stream)
        if img_path is not None:
            fig, ax = plt.subplots(1, 1)
            ax.scatter(plane_intersects_2D[:, 0], plane_intersects_2D[:, 1], lw=0, c='blue', s=2, alpha=0.3)
            ax.scatter(0, 0, lw=0, c='black', s=10, marker='*')  # solar position
            ax.set(xlabel='X stream plane', ylabel='Y stream plane', xlim=(-1500,1500), ylim=(-1500,1500))
            fig.tight_layout()
            plt.savefig(img_path, dpi=250)
            plt.close()
        print ' Density field from simulated intersections'
        stars_density = KernelDensity(bandwidth=self.density_bandwith, kernel=self.density_kernel).fit(plane_intersects_2D)
        grid_pos = np.linspace(-self.grid_density_size, self.grid_density_size, self.grid_density_bins)
        _x, _y = np.meshgrid(grid_pos, grid_pos)
        density_field = stars_density.score_samples(np.vstack((_x.ravel(), _y.ravel())).T) + np.log(plane_intersects_2D.shape[0])
        density_field = np.exp(density_field) * 1e3

        # output pretty png image of the density plot
        fig, ax = plt.subplots(1, 1)
        im_ax = ax.imshow(density_field.reshape(_x.shape), interpolation=None, cmap='seismic',
                          origin='lower', vmin=0.)
        ax.scatter(self.density_peaks[:, 1], self.density_peaks[:, 0], lw=0, s=8, c='#00ff00')
        fig.colorbar(im_ax)
        ax.set_axis_off()
        fig.tight_layout()
        plt.savefig(img_path[:-4]+'_d.png', dpi=250)
        plt.close()

        # statistical descriptions of the given field
        min_d = np.min(density_field)
        max_d = np.max(density_field)
        counts_d, bin_d = np.histogram(density_field, bins=100, range=(1e-5, np.percentile(density_field,98)))
        # analyse peak by peak
        print ' Analysing simulation density field at peaks in data'
        for i_p in range(n_m_peaks):
            x_peak, y_peak = self.meaningful_peaks[i_p]
            idx_density_pos = np.argmin((_x.ravel()-x_peak)**2 + (_y.ravel()-y_peak)**2)
            density_mag_peak = density_field[idx_density_pos]
            n_stars = np.sqrt((plane_intersects_2D[:,0]-x_peak)**2 + (plane_intersects_2D[:,1]-y_peak)**2) < self.density_bandwith
            n_stars = np.sum(n_stars)
            if txt_out is not None:
                txt_w = open(txt_out, 'a')
                txt_w.write("Peak location  X':"+str(x_peak)+"   Y':"+str(y_peak)+" \n")
                txt_w.write(" N sim stars: " + str(n_stars)+" \n")
                txt_w.write(" Peak mag   : " + str(np.int(100.*(density_mag_peak-min_d)/(max_d-min_d)))+"%"+" \n")
                txt_w.write('\n')
                txt_w.close()

    def show_dbscan_field(self, samples=10., eps=50, peaks=False, GUI=False, path='density.png'):
        plot_lim = (-1500, 1500)
        # select the data
        data_use = self.plane_intersects_2d
        # fit the data
        print 'DBSCAN running'
        db_fit = DBSCAN(min_samples=samples, eps=eps, algorithm='auto', metric='euclidean', n_jobs=12).fit(data_use)
        print 'DBSCAN finished'
        self.cluster_labels = db_fit.labels_
        # create plot
        fig, ax = plt.subplots(1, 1)
        # plot the points that were not signed to ny of the clusters
        idx_background = self.cluster_labels == -1
        if np.sum(idx_background) > 0:
            ax.scatter(data_use[idx_background, 0], data_use[idx_background, 1], c='black', s=2, lw=0, alpha=0.2)
        # plot clusters by colour
        idx_clusters = self.cluster_labels >= 0
        ax.scatter(data_use[idx_clusters, 0], data_use[idx_clusters, 1], c=self.cluster_labels[idx_clusters],
                   s=2, lw=0, cmap='jet')
        ax.scatter(0, 0, lw=0, c='black', s=10, marker='*')  # solar position
        # # TODO
        # if peaks:
        #     # determine peaks in density field
        #     self._compute_density_field_peaks()
        #     ax.scatter(self.density_peaks[:, 1], self.density_peaks[:, 0], lw=0, s=8, c='#00ff00')
        # fig.colorbar()
        ax.set(xlabel='X stream plane', ylabel='Y stream plane', xlim=plot_lim, ylim=plot_lim)
        fig.tight_layout()
        if GUI:
            fig.set_size_inches(5.6, 4)
            return fig
        elif path is not None:
            plt.savefig(path, dpi=250)
        else:
            plt.show()
        plt.close()

    def evaluate_dbscan_field(self, path=None, MC=False):
        if path is not None:
            save_output = True
            txt_out = open(path, 'w')
        else:
            save_output = False
        # get unique numbers of objects
        if MC:
            # use MC simulated data
            star_ids = self.data_MC['id_uniq']
        else:
            # use original dataset
            star_ids = self.input_data['id_uniq']
        # evluate number of labels
        db_label, c_label = np.unique(self.cluster_labels, return_counts=True)
        # loop through every label
        for i_l in range(len(db_label)):
            if db_label[i_l] == -1:
                continue
            idx_label = self.cluster_labels == db_label[i_l]
            selected_stars = star_ids[idx_label]
            star_id, c_star_id = np.unique(selected_stars, return_counts=True)
            c_mean = np.mean(c_star_id)
            if c_mean > 2:
                out_str = 'Label: '+str(db_label[i_l])+'\n  Items: '+str(c_label[i_l])+'\n'
                out_str += '  Uniq: '+str(len(star_id))+'\n  Max:  '+str(np.max(c_star_id))+'\n  Mean: '+str(c_mean)+'\n \n'
                if save_output:
                    txt_out.write(out_str)
                else:
                    print out_str
        if save_output:
            txt_out.close()

    def phase_intersects_analysis(self, GUI=False, path='phase.png', phase_step=2., parsec_step=10., phs_std_multi=2.):
        # compute phases of intersects
        phase_ang = np.rad2deg(np.arctan2(self.plane_intersects_2d[:, 0], self.plane_intersects_2d[:, 1]))
        phase_dist = np.sqrt(np.sum(self.plane_intersects_2d**2, axis=1))
        idx_neg = phase_ang < 0.
        if np.sum(idx_neg) > 0:
            phase_ang[idx_neg] += 360.

        # compute distribution of the phases
        phase_hist, phase_bins = np.histogram(phase_ang, range=(0., 360.), bins=360./phase_step)
        phase_hist_pos = phase_bins[:-1]+phase_step/2.

        def fit_baseline_cheb(x, y, c_deg, n_step=2):
            # initial fit before any point removal from the fit
            chb_coef = np.polynomial.chebyshev.chebfit(x, y, c_deg)
            base = np.polynomial.chebyshev.chebval(x, chb_coef)
            for i_s in range(n_step):
                y_diff = y - base
                y_diff_std = np.nanstd(y_diff)
                # removal of upper outliers
                idx_up_use = np.logical_and(y_diff > 0, np.abs(y_diff) < 0.5 * y_diff_std)
                # removal of outliers below baseline
                idx_low_use = np.logical_and(y_diff < 0, np.abs(y_diff) < 2. * y_diff_std)
                # final idx to be used
                idx_use = np.logical_or(idx_up_use, idx_low_use)
                # refit
                chb_coef = np.polynomial.chebyshev.chebfit(phase_hist_pos[idx_use], phase_hist[idx_use], 15)
                base = np.polynomial.chebyshev.chebval(phase_hist_pos, chb_coef)
            return base

        # fit gaussian function(s) to the hist-baseline function
        def gaussian_fit(parameters, data, phs, ref_data, evaluate=True):
            n_keys = (len(parameters)) / 3
            function_val = np.array(ref_data)
            for i_k in range(n_keys):
                function_val += parameters['amp' + str(i_k)] * np.exp(
                    -0.5 * (parameters['phs' + str(i_k)] - phs) ** 2. / parameters['std' + str(i_k)] ** 2.)
            if evaluate:
                likelihood = np.power(data - function_val, 2)
                return likelihood
            else:
                return function_val

        # ----------------------
        # ----- Phase part -----
        # ----------------------

        filtered_curve = savgol_filter(phase_hist, window_length=7, polyorder=5)
        baseline = fit_baseline_cheb(phase_hist_pos, phase_hist, 25, n_step=4)
        # replace curve with baseline points
        idx_replace = (filtered_curve - baseline) < 0
        filtered_curve[idx_replace] = baseline[idx_replace]
        peaks_max = argrelextrema(filtered_curve, np.greater, order=4, mode='wrap')[0]
        residual = filtered_curve - baseline

        # remove peaks bellow baseline - there should be no one like this when replacing filtered with baseline for low values
        idx_peaks_ok = residual[peaks_max] > 0
        peaks_max = peaks_max[idx_peaks_ok]

        # remove insignificant peaks or based on their height above the baseline
        idx_peaks_ok = baseline[peaks_max] * 0.15 < residual[peaks_max]
        peaks_max = peaks_max[idx_peaks_ok]

        # Parameters class with the values to be fitted to the filtered function
        fit_param = Parameters()
        fit_keys = list([])
        for i_p in range(len(peaks_max)):
            key_std = 'std' + str(i_p)
            fit_param.add(key_std, value=4., min=0.5, max=10., vary=True)
            fit_keys.append(key_std)
            key_amp = 'amp' + str(i_p)
            fit_param.add(key_amp, value=500., min=1., max=2000., vary=True)
            fit_keys.append(key_amp)
            key_wvl = 'phs' + str(i_p)
            peak_loc = phase_hist_pos[peaks_max[i_p]]
            fit_param.add(key_wvl, value=peak_loc, min=peak_loc - 5., max=peak_loc + 5., vary=True)
            fit_keys.append(key_wvl)

        # perform the actual fit itself
        fit_res = minimize(gaussian_fit, fit_param, args=(filtered_curve, phase_hist_pos, baseline), method='leastsq')
        # fit_res.params.pretty_print()  # name of the function explains everything
        report_fit(fit_res)
        fitted_curve = gaussian_fit(fit_res.params, 0., phase_hist_pos, baseline, evaluate=False)

        # extract phase results - analyse widths and amplitudes of peaks
        final_phase_peaks = list([])
        res_param = fit_res.params
        for i_k in range(len(res_param) / 3):
            p_phs = res_param['phs' + str(i_k)].value
            p_std = res_param['std' + str(i_k)].value
            # evaluate number of points in the selection and remove ones with low number of
            idx_in = np.logical_and(phase_ang < p_phs+phs_std_multi*p_std, phase_ang > p_phs-phs_std_multi*p_std)
            if np.sum(idx_in) > self.n_samples_MC:  # so we have number of data worth for at least one MC star
                final_phase_peaks.append([p_phs, p_std])

        # plot results
        fig, ax = plt.subplots(1, 1)
        ax.plot(phase_hist_pos, phase_hist, lw=0.5, color='black')
        ax.plot(phase_hist_pos, baseline, lw=1, color='green')
        ax.plot(phase_hist_pos, filtered_curve, lw=1, color='blue')
        for i_p in peaks_max:
            ax.axvline(x=phase_hist_pos[i_p], lw=0.5, color='green')
        ax.plot(phase_hist_pos, fitted_curve, lw=1, color='red')
        ax.set(ylim=(0., np.max(phase_hist) * 1.05), xlim=(0., 360.))

        # plot fitted peak (center and std) to the graph
        for i_k in range(len(final_phase_peaks)):
            p_phs = final_phase_peaks[i_k][0]
            p_std = phs_std_multi * final_phase_peaks[i_k][1]
            ax.axvline(x=p_phs, lw=1, color='black')
            ax.axvspan(p_phs - p_std, p_phs + p_std, facecolor='black', alpha=0.05)
        fig.tight_layout()

        # settings how to show or store derived plot
        if GUI:
            fig.set_size_inches(5.6, 4)
            return fig
        elif path is not None:
            plt.savefig(path, dpi=250)
        else:
            plt.show()
        plt.close()

        # # -----------------------
        # # ----- Radial part -----
        # # -----------------------
        # for i_k in range(len(final_phase_peaks)):
        #     p_phs = final_phase_peaks[i_k][0]
        #     p_std = phs_std_multi * final_phase_peaks[i_k][1]
        #     idx_ang_sel = np.logical_and(phase_ang < p_phs+phs_std_multi*p_std, phase_ang > p_phs-phs_std_multi*p_std)
        #     parsec_dist = phase_dist[idx_ang_sel]
        #     # do a similar thing as before for the phase analysis, but now for the distances inside a triangle
        #     max_parsec = 1200.
        #     parsec_hist, parsec_bins = np.histogram(parsec_dist, range=(0, max_parsec), bins=max_parsec/parsec_step)
        #     parsec_hist_pos = parsec_bins[:-1] + parsec_step / 2.
        #     print p_phs
        #     print parsec_hist
        #     plt.title('Peak at '+str(p_phs))
        #     plt.bar(parsec_hist_pos, parsec_hist, width=parsec_step, align='center', color='black', alpha=0.25, linewidth=0)
        #     plt.savefig('peak_'+str(i_k)+'.png', dpi=250)
        #     plt.close()

        # TODO: select viable overdensities


def scatter_to_2d(x, y, range=[-1500, 1500], steps=10):
    data, x_edges, y_edges = np.histogram2d(x, y, bins=steps, range=[range, range])
    step = x_edges[1] - x_edges[0]
    edges = x_edges[:-1] + step/2.  # x and y edges are the same
    return data, edges



# WATERSHED TEST ON DENSITY IMAGE
# from scipy import ndimage as ndi
# local_maxi = peak_local_max(self.density_field, indices=False, min_distance=2.*self.density_bandwith, num_peaks=10,
#                                     threshold_abs=1., threshold_rel=None)
# markers = ndi.label(local_maxi)[0]
# markers = np.zeros_like(self.density_field)
# for i_m in range(len(self.density_peaks[:, 1])):
#     print i_m
#     markers[self.density_peaks[i_m, 1], self.density_peaks[i_m, 0]] = i_m + 1  # initial markers
# markers[0, 0] = i_m+1
# markers[0, -1] = i_m + 1
# markers[-1, 0] = i_m + 1
# markers[-1, -1] = i_m + 1
# watershed_img = watershed_ift(np.uint16(self.density_field*100), np.int16(markers))
# print watershed_img
# print np.min(markers), np.max(markers)
# print np.min(watershed_img), np.max(watershed_img)
# for i_m in range(1,11):
#     print np.sum(watershed_img == i_m)
# im_ax = ax.imshow(watershed_img, interpolation=None, cmap='seismic', origin='lower', vmin=1.)
