import astropy.units as u
import astropy.coordinates as coord
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


class STREAM:
    def __init__(self, data, uvw_vel=None, xyz_vel=None, radiant=None):
        """

        :param data:
        :param uvw_vel:
        :param xyz_vel:
        :param radiant:
        """
        self.input_data = data
        self.radiant = radiant
        # transform coordinates in cartesian coordinate system
        self.cartesian = coord.SkyCoord(ra=data['ra_gaia']*u.deg,
                                        dec=data['dec_gaia']*u.deg,
                                        distance=data['parsec']*u.pc).cartesian
        if self.radiant is not None:
            self.radiant_cartesian = coord.SkyCoord(ra=radiant[0]*u.deg,
                                                    dec=radiant[1]*u.deg,
                                                    distance=3000).cartesian
        # prepare labels that will be used later on
        self.cartesian_rotated = None
        self.stream_params = None
        # store galactic and cartesian velocities
        if uvw_vel is not None:
            self.uvw_vel = uvw_vel
        else:
            self.uvw_vel = None
        if xyz_vel is not None:
            self.xyz_vel = xyz_vel
        else:
            self.xyz_vel = None

    def _rotate_coordinate_system(self):
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
        old_coordinates = np.transpose(np.vstack((self.cartesian.x, self.cartesian.y, self.cartesian.z)))
        new_coordinates = old_coordinates.value.dot(rot_matrix)
        self.cartesian_rotated = coord.SkyCoord(x=new_coordinates[:, 0]*u.pc, y=new_coordinates[:, 1]*u.pc, z=new_coordinates[:, 2]*u.pc,
                                                frame='icrs', representation='cartesian').cartesian

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

    def stream_show(self, path=None, view_pos=None):
        """

        :param path:
        :return:
        """
        plot_lim = (-2000, 2000)
        fig = plt.subplot(111, projection='3d')
        fig.scatter(0, 0, 0, c='black', marker='*', s=20)
        fig.scatter(self.cartesian.x, self.cartesian.y, self.cartesian.z,
                    c='blue', depthshade=False, alpha=0.35, s=20, lw=0)
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

    def plot_velocities(self, uvw=False, xyz=False, path='vel.png'):
        plot_lim = (-20, 20)
        if xyz and self.xyz_vel is not None:
            plot_data = self.xyz_vel
            labels = ['X', 'Y', 'Z']
        elif uvw and self.uvw_vel is not None:
            plot_data = self.uvw_vel
            labels = ['U', 'V', 'W']
        # Create a plot
        plot_comb = [[0,1], [2,1], [0,2]]
        plot_pos = [[0,0], [0,1], [1,0]]
        fig, ax = plt.subplots(2, 2)
        for i_c in range(len(plot_comb)):
            fig_pos = (plot_pos[i_c][0], plot_pos[i_c][1])
            ax[fig_pos].scatter(plot_data[:,plot_comb[i_c][0]], plot_data[:,plot_comb[i_c][1]], lw=0, c='black', s=2)
            ax[fig_pos].set(xlabel=labels[plot_comb[i_c][0]], ylabel=labels[plot_comb[i_c][1]])#, xlim=plot_lim, ylim=plot_lim)
        plt.savefig(path, dpi=250)
        plt.close()

    def estimate_stream_dimensions(self, path=None, color=None):
        """

        :param path:
        :param color:
        :return:
        """
        plot_lim = (-2000, 2000)
        # first transform coordinate system in that way that radian lies on a z axis
        if self.cartesian_rotated is None:
            self._rotate_coordinate_system()
        # compute stream parameters
        self.stream_params = self._determine_stream_param(method='mass')
        # plot results
        plt.scatter(self.stream_params[0], self.stream_params[1], c='black', marker='+', s=15)
        ax = plt.gca()
        c1 = plt.Circle((self.stream_params[0].value, self.stream_params[1].value), self.stream_params[2].value, color='0.85', fill=False)
        ax.add_artist(c1)
        # plot stream in xy plane
        plt.scatter(0, 0, c='black', marker='*', s=15)
        if color is None:
            plt.scatter(self.cartesian_rotated.x, self.cartesian_rotated.y,
                        c='blue', alpha=0.35, s=15, lw=0)
        else:
            plt.scatter(self.cartesian_rotated.x, self.cartesian_rotated.y,
                        c=color, s=15, lw=0, cmap='jet')
            plt.colorbar()
        plt.axis('equal')
        plt.xlim(plot_lim)
        plt.ylim(plot_lim)
        plt.xlabel('X\' [pc]')
        plt.ylabel('Y\' [pc]')
        plt.title('radius {:4.0f}   length {:4.0f}'.format(self.stream_params[2], self.stream_params[3]))
        plt.tight_layout()
        if path is not None:
            plt.savefig(path, dpi=250)
        plt.close()
        # also return computed parameters
        return self.stream_params

