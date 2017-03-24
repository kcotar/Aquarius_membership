import astropy.units as u
import astropy.coordinates as coord
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np



class STREAM:
    def __init__(self, data, radiant=None):
        self.input_data = data
        self.radiant = radiant
        # transform coordinates in cartesian coordinate system
        self.cartesian = coord.SkyCoord(ra=data['ra_gaia']*u.deg,
                                        dec=data['dec_gaia']*u.deg,
                                        distance=data['parsec']*u.pc).cartesian

    def stream_show(self, path=None, radiant=None):
        plot_lim = (-2000, 2000)
        fig = plt.subplot(111, projection='3d')
        fig.scatter(0, 0, 0, c='black', marker='*', s=20)
        fig.scatter(self.cartesian.x, self.cartesian.y, self.cartesian.z,
                    c='blue', depthshade=False, alpha=0.35, s=50, lw=0, marker='.')
        # add line that connects point of radiant and anti-radiant
        if radiant is not None:
            # compute the cartesian coordinates of both points
            rad_coord = coord.SkyCoord(ra=radiant[0]*u.deg, dec=radiant[1]*u.deg,
                                       distance=np.sqrt(3 * plot_lim[0]**2)).cartesian
            fig.plot([-rad_coord.x, rad_coord.x], [-rad_coord.y, rad_coord.y], [-rad_coord.z, rad_coord.z])
            # compute elevation and azimuth of the line that crosses trough Earth and radiant point
            azimuth = np.rad2deg(np.arctan2(rad_coord.y, rad_coord.x))
            elevation = np.rad2deg(np.arctan2(rad_coord.z, np.sqrt(rad_coord.x**2+rad_coord.y**2)))
            fig.view_init(elev=radiant[1], azim=radiant[0])
        fig.set(xlim=plot_lim, ylim=plot_lim, zlim=plot_lim, xlabel='X [pc]', ylabel='Y [pc]', zlabel='Z [pc]')
        if path is None:
            plt.show()
        else:
            plt.savefig(path)
        plt.close()
