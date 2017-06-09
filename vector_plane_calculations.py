import numpy as np


def points_distance(xyz_1, xyz_2):
    """

    :param xyz_1:
    :param xyz_2:
    :return:
    """
    if len(xyz_1.shape) >= 2:
        distance = np.sqrt(np.sum((xyz_1 - xyz_2)**2, axis=1))
    else:
        distance = np.sqrt(np.sum((xyz_1 - xyz_2)**2))
    return distance


def mean_velocity(xyz_vel_stars):
    """
    Compute mean velocity of observed stars in every coordinate
    :param xyz_vel_stars: velocity of the stars array(n_stars, 3) in cartesian coordinate system
    :return:
    """
    return np.nanmean(xyz_vel_stars, axis=0)


def _transform_cartesian_coord_to_array(coord):
    return np.transpose(np.vstack((coord.x.value, coord.y.value, coord.z.value)))


def stream_plane_vector_intersect(xyz_stars_c, xyz_vel_stars, xyz_vel_stream):
    """
    Compute intersect point(s) between stars trajectory defined by its xyz position/velocity
    and plane that is perpendicular to the stream vector and has origin in Earth center
    :param xyz_stars_c: position of the stars array(n_stars, 3) in cartesian coordinate system
    :param xyz_vel_stars: velocity of the stars array(n_stars, 3) in cartesian coordinate system
    :param xyz_vel_stream: stream velocity array(3) in cartesian coordinate system
    :return:
    """
    xyz_stars = _transform_cartesian_coord_to_array(xyz_stars_c)
    # some quick checks
    if xyz_stars.shape != xyz_vel_stars.shape:
        raise ArithmeticError('Stars arrays have different sizes.')
    if len(xyz_stars.shape) >= 2:
        # determine vector scaling factors
        vector_scale = -1.*(xyz_vel_stream[0]*xyz_stars[:,0] + xyz_vel_stream[1]*xyz_stars[:,1] + xyz_vel_stream[2]*xyz_stars[:,2])\
                       /(xyz_vel_stream[0]*xyz_vel_stars[:,0] + xyz_vel_stream[1]*xyz_vel_stars[:,1] + xyz_vel_stream[2]*xyz_vel_stars[:,2])
        # compute intersection coordinates
        intersects = xyz_stars + xyz_vel_stars*vector_scale.reshape(len(vector_scale),1)
    else:
        # determine vector scaling factors
        vector_scale = -1.*(xyz_vel_stream[0]*xyz_stars[0] + xyz_vel_stream[1]*xyz_stars[1] + xyz_vel_stream[2]*xyz_stars[2]) \
                       /(xyz_vel_stream[0]*xyz_vel_stars[0] + xyz_vel_stream[1]*xyz_vel_stars[1] + xyz_vel_stream[2]*xyz_vel_stars[2])
        # compute intersection coordinates
        intersects = xyz_stars + xyz_vel_stars*vector_scale
    return intersects


def stream_plane_vector_angle(xyz_stars_c, xyz_vel_stars, xyz_vel_stream):
    """
    Compute angle between the plane that is perpendicular to the stream vector and has origin in Earth center
    and star trajectory defined by its xyz position/velocity.
    :param xyz_stars_c: position of the stars array(n_stars, 3) in cartesian coordinate system
    :param xyz_vel_stars: velocity of the stars array(n_stars, 3) in cartesian coordinate system
    :param xyz_vel_stream: stream velocity array(3) in cartesian coordinate system
    :return:
    """
    xyz_stars = _transform_cartesian_coord_to_array(xyz_stars_c)
    # length of stream velocity vector
    stream_vec_length = np.sqrt(np.sum(xyz_vel_stream**2))
    if len(xyz_stars.shape) >= 2:
        #
        dist = np.abs(xyz_vel_stream[0]*xyz_vel_stars[:,0] + xyz_vel_stream[1]*xyz_vel_stars[:,1] + xyz_vel_stream[2]*xyz_vel_stars[:,2])
        # length of stars velocity vector
        star_vec_length = np.sqrt(np.sum(xyz_vel_stars**2, axis=1))
    else:
        #
        dist = np.abs(xyz_vel_stream[0]*xyz_vel_stars[0] + xyz_vel_stream[1]*xyz_vel_stars[1] + xyz_vel_stream[2]*xyz_vel_stars[2])
        # length of star velocity vector
        star_vec_length = np.sqrt(np.sum(xyz_vel_stars**2))
    # final impact angle calculation
    angle = 90 - np.rad2deg(np.arccos(dist/(stream_vec_length*star_vec_length)))
    return angle  # in degrees


# # test
# import astropy.coordinates as coord
# import astropy.units as un
# stream_vel = np.array([5,6,7])
# star_vel = np.array([[45,6,27],[55,6,7],[55,6,7],[55,6,7]])
# star_pos = np.array([[2,3,43],[2,3,4],[22,3,54],[2,3,4],[12,3,14],[23,23,4]])
# star_pos = coord.SkyCoord(ra=np.array([23.,45.,45,45])*un.deg,
#                           dec=np.array([41.,75.,13,78])*un.deg,
#                           distance=1e3/np.array([1.,3.,3,2])*un.pc).cartesian
# print stream_plane_vector_intersect(star_pos, star_vel, stream_vel)
# print stream_plane_vector_angle(star_pos, star_vel, stream_vel)