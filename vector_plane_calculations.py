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


def median_velocity(xyz_vel_stars):
    """
    Compute median velocity of observed stars in every coordinate
    :param xyz_vel_stars: velocity of the stars array(n_stars, 3) in cartesian coordinate system
    :return:
    """
    return np.nanmedian(xyz_vel_stars, axis=0)


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
    # check if star position are given as ndarray or astropy.coordinates that has to be converted into ndarray
    if str(type(xyz_stars_c)) == "<type 'numpy.ndarray'>":
        xyz_stars = np.array(xyz_stars_c)
    else:
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


def stream_plane_vector_angle(xyz_vel_stars, xyz_vel_stream):
    """
    Compute angle between the plane that is perpendicular to the stream vector and has origin in Earth center
    and star trajectory defined by its xyz position/velocity.
    :param xyz_vel_stars: velocity of the stars array(n_stars, 3) in cartesian coordinate system
    :param xyz_vel_stream: stream velocity array(3) in cartesian coordinate system
    :return:
    """
    # length of stream velocity vector
    stream_vec_length = np.sqrt(np.sum(xyz_vel_stream**2))
    if len(xyz_vel_stars.shape) >= 2:
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


def _vector_vel(vect):
    """

    :param vect:
    :return:
    """
    return np.sqrt(np.sum(vect ** 2))


def intersects_to_2dplane(intersects, xyz_vel_stream):
    """

    :param intersects:
    :param xyz_vel_stream:
    :return:
    """
    if xyz_vel_stream[2] == 0:
        xyz_vel_stream = np.array(xyz_vel_stream)
        xyz_vel_stream[2] += 0.001  # for numerical stability of derivatives
    # select two random vectors that are defined on the plane
    vect_1 = np.array([1., 0, -1. * xyz_vel_stream[0] / xyz_vel_stream[2]])
    vect_2 = np.array([0, 1., -1. * xyz_vel_stream[1] / xyz_vel_stream[2]])
    # crete unit vectors
    vect_1_unit = vect_1 / _vector_vel(vect_1)
    vect_2_unit = vect_2 / _vector_vel(vect_2)
    # compute new base vectors which will be used to transform points to 2D plane
    base_1 = vect_1_unit
    base_2 = vect_2 - np.sum(vect_1_unit*vect_2)*vect_1_unit
    base_2 = base_2 / _vector_vel(base_2)
    # transform points
    intersects_new = np.array([base_1, base_2]).dot(intersects.T).T
    return intersects_new


# # test
# import astropy.coordinates as coord
# import astropy.units as un
# stream_vel = np.array([5,6,7])
# star_vel = np.array([[45,6,27],[55,6,7],[55,6,7],[55,6,7],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1]])
# star_pos = np.array([[2,3,43],[2,3,4],[22,3,54],[2,3,4],[12,3,14],[23,23,4],[2,3,6],[7,5,6],[10,5,6],[8,6,9]])
# star_pos = coord.SkyCoord(ra=np.array([23.,45.,45,45,3,5,41,65,84,12])*un.deg,
#                           dec=np.array([41.,75.,13,78,-2,-36,-9,6,5,12])*un.deg,
#                           distance=1e3/np.array([1.,3.,3,2,4,5,1,2,3,1])*un.pc).cartesian
# inters = stream_plane_vector_intersect(star_pos, star_vel, stream_vel)
# angles = stream_plane_vector_angle(star_vel, stream_vel)
# inters_new = intersects_to_2dplane(inters, stream_vel)
