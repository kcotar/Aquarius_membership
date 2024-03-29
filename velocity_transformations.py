import numpy as np


# --------------------------------------------------------
# ---------------- Constants -----------------------------
# --------------------------------------------------------
F = 212.  # (mas/yr)*pc/(km/s)
# OR maybe even more correct value of this constant
F = 1./4.74047*1e3

# --------------------------------------------------------
# ----NOTE: all angle values in the following equations should be transformed to radians prior to the execution of the function
# --------------------------------------------------------
def compute_xyz_vel(ra, dec, rv):
    """

    :param ra:
    :param dec:
    :param rv:
    :return:
    """
    vx = np.cos(dec) * np.cos(ra) * rv
    vy = np.cos(dec) * np.sin(ra) * rv
    vz = np.sin(dec) * rv
    if np.size(vx) <= 1:
        return np.hstack((vx, vy, vz))
    else:
        return np.transpose(np.vstack((vx, vy, vz)))


def compute_rv(ra, dec, vel):
    """

    :param ra:
    :param dec:
    :param vel:
    :return:
    """
    unit_vect = compute_xyz_vel(ra, dec, 1.)
    return np.sum(unit_vect * vel, axis=1)


def compute_pmra(ra, dec, dist, vel):
    """

    :param ra:
    :param dec:
    :param dist: in parsecs
    :param vel:
    :return:
    """
    return F / dist * (-vel[0] * np.sin(ra) + vel[1] * np.cos(ra))


def compute_pmdec(ra, dec, dist, vel):
    """

    :param ra:
    :param dec:
    :param dist: in parsecs
    :param vel:
    :return:
    """
    return F / dist * (-vel[0] * np.cos(ra) * np.sin(dec) - vel[1] * np.sin(ra) * np.sin(dec) + vel[2] * np.cos(dec))


def compute_distance_pmra(ra, dec, pmra, vel, parallax=False):
    """

    :param ra:
    :param dec:
    :param vel:
    :param pmra:
    :return:
    """
    # compute distance in parsecs
    dist = F / pmra * (-vel[0] * np.sin(ra) + vel[1] * np.cos(ra))
    if parallax:
        # transform to parallax value if requested
        return 1./dist*1e3
    else:
        return dist


def compute_distance_pmdec(ra, dec, pmdec, vel, parallax=False):
    """

    :param ra:
    :param dec:
    :param vel:
    :param pmra:
    :return:
    """
    # compute distance in parsecs
    dist = F / pmdec * (-vel[0] * np.cos(ra) * np.sin(dec) - vel[1] * np.sin(ra) * np.sin(dec) + vel[2] * np.cos(dec))
    if parallax:
        # transform to parallax value if requested
        return 1./dist*1e3
    else:
        return dist
