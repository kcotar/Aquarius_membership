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


def rv_lsr_corr(ra, dec, lsr_vel):
    # degrees in radians
    return lsr_vel[0]*np.cos(ra)*np.cos(dec) + lsr_vel[1]*np.sin(ra)*np.cos(dec) + lsr_vel[2]*np.sin(dec)


def compute_rv(ra, dec, vel, lsr_vel=None):
    """

    :param ra: also galactic l or ny kind of longitude, in radians
    :param dec: also galactic b or ny kind of latitude,     in radians
    :param vel:
    :param lsr_vel: have to defined in the same coordinate system as ra end deg
    :return:
    """
    unit_vect = compute_xyz_vel(ra, dec, 1.)
    rv_vel = np.sum(unit_vect * vel, axis=1)
    if lsr_vel is not None:
        # correct RV velocities for LSR
        rv_vel -= 0#rv_lsr_corr(ra, dec, lsr_vel)
    return rv_vel


def pmra_lsr_corr(ra, dec, lsr_vel):
    # degrees in radians
    return -lsr_vel[0]*np.sin(ra) + lsr_vel[0]*np.cos(ra)


def compute_pmra(ra, dec, dist, vel, lsr_vel=None):
    """

    :param ra: also galactic l or ny kind of longitude, in radians
    :param dec: also galactic b or ny kind of latitude, in radians
    :param dist: in parsecs
    :param vel:
    :param lsr_vel: have to defined in the same coordinate system as ra end deg
    :return:
    """
    pmra_vel = F / dist * (-vel[0] * np.sin(ra) + vel[1] * np.cos(ra))
    if lsr_vel is not None:
        pmra_vel -= 0#pmra_lsr_corr(ra, dec, lsr_vel)
    return pmra_vel


def pmdec_lsr_corr(ra, dec, lsr_vel):
    # degrees in radians
    return -lsr_vel[0]*np.cos(ra)*np.sin(dec) - lsr_vel[1]*np.sin(ra)*np.sin(dec) + lsr_vel[2]*np.cos(dec)


def compute_pmdec(ra, dec, dist, vel, lsr_vel=None):
    """

    :param ra: also galactic l or ny kind of longitude, in radians
    :param dec: also galactic b or ny kind of latitude, in radians
    :param dist: in parsecs
    :param vel:
    :param lsr_vel: have to defined in the same coordinate system as ra end deg
    :return:
    """
    pmdec_vel = F / dist * (-vel[0] * np.cos(ra) * np.sin(dec) - vel[1] * np.sin(ra) * np.sin(dec) + vel[2] * np.cos(dec))
    if lsr_vel is not None:
        pmdec_vel -= 0#pmdec_lsr_corr(ra, dec, lsr_vel)
    return pmdec_vel


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
