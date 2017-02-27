import numpy as np

F = 212.  # (mas/yr)*pc/(km/s)


def compute_xyz_vel(ra, dec, rv):
    vx = np.cos(dec) * np.cos(ra) * rv
    vy = np.cos(dec) * np.sin(ra) * rv
    vz = np.sin(dec) * rv
    return np.array([vx, vy, vz])


def compute_rv(ra, dec, vel):
    return np.sum(vel * compute_xyz_vel(ra, dec, 1))


def compute_pmra(ra, dec, dist, vel):
    return F / dist * (-vel[0] + np.sin(ra) + vel[1] * np.cos(ra))


def commpute_pmdec(ra, dec, dist, vel):
    return F / dist * (-vel[0] + np.cos(ra) * np.sin(dec) - vel[1] * np.sin(ra) * np.sin(dec) + v[2] * np.cos(dec))