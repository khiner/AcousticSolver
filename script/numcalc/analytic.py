"""Independent spherical-wave solutions for NumCalc verification."""
import numpy as np
from scipy.special import spherical_jn, spherical_yn, eval_legendre


def sphere_plane(receivers, frequency, radius, kind='velocity', admittance=0, sound_speed=343):
    k = 2*np.pi*frequency/sound_speed
    x = np.asarray(receivers)
    r = np.linalg.norm(x, axis=1)
    n = np.arange(max(30, int(k*radius)+20))
    ka = k*radius
    j = spherical_jn(n, ka)
    h = j + 1j*spherical_yn(n, ka)
    dj = spherical_jn(n, ka, True)
    dh = dj + 1j*spherical_yn(n, ka, True)
    # Exterior outward mesh normal; passive admittance gives dp/dr = -ik Y p.
    coefficient = -j/h if kind == 'pressure' else -(dj+1j*admittance*j)/(dh+1j*admittance*h)
    kr = k*r[:, None]
    radial = spherical_jn(n, kr) + coefficient*(spherical_jn(n, kr)+1j*spherical_yn(n, kr))
    return np.sum((2*n+1)*1j**n*radial*eval_legendre(n, x[:, 0, None]/r[:, None]), axis=1)


def relative(a, b):
    return float(np.linalg.norm(np.asarray(a)-b)/max(np.linalg.norm(b), 1e-30))
