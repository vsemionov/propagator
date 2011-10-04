#!/usr/bin/env python


import sys
import os
import math

import numpy as np
from scipy.linalg import solve_banded

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt


c0 = 3.0e8

# begin independent config params
sizex = 1.0e-3
sizey = 1.0e-3
sizez = 0.5

countx = 65
county = 65
countz = 51

countx_scale = 2
county_scale = 2
countz_scale = 1

amplitude = 1.0

wavelen = 632.8e-9

beamwidth = 0.4e-3

sigma_max = 3.0e16 # max value at borders
sigma_power = 2 # exponent of the absorption profile
pml_border_size = 0.046875 # border size, relative to window size

output_dir = "propagator"
# end independent config params

# begin dependent config params
lowx = -0.5 * sizex
highx = 0.5 * sizex
lowy = -0.5 * sizey
highy = 0.5 * sizey
lowz = 0.0
highz = sizez

dx = sizex / (countx - 1)
dy = sizey / (county - 1)
dz = sizez / (countz - 1)
dx2 = dx * dx
dy2 = dy * dy
dz2 = dz * dz

K = 2.0 * math.pi / wavelen
omega = K * c0

w0 = 0.5 * beamwidth
w02 = w0 * w0
# end dependent config params


def gaussian(z):
    z2 = z * z

    zR = math.pi * w02 / wavelen
    zR2 = zR * zR

    wz = w0 * math.sqrt(1 + z2 / zR2)
    wz2 = wz * wz

    Rz_inv = z / (z2 + zR2)

    zeta = math.atan(z / zR)

    exparg = (-(1 / wz2) +  1j * K * 0.5 * Rz_inv) * R2
    exparg += -1j * zeta

    field = amplitude * (w0 / wz) * np.exp(exparg)
    return field

def calc_tbc(u1, u2):
    k = -(1j / dx) * np.log(u2 / u1)
    km = np.maximum(k.real, 0) + 1j * k.imag
    tbc = np.exp(1j * dx * km)
    return tbc

def step_analytic(k, field):
    return gaussian(lowz + k * dz)

def step_reflect(k, field):
    tmp_field = field.copy()

    A = 1j / (2.0 * K)
    Ax = A / dx2
    Ay = A / dy2
    B = 2.0 / dz

    global xab
    if k == 1:
        ones = np.ones(countx)
        diaga = -Ax * ones
        diagb = (B + 2.0 * Ax) * ones
        diagc = -Ax * ones
        xab = np.matrix([diaga, diagb, diagc])

    lxab = xab

    resvecs = (B - 2.0 * Ay) * field
    resvecs[:, 1:] += Ay * field[:, :-1]
    resvecs[:, :-1] += Ay * field[:, 1:]

    for n, rv in enumerate(resvecs.T):
        U = solve_banded((1, 1), lxab, rv)
        tmp_field[:, n] = U

    global yab
    if k == 1:
        ones = np.ones(county)
        diaga = -Ay * ones
        diagb = (B + 2.0 * Ay) * ones
        diagc = -Ay * ones
        yab = np.matrix([diaga, diagb, diagc])

    lyab = yab

    resvecs = (B - 2.0 * Ax) * tmp_field
    resvecs[1:] += Ax * tmp_field[:-1]
    resvecs[:-1] += Ax * tmp_field[1:]

    for m, rv in enumerate(resvecs):
        U = solve_banded((1, 1), lyab, rv)
        field[m] = U

    return field

def step_tbc(k, field):
    tmp_field = field.copy()

    A = 1j / (2.0 * K)
    Ax = A / dx2
    Ay = A / dy2
    B = 2.0 / dz

    global xab
    if k == 1:
        ones = np.ones(countx)
        diaga = -Ax * ones
        diagb = (B + 2.0 * Ax) * ones
        diagc = -Ax * ones
        xab = np.matrix([diaga, diagb, diagc])

    lxab = xab.copy()

    resvecs = (B - 2.0 * Ay) * field
    resvecs[:, 1:] += Ay * field[:, :-1]
    resvecs[:, :-1] += Ay * field[:, 1:]

    tbc_y_low = calc_tbc(field[:, 1], field[:, 0])
    tbc_y_high = calc_tbc(field[:, -2], field[:, -1])
    resvecs[:, 0] += Ay * tbc_y_low * field[:, 0]
    resvecs[:, -1] += Ay * tbc_y_high * field[:, -1]

    for n, rv in enumerate(resvecs.T):
        diagb = lxab[1]
        tbc_x_low = calc_tbc(field[1, n], field[0, n]);
        tbc_x_high = calc_tbc(field[-2, n], field[-1, n]);
        diagb[0] = (B + 2.0 * Ax) - (Ax * tbc_x_low)
        diagb[-1] = (B + 2.0 * Ax) - (Ax * tbc_x_high)
        U = solve_banded((1, 1), lxab, rv)
        tmp_field[:, n] = U

    global yab
    if k == 1:
        ones = np.ones(county)
        diaga = -Ay * ones
        diagb = (B + 2.0 * Ay) * ones
        diagc = -Ay * ones
        yab = np.matrix([diaga, diagb, diagc])

    lyab = yab.copy()

    resvecs = (B - 2.0 * Ax) * tmp_field
    resvecs[1:] += Ax * tmp_field[:-1]
    resvecs[:-1] += Ax * tmp_field[1:]

    tbc_x_low = calc_tbc(tmp_field[1], tmp_field[0])
    tbc_x_high = calc_tbc(tmp_field[-2], tmp_field[-1])
    resvecs[0] += Ax * tbc_x_low * tmp_field[0]
    resvecs[-1] += Ax * tbc_x_high * tmp_field[-1]

    for m, rv in enumerate(resvecs):
        diagb = lyab[1]
        tbc_y_low = calc_tbc(field[m, 1], field[m, 0]);
        tbc_y_high = calc_tbc(field[m, -2], field[m, -1]);
        diagb[0] = (B + 2.0 * Ay) - (Ay * tbc_y_low)
        diagb[-1] = (B + 2.0 * Ay) - (Ay * tbc_y_high)
        U = solve_banded((1, 1), lyab, rv)
        field[m] = U

    return field

def init_globals():
    global X, Y, Z
    X = np.linspace(lowx, highx, countx, retstep=False)
    Y = np.linspace(lowy, highy, county, retstep=False)
    Z = np.linspace(lowz, highz, countz, retstep=False)
    global XY, YX, R2
    XY, YX = np.meshgrid(X, Y)
    R2 = (XY * XY) + (YX * YX)
    global XZ, ZX
    XZ, ZX = np.meshgrid(X, Z)

def init_dir(name=None):
    if name is None:
        dirname = output_dir
    else:
        dirname = os.path.join(output_dir, name)
    if not os.path.isdir(dirname):
        os.mkdir(dirname)

def init_field():
    field = gaussian(lowz)
    return field

def propagate(method):
    init_dir(method)
    step = getattr(sys.modules[__name__], "step_" + method)
    vals = np.zeros((countz, countx), np.complex)
    field = init_field()
    vals[0] = field[:, county/2]
    for k in range(1, countz):
        field = step(k, field)
        vals[k] = field[:, county/2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    vals = (np.absolute(vals) ** 2)[::countz_scale, ::countx_scale]
    zx = ZX[::countz_scale, ::countx_scale]
    xz = XZ[::countz_scale, ::countx_scale]
    ax.plot_wireframe(zx, xz, vals)
    plt.title(method)
    plt.show()

init_globals()
init_dir()
propagate("analytic")
propagate("reflect")
propagate("tbc")
