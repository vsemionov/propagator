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

tbc_min = 1.0e-9

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


def solve_tridiag(a, b, c, r):
    ab = np.matrix([a, b, c])
    return solve_banded((1, 1), ab, r)

def gaussian(z):
    z2 = z * z

    zR = math.pi * w02 / wavelen
    zR2 = zR * zR

    wz = w0 * math.sqrt(1 + z2 / zR2)
    wz2 = wz * wz

    #Rz = z * (1 + zR2 / z2)
    Rz_inv = z / (z2 + zR2)

    zeta = math.atan(z / zR)

    exparg = (-(1 / wz2) +  1j * K * 0.5 * Rz_inv) * R2
    exparg += -1j * zeta

    field = amplitude * (w0 / wz) * np.exp(exparg)
    return field

def step_analytic(k, field):
    return gaussian(lowz + k * dz)

def step_reflect(k, field):
    tmp_field = field.copy()

    A = 1j / (2.0 * K)
    Ax = A / dx2
    Ay = A / dy2
    B = 2.0 / dz

    ones = np.ones(countx)
    diaga = -Ax * ones
    diagb = (B + 2.0 * Ax) * ones
    diagc = -Ax * ones

    for n in range(county):
        resvec = (B - 2.0 * Ay) * field[:, n]
        if n > 0:
            resvec += Ay * field[:, n-1]
        if n < (county - 1):
            resvec += Ay * field[:, n+1]

        U = solve_tridiag(diaga, diagb, diagc, resvec)
        tmp_field[:, n] = U

    ones = np.ones(county)
    diaga = -Ay * ones
    diagb = (B + 2.0 * Ay) * ones
    diagc = -Ay * ones

    for m in range(countx):
        resvec = (B - 2.0 * Ax) * tmp_field[m]
        if m > 0:
            resvec += Ax * tmp_field[m-1]
        if m < (countx - 1):
            resvec += Ax * tmp_field[m+1]

        U = solve_tridiag(diaga, diagb, diagc, resvec)
        field[m] = U

    return field

def init_globals():
    global X, Y, XX, YY, R2
    X = np.linspace(lowx, highx, countx, retstep=False)
    Y = np.linspace(lowy, highy, county, retstep=False)
    XX, YY = np.meshgrid(X, Y)
    R2 = (XX * XX) + (YY * YY)

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
    field = init_field()
    for k in range(countz):
        field = step(k, field)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(XX, YY, np.absolute(field) ** 2)
    plt.show()


init_globals()
init_dir()
propagate("analytic")
propagate("reflect")
