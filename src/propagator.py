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

class propagator_analytic:
    def step(self, k, field):
        return gaussian(lowz + k * dz)

class propagator_reflect:
    def __init__(self):
        self.A = 1j / (2.0 * K)
        self.Ax = self.A / dx2
        self.Ay = self.A / dy2
        self.B = 2.0 / dz

        ones = np.ones(countx)
        diaga = -self.Ax * ones
        diagb = (self.B + 2.0 * self.Ax) * ones
        diagc = -self.Ax * ones
        self.xab = np.array([diaga, diagb, diagc])

        ones = np.ones(county)
        diaga = -self.Ay * ones
        diagb = (self.B + 2.0 * self.Ay) * ones
        diagc = -self.Ay * ones
        self.yab = np.array([diaga, diagb, diagc])

    def step(self, k, field):
        tmp_field = field.copy()

        lxab = self.xab

        resvecs = (self.B - 2.0 * self.Ay) * field
        resvecs[:, 1:] += self.Ay * field[:, :-1]
        resvecs[:, :-1] += self.Ay * field[:, 1:]

        for n, rv in enumerate(resvecs.T):
            U = solve_banded((1, 1), lxab, rv)
            tmp_field[:, n] = U

        lyab = self.yab

        resvecs = (self.B - 2.0 * self.Ax) * tmp_field
        resvecs[1:] += self.Ax * tmp_field[:-1]
        resvecs[:-1] += self.Ax * tmp_field[1:]

        for m, rv in enumerate(resvecs):
            U = solve_banded((1, 1), lyab, rv)
            field[m] = U

        return field

class propagator_tbc:
    def __init__(self):
        self.A = 1j / (2.0 * K)
        self.Ax = self.A / dx2
        self.Ay = self.A / dy2
        self.B = 2.0 / dz

        ones = np.ones(countx)
        diaga = -self.Ax * ones
        diagb = (self.B + 2.0 * self.Ax) * ones
        diagc = -self.Ax * ones
        self.xab = np.array([diaga, diagb, diagc])

        ones = np.ones(county)
        diaga = -self.Ay * ones
        diagb = (self.B + 2.0 * self.Ay) * ones
        diagc = -self.Ay * ones
        self.yab = np.array([diaga, diagb, diagc])

    @staticmethod
    def calc_tbc(u1, u2):
        k = -(1j / dx) * np.log(u2 / u1)
        km = np.maximum(k.real, 0) + 1j * k.imag
        tbc = np.exp(1j * dx * km)
        return tbc

    def step(self, k, field):
        tmp_field = field.copy()

        lxab = self.xab.copy()

        resvecs = (self.B - 2.0 * self.Ay) * field
        resvecs[:, 1:] += self.Ay * field[:, :-1]
        resvecs[:, :-1] += self.Ay * field[:, 1:]

        tbc_y_low = self.__class__.calc_tbc(field[:, 1], field[:, 0])
        tbc_y_high = self.__class__.calc_tbc(field[:, -2], field[:, -1])
        resvecs[:, 0] += self.Ay * tbc_y_low * field[:, 0]
        resvecs[:, -1] += self.Ay * tbc_y_high * field[:, -1]

        for n, rv in enumerate(resvecs.T):
            diagb = lxab[1]
            tbc_x_low = self.__class__.calc_tbc(field[1, n], field[0, n])
            tbc_x_high = self.__class__.calc_tbc(field[-2, n], field[-1, n])
            diagb[0] = (self.B + 2.0 * self.Ax) - (self.Ax * tbc_x_low)
            diagb[-1] = (self.B + 2.0 * self.Ax) - (self.Ax * tbc_x_high)
            U = solve_banded((1, 1), lxab, rv)
            tmp_field[:, n] = U

        lyab = self.yab.copy()

        resvecs = (self.B - 2.0 * self.Ax) * tmp_field
        resvecs[1:] += self.Ax * tmp_field[:-1]
        resvecs[:-1] += self.Ax * tmp_field[1:]

        tbc_x_low = self.__class__.calc_tbc(tmp_field[1], tmp_field[0])
        tbc_x_high = self.__class__.calc_tbc(tmp_field[-2], tmp_field[-1])
        resvecs[0] += self.Ax * tbc_x_low * tmp_field[0]
        resvecs[-1] += self.Ax * tbc_x_high * tmp_field[-1]

        for m, rv in enumerate(resvecs):
            diagb = lyab[1]
            tbc_y_low = self.__class__.calc_tbc(tmp_field[m, 1], tmp_field[m, 0])
            tbc_y_high = self.__class__.calc_tbc(tmp_field[m, -2], tmp_field[m, -1])
            diagb[0] = (self.B + 2.0 * self.Ay) - (self.Ay * tbc_y_low)
            diagb[-1] = (self.B + 2.0 * self.Ay) - (self.Ay * tbc_y_high)
            U = solve_banded((1, 1), lyab, rv)
            field[m] = U

        return field

class propagator_pml:
    def __init__(self):
        self.A = 1j / (2.0 * K)
        self.Ax = self.A / dx2
        self.Ay = self.A / dy2
        self.B = 2.0 / dz

        X_pre = np.linspace(lowx - dx/2, highx - dx/2, countx, retstep=False)
        Y_pre = np.linspace(lowy - dy/2, highy - dy/2, county, retstep=False)
        XY_pre, YX_pre = np.meshgrid(X_pre, Y_pre)

        X_post = np.linspace(lowx + dx/2, highx + dx/2, countx, retstep=False)
        Y_post = np.linspace(lowy + dy/2, highy + dy/2, county, retstep=False)
        XY_post, YX_post = np.meshgrid(X_post, Y_post)

        self.sigmax = self.__class__.calc_sigma(XY, lowx, highx)
        self.sigmay = self.__class__.calc_sigma(YX, lowy, highy)

        self.sigmax_pre = self.__class__.calc_sigma(XY_pre, lowx, highx)
        self.sigmay_pre = self.__class__.calc_sigma(YX_pre, lowy, highy)

        self.sigmax_post = self.__class__.calc_sigma(XY_post, lowx, highx)
        self.sigmay_post = self.__class__.calc_sigma(YX_post, lowy, highy)

        self.a = 1.0 / ((1.0 + 1j * self.sigmax / omega) * (1.0 + 1j * self.sigmax_pre / omega))
        self.c = 1.0 / ((1.0 + 1j * self.sigmax / omega) * (1.0 + 1j * self.sigmax_post / omega))
        self.b = -(self.a + self.c)

        self.d = 1.0 / ((1.0 + 1j * self.sigmay / omega) * (1.0 + 1j * self.sigmay_pre / omega))
        self.f = 1.0 / ((1.0 + 1j * self.sigmay / omega) * (1.0 + 1j * self.sigmay_post / omega))
        self.e = -(self.d + self.f)

        diaga = -self.Ax * self.a[:-1, 0]
        diaga = np.insert(diaga, 0, 0.0)
        diagb = self.B - self.Ax * self.b[:, 0]
        diagc = -self.Ax * self.c[:, 0]
        self.xab = np.array([diaga, diagb, diagc])

        diaga = -self.Ay * self.d[0, :-1]
        diaga = np.insert(diaga, 0, 0.0)
        diagb = self.B - self.Ay * self.e[0]
        diagc = -self.Ay * self.f[0]
        self.yab = np.array([diaga, diagb, diagc])

    @staticmethod
    def calc_sigma(q, qmin, qmax):
        def calc_pml_depth(q, qmin, qmax):
            pml_width = (qmax - qmin) * pml_border_size
            qlow = qmin + pml_width
            qhigh = qmax - pml_width

            d = np.where(q <= qlow, (q - qlow) / pml_width, 0.0)
            d = np.where(q >= qhigh, (q - qhigh) / pml_width, d)

            return d

        d = calc_pml_depth(q, qmin, qmax)

        s = np.where(d != 0.0, np.fabs(d) ** sigma_power, 0.0)

        return s * sigma_max

    def step(self, k, field):
        tmp_field = field.copy()

        lxab = self.xab

        resvecs = (self.B + self.Ay * self.e) * field
        resvecs[:, 1:] += self.Ay * self.d[:, 1:] * field[:, :-1]
        resvecs[:, :-1] += self.Ay * self.f[:, :-1] * field[:, 1:]

        for n, rv in enumerate(resvecs.T):
            U = solve_banded((1, 1), lxab, rv)
            tmp_field[:, n] = U

        lyab = self.yab

        resvecs = (self.B + self.Ax * self.b) * tmp_field
        resvecs[1:] += self.Ax * self.a[1:] * tmp_field[:-1]
        resvecs[:-1] += self.Ax * self.c[:-1] * tmp_field[1:]

        for m, rv in enumerate(resvecs):
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

def propagate(name):
    init_dir(name)
    propagator = getattr(sys.modules[__name__], "propagator_" + name)()
    vals = np.zeros((countz, countx), np.complex)
    field = init_field()
    vals[0] = field[:, county/2]
    for k in range(1, countz):
        field = propagator.step(k, field)
        vals[k] = field[:, county/2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    vals = (np.absolute(vals) ** 2)[::countz_scale, ::countx_scale]
    zx = ZX[::countz_scale, ::countx_scale]
    xz = XZ[::countz_scale, ::countx_scale]
    ax.plot_wireframe(zx, xz, vals)
    plt.title(name)
    plt.show()

init_globals()
init_dir()
propagate("analytic")
propagate("reflect")
propagate("tbc")
propagate("pml")
