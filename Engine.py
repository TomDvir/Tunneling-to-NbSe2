## FMcMillen model functions

import numpy
import matplotlib.pyplot as plt
from cmath import sqrt
import time
from lmfit import Parameters, minimize, fit_report
import scipy.io
from scipy.interpolate import interp1d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def FermiDirac(E, T):
    return numpy.array(1 / (1 + numpy.exp(E / T)))

def SelfConsEq(Delta, E, Delta0, Gamma):
    if Delta[0].imag > 0:
        Delta[0] = Delta[0].conjugate()
    if Delta[1].imag > 0:
        Delta[1] = Delta[1].conjugate()

    NewDelta = numpy.ndarray(2, complex)
    NewDelta[0] = (Delta0[0] + Gamma[0] * Delta[1] / sqrt(Delta[1] ** 2 - E ** 2)) / \
                  (1 + Gamma[0] / sqrt(Delta[1] ** 2 - E ** 2))
    NewDelta[1] = (Delta0[1] + Gamma[1] * Delta[0] / sqrt(Delta[0] ** 2 - E ** 2)) / \
                  (1 + Gamma[1] / sqrt(Delta[0] ** 2 - E ** 2))

    return NewDelta

def Iterations(prevDelta, E, Delta0, Gamma):
    errors = numpy.array([1, 1])
    index = 0
    prevDelta = prevDelta + 0.1 * 1j
    while (max(abs(errors)) > 1e-6) and (index < 1e1):
        index = index + 1
        newDelta = SelfConsEq(prevDelta, E, Delta0, Gamma)
        errors = prevDelta - newDelta
        prevDelta = newDelta
    return newDelta, errors

def AG_DOS(E, Delta, alpha):
    Delta = numpy.array(Delta)
    E = numpy.array(E)

    xi = E / Delta * (1 + 0j)
    alpha = numpy.ones(len(Delta)) * alpha * (1 + 0j)

    A = 2 - 2 * alpha ** 2 + xi ** 2
    B = (-1 + alpha ** 2 + xi ** 2) ** 2
    C = -1 + alpha ** 6 + 3 * xi ** 2 - 3 * xi ** 4 + xi ** 6 \
        + 3 * alpha ** 4 * (-1 + xi ** 2) + 3 * alpha ** 2 * (1 + 16 * xi ** 2 + xi ** 4)
    D = 6 * numpy.sqrt(3) * numpy.sqrt(alpha ** 2 * xi ** 2 * (alpha ** 6 + 3 * alpha ** 4 * (-1 + xi ** 2)\
        + (-1 + xi ** 2) ** 3 + 3 * alpha ** 2 * (1 + 7 * xi ** 2 + xi ** 4)))
    E = (C + D) ** (1 / 3.0)

    u = 1 / 6 * (3 * xi + sqrt(3) * numpy.sqrt((A + B / E + E)) - sqrt(3) * numpy.sqrt(2 * A - B / E - E \
             - 6 * sqrt(3) * (1 + alpha ** 2) * xi / numpy.sqrt(A + B / E + E)))

    return (u / numpy.sqrt(u ** 2 - 1)).real

def CalculateDOS(Energys, Delta0, Gamma, Neff, alpha=0,dynes = 0):
    Neff = numpy.array(Neff)
    Gamma = numpy.array(Gamma)
    Delta0 = numpy.array(Delta0)
    Energys = numpy.array(Energys)

    initDelta = numpy.array([0.5 - 0.1 * 1j, 0 + 0.1 * 1j])
    sol, err = Iterations(initDelta, 0, Delta0, Gamma)

    # Ns = [0] * (len(Energys))
    # Deltas = [0] * (len(Energys))
    # errors = [0] * (len(Energys))
    N = numpy.ndarray(len(Energys))

    for k in range(len(Energys)):
        initDelta = sol
        sol, err = Iterations(initDelta, Energys[k], Delta0, Gamma)
        # Deltas = sol
        # errors = err
        u = Energys[k]  / sol + 1j * dynes
        if alpha == 0.0:
            Ns = Neff * (u / (u ** 2 - 1) ** 0.5).real
        else:
            Ns = Neff * AG_DOS(Energys[k], sol, alpha)
        N[k] = sum(Ns)

    return N

def IofV(Vs, Delta0, Gamma, Neff, T, alpha=0,dynes = 0):
    energys = numpy.linspace(0, 6, 1000)
    # energys = numpy.linspace(0, 6, 11)
    N = CalculateDOS(energys, Delta0, Gamma, Neff, alpha,dynes)

    energys = numpy.concatenate([numpy.flipud(-energys), energys])
    N = numpy.concatenate([numpy.flipud(N), N])
    N = N / N[0]

    Is = numpy.ndarray(len(Vs))

    FDS = FermiDirac(energys, T)
    for k in range(len(Vs)):
        FDN = FermiDirac(energys - Vs[k], T)
        Is[k] = sum(N * (FDN - FDS))

    slope = (sum(FermiDirac(energys - Vs[len(Vs) - 1], T) - FDS) - sum(FermiDirac(energys - Vs[0], T) - FDS)) / (
    Vs[len(Vs) - 1] - Vs[0])

    return Is / slope

