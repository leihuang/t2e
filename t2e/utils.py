"""
"""

from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import fsolve


def simulate_times_h0(func_h0, r, seed=0, **kwargs):
    """

    :param func_h0:
    :param r:
    :param seed:
    :param kwargs:
    :return:
    """
    np.random.seed(seed)
    p = np.random.rand(len(r))

    if 'x0' not in kwargs:
        kwargs['x0'] = [1e-3]
    if 'maxfev' not in kwargs:
        kwargs['maxfev'] = 10000
    if 'xtol' not in kwargs:
        kwargs['xtol'] = 1e-6

    func_h = lambda t: r_ * func_h0(t)
    func_H0 = lambda t: quad(func_h0, 0, t, epsabs=1e-6, epsrel=1e-6)[0]

    t, d = [], []
    for r_, p_ in zip(r, p):
        g = lambda t: r_ * func_H0(t) + np.log(p_)
        t_ = fsolve(g, fprime=func_h, **kwargs)[0]
        t.append(t_)
        d.append(g(t_))
    return np.array(t), np.array(d)


def simulate_times_H0(func_H0, r, func_h0=None, seed=0, **kwargs):
    """

    :param func_H0:
    :param r:
    :param func_h0:
    :param seed:
    :param kwargs:
    :return:
    """
    np.random.seed(seed)
    p = np.random.rand(len(r))

    if 'x0' not in kwargs:
        kwargs['x0'] = [1e-3]
    if 'maxfev' not in kwargs:
        kwargs['maxfev'] = 10000
    if 'xtol' not in kwargs:
        kwargs['xtol'] = 1e-6

    t, d = [], []
    for r_, p_ in zip(r, p):
        g = lambda t: r_ * func_H0(t) + np.log(p_)
        if func_h0 is not None:
            func_h = lambda t: r_ * func_h0(t)
            t_ = fsolve(g, fprime=func_h, **kwargs)[0]
        else:
            t_ = fsolve(g, fprime=None, **kwargs)[0]
        t.append(t_)
        d.append(g(t_))
    return np.array(t), np.array(d)


def simulate_times_iH0(func_iH0, r, seed=0):
    """Simulate event times using inverse baseline cumulative hazard function.

    :param func_iH0:
    :param r:
    :param seed:
    :return:
    """
    np.random.seed(seed)
    p = np.random.rand(len(r))
    return np.array([func_iH0(-np.log(p_)/r_) for r_, p_ in zip(r, p)])


def get_cdf(x):
    """

    :param x:
    :return:
    """
    y = np.array([i/len(x) for i in range(len(x))])
    return np.sort(x), y


def get_censored_data(times_event, times_censoring):
    """

    :param times_event:
    :param times_censoring:
    :return:
    """
    t = np.min(np.array([times_event, times_censoring]), axis=0)
    s = np.array(np.array(times_event) < np.array(times_censoring), dtype='int')
    return pd.DataFrame([t, s], index=['time', 'status']).T

