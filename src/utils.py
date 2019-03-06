"""
"""

from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import fsolve


def simulate_times(func_h0, r, seed=0, **kwargs):
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

    t, d = [], []
    for r_, p_ in zip(r, p):
        func_h = lambda t: func_h0(t) * r_
        def func_H(t):
            return quad(func_h, 0, t, epsabs=1e-6, epsrel=1e-6)[0]
        g = lambda t: func_H(t) + np.log(p_)
        t_ = fsolve(g, fprime=func_h, **kwargs)[0]
        t.append(t_)
        d.append(g(t_))
    return np.array(t), np.array(d)