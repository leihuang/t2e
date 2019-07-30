"""
"""

from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

from lifelines import KaplanMeierFitter



def simulate_times_h0(func_h0, r, seed=None, **kwargs):
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


def simulate_times_H0(func_H0, r, func_h0=None, seed=None, **kwargs):
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


def simulate_times_iH0(func_iH0, r, seed=None):
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


def get_calibration(probs_pred, times, status, bins=5):
    """

    :param probs_pred:
    :param times:
    :param status:
    :return:
    """
    probs_pred = pd.Series(probs_pred)


    kmf = KaplanMeierFitter()
    kmf.fit(times, status)




def get_calibration2(probs, times, events, year, bins=5, return_cut=False):
    """Compute calibration results using Kaplan-Meier estimates as observed probabilities.
    The function is different from `get_calibration` in that it uses `pd.qcut`. 

    :param probs: a sequence of predicted probabilities
    :param times: event times
    :param events: booleans indicating censorship
    :param year: the year for which calibration is calculated
    :param bins: integer or a pd.IntervalIndex object 
      (eg, the `categories` attribute of the object returned by `pd.qcut`)
    :return: two pd.DataFrames for predicted and observed probabilities with standard errors
    """
    def _expand_ends(interval_index):
        first = pd.Interval(-0.01, interval_index[0].right, closed='right')
        last = pd.Interval(interval_index[-1].left, 1, closed='right')
        return pd.IntervalIndex([first] + list(interval_index)[1:-1] + [last])
    
    probs = pd.Series(probs)
    probs.sort_values(inplace=True)
    
    # the returned `cut` is a pd.Series
    if isinstance(bins, pd.IntervalIndex):
        bins = _expand_ends(bins)
        cut = pd.cut(probs, bins=bins)
    else:
        cut = pd.qcut(probs, q=bins)
    
    y = pd.DataFrame({'time':times, 'status':events})

    outs_pred = []
    outs_obs = []
    
    for i, interval in enumerate(cut.cat.categories):
        idxs_bin = cut[cut==interval].index
        if len(idxs_bin) == 0:
            continue
            
        probs_bin = probs.loc[idxs_bin]
        y_bin = y.loc[idxs_bin]
    
        # predicted probability for the bin with standard error
        pred = np.mean(probs_bin)
        se_pred = np.std(probs_bin)
        
        kmf = KaplanMeierFitter().fit(y_bin.time, y_bin.status)
        idx = np.where((kmf.survival_function_.index.values - year) <0)[0][-1]
        #obs = kmf.predict(year)
        obs = kmf.survival_function_.iloc[idx][0]
        
        # get the confidence interval at the nearest time to compute the standard error
        time_ = kmf.timeline[(np.abs(kmf.timeline - year)).argmin()] 
        confint = kmf.confidence_interval_.loc[time_].values 
        se_obs = (confint[0] - confint[1]) / 2 / 1.96

        outs_pred.append((pred, se_pred))
        outs_obs.append((obs, se_obs))

    df_pred = pd.DataFrame(outs_pred, columns=['prob', 'se'])
    df_obs = pd.DataFrame(outs_obs, columns=['prob', 'se'])
    
    if return_cut:
        return df_pred, df_obs, cut
    else:
        return df_pred, df_obs
  

def plot_calibration(df_pred, df_obs, plot_risk=False, plot_xerr=True, plot_yerr=True, 
                     figsize=(4,4), xylim=None, ax=None):
    """Plot calibration plot using the output of `get_calibration`.

    :param df_pred, df_obs: two pandas dataframes outputted from `get_calibration`
    :param plot_risk: a boolean indicating whether to plot risk or survival
    :param plot_xerr, plot_yerr: booleans indicating whether to plot error bars
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, frameon=True)
        show_fig = True
    else:
        show_fig = False
        
    if plot_risk:
        x = 1 - df_pred.prob
        y = 1 - df_obs.prob
        xlabel = 'Predicted risk'
        ylabel = 'Observed risk'
    else:
        x = df_pred.prob
        y = df_obs.prob
        xlabel = 'Predicted survival prob'
        ylabel = 'Observed survival prob'
    if plot_xerr:
        xerr = df_pred.se
    else:
        xerr = None
    if plot_yerr:
        yerr = df_obs.se
    else:
        yerr = None
    ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='.', ms=5)
    ax.plot([-0.05, 1.05], [-0.05, 1.05], '-r')
    if xylim is None:
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
    else:
        ax.set_xlim(*xylim)
        ax.set_ylim(*xylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    if show_fig:
        plt.subplots_adjust(left=0.2, bottom=0.2)
        plt.show()
    else:
        return ax