# -*- coding: utf-8 -*-
"""
biosppy.hrv
-------------------

This module provides computation and visualization of Heart-Rate Variability
metrics.
 

:copyright: (c) 2015-2018 by Instituto de Telecomunicacoes
:license: BSD 3-clause, see LICENSE for more details.

"""

# Imports 
# compat
from __future__ import absolute_import, division, print_function

# 3rd party
import numpy as np
from scipy.signal import welch
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
import warnings

# local
from biosppy import utils
from biosppy.signals import tools as st

# Global variables
FBANDS = {'ulf': [0, 0.003],
          'vlf': [0.003, 0.04],
          'lf': [0.04, 0.15],
          'hf': [0.15, 0.4],
          'vhf': [0.4, 0.5]
          }


def hrv(rpeaks=None, sampling_rate=1000., rri=None, parameters='auto',
        binsize=1 / 128, freq_method='FFT', show=False):
    """ Returns an HRV report with the chosen features, along with informative plots.
     RR-intervals and NN-intervals are used interchangeably.

    Parameters
    ----------
    rpeaks : array
        R-peak index locations.
    sampling_rate : int, float, optional
        Sampling frequency (Hz). Default: 1000.0 Hz.
    rri : array, optional
        RR-intervals (ms). Providing this parameter overrides the computation of
        RR-intervals from rpeaks.
    parameters : str, optional
        If 'auto' computes the recommended HRV features. If 'time' computes 
        only time-domain features. If 'frequency' computes only 
        frequency-domain features. If 'non-linear' computes only non-linear 
        features. If 'all' computes all available HRV features. Default: 'auto'.
    binsize : float, optional
        Binsize for RRI histogram (s). Default: 1/128 s.
    freq_method : str, optional
        Method for spectral estimation. If 'FFT' uses Welch's method. If 'AR' 
        uses the autoregressive method. Default: 'FFT'.
    show : bool, optional
        If True, show a summary plot. Default: False.
    Returns
    -------
    rri : array
        RR-intervals (ms).
    hr : array
        Heart rate (bpm).
    hr_min : float
        Minimum heart rate (bpm).
    hr_max : float
        Maximum heart rate (bpm).
    hr_minmax :  float
        Difference between the highest and the lowest heart rate (bpm).
    hr_avg : float
        Average heart rate (bpm).
    rmssd : float
        RMSSD - Root mean square of successive RR interval differences (ms).
    nn50 : int
        NN50 - Number of successive RR intervals that differ by more than 50ms.
    pnn50 : float
        pNN50 - Percentage of successive RR intervals that differ by more than 
        50ms.
    sdnn: float
       SDNN - Standard deviation of RR intervals (ms).
    hti : float
        HTI - HRV triangular index - Integral of the density of the RR interval 
        histogram divided by its height.
    tinn : float
        TINN - Baseline width of RR interval histogram (ms).
    sdann : float
        SDANN - Standard deviation of the average RR intervals for each 5 min
        segment of a 24 h HRV recording (ms).
    sdnni : float
        SDNN Index - Mean of the standard deviations of all the RR intervals
        for each 5 min segment of a 24 h HRV recording (ms).
    lf_peak : float
        Peak frequency of the low-frequency band (Hz).
    lf_pwr : float
        Absolute power of the low-frequency band (ms^2).
    lf_rpwr : float
        Relative power of the low-frequency band (nu).
    hf_peak :  float
        Peak frequency of the high-frequency band (Hz).
    hf_pwr : float
        Absolute power of the high-frequency band (ms^2).
    hf_rpwr : float
        Relative power of the high-frequency band (nu).
    lf_hf : float
        Ratio of low-frequency band to high-frequency band power.
    vlf_pwr : float
        Absolute power of the very-low-frequency band (ms^2).
    ulf_pwr : float
        Absolute power of the ultra-low-frequency band (ms^2).
    s : float
        S - Area of the ellipse of the Poincaré plot (ms^2).
    sd1 : float
        SD1 - Poincaré plot standard deviation perpendicular to the identity
        line (ms).
    sd2 : float
        SD2 - Poincaré plot standard deviation along the identity line (ms).
    sd12 : float
        SD1/SD2 - SD1 to SD2 ratio.
    sampen : float
        SampEn - Sample entropy of the RRI time series.
    dfa_a1 : float
        DFA alpha-1 - Short-term fractal exponent of the Detrended Fluctuation
        Analysis (DFA).
    dfa_a2 : float
        DFA alpha-2 - Long-term fractal exponent of the Detrended Fluctuation
        Analysis (DFA)
    d2 : float
        D2 - Correlation Dimension (CD).
    """

    # check inputs
    if rpeaks is None and rri is None:
        raise IOError("Please specify an R-Peak or RRI list or array.")

    parameters_list = ['auto', 'time', 'frequency', 'non-linear', 'all']
    if parameters not in parameters_list:
        raise IOError(f"'{parameters}' is not an available input. Enter one from: {parameters_list}.")

    # ensure input format
    sampling_rate = float(sampling_rate)

    # initialize outputs
    args, names = (), ()

    # compute RRIs
    if rri is None:
        rpeaks = np.array(rpeaks, dtype=float)
        rri = compute_rri(rpeaks=rpeaks, sampling_rate=sampling_rate)

    # compute rpeaks from rri
    if rpeaks is None:
        rpeaks = np.cumsum(rri)
        sampling_rate = 1000.  # because rri is in ms

    # compute duration
    duration = np.sum(rri) / 1000.  # seconds

    args = args + (rri,)
    names = names + ('rri',)

    if parameters == 'time':
        return hrv_timedomain(rri=rri, duration=duration, binsize=binsize, show=show)

    if parameters == 'frequency':
        return hrv_frequencydomain(rri=rri, duration=duration, freq_method=freq_method, show=show)

    if parameters == 'non-linear':
        return hrv_nonlinear(rri=rri, duration=duration, show=show)

    if parameters == 'auto' or parameters == 'all':

        if parameters == 'all':
            duration = np.inf

        if duration < 10:
            raise IOError("Signal must be longer than 10 seconds.")

        if duration >= 10:
            # compute HR and HR Max - HR Min
            hr_idx, hr = st.get_heart_rate(rpeaks, sampling_rate)
            hr_vals = hr_metrics(hr)

            args = args + (hr)
            args += tuple(hr_vals)
            names = names + ('hr', 'hr_min', 'hr_max', 'hr_minmax', 'hr_avg')

            if show:
                plot_hr(hr_idx=hr_idx, hr=hr, sampling_rate=sampling_rate)

            # compute time-domain features
            hrv_td = hrv_timedomain(rri=rri, duration=duration,
                                    binsize=binsize, show=show)

            args = args + tuple(hrv_td)
            names = names + tuple(hrv_td.keys())

        if duration >= 20:
            # compute frequency-domain features
            hrv_fd = hrv_frequencydomain(rri=rri, duration=duration,
                                         freq_method=freq_method, show=show)

            args = args + tuple(hrv_fd)
            names = names + tuple(hrv_fd.keys())

        if duration >= 90:
            # compute non-linear features
            hrv_nl = hrv_nonlinear(rri=rri, duration=duration, show=show)

            args = args + tuple(hrv_nl)
            names = names + tuple(hrv_nl.keys())

    return utils.ReturnTuple(args, names)


def hr_metrics(hr):
    """Computes hr metrics given a hr signal

    :param hr: signal in beats per minute
    :return: min, max, minmax and avg
    """
    hr_min = hr.min()
    hr_max = hr.max()
    hr_minmax = hr.max() - hr.min()
    hr_avg = hr.mean()

    args = [hr_min, hr_max, hr_minmax, hr_avg]
    names = ['hrmin', 'hrmax', 'hrminmax', 'hravg']

    return utils.ReturnTuple(args, names)

def compute_rri(rpeaks=None, sampling_rate=1000.):
    """ Computes RR intervals from a list of R-peaks.

    Parameters
    ----------
    rpeaks : array
        R-peak index locations.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).

    Returns
    -------
    rri : array
        RR-intervals (ms).
    """

    # difference of R peaks converted to ms
    rri = (1000. * np.diff(rpeaks)) / sampling_rate

    # check if rri is within physiological parameters
    if rri.min() < 400 or rri.min() > 1400:
        warnings.warn("RR-intervals appear to be out of normal parameters. Check input values.")

    return rri


def filter_rri(rri=None, threshold=1200):
    """

    Parameters
    ----------
    rri : array
        RR-intervals (default: ms).
    threshold : int, float, optional
        Maximum rri value to accept (ms).
    """

    rri_filt = rri[np.where(rri < threshold)]

    return rri_filt


def hrv_timedomain(rri=None, duration=None, binsize=1 / 128, show=False):
    """

    Parameters
    ----------
    rri : array
        RR-intervals (ms).
    duration : int, optional
        Duration of the signal (s).
    binsize :  float, optional
        Binsize for RRI histogram (s). Default: 1/128 s.
    show : bool, optional
        If True, show a summary plot.

    Returns
    -------
    rmssd : float
        RMSSD - Root mean square of successive RR interval differences (ms).
    nn50 : int
        NN50 - Number of successive RR intervals that differ by more than 50ms.
    pnn50 : float
        pNN50 - Percentage of successive RR intervals that differ by more than 
        50ms.
    hti : float
        HTI - HRV triangular index - Integral of the density of the RR interval 
        histogram divided by its height.
    tinn : float
        TINN - Baseline width of RR interval histogram (ms).
    sdann : float
        SDANN - Standard deviation of the average NN intervals for each 5 min 
        segment of a 24 h HRV recording (ms).
    sdnni : float
        SDNN Index - Mean of the standard deviations of all the NN intervals 
        for each 5 min segment of a 24 h HRV recording (ms).
    """

    # check inputs
    if rri is None:
        raise TypeError("Please specify an RRI list or array.")

    # ensure numpy
    rri = np.array(rri, dtype=float)

    if duration is None:
        duration = np.sum(rri) / 1000.  # seconds

    if duration < 10:
        raise IOError("Signal must be longer than 10 seconds.")

    # initialize outputs
    args, names = (), ()

    # compute the difference between RRIs
    rri_diff = np.diff(rri)
    rri_mean = rri.mean()
    args = args + (rri_mean,)
    names = names + ('meanrr',)

    if duration >= 10:
        # compute RMSSD
        rmssd = (rri_diff ** 2).mean() ** 0.5

        args = args + (rmssd,)
        names = names + ('rmssd',)

    if duration >= 20:
        # since rri is in ms th50 = 50ms
        th50 = 50

        # compute NN50 and pNN50
        nntot = len(rri_diff)
        nn50 = len(np.argwhere(abs(rri_diff) > th50))
        pnn50 = 100 * (nn50 / nntot)

        args = args + (nn50, pnn50)
        names = names + ('nn50', 'pnn50')

    if duration >= 60:
        # compute SDNN
        sdnn = rri.std()

        args = args + (sdnn,)
        names = names + ('sdnn',)

    if duration >= 90:
        # compute geometrical features (histogram)
        hti, tinn = compute_geometrical(rri=rri, binsize=binsize, show=show)

        args = args + (hti, tinn)
        names = names + ('hti', 'tinn')

    if duration >= 86400:
        # TODO: compute SDANN and SDNN Index
        sdann, sdnni = None, None

        args = args + (sdann, sdnni)
        names = names + ('sdann', 'sdnni')

    return utils.ReturnTuple(args, names)


def hrv_frequencydomain(rri=None, duration=None, freq_method='FFT', fbands=None, show=False):
    """

    Parameters
    ----------
    rri : array
        RR-intervals (ms).
    duration : int, optional
        Duration of the signal (s).
    freq_method : str, optional
        Method for spectral estimation. If 'FFT' uses Welch's method. If 'AR' 
        uses the autoregressive method.
    fbands : dict, optional
        Dictionary specifying the desired HRV frequency bands.
    show : bool, optional
        If True, show a summary plot.

    Returns
    -------
    vlf_peak : float
        Peak frequency (Hz) of the very-low-frequency band (0.0033–0.04 Hz) in
        normal units.
    vlf_pwr : float
        Relative power of the very-low-frequency band (0.0033–0.04 Hz) in 
        normal units.
    lf_peak : float
        Peak frequency (Hz) of the low-frequency band (0.04–0.15 Hz).
    lf_pwr : float
        Relative power of the low-frequency band (0.04–0.15 Hz) in normal 
        units.
    hf_peak : float
        Peak frequency (Hz)  of the high-frequency band (0.15–0.4 Hz).
    hf_pwr : float
        Relative power of the high-frequency band (0.15–0.4 Hz) in normal 
        units.
    lf_hf : float
        Ratio of LF-to-HF power.
    total_pwr : float
        Total power.
    """

    # check inputs
    if rri is None:
        raise TypeError("Please specify an RRI list or array.")

    freq_methods = ['FFT', 'AR']
    if freq_method not in freq_methods:
        raise IOError(f"'{freq_method}' is not an available input. Enter one from: {freq_methods}.")

    if fbands is None:
        fbands = FBANDS

    # ensure numpy
    rri = np.array(rri, dtype=float)

    # ensure minimal duration
    if duration is None:
        duration = np.sum(rri) / 1000.  # seconds

    if duration < 20:
        raise IOError("Signal must be longer than 20 seconds.")

    # initialize outputs
    args, names = (), ()

    if duration >= 20:

        # compute frequencies and powers
        if freq_method == 'FFT':
            frequencies, powers = welch(rri, fs=1., scaling='density')

        if freq_method == 'AR':
            # TODO: develop AR method
            print('AR method not available. Using FFT instead.')
            return hrv_frequencydomain(rri=rri, duration=duration, freq_method='FFT')

        powers = powers / 1000. ** 2  # to ms^2/Hz

        # compute frequency bands
        fb_out = compute_fbands(frequencies=frequencies, powers=powers, method_name=freq_method, show=False)

        args = args + tuple(fb_out)
        names = names + tuple(fb_out.keys())

        # compute LF/HF ratio
        lf_hf = fb_out['lf_pwr'] / fb_out['hf_pwr']

        args += (lf_hf,)
        names += ('lf_hf',)

        # plot
        if show:
            plot_fbands(frequencies, powers, fbands, freq_method, LF_HF=lf_hf)

    # if duration >= 270:
    #     args = args + (fb_out['vlf_pwr'],)
    #     names = names + ('vlf_pwr',)
    #
    # if duration >= 86400:
    #     args = args + (fb_out['ulf_pwr'],)
    #     names = names + ('ulf_pwr',)

    return utils.ReturnTuple(args, names)


def hrv_nonlinear(rri=None, duration=None, show=False):
    """

    Parameters
    ----------
    rri : array
        RR-intervals (ms).
    duration : int, optional
        Duration of the signal (s).
    show : bool, optional
        If True, show a summary plot.

    Returns
    -------
    s : float
        S - Area of the ellipse of the Poincaré plot (ms^2).
    sd1 : float
        SD1 - Poincaré plot standard deviation perpendicular to the identity 
        line (ms).
    sd2 : float
        SD2 - Poincaré plot standard deviation along the identity line (ms).
    sd12 : float
        SD1/SD2 - SD1 to SD2 ratio.
    sampen : float
        SampEn - Sample entropy of the RRI time series.
    dfa_a1 : float
        DFA alpha-1 - Short-term fractal exponent of the Detrended Fluctuation 
        Analysis (DFA).
    dfa_a2 : float
        DFA alpha-2 - Long-term fractal exponent of the Detrended Fluctuation 
        Analysis (DFA)
    d2 : float
        D2 - Correlation Dimension (CD).
    """

    if rri is None:
        raise TypeError("Please specify an RRI list or array.")

    # ensure numpy
    rri = np.array(rri, dtype=float)

    if duration is None:
        duration = np.sum(rri) / 1000.  # seconds

    if duration < 90:
        raise IOError("Signal duration must be greater than 90 seconds.")

    # initialize outputs
    args, names = (), ()
    rri -= rri.mean()

    if duration >= 90:
        # compute SD1, SD2, SD1/SD2 and S
        s, sd1, sd2, sd12 = compute_poincare(rri=rri, show=show)
        sd12 = sd1 / sd2
        sd21 = sd2 / sd1

        args = args + (s, sd1, sd2, sd12, sd21)
        names = names + ('s', 'sd1', 'sd2', 'sd12', 'sd21')

    if duration >= 180:
        # TODO: compute SampEn
        sampen = None

        # TODO: perform DFA
        dfa_a1, dfa_a2 = None, None

        args = args + (sampen, dfa_a1, dfa_a2)
        names = names + ('sampen', 'dfa_a1', 'dfa_a2')

    if duration >= 300:
        # TODO: compute D2
        d2 = None

        args = args + (d2,)
        names = names + ('d2',)

    return utils.ReturnTuple(args, names)


# TODO: add documentation
def plot_hr(hr_idx, hr, sampling_rate=1000., plot_stats=True):
    """

    Parameters
    ----------
    hr_idx : array
        Heart rate location indices.
    hr : array
        Instantaneous heart rate (bpm).
    sampling_rate : float
        Sampling rate (Hz).
    plot_stats : bool
        Chose to plot basic statistics in the plot.
    """
    hr_ts = hr_idx / sampling_rate
    hr_mean = hr.mean()
    hrmaxmin = hr.max() - hr.min()

    # plot
    plt.figure()
    plt.title('Instantaneous Heart Rate')
    plt.ylabel('Heart Rate (bpm)')
    plt.xlabel('Time (s)')

    plt.plot(hr_ts, hr, color='#85B3D1FF', label='Heart Rate')

    if plot_stats:
        plt.plot([hr_ts.min(), hr_ts.max()], [hr_mean, hr_mean], linestyle='--',
                 linewidth=1, color='0.2', label='Mean (%.1f bpm)' % hr_mean)
        plt.plot([hr_ts.min(), hr_ts.max()], [hr.min(), hr.min()], linestyle='--',
                 linewidth=1, color='#A13941FF', label='Max-Min (%.1f bpm)' % hrmaxmin)
        plt.plot([hr_ts.min(), hr_ts.max()], [hr.max(), hr.max()], linestyle='--',
                 linewidth=1, color='#A13941FF')

    plt.grid()
    plt.legend(loc='upper right')
    plt.show()


# TODO: add documentation
def compute_poincare(rri, show=False):
    x = rri[:-1]
    y = rri[1:]

    # compute SD1, SD2 and S
    x1 = (x - y) / np.sqrt(2)
    x2 = (x + y) / np.sqrt(2)
    sd1 = x1.std()
    sd2 = x2.std()
    s = np.pi * sd1 * sd2
    sd12 = sd1 / sd2

    # plot
    if show:
        plot_poincare(rri=rri,
                      x=x,
                      y=y,
                      sd1=sd1,
                      sd2=sd2,
                      s=s,
                      sd12=sd12)

    # output
    args = (s, sd1, sd2, sd12)
    names = ('s', 'sd1', 'sd2', 'sd12')

    return utils.ReturnTuple(args, names)


def plot_poincare(rri, x, y, sd1, sd2, s, sd12):

    rr_mean = rri.mean()

    fig, ax = plt.subplots()

    ax.set_title('Poincaré Plot')
    ax.set_xlabel('$RR_i$ (ms)')
    ax.set_ylabel('$RR_{i+1}$ (ms)')

    # plot Poincaré data points
    ax.scatter(x, y, marker='.', color='#85B3D1FF', alpha=0.5, s=100, zorder=1)
    ax.set_xlim([np.min(rri) - 50, np.max(rri) + 50])
    ax.set_ylim([np.min(rri) - 50, np.max(rri) + 50])
    ax.set_aspect(1. / ax.get_data_ratio())

    # draw identity line (RRi+1=RRi)
    lims = [np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()])]  # max of both axes

    ax.plot(lims, lims, linewidth=0.7, color='grey', linestyle='--',
            zorder=2, label='Identity line')

    # draw ellipse
    ellipse = patches.Ellipse((rr_mean, rr_mean), sd1 * 2, sd2 * 2, angle=-45,
                              linewidth=1, edgecolor='#2E5266FF', facecolor='None',
                              label='S = %.1f' % s, zorder=3)
    ax.add_artist(ellipse)

    # draw SD1 and SD2
    ax.arrow(rr_mean, rr_mean, sd1 * np.cos(3 * np.pi / 4), sd1 * np.sin(3 * np.pi / 4),
             facecolor='#A13941FF', edgecolor='#A13941FF', linewidth=2,
             length_includes_head=True, head_width=4, head_length=4,
             label='SD1 = %.1f ms' % sd1, zorder=3)

    ax.arrow(rr_mean, rr_mean, sd2 * np.cos(np.pi / 4), sd2 * np.sin(np.pi / 4),
             facecolor='#DDB65DFF', edgecolor='#DDB65DFF', linewidth=2,
             length_includes_head=True, head_width=4, head_length=4,
             label='SD2 = %.1f ms' % sd2, zorder=3)

    # draw SD1 and SD2 axes
    f = 4
    ax.add_artist(
        lines.Line2D([rr_mean - f * sd1 * np.cos(3 * np.pi / 4), rr_mean + f * sd1 * np.cos(3 * np.pi / 4)],
                     [rr_mean + f * sd1 * np.cos(3 * np.pi / 4), rr_mean - f * sd1 * np.cos(3 * np.pi / 4)],
                     lw=1, color='0.2'))

    ax.add_artist(lines.Line2D([rr_mean - f * sd2 * np.cos(np.pi / 4), rr_mean + f * sd2 * np.cos(np.pi / 4)],
                               [rr_mean - f * sd2 * np.cos(np.pi / 4), rr_mean + f * sd2 * np.cos(np.pi / 4)],
                               lw=1, color='0.2'))

    # add SD1/SD2
    handles, labels = fig.gca().get_legend_handles_labels()
    sd12_patch = patches.Patch(color='white', alpha=0)
    handles.extend([sd12_patch])
    labels.extend(['SD1/SD2 = %.2f' % sd12])

    # change handle for ellipse
    handles[1] = plt.Line2D([], [], color='#2E5266FF', marker="o", markersize=10, linewidth=0,
                            markerfacecolor='none')

    ax.legend(handles=handles, labels=labels, loc='upper right')

    # plot grid
    ax.grid()
    ax.set_axisbelow(True)


# TODO: add documentation
def compute_geometrical(rri, binsize=1 / 128, show=False):
    binsize = binsize * 1000  # to ms

    # create histogram
    tmin = rri.min()
    tmax = rri.max()
    bins = np.arange(tmin, tmax + binsize, binsize)
    nn_hist = np.histogram(rri, bins)

    # histogram peak
    max_count = np.max(nn_hist[0])
    peak_hist = np.argmax(nn_hist[0])

    # compute HTI
    hti = len(rri) / max_count

    # possible N and M values
    n_values = bins[:peak_hist]
    m_values = bins[peak_hist + 1:]

    # find triangle with base N and M that best approximates the distribution
    error_min = np.inf
    n = 0
    m = 0
    q_hist = None

    for n_ in n_values:

        for m_ in m_values:

            t = np.array([tmin, n_, nn_hist[1][peak_hist], m_, tmax + binsize])
            y = np.array([0, 0, max_count, 0, 0])
            q = interp1d(x=t, y=y, kind='linear')
            q = q(bins)

            # compute the sum of squared differences
            error = np.sum((nn_hist[0] - q[:-1]) ** 2)

            if error < error_min:
                error_min = error
                n, m, q_hist = n_, m_, q

    # compute TINN
    tinn = m - n

    # plot
    if show:
        plot_hist(rri=rri,
                  bins=bins,
                  q_hist=q_hist,
                  hti=hti,
                  tinn=tinn)

    # output
    args = (hti, tinn)
    names = ('hti', 'tinn')

    return utils.ReturnTuple(args, names)


def plot_hist(rri, bins, q_hist, hti, tinn):
    fig, ax = plt.subplots()
    ax.hist(rri, bins, facecolor='#85B3D1FF', edgecolor='0.2', label='HTI: %.1f' % hti)
    ax.set_title('RRI Distribution')
    ax.set_xlabel('RR Interval (ms)')
    ax.set_ylabel('Count')
    ax.locator_params(axis='y', integer=True)
    ax.plot(bins, q_hist, color='#A13941FF', linewidth=1.5, label='TINN: %.1f ms' % tinn)
    ax.legend()


# TODO: add documentation
def compute_fbands(frequencies, powers, method_name, fbands=None, show=False):
    """

    Parameters
    ----------
    fbands
    frequencies : array
    powers : array
    method_name : str
    show : bool, optional

    Returns
    -------
    lf_peak : float
        Peak frequency of the low-frequency band (Hz).
    lf_pwr : float
        Absolute power of the low-frequency band (ms^2).
    lf_rpwr : float
        Relative power of the low-frequency band (nu).
    hf_peak :  float
        Peak frequency of the high-frequency band (Hz).
    hf_pwr : float
        Absolute power of the high-frequency band (ms^2).
    hf_rpwr : float
        Relative power of the high-frequency band (nu).
    lf_hf : float
        Ratio of low-frequency band to high-frequency band power.
    vlf_pwr : float
        Absolute power of the very-low-frequency band (ms^2).
    ulf_pwr : float
        Absolute power of the ultra-low-frequency band (ms^2).
    """

    # initialize outputs
    args = tuple()
    names = tuple()
    band_nu = np.argwhere((frequencies > 0.04) & (frequencies < 0.5)).reshape(-1)
    total_pwr = np.sum(powers[band_nu])

    if fbands is None:
        fbands = FBANDS

    # compute power, peak and relative power for each frequency band
    for fband in fbands.keys():
        band = np.argwhere((frequencies > fbands[fband][0]) & (frequencies < fbands[fband][-1])).reshape(-1)
        # check if it's possible to compute the frequency band
        if len(band) == 0:
            continue
        pwr = np.sum(powers[band])
        peak = frequencies[band][np.argmax(powers[band])]
        rpwr = 100 * (pwr / total_pwr)

        args += (pwr, peak, rpwr)
        names += (fband + '_pwr', fband + '_peak', fband + '_rpwr')

    if show:
        plot_fbands(frequencies=frequencies,
                    powers=powers,
                    fbands=fbands,
                    method_name=method_name)

    return utils.ReturnTuple(args, names)


def plot_fbands(frequencies, powers, fbands=None, method_name=None, yscale='linear', **legends):
    if fbands is None:
        fbands = FBANDS

    fig, ax = plt.subplots()

    # figure attributes
    if method_name is None:
        ax.set_title(f'Power Spectral Density')
    else:
        ax.set_title(f'Power Spectral Density ({method_name})')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power (ms$^2$/Hz)')
    ax.set_yscale(yscale)

    # plot spectrum
    ax.plot(frequencies, powers, linewidth=1, color='0.2', label='_nolegend_')
    ax.margins(0)

    # plot frequency bands
    cmap_ = plt.get_cmap('Set2')
    cmap = iter(list(map(lambda i: cmap_(i), range(len(fbands.keys())))))

    for fband in fbands.keys():
        band = np.argwhere((frequencies > fbands[fband][0]) & (frequencies < fbands[fband][-1])).reshape(-1)
        color = next(cmap)
        if len(band) > 0:
            ax.fill_between(frequencies[band], powers[band], color=color, label=fband.upper())

    # update figure legend
    handles, labels = fig.gca().get_legend_handles_labels()
    if legends.__len__() != 0:
        for key, value in legends.items():
            new_patch = patches.Patch(color='white', alpha=0)
            handles.extend([new_patch])
            labels.extend(['%s = %.2f' % (key, value)])
    ax.legend(handles=handles, labels=labels, loc='upper right')
