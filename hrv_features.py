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

# local
from biosppy import utils
from biosppy.signals import tools as st


# TODO: complete documentation
def hrv(rpeaks=None, sampling_rate=1000., rri=None, duration=None, parameters='auto',
        binsize=1/128, freq_method='FFT', show=False):
    """ Returns an HRV report.

    Parameters
    ----------
    rpeaks : array
        R-peak index locations.
    rri : array, optional
        RR-intervals (ms).
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    duration : int, float, optional
        Duration of the signal (s).
    parameters : str, optional
        If 'auto' computes the recommended HRV features. If 'all' computes all
        available HRV features.
    binsize : float, optional
        Binsize for RRI histogram (s). Default: 1/128 s.
    freq_method : str, optional
        Method for spectral estimation. If 'FFT' uses Welch's method. If 'AR' 
        uses the autoregressive method.
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
    if rpeaks is None and rri is None:
        raise IOError("Please specify an R-Peak or RRI list or array.")

    if parameters != 'auto' and parameters != 'all':
        raise IOError(f"'{parameters}' is not an available input. Enter 'auto' or 'all'.")

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
        sampling_rate = 1000. # because rri is in ms
        
    args = args + (rri,)
    names = names + ('rri',)

    if parameters == 'auto':

        if duration is None:
            duration = np.sum(rri) / 1000.  # seconds

        if duration < 10:
            raise IOError("Signal must be longer than 10 seconds.")

        if duration >= 10:
            # compute HR and HR Max - HR Min
            hr_idx, hr = st.get_heart_rate(rpeaks, sampling_rate)
            hrminmax = hr.max() - hr.min()

            args = args + (hr, hrminmax)
            names = names + ('hr', 'hrminmax')

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

    if parameters == 'all':
        return hrv(rpeaks=rpeaks, sampling_rate=sampling_rate,
                   parameters='auto', duration=np.inf, show=show)

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

    # TODO : check values below
    # check if rri is within physiological parameters 
    if rri.min() < 400 or rri.min() > 1400:
        raise Warning("RR-intervals appear to be out of normal parameters. Check input values.")

    return rri


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


def hrv_frequencydomain(rri=None, duration=None, freq_method='FFT', show=False):
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

    if rri is None:
        raise TypeError("Please specify an RRI list or array.")

    # ensure numpy
    rri = np.array(rri, dtype=float)

    if duration is None:
        duration = np.sum(rri) / 1000.  # seconds

    if duration < 20:
        raise IOError("Signal must be longer than 20 seconds.")

    # initialize outputs
    args, names = (), ()

    if duration >= 20:

        if freq_method == 'FFT':
            frequencies, powers = welch(rri, fs=1., scaling='density')

        if freq_method == 'AR':
            # TODO: develop AR method
            print('AR method not available. Using FFT instead.')
            return hrv_frequencydomain(rri=rri, duration=duration, method='FFT')

        fout = compute_fbands(frequencies=frequencies, powers=powers,
                              method_name=freq_method, show=show)

        args = args + (fout['lf_peak'], fout['lf_pwr'], fout['lf_rpwr'],
                       fout['hf_peak'], fout['hf_pwr'], fout['hf_rpwr'],
                       fout['vhf_pwr'], fout['lf_hf'])
        names = names + ('lf_peak', 'lf_pwr', 'lf_rpwr', 'hf_peak', 'hf_pwr',
                         'hf_rpwr', 'vhf_pwr', 'lf_hf')

    if duration >= 270:
        args = args + (fout['vlf_pwr'],)
        names = names + ('vlf_pwr',)

    if duration >= 86400:
        args = args + (fout['ulf_pwr'],)
        names = names + ('ulf_pwr',)

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
        SampEn - Sample Entropy of the RR-intervals.
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

    if duration >= 90:
        # compute SD1, SD2, SD1/SD2 and S
        s, sd1, sd2, sd12 = compute_poincare(rri=rri, show=show)
        sd12 = sd1 / sd2

        args = args + (s, sd1, sd2, sd12)
        names = names + ('s', 'sd1', 'sd2', 'sd12')

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
        rr_mean = np.mean(rri)

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

    # output
    args = (s, sd1, sd2, sd12)
    names = ('s', 'sd1', 'sd2', 'sd12')

    return utils.ReturnTuple(args, names)


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
        fig, ax = plt.subplots()
        ax.hist(rri, bins, facecolor='#85B3D1FF', edgecolor='0.2', label='HTI: %.1f' % hti)
        ax.set_title('RRI Distribution')
        ax.set_xlabel('RR Interval (ms)')
        ax.set_ylabel('Count')
        ax.locator_params(axis='y', integer=True)
        ax.plot(bins, q_hist, color='#A13941FF', linewidth=1.5, label='TINN: %.1f ms' % tinn)
        ax.legend()

    # output
    args = (hti, tinn)
    names = ('hti', 'tinn')

    return utils.ReturnTuple(args, names)


# TODO: add documentation
def compute_fbands(frequencies, powers, method_name, show=False):
    powers = powers / 1000. ** 2  # to ms^2/Hz
    total_pwr = np.sum(powers)

    # ULF band
    ulf_band = np.argwhere((frequencies > 0) & (frequencies < 0.003)).reshape(-1)
    ulf_pwr = np.sum(powers[ulf_band])
    ulf_rpwr = ulf_pwr / total_pwr

    # VLF band
    vlf_band = np.argwhere((frequencies > 0.003) & (frequencies < 0.04)).reshape(-1)
    vlf_pwr = np.sum(powers[vlf_band])
    vlf_rpwr = vlf_pwr / total_pwr

    # LF band
    lf_band = np.argwhere((frequencies > 0.04) & (frequencies < 0.15)).reshape(-1)
    lf_peak = frequencies[lf_band][np.argmax(powers[lf_band])]
    lf_pwr = np.sum(powers[lf_band])
    lf_rpwr = lf_pwr / total_pwr

    # HF band
    hf_band = np.argwhere((frequencies > 0.15) & (frequencies < 0.4)).reshape(-1)
    hf_peak = frequencies[hf_band][np.argmax(powers[hf_band])]
    hf_pwr = np.sum(powers[hf_band])
    hf_rpwr = hf_pwr / total_pwr

    # VHF band
    vhf_band = np.argwhere((frequencies > 0.4) & (frequencies <= 0.5)).reshape(-1)
    vhf_pwr = np.sum(powers[vhf_band])
    vhf_rpwr = vhf_pwr / total_pwr

    # compute LF/HF ratio
    lf_hf = lf_pwr / hf_pwr

    if show:
        fig, ax = plt.subplots()

        ax.set_title(f'Power Spectral Density ({method_name})')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power (ms$^2$/Hz)')

        # plot spectrum
        ax.plot(frequencies, powers, linewidth=1, color='0.2', label='_nolegend_')
        ax.set_xlim([0, 0.5])
        ax.set_ylim([0, np.max(ax.get_ylim())])

        # plot ULF, VLF, LF, HF and VHF
        cmap = plt.get_cmap('Set2')

        ax.fill_between(frequencies[ulf_band], powers[ulf_band], color=cmap(0), label='ULF (%.2f)' % ulf_rpwr)
        ax.fill_between(frequencies[vlf_band], powers[vlf_band], color=cmap(1), label='VLF (%.2f)' % vlf_rpwr)
        ax.fill_between(frequencies[lf_band], powers[lf_band], color=cmap(2), label='LF (%.2f)' % lf_rpwr)
        ax.fill_between(frequencies[hf_band], powers[hf_band], color=cmap(3), label='HF (%.2f)' % hf_rpwr)
        ax.fill_between(frequencies[vhf_band], powers[vhf_band], color=cmap(4), label='VHF (%.2f)' % vhf_rpwr)

        # add LF/HF legend
        handles, labels = fig.gca().get_legend_handles_labels()
        lfhf_patch = patches.Patch(color='white', alpha=0)
        handles.extend([lfhf_patch])
        labels.extend(['LF/HF = %.2f' % lf_hf])

        ax.legend(handles=handles, labels=labels, loc='upper right')

    # output
    args = (ulf_pwr, vlf_pwr, lf_peak, lf_pwr, lf_rpwr, hf_peak, hf_pwr,
            hf_rpwr, vhf_pwr, lf_hf)
    names = ('ulf_pwr', 'vlf_pwr', 'lf_peak', 'lf_pwr', 'lf_rpwr', 'hf_peak',
             'hf_pwr', 'hf_rpwr', 'vhf_pwr', 'lf_hf')

    return utils.ReturnTuple(args, names)
