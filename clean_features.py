# Python script to take segments and perform smooothing, standarisation and interpolation, in order to return
# cleaner signals
# Created by: Mariana Abreu
# Last update: 27-09-2022

import biosppy as bp
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def standardise(sig):
    # signal standarisation
    return (sig - np.mean(sig)) / np.std(sig)


def smooth(sig):
    # signal smoothing
    return bp.signals.tools.smoother(sig, size=6)['signal']


def clean_segment(segment, list_features):
    # signal interpolation, standarisation and smoothing
    # processes all features
    clean_segment = pd.DataFrame()
    xrange = (segment['t0'] - segment['t0'].iloc[0]).astype('timedelta64[s]')
    xnew = np.arange(0, xrange.iloc[-1], int(xrange.iloc[-1] / 100))

    for feat in list_features:
        f = interp1d(x=xrange, y=smooth(standardise(segment[feat].values)), kind='quadratic')
        ynew = f(xnew)
        clean_segment[feat] = ynew
    clean_segment['label'] = segment['label'].iloc[0]
    clean_segment['time2sz'] = xnew - xnew[-1]

    return clean_segment


file = pd.read_parquet('data\\features_p386_seizure.parquet')
# separate into segments
segments = [file.loc[file['label'] == seizure] for seizure in sorted(set(file['label']))]
# clean one feature
segment = segments[0]
list_features = [feat for feat in segment if feat not in ['label', 't0', 't1', 'dfa_a1', 'dfa_a2', 'sampen', 'd2']]
clean_seg = clean_segment(segments[0], list_features)
# convert time to time until seizure
