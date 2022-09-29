# Python script to take segments and perform smooothing, standarisation and interpolation, in order to return
# cleaner signals
# Created by: Mariana Abreu
# Last update: 27-09-2022

import os

import biosppy as bp
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def standardise(sig):
    # signal standarisation
    return (sig - np.mean(sig)) / np.std(sig)


def smooth(sig):
    # signal smoothing
    return bp.signals.tools.smoother(sig, size=10)['signal']


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


def clean_features(patient, type='seizure'):
    # select patient features
    file = pd.read_parquet(os.path.join('C:\\Users\\Mariana', 'PycharmProjects', 'MapME', 'data',
                           f'features_p{patient}_{type}.parquet'))
    # separate into segments
    if type == 'seizure':
        list_labels = sorted(set(file['label']))
        segments = [file.loc[file['label'] == label] for label in list_labels]
    else:
        range_time = file['t0'].iloc[::120]
        segments = [file.loc[file['t0'].between(range_time.iloc[i], range_time.iloc[i+1])] for i in range(len(range_time)-1)]

    list_features = [feat for feat in file.columns if feat not in ['ulf_peak', 'ulf_pwr', 'ulf_rpwr', 'label', 't0',
                                                                   't1', 'dfa_a1', 'dfa_a2', 'sampen', 'd2']]
    # clean one feature
    i = 0
    clean_file = pd.DataFrame()
    for segment in segments:
        clean_seg = clean_segment(segment, list_features)
        if type == 'baseline':
            clean_seg['label'] = ['baseline' + str(i)] * len(clean_seg)
        clean_file = pd.concat((clean_file, clean_seg), ignore_index=True)
        i += 1
    clean_file = clean_file.drop_duplicates()
    clean_file.to_parquet(os.path.join('C:\\Users\\Mariana', 'PycharmProjects', 'MapME', 'data',
                                       f'clean_features_p{patient}_{type}.parquet'))
    # convert time to time until seizure


for patient in ['400', '413', '386', '365', '312', '326', '352', '358']:
    clean_features(patient, type='baseline')
