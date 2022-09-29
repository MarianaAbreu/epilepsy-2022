# Python script to extract second-level features from feature segments of 1 hour
# Created by: Mariana Abreu
# Last update: 27-09-2022

import numpy as np
import pandas as pd
from scipy.stats import skew


def statistics_(sig):
    # minimum
    min_sig = np.min(sig)
    # min max amplitude
    minmax = np.max(sig) - np.min(sig)
    # derivative
    deriv = minmax / (np.argmax(sig) - np.argmin(sig))
    if np.argmax(sig) < np.argmin(sig):
        deriv *= -1
    skewness = skew(sig)
    # median of seconds half minus median of first half if sig duration > 20 min
    mid_len = len(sig)//2
    trend_ratio = np.median(sig[mid_len:]) / np.median(sig[:mid_len])
    trend_diff = np.median(sig[mid_len:]) - np.median(sig[:mid_len])

    return {'min': np.min(sig), 'minmax': minmax, 'deriv': deriv, 'skewness': skewness, 'trend_ratio': trend_ratio,
            'trend_diff': trend_diff}


def get_level2(patient, list_features=None, type='seizure'):
    # calculate level2 features for each feature
    if list_features is None:
        list_features = ['meanrr', 'rmssd', 'nn50', 'pnn50', 'sdnn', 'hti', 'tinn', 'vlf_pwr', 'vlf_peak', 'vlf_rpwr',
                     'lf_pwr',
                     'lf_peak', 'lf_rpwr', 'lfnupwr', 'hf_pwr', 'hf_peak', 'hf_rpwr', 'hfnupwr', 'vhf_pwr', 'vhf_peak',
                     'vhf_rpwr', 'lf_hf', 's', 'sd1', 'sd2', 'sd12', 'sd21', 'hrmin', 'hrmax', 'hrminmax', 'hravg']

    pat1_seizures = pd.read_parquet(f'..\data\clean_features_p{patient}_{type}.parquet')
    pat1_level2features = pd.DataFrame()

    for seizure in sorted(set(pat1_seizures['label'])):
        seiz_table = pat1_seizures.loc[pat1_seizures['label']==seizure]
        for feat in list_features:
            statistics = statistics_(seiz_table[feat].values)
            statistics['label'] = seizure
            statistics['len'] = len(seiz_table)
            statistics['feature'] = feat
            pat1_level2features = pd.concat((pat1_level2features, pd.DataFrame(statistics, index=[0])), ignore_index=True)
    pat1_level2features.to_parquet(f'..\data\level2_features_p{patient}_{type}.parquet')


for patient in ['312', '386', '391', '365', '326', '400', '413', '358', '352']:
    for type_ in ['seizure', 'baseline']:
        try:
            get_level2(patient, type=type_)
        except Exception as e:
            print(e)
