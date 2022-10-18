# Python script to extract second-level features from feature segments of 1 hour
# Created by: Mariana Abreu
# Last update: 27-09-2022
import os
from datetime import timedelta

import numpy as np
import pandas as pd
from scipy.stats import skew



def statistics_onset_1H(sig_onset, sig_1h, feature):
    # minimum
    # min max amplitude
    minmax_onset = np.max(sig_onset) - np.min(sig_onset)
    # minmax_1h = np.max(sig_1h) - np.min(sig_1h)
    # derivative
    deriv_onset = minmax_onset / (np.argmax(sig_onset) - np.argmin(sig_onset))
    # deriv_1h = minmax_1h / (np.argmax(sig_1h) - np.argmin(sig_1h))

    if np.argmax(sig_onset) < np.argmin(sig_onset):
        deriv_onset *= -1
    skewness = skew(sig_onset)
    # median of seconds half minus median of first half if sig duration > 20 min
    # mid_len = len(sig) // 2
    trend_ratio = np.median(sig_onset) / np.median(sig_1h)
    trend_diff = np.median(sig_onset) - np.median(sig_1h)

    stats = {feature + '-minmax': minmax_onset, feature + '-deriv': deriv_onset, feature + '-skewness': skewness, feature + '-trend_ratio': trend_ratio,
            feature + '-trend_diff': trend_diff}
    return stats


def statistics_(sig, feature):
    # minimum
    # min max amplitude
    minmax = np.max(sig) - np.min(sig)
    # derivative
    deriv = minmax / (np.argmax(sig) - np.argmin(sig))
    if np.argmax(sig) < np.argmin(sig):
        deriv *= -1
    skewness = skew(sig)
    # median of seconds half minus median of first half if sig duration > 20 min
    mid_len = len(sig) // 2
    trend_ratio = np.median(sig[mid_len:]) / np.median(sig[:mid_len])
    trend_diff = np.median(sig[mid_len:]) - np.median(sig[:mid_len])

    stats = {feature + '-minmax': minmax, feature + '-deriv': deriv, feature + '-skewness': skewness, feature + '-trend_ratio': trend_ratio,
            feature + '-trend_diff': trend_diff}
    return stats


def get_level2(patient, list_features=None, type='seizure', save_dir=''):
    # calculate level2 features for each feature
    if list_features is None:
        list_features = ['lf_hf', 'hr_avg']

    pat1_seizures = pd.read_parquet(os.path.join(save_dir, f'features_p{patient}_{type}.parquet'))
    pat1_level2features = pd.DataFrame()
    seizures_csv = pd.read_csv(os.path.join(save_dir, f'seizure_label_p{patient}'))
    seizure_times = pd.to_datetime(seizures_csv['Date'])

    for seizure_idx in range(len(seizures_csv)):
        new_row_array = []
        new_row_names = []
        seizure_label = 'seizure' + seizures_csv['Type'].iloc[seizure_idx]
        seizure_time = seizure_times.iloc[seizure_idx]
        seiz_table = pat1_seizures.loc[pat1_seizures['label'] == seizure_label]
        seiz_table_onset = seiz_table.loc[seiz_table['t0'].between(seizure_time - timedelta(minutes=1),
                                                                   seizure_time + timedelta(minutes=1))]
        for feat in list_features:
            statistics = statistics_onset_1H(seiz_table[feat].values, feat)
            statistics['label'] = seizure_label
            statistics['len'] = len(seiz_table)

            pat1_level2features = pd.concat((pat1_level2features, pd.DataFrame(statistics, index=[0])),
                                            ignore_index=True)
    pat1_level2features.to_parquet(os.path.join(save_dir, f'..\data\level2_features_new_p{patient}_{type}.parquet'))


## example
save_dir = 'C:\\Users\\Mariana\\OneDrive - Universidade de Lisboa\\G5\\data'


for patient in ['413', '400', '358', '312', '326', '365', '386', '391']:
    for type in ['baseline', 'seizure']:
        features_data = pd.read_parquet(os.path.join(save_dir, 'features_p' + patient + '_' + type + '.parquet'))

        list_features = [feat for feat in features_data.columns if feat not in ['t0', 't1', 'label', 'time2end']]

get_level2(patient, list_features=list_features, type=type, save_dir=save_dir)
