# create dataset X y with patients separated

import os

import numpy as np
import pandas as pd


def create_dataset(path_to_files='..\data', str_files='', list_features=[]):
    """
    Create a dataset of X and y based on str files. X and y are dict of arrays, and each patient is one
    entry of the dict
    :param path_to_files:
    :param str_files:
    :return:
    """
    # list of hrv features
    hrv_feats = ['hr_min',
                 'hr_max',
                 'hr_minmax',
                 'hr_avg',
                 'hr_med',
                 'rr_mean',
                 'rmssd',
                 'nn50',
                 'pnn50',
                 'sdnn',
                 'hti',
                 'tinn',
                 'ulf_pwr',
                 'ulf_peak',
                 'ulf_rpwr',
                 'vlf_pwr',
                 'vlf_peak',
                 'vlf_rpwr',
                 'lf_pwr',
                 'lf_peak',
                 'lf_rpwr',
                 'hf_pwr',
                 'hf_peak',
                 'hf_rpwr',
                 'vhf_pwr',
                 'vhf_peak',
                 'vhf_rpwr',
                 'lf_hf',
                 's',
                 'sd1',
                 'sd2',
                 'sd12',
                 'sd21']
    # list of statistical features calculated for each hrv feature
    stats_feats = ['minmax', 'deriv', 'trend_ratio', 'trend_diff', 'skewness']
    file_names = [file for file in os.listdir(path_to_files) if str_files in file]
    X = {}
    y = {}

    for file in file_names:
        patient = file.split('_')[2]
        data = pd.read_parquet(os.path.join(path_to_files, file))

        labels = sorted(set(data['label']))
        all_feats_names = [hrv_feat + '-' + col for col in data.columns if col not in ['label', 'len', 'feature']
                           for hrv_feat in hrv_feats]

        table_all_feats = pd.DataFrame()

        for label in labels:
            label_feats = data.loc[data['label'] == label]

            one_segment_feats = np.hstack(label_feats[stats_feats].values)
            table_all_feats = pd.concat((table_all_feats, pd.DataFrame([one_segment_feats], columns=all_feats_names)),
                                        ignore_index=True)

        feats_included = list_features if list_features != [] else all_feats_names
        table_feats_included = table_all_feats[feats_included]
        if 'baseline' in labels[0]:
            labels_binary = np.zeros((len(labels)))
        elif 'seizure' in labels[0]:
            labels_binary = np.ones((len(labels)))

        if patient not in X.keys():
            X[patient] = table_feats_included.values
            y[patient] = labels_binary
        else:
            X[patient] = np.vstack((X[patient], table_feats_included.values))
            y[patient] = np.hstack((y[patient], labels_binary))

    return X, y, feats_included

# X, y, feats_included = create_dataset('level2_features')
