from datetime import timedelta
from os import path

import numpy as np
import pandas as pd
import pylab as p
from fastdtw import fastdtw
from matplotlib import pyplot as plt
from scipy.signal import resample
from scipy.spatial.distance import euclidean, cosine, braycurtis, canberra, chebyshev, cityblock, correlation, mahalanobis, seuclidean, sqeuclidean
from scipy.stats import pearsonr
# from sklearn.metrics import roc_auc_score, roc_curve
import seaborn as sns


def calculate_distance(sig1, sig2, distances=['pearsonr', 'euclidean', 'dtw', 'cosine', 'chebyshev', 'braycurtis',
                                              'canberra', 'correlation', 'cityblock', 'mahalanobis', 'seuclidean',
                                              'sqeuclidean'], resample_bool=False):
    """
    Calculate distance between two signals
    :param sig1: signal 1
    :param sig2: signal 2
    :param distances: list of distances to use
    :return:
    """
    # if (len(sig1) < 1) | (len(sig2) <1):
    #    return {dict: 0 for dict in distances}

    if len(sig2) != len(sig1):
        # print(f'samples have a difference in points of {abs(len(sig2)-len(sig1))}')
        if resample_bool:
            sig1 = resample(sig1, len(sig2))
        else:
            min_len = np.min([len(sig1), len(sig2)])
            sig1 = sig1.iloc[len(sig1)-min_len:]
            sig2 = sig2.iloc[len(sig2)-min_len:]
    dist_dict = {}

    if 'pearsonr' in distances:
        dist_dict['pearsonr'] = pearsonr(sig1, sig2)[0]
    if 'euclidean' in distances:
        dist_dict['euclidean'] = euclidean(sig1, sig2)
    if 'dtw' in distances:
        dist_dict['dtw'], _ = fastdtw(sig1, sig2, dist=euclidean)
    if 'cosine' in distances:
        dist_dict['cosine'] = cosine(sig1, sig2)
    if 'cityblock' in distances:
        dist_dict['cityblock'] = cityblock(sig1, sig2)
    if 'chebyshev':
        dist_dict['chebyshev'] = chebyshev(sig1, sig2)
    if 'braycurtis' in distances:
        dist_dict['braycurtis'] = braycurtis(sig1, sig2)
    if 'canberra' in distances:
        dist_dict['canberra'] = canberra(sig1, sig2)
    if 'correlation' in distances:
        dist_dict['correlation'] = correlation(sig1, sig2)
    if 'mahalanobis' in distances:
        dist_dict['mahalanobis'] = mahalanobis(sig1, sig2)
    if 'seuclidean' in distances:
        dist_dict['seuclidean'] = seuclidean(sig1, sig2)
    if 'sqeuclidean' in distances:
        dist_dict['sqeuclidean'] = sqeuclidean(sig1, sig2)

    return dist_dict


def calculate_distance_train(sig1, sig2):
    """
    Since the training only requires two specific distances, this function is a simplified version of calculate
    distances for faster processing
    """
    if len(sig2) != len(sig1):
        # print(f'samples have a difference in points of {abs(len(sig2)-len(sig1))}')
        min_len = np.min([len(sig1), len(sig2)])
        sig1 = sig1[len(sig1)-min_len:]
        sig2 = sig2[len(sig2)-min_len:]
    dist_dict = {'pearsonr': pearsonr(sig1, sig2)[0]}
    dist_dict['dtw'], _ = fastdtw(sig1, sig2, dist=euclidean)
    return dist_dict

