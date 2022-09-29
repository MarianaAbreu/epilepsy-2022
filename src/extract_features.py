# This is a python script to extract hrv features from the ECG signal
# Created by: Mariana Abreu
# last update: 21-09-2022

import os
import time
from datetime import timedelta

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

# get seizures

import hrv

import extract_rri


def get_all_feats(segment):
    """Calculate all features based on RR intervals

    :param segment:
    :return:
    """
    new_row = np.array([])
    columns = []
    # time domain features
    hrv_td = hrv.hrv_timedomain(rri=segment)
    columns += hrv_td.keys()
    new_row = np.hstack((new_row, hrv_td))
    # frequency domain features
    # segment_1hz = resample(segment, len(segment)*1)
    hrv_fq = hrv.hrv_frequencydomain(rri=segment)
    new_row = np.hstack((new_row, hrv_fq))
    columns += hrv_fq.keys()
    # nonlinear features
    hrv_nl = hrv.hrv_nonlinear(rri=segment)
    new_row = np.hstack((new_row, hrv_nl))
    columns += hrv_nl.keys()
    # heart rate features
    hr = (60 * segment) / 1000
    hrv_metrics = hrv.hr_metrics(hr)
    new_row = np.hstack((new_row, hrv_metrics))
    columns += hrv_metrics.keys()

    return dict(zip(columns, new_row.T))


def get_seizures_df(rri_signal, seizure_dates):
    """
    Get rri signal only around seizures, 1 hour prior and 1 hour after each seizure
    :param rri_signal: dataframe of rri values
    :return: seizure_signal
    """
    seizures_df = pd.DataFrame()
    i = 0
    for sd in seizure_dates:
        next_df = rri_signal.loc[rri_signal['dates'].between(sd-timedelta(hours=1), sd+timedelta(minutes=5))]
        next_df['id'] = np.ones(len(next_df)) * i
        seizures_df = pd.concat((seizures_df, next_df), ignore_index=True)
        i += 1
    seizures_df = seizures_df.drop_duplicates()
    return seizures_df


def get_baseline_df(rri_signal, seizure_dates):
    """
    Get rri signal only outside seizures, 1 hour prior and 1 hour after each seizure is excluded
    :param rri_signal: dataframe of rri values
    :return: rri signal
    """
    i = 0
    baseline_df = pd.DataFrame()
    for sd in seizure_dates:
        next_df = rri_signal.loc[~rri_signal['dates'].between(sd-timedelta(hours=2), sd + timedelta(hours=1))]
        next_df['id'] = np.ones(len(next_df)) * i
        baseline_df = pd.concat((baseline_df, next_df), ignore_index=True)
        i += 1

    baseline_df = baseline_df.drop_duplicates()
    return baseline_df


def get_features_good_intervals(rri_signal, type='baseline', patient=''):
    """
    Calculate features on selected good intervals
    :param rri_signal:
    :param type:
    :param patient:
    :return:
    """
    feats_df = pd.DataFrame()
    if type == 'baseline':
        rri_signal = get_baseline_df(rri_signal, seizure_dates)
    elif type == 'seizure':
        rri_signal = get_seizures_df(rri_signal, seizure_dates)

    intervals = pd.date_range(rri_signal.iloc[0]['dates'], rri_signal.iloc[-1]['dates'], freq=str(overlap) + 'S')

    for time0 in intervals[:-1]:

        time1 = time0 + timedelta(seconds=window)
        segment = rri_signal.loc[rri_signal['dates'].between(time0, time1)]
        # at least 200 points
        if len(segment) < 200:
            continue

        get_features = get_all_feats(segment['rri'])
        get_features['t0'] = time0
        get_features['t1'] = time1
        get_features['label'] = type + str(segment['id'].iloc[0])
        feats_df = pd.concat((feats_df, pd.DataFrame(get_features, index=[0])), ignore_index=True)
    feats_df.to_parquet('data\\features_'+patient+'_'+type+'.parquet')


for patient_num in ['312', '326', '391', '365', '358', '352', '386', '400', '413']:
    print('Processing features for patient ', patient_num)
    time_start = time.time()

    patient = 'p' + patient_num
    old_patient = 'PAT_' + patient[1:]
    if not os.path.isdir('G:\\PreEpiSeizures\\Patients_HEM\\Retrospective\\'+old_patient):
        old_patient += '_EXAMES'
    try:
        seizures = pd.read_csv('data\\seizure_label_'+patient, index_col=0)
    except:
        try:
            seizures = pd.read_csv('G:\\PreEpiSeizures\\Patients_HEM\\Retrospective\\' + old_patient + '\\seizure_label',
                                   index_col=0)
        except:
            seizures = pd.read_csv('G:\\PreEpiSeizures\\Patients_HEM\\Retrospective\\' + old_patient + '_EXAMES' +
                                   '\\seizure_label', index_col=0)
        seizures.to_csv('data\\seizure_label_'+patient)

    # get seizure times
    seizure_dates = pd.to_datetime(seizures['Date'], dayfirst=True)

    # get rr intervals
    if not os.path.isfile(f'data\\{patient}_rri.parquet'):
        rri_signal = extract_rri.extract_rri(old_patient)
    else:
        rri_signal = pd.read_parquet('data\\'+patient+'_rri.parquet')
    # pass string of times to datetime
    rri_signal['dates'] = pd.to_datetime(rri_signal['index'])

    # window in seconds
    window = 300
    overlap = 30

    if not os.path.isfile(f'data\\features_{patient}_seizure.parquet'):

        try:
            get_features_good_intervals(rri_signal, type='seizure', patient=patient)
        except Exception as e:
            print(e)
    # if not os.path.isfile(f'data\\features_{patient}_baseline.parquet'):
    try:
        get_features_good_intervals(rri_signal, type='baseline', patient=patient)
    except Exception as e:
        print(e)

    print(f'Time to process ... {time.time()-time_start}')
