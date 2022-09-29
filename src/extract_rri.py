# This is a python script to extract rr-intervals from the ECG signal
# Created by: Mariana Abreu
# last update: 21-09-2022

import os

import numpy as np
import pandas as pd
import matplotlib


# directory

patient = 'PAT_391_EXAMES'


def extract_rri(patient):
    """
    Function to extract rri signals from ECG time series of the entire hospital acquisition
    :return:
    """

    dir = 'G:\\PreEpiSeizures\\Patients_HEM\\Retrospective\\' + patient + '\\ECG'
    sampling_rate = 256

    rri_data = pd.DataFrame()

    for file in os.listdir(dir):
        rri_df = pd.DataFrame()

        data = pd.read_parquet(os.path.join(dir, file))

        rpeaks = data['rpeaks'].dropna().index

        rri_df['rri'] = 1000 * np.diff(rpeaks) / sampling_rate
        rri_df['index'] = data['index'][rpeaks[1:]].values
        rri_df['fileid'] = [file.split('_')[-2]] * len(rri_df)
        rri_data = pd.concat((rri_data, rri_df), ignore_index=True)

    new_patient = 'p'+patient.split('_')[1]
    rri_data.to_parquet('C:\\Users\\Mariana\\PycharmProjects\\epilepsy-2022\\data\\'+new_patient+'_rri.parquet',
                        engine='fastparquet', compression='gzip')
    return rri_data