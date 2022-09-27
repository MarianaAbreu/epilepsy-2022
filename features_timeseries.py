import os

def standardize(sig):
    # sig should be a dataframe where each column is one feature and each row a new segment
    # standardise a seizure interval
    return (sig - np.mean(axis=1)) / np.std(axis=1)