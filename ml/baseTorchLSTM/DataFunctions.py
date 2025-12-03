import tensorflow as tf
import keras
from keras import layers

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# load dataset and remove missing value from data
def getDataframe(path):
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print("ERROR LOADING DF")
        print(e)
        raise  #if try fails the code continues running after the exception so df wont be defined since it is defined in try.
    df_clean = df.dropna()
    return df_clean

#print data about the dataset
def printDataDetails(df):
    print("shape: ", df.shape)
    print("columns:", df.columns.tolist())
    print("head: ")
    print(df.head())


def breakDataIntoSequences(data, seq_length, pred_col):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        y.append(
            data[i + seq_length, pred_col] - data[i + seq_length - 1, pred_col]
        )  # changed
    return np.array(X), np.array(y)


def scaleData(df):
    data = df.to_numpy()

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    return scaled_data, scaler


def generateSplit(X, y, split_frac = 0.8):

    split = int(split_frac * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    return X_train, X_test, y_train, y_test
