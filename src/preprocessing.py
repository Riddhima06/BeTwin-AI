import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
def load_data(path):
    df = pd.read_csv(path, sep=" ", header=None)
    df = df.dropna(axis=1)
    return df

def add_rul(df):
    max_cycle = df.groupby(0)[1].max().reset_index()
    max_cycle.columns = [0, "max"]
    df = df.merge(max_cycle, on=0)
    df["RUL"] = df["max"] - df[1]
    df.drop("max", axis=1, inplace=True)
    return df

def scale_data(df):
    scaler = MinMaxScaler()
    sensor_cols = df.columns[2:-1]
    df[sensor_cols] = scaler.fit_transform(df[sensor_cols])
    return df, scaler

def create_sequences(df, seq_length):
    X = []
    y = []

    for engine_id in df[0].unique():
        engine_df = df[df[0] == engine_id]
        values = engine_df.values

        for i in range(len(values) - seq_length):
            X.append(values[i:i+seq_length, 2:-1])
            y.append(values[i+seq_length, -1])

    return np.array(X), np.array(y)