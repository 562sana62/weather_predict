import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_data(path):

    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    features = df[['Temp','Humidity','Pressure','Rainfall']]

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(features)

    return scaled, scaler

def create_sequences(data, seq_length=30):
    X = []
    y = []
    for i in range(len(data)-seq_length-5):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length:i+seq_length+5][:,[0,3]])

    return np.array(X), np.array(y)