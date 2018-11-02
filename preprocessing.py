import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def split_sequence(sequences, cold_start_window, pred_window, features):
    """
    Split a multivariate series into training samples

    Args:
        sequences: a numpy array of feature values of one training series
        cold_start_window: number of cold start days provided
        pred_window: size of the prediction window ('hourly', 'daily', 'weekly')
        features: a list of the names of the features being used

    Returns:
        X: training data
        y: target
    """     
    strides = 24
    
    pred_window_to_steps = {'hourly': 24, 'daily': 24*7, 'weekly': 24*7*2}
    steps = pred_window_to_steps[pred_window]
    cold_start = cold_start_window * 24

    X, y = list(), list()

    for i in range(0, len(sequences), strides):
        # find the end of this pattern
        end_idx = i + cold_start
        out_end_idx = end_idx + steps
        
        # check if we are beyond the sequence
        if out_end_idx > len(sequences):
            break
            
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_idx,], sequences[end_idx:out_end_idx, 0]
        X.append(seq_x)
        y.append(seq_y)
    
    return np.array(X), np.array(y)

def prepare_training_data(series_data, cold_start_window, pred_window, num_features, features):
    """
    Prepare data for training in neural networks

    Args:
        series_data: a dataframe of a train series
        cold_start_window: number of cold start days provided
        pred_window: size of the prediction window ('hourly', 'daily', 'weekly')
        num_features: number of features
        features: a list of the names of the features being used

    Returns:
        X: training data
        y: target
        scaler: scaler instance
    """  
    if len(features) != num_features:
        raise ValueError('The number of features must match!')
        
    # scale training data
    scaler = MinMaxScaler(feature_range=(-1,1))

    # loop over features and concatenate the values
    sequences = np.empty((series_data.shape[0], 1))
    for feature in features:
        vals = scaler.fit_transform(series_data.eval(feature).values.reshape(-1,1))
        sequences = np.concatenate([sequences, vals], axis = 1)
    
    sequences = sequences[:, 1:]
    
    X, y = split_sequence(sequences, cold_start_window, pred_window, features)
    
    X = X.reshape(X.shape[0], X.shape[1], num_features)
    y = y.reshape(y.shape[0], y.shape[1])
    
    return X, y, scaler