import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

def split_test_sequence(sequences, cold_start_window, pred_window):
    """
    Split a univariate consumption series into training samples

    Args:
        sequences: a numpy array of feature values of one training series
        cold_start_window: number of cold start days provided
        pred_window: size of the prediction window ('hourly', 'daily', 'weekly')

    Returns:
        X: training data
    """
    strides = 24
    
    pred_window_to_steps = {'hourly': 24, 'daily': 24*7, 'weekly': 24*7*2}
    steps = pred_window_to_steps[pred_window]
    cold_start = cold_start_window * 24

    X = list()

    for i in range(0, len(sequences), strides):
        # find the end of this pattern
        end_idx = i + cold_start
        
        if end_idx > len(sequences):
            break
            
        # gather input part of the pattern
        seq_x = sequences[i:end_idx, ]
        X.append(seq_x)
    
    return np.array(X)

def prepare_testing_data(series_data, cold_start_window, pred_window, num_features, features):
    """
    Prepare data for training in neural networks

    Args:
        series_data: a dataframe of a train series
        cold_start_window: number of cold start days provided
        pred_window: size of the prediction window ('hourly', 'daily', 'weekly')
        num_features: number of features

    Returns:
        X: testing data
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
    
    scaler.fit_transform(series_data.consumption.values.reshape(-1,1))
    
    X = split_test_sequence(sequences, cold_start_window, pred_window)  
    X = X.reshape(X.shape[0], X.shape[1], num_features)
    
    return X, scaler

def make_forecasts(submission, features, model):
    pred_window_to_num_preds = {'hourly': 24, 'daily': 7, 'weekly': 2}
    pred_window_to_num_pred_hours = {'hourly': 24, 'daily': 7 * 24, 'weekly': 2 * 7 * 24}

    num_test_series = submission.series_id.nunique()
    num_features = len(features)

    model.reset_states()

    for ser_id, pred_df in tqdm(submission.groupby('series_id'), 
                                total=num_test_series, 
                                desc="Forecasting from Cold Start Data"):
            
        # get info about this series' prediction window
        pred_window = pred_df.prediction_window.unique()[0]
        num_preds = pred_window_to_num_preds[pred_window]
        num_pred_hours = pred_window_to_num_pred_hours[pred_window]

        # prepare cold start data
        ser_data = cold_start_test[cold_start_test.series_id == ser_id]
        
        cold_start_window = 1
        
        cold_X, scaler = prepare_testing_data(ser_data, cold_start_window, pred_window, num_features, features)
            
        # generate forecasts
        preds_scaled = model.predict(cold_X)
        preds = scaler.inverse_transform(preds_scaled[-1].reshape(-1, 1))
        
        # reduce by taking sum over each sub window in pred window
        reduced_preds = [pred.sum() for pred in np.split(preds, num_preds)]
        
        # store result in submission DataFrame
        ser_id_mask = submission.series_id == ser_id
        submission.loc[ser_id_mask, 'consumption'] = reduced_preds

    return submission