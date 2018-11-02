from preprocessing import split_sequence, prepare_training_data
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam

def lstm_model(cold_start_window, pred_window, num_features, num_hidden_states, learning_rate, decay_rate):
    """
    Define a model
    
    Args:
        cold_start_window: number of cold start days provided
        pred_window: size of the prediction window ('hourly', 'daily', 'weekly')
        num_features: number of features
        num_hidden_states: number of LSTM hidden states
        learning_rate: learning rate of the model
        decay_rate: decay rate of the learning rate

    Returns:
        a Keras model object
    """  
    cold_start = cold_start_window * 24
    
    pred_window_to_num_preds = {'hourly': 24, 'daily': 24*7, 'weekly': 24*7*2}
    num_preds = pred_window_to_num_preds[pred_window]
        
    model = Sequential()
    model.add(LSTM(num_hidden_states, activation='relu', return_sequences=True, input_shape=(cold_start, num_features)))
    model.add(LSTM(num_hidden_states, activation='relu')) 
    model.add(Dense(num_preds)) 
    
    adam = Adam(lr = learning_rate, beta_1 = 0.9, beta_2 = 0.999, epsilon= None, decay=decay_rate, amsgrad=False)
    model.compile(optimizer='adam', loss='mae')
    
    return model
    
def calc_valid_error(valid_df, model, cold_start_window, pred_window, num_features, features):
    """
    Calculate validation error
    
    Args:
        valid_df: validation Pandas dataframe
        model: a model object that has been trained
        
    Returns:
        mean absolute error score
    """
    mean_absolute_error = 0
    num_valid_series = valid_df.series_id.nunique()
    
    for ser_id, ser_data in valid_df.groupby('series_id'):  
        X_valid, y_valid, scaler = prepare_training_data(ser_data, cold_start_window, pred_window, num_features, features)
        y_pred = model.predict(X_valid)
        error = np.sum(abs(y_pred[-1] - y_valid[-1])) / np.mean(y_valid[-1])
        mean_absolute_error += error
    
    normalized_mean_absolute_error = mean_absolute_error / (num_valid_series * 672)
    
    return normalized_mean_absolute_error

def training(train_df, valid_df, cold_start_window, pred_window, num_features, features, epochs, num_passes_through_data, num_hidden_states):
    """
    Train and save the model object
    
    Args:
        train_df: training Pandas dataframe
        cold_start_window: number of cold start days provided
        pred_window: size of the prediction window ('hourly', 'daily', 'weekly')
        num_features: number of features

    Returns:
        a Keras model object
    """  
    num_training_series = train_df.series_id.nunique()
    num_passes_through_data = num_passes_through_data
    
    model = lstm_model(cold_start_window, pred_window, num_features, num_hidden_states, 0.05, 0)
   
    for i in tqdm(range(num_passes_through_data), 
              total=num_passes_through_data, 
              desc='Learning Consumption Trends - Epoch'):
        
        # reset the LSTM state for training on each series
        for ser_id, ser_data in train_df.groupby('series_id'):
            print(ser_id)
            # prepare the data
            X, y, scaler = prepare_training_data(ser_data, cold_start_window, pred_window, num_features, features)

            # fit the model: note that we don't shuffle batches (it would ruin the sequence)
            # and that we reset states only after an entire X has been fit, instead of after
            # each (size 1) batch, as is the case when stateful=False
            model.fit(X, y, epochs=epochs, verbose=1, shuffle=False)
            model.reset_states()
            
        mean_absolute_error = calc_valid_error(valid_df, model, cold_start_window, pred_window, num_features, features)
        print("mean_absolute_error:", mean_absolute_error)
    
    return model