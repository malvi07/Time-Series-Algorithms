import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Sequence
import datetime
import matplotlib.pyplot as plt
import gc
import openmeteo_requests
import os
import requests_cache
from retry_requests import retry
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

import pickle
import re
from sklearn.linear_model import LinearRegression

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.utils.missing_values import fill_missing_values
from darts.models import TFTModel
from tqdm.autonotebook import tqdm
from darts.models import TiDEModel


# TO GET THE OPENMETEO DATA
def fetch_weather_data(latitude, longitude, elevation, start_date, end_date, timezone='America/New_York', model='best_match'):
#    weather_data = fetch_weather_data(latitude=lat, longitude=lon, elevation =0, start_date=start_date, end_date=end_date, timezone='America/Santiago', model = 'era5_seamless')
#    weather_data = fetch_weather_data(latitude=lat, longitude=lon, elevation = 0, start_date=start_date, end_date=end_date, timezone='America/Santiago', model = 'ecmwf_ifs')

    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # Define the URL and parameters for the API request
    url = "https://archive-api.open-meteo.com/v1/archive"
    #"start_date": "2017-01-01",
    #"end_date": "2023-12-31",
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        #"hourly": ["wind_speed_10m", "wind_direction_10m"],
        "hourly": ["temperature_2m", "relative_humidity_2m", "rain", "pressure_msl", "surface_pressure", "wind_speed_10m", "wind_direction_10m"],
        "wind_speed_unit": "ms",
        "models": model,
        #"elevation": elevation,
        "timezone": timezone #"auto"
    }

    # Make the API request
    responses = openmeteo.weather_api(url, params=params)

    # Process the response
    response = responses[0]

    # Process hourly data
    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
    hourly_rain = hourly.Variables(2).ValuesAsNumpy()
    hourly_pressure_msl = hourly.Variables(3).ValuesAsNumpy()
    hourly_surface_pressure = hourly.Variables(4).ValuesAsNumpy()
    hourly_wind_speed_10m = hourly.Variables(5).ValuesAsNumpy()
    hourly_wind_direction_10m = hourly.Variables(6).ValuesAsNumpy()

    hourly_data = {"date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s"),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s"),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        )}
    hourly_data["temperature_2m"] = hourly_temperature_2m
    hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
    hourly_data["rain"] = hourly_rain
    hourly_data["pressure_msl"] = hourly_pressure_msl
    hourly_data["surface_pressure"] = hourly_surface_pressure
    hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
    hourly_data["wind_direction_10m"] = hourly_wind_direction_10m

    hourly_dataframe = pd.DataFrame(data=hourly_data)
    return hourly_dataframe


def predict_and_evaluate(model, ts_list, future_cov_list, past_cov_list, start_time = '2023-09-01 11:00:00'):
    # each day at 10 AM predict the next day (midnight to midnight)
    start = pd.Timestamp(start_time)
    stride = 24

    all_forecasts= pd.DataFrame()

    fcsts = model.historical_forecasts(
        series = ts_list,
        past_covariates = past_cov_list,
        future_covariates = future_cov_list,
        stride=stride,
        start=start,
        forecast_horizon=240,
        last_points_only = False,
        retrain=False,
        verbose=False,
    )
    
    all_forecasts = pd.DataFrame()

    # for each forecasted day at the init time (start plus stride*i)
    fcst_id = pd.DataFrame()
    for j, fcst_day in enumerate(fcsts[0]):
        # convert to list
        fcst_id_day = fcst_day.pd_dataframe()
        fcst_id_day.reset_index(drop=False, inplace=True)
        fcst_id_day[f'init_time'] = start + pd.Timedelta(hours=stride*j-1)

        #concatenate into one df
        if fcst_id.empty:
            fcst_id = fcst_id_day
        else:
            fcst_id = pd.concat([fcst_id, fcst_id_day])

    # merge into one DataFrame
    if all_forecasts.empty:
        all_forecasts = fcst_id
    else:
        all_forecasts = pd.merge(left=all_forecasts, right=fcst_id, how='left', on=['time', 'init_time'])


    return all_forecasts


def train_TFT_model(args, ts_train_list, past_cov_list, future_cov_list, NUM_EPOCHS):
    # Convert args to model parameters
    model_params = {
        "output_chunk_length": 240,  # pred_len
        "n_epochs": NUM_EPOCHS,
        "add_encoders": {
        'datetime_attribute': {'past': ['hour', 'day_of_week', 'month'],'future': ['hour', 'day_of_week', 'month']},
        'transformer': Scaler(),
        },
        "input_chunk_length": args.input_chunk_length,
        "hidden_size": args.hidden_size,
        "lstm_layers": args.lstm_layers,
        "num_attention_heads": args.num_attention_heads,
        "dropout": args.dropout,
        "hidden_continuous_size": args.hidden_continuous_size,
        "batch_size": args.batch_size,
        "optimizer_kwargs": {'lr': args.learning_rate},
    }

    model = TFTModel(
        **model_params,
        pl_trainer_kwargs={
        "accelerator": "auto",
        "devices":"auto",
        },
    )
    model.fit(ts_train_list,
        past_covariates = past_cov_list,
        future_covariates= future_cov_list,
        verbose=True)

    return model

# best hyperparams out of tuning job
class Args:
    def __init__(self):
        self.input_chunk_length = 240
        self.hidden_size = 16
        self.lstm_layers = 3
        self.num_attention_heads = 3
        self.hidden_continuous_size = 8
        self.batch_size = 128
        self.learning_rate = 0.004853746699563581
        self.dropout = 0.0

def get_args():
    '''Provides hyperparameters for testing.'''
    return Args()



def preprocess_data(df, weather_df, training_split_date = '2023-05-01', val_split_date = '2023-07-01'):
    df["time"] = pd.to_datetime(df["time"])
    weather_df["time"] = pd.to_datetime(weather_df["time"])

    duplicates = df.duplicated(subset=['time'])
    print ("number of duplicates: ", duplicates.sum())
    if duplicates.any():
        print("There are duplicates in the df time column.")
    else:
        print("There are no duplicates in the df time column.")

    duplicates = weather_df.duplicated(subset=['time'])
    if duplicates.any():
        print("There are duplicates in the weather df time column.")
    else:
        print("There are no duplicates in the weather df time column.")

    # PREPROCESSING
    orig_len = len(df)
    print("Original Length: ", orig_len)
    df = df.set_index("time")
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    df = df.resample('h')
    df = df.interpolate(method='linear', limit_direction=None)
    df = df.reset_index()
    print("Rows Added: ", len(df) - orig_len)

    # build lists of time series objects (each element in the list is a different location)
    ts_train_list = [] # use for trainig; until and including 2022
    ts_val_list = [] # use for hyperparameter tuning; 2023
    ts_list = [] #both together; everything

    future_cov_list = []
    past_cov_list = []


    target_column = 'value_avbl'
    ts = TimeSeries.from_dataframe(df, time_col='time', value_cols=target_column, fill_missing_dates=True, freq='h')
    ts = fill_missing_values(ts)
    ts_train, ts_val = ts.split_before(pd.Timestamp(training_split_date))
    ts_val, _ = ts_val.split_before(pd.Timestamp(val_split_date))
    ts_train_list.append(ts_train)
    ts_val_list.append(ts_val)
    ts_list.append(ts)
    
    # COVARIATES
    # future
    location_df = weather_df

    # future: wind_speed_10m_ifs, wind_direction_10m_ifs
    future_cov_columns = ['wind_speed_10m', 'wind_direction_10m']
    future_cov = TimeSeries.from_dataframe(location_df, time_col='time', value_cols=future_cov_columns, fill_missing_dates=True, freq='h')
    future_cov = fill_missing_values(future_cov)
    future_cov_list.append(future_cov)
    
    # past: wind_speed_10m_ifs, wind_direction_10m_ifs, wind_speed_10m_era, wind_direction_10m_era
    past_cov_columns = [
        'wind_speed_10m', 'wind_direction_10m'
    ]
    past_cov = TimeSeries.from_dataframe(location_df, time_col='time', value_cols=past_cov_columns, fill_missing_dates=True, freq='h')
    past_cov = fill_missing_values(past_cov)
    past_cov_list.append(past_cov)

    return ts_train_list, ts_val_list, ts_list, future_cov_list, past_cov_list


def train_TIDE_model(ts_train_list, ts_val_list, future_cov_list, past_cov_list, n_epochs):
    # Convert args to model parameters
    print ("ENTERING THE TIDE MODEL")
    MAPE_351 = {
        "input_chunk_length": 720,  # hist_len
        "num_encoder_layers": 2,  # num_layers
        "num_decoder_layers": 2,  # num_layers
        "decoder_output_dim": 8,  # decoder_output_dim
        "hidden_size": 512,  # hidden_size
        "temporal_width_past": 4,  # Default or based on your dataset
        "temporal_width_future": 4,  # Default or based on your dataset
        "temporal_decoder_hidden": 32,  # final_decoder_hidden
        "use_layer_norm": False,  # layer_norm
        "dropout": 0.15,  # dropout_rate

        # not tuned
        "use_static_covariates": False,
        "output_chunk_length": 37,  # pred_len
        "n_epochs": n_epochs,
    }

    model = TiDEModel(
        **MAPE_351,
        pl_trainer_kwargs={
          "accelerator": "auto",
          "devices":"auto"
        },
        add_encoders = {
          'datetime_attribute': {'past': ['hour', 'day_of_week', 'month'],'future': ['hour', 'day_of_week', 'month']},
          'transformer': Scaler(),
        },
        model_name = 'tide',
        save_checkpoints=True,
        force_reset=True
    )
    
    # check that everything is correct
    print(f'ts_train_list: {len(ts_train_list)}')
    print(f'ts_val_list: {len(ts_val_list)}')
    print(f'future_cov_list: {len(future_cov_list)}')
    print(f'past_cov_list: {len(past_cov_list)}')

    #model.fit(ts_train_list, past_covariates= past_cov_list, verbose=True)

    model.fit(ts_train_list,
              future_covariates=future_cov_list,
              past_covariates=past_cov_list,
              val_series = ts_val_list,
              val_future_covariates = future_cov_list,
              val_past_covariates = past_cov_list,
              verbose=True)
    
    #load best model on validation set to avoid overfitting
    #model = TiDEModel.load_from_checkpoint(model_name = 'tide', best=True)
    print ("DONE WITH THE TIDE MODEL")

    return model
