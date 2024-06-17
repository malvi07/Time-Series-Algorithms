import numpy as np 
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler
from darts.models import TFTModel
from darts.metrics import mape
from darts.utils.statistics import check_seasonality, plot_acf
from darts.datasets import AirPassengersDataset, IceCreamHeaterDataset
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.likelihood_models import QuantileRegression
import warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.CRITICAL)

df=pd.read_csv('/kaggle/input/wind-turbine-scada-dataset/T1.csv')

#Changing format of the 'Date/Time' colmn
df['Date/Time'] = pd.to_datetime(df['Date/Time'], format='%d %m %Y %H:%M')

#Setting the index of the dataframe to the Date/Time Column
df.set_index('Date/Time', inplace=True)

series = TimeSeries.from_dataframe(df,fill_missing_dates=True, freq='10T')

#Splitting data into train, validation, and test series.
total_rows = len(series)
train_ratio = 0.70
validation_ratio = 0.15
end_train = int(train_ratio * total_rows)
end_valid = end_train + int(validation_ratio * total_rows)
train_series, remainder = series.split_before(end_train)
valid_series, test_series = remainder.split_before(end_valid - end_train)
# Verify the lengths of the splits
print(f"Train length: {len(train_series)}")
print(f"Validation length: {len(valid_series)}")
print(f"Test length: {len(test_series)}")
print(f'Series length: {len(series)}')

#Normalize the time series. Avoid fitting the transformer on the validation set.
transformer = Scaler()
train_transformed = transformer.fit_transform(train_series)
val_transformed = transformer.transform(valid_series)
series_transformed = transformer.transform(series)

# create year, month and integer index covariate series
covariates = datetime_attribute_timeseries(series, attribute="year", one_hot=False)
covariates = covariates.stack(
    datetime_attribute_timeseries(series, attribute="month", one_hot=False)
)
covariates = covariates.stack(
    TimeSeries.from_times_and_values(
        times=series.time_index,
        values=np.arange(len(series)),
        columns=["linear_increase"],
    )
)
covariates = covariates.astype(np.float32)


# transform covariates
cov_train, remainder_cov = covariates.split_before(end_train)
cov_valid, cov_test = remainder_cov.split_before(end_valid - end_train)
scaler_covs = Scaler()
scaler_covs.fit(cov_train)
covariates_transformed = scaler_covs.transform(covariates)

#Creating the TFT Model
model = TFTModel(
    input_chunk_length=168,
    output_chunk_length=24,
    hidden_size=32,
    lstm_layers=2,
    dropout=0.1,
    loss_fn=torch.nn.MSELoss(),
    optimizer_cls=torch.optim.Adam,
    optimizer_kwargs={'lr':0.001},
    batch_size=32,
    n_epochs=100,
    log_tensorboard=True,
    num_attention_heads=4,
    model_name='WindPower_TFT_baseline'    
)

model.fit(series=train_transformed,
         future_covariates=covariates_transformed, verbose=True)
