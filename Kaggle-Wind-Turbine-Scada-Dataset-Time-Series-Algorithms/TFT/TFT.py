import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler
from darts.models import TFTModel
from darts.metrics import mape
from darts.metrics import mae, rmse
from darts.utils.statistics import check_seasonality, plot_acf
from darts.datasets import AirPassengersDataset, IceCreamHeaterDataset
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.likelihood_models import QuantileRegression

import warnings
warnings.filterwarnings("ignore")

import logging
logging.disable(logging.CRITICAL)

#Inspecting dataset
df=pd.read_csv('/kaggle/input/wind-turbine-scada-dataset/T1.csv')
df.columns

#Count of 0 values
zero_count = (df['LV ActivePower (kW)'] == 0).sum()
# Count of negative values
negative_count = (df['LV ActivePower (kW)'] < 0).sum()
print(zero_count, negative_count)

# Calculate the mean of the positive values in the column
mean_value = df[df['LV ActivePower (kW)'] > 0]['LV ActivePower (kW)'].mean()
# Replace negative values with the mean
df.loc[df['LV ActivePower (kW)'] < 0, 'LV ActivePower (kW)'] = mean_value

#Changing format of the 'Date/Time' colmn
df['Date/Time'] = pd.to_datetime(df['Date/Time'], format='%d %m %Y %H:%M')
#Making the index of the dataset the 'Date/Time' column
df.set_index('Date/Time', inplace=True)

#Convertint the dataframe into a TimeSeries
series = TimeSeries.from_dataframe(df,fill_missing_dates=True, freq='10T')

# Splitting data into train, validation, and test series using time.
train_series, remainder = series.split_before(0.70)
valid_series, test_series = remainder.split_before(0.85)  # Adjusted to 0.85 because it's relative to the start of the remainder

# Verify the lengths of the splits
print(f"Train length: {len(train_series)}")
print(f"Validation length: {len(valid_series)}")
print(f"Test length: {len(test_series)}")
print(f'Series length: {len(series)}')

#Normalize the time series.
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
total_rows = len(series)
train_ratio = 0.70
end_train = int(train_ratio * total_rows)
validation_ratio = 0.15
end_valid = end_train + int(validation_ratio * total_rows)
cov_train, remainder_cov = covariates.split_before(end_train)
cov_valid, cov_test = remainder_cov.split_before(end_valid - end_train)
scaler_covs = Scaler()
scaler_covs.fit(cov_train)
covariates_transformed = scaler_covs.transform(covariates)


# Check NaN values in the transformed training, validation, and covariates series
print("Training NaN count:", train_transformed.pd_dataframe().isna().sum().sum())
print("Validation NaN count:", val_transformed.pd_dataframe().isna().sum().sum())
print("Covariates NaN count:", covariates_transformed.pd_dataframe().isna().sum().sum())

# Check which columns have NaN values
train_nan_columns = train_transformed.pd_dataframe().isna().sum()
val_nan_columns = val_transformed.pd_dataframe().isna().sum()
print("NaN counts in training columns:", train_nan_columns)
print("NaN counts in validation columns:", val_nan_columns)

# Apply forward fill
train_filled = train_transformed.pd_dataframe().fillna(method='ffill').pipe(TimeSeries.from_dataframe)
val_filled = val_transformed.pd_dataframe().fillna(method='ffill').pipe(TimeSeries.from_dataframe)


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
    n_epochs=10,
    log_tensorboard=True,
    num_attention_heads=4,
    model_name='WindPower_TFT_baseline',
    add_relative_index=True
)

#Training the model
model.fit(
    series=train_filled, 
    past_covariates=covariates_transformed,
    val_series=val_filled, 
    val_past_covariates=covariates_transformed
)


#Testing on the validation dataset
all_predictions = []

timestamps = val_filled.time_index

for i in range(0, len(val_filled) - model.input_chunk_length, model.output_chunk_length):
    start_ts = timestamps[i]
    end_ts = timestamps[i + model.input_chunk_length - 1]
    
    past_covariates_slice = covariates_transformed.slice(start_ts, end_ts)
    val_filled_slice = val_filled.slice(start_ts, end_ts)
    
    pred = model.predict(n=model.output_chunk_length, series=val_filled_slice, past_covariates=past_covariates_slice)
    
    all_predictions.append(pred)

all_predictions = concatenate(all_predictions)

aligned_predicted_lv_activepower = all_predictions.slice_intersect(val_filled)

actual_lv_activepower = val_filled['LV ActivePower (kW)']
predicted_lv_activepower = aligned_predicted_lv_activepower['LV ActivePower (kW)']

validation_mae = mae(actual_lv_activepower, predicted_lv_activepower)
validation_rmse = rmse(actual_lv_activepower, predicted_lv_activepower)

print(f"Validation MAE: {validation_mae}")
print(f"Validation RMSE: {validation_rmse}")






# Plot actual vs predicted LV ActivePower (kW)
plt.figure(figsize=(12, 6))

actual_lv_activepower.plot(label='Actual LV ActivePower (kW)', lw=2, color='blue', alpha=0.6)
predicted_lv_activepower.plot(label='Predicted LV ActivePower (kW)', lw=2, color='red', alpha=0.6)

plt.legend()
plt.title('Actual vs Predicted LV ActivePower (kW)')
plt.xlabel('Time')
plt.ylabel('LV ActivePower (kW)')
plt.grid(True)
plt.show()

#Making predictions on the test dataset
test_transformed = transformer.transform(test_series)

test_covariates = datetime_attribute_timeseries(test_transformed, attribute="year", one_hot=False)
test_covariates = test_covariates.stack(datetime_attribute_timeseries(test_transformed, attribute="month", one_hot=False))
test_covariates = test_covariates.stack(
    TimeSeries.from_times_and_values(
        times=test_transformed.time_index,
        values=np.arange(len(test_transformed)),
        columns=["linear_increase"]
    )
)

test_filled = test_transformed.pd_dataframe().fillna(method='ffill').pipe(TimeSeries.from_dataframe)
test_covariates_transformed = scaler_covs.transform(test_covariates)




test_predictions = []

test_timestamps = test_filled.time_index

for i in range(0, len(test_filled) - model.input_chunk_length, model.output_chunk_length):
    start_ts = test_timestamps[i]
    end_ts = test_timestamps[i + model.input_chunk_length - 1]
    
    test_past_covariates_slice = test_covariates_transformed.slice(start_ts, end_ts)
    test_filled_slice = test_filled.slice(start_ts, end_ts)
    
    test_pred = model.predict(n=model.output_chunk_length, series=test_filled_slice, past_covariates=test_past_covariates_slice)
    
    test_predictions.append(test_pred)

test_predictions = concatenate(test_predictions)

aligned_test_predicted_lv_activepower = test_predictions.slice_intersect(test_filled)

actual_test_lv_activepower = test_filled['LV ActivePower (kW)']
predicted_test_lv_activepower = aligned_test_predicted_lv_activepower['LV ActivePower (kW)']

test_mae = mae(actual_test_lv_activepower, predicted_test_lv_activepower)
test_rmse = rmse(actual_test_lv_activepower, predicted_test_lv_activepower)

print(f"Test MAE: {test_mae}")
print(f"Test RMSE: {test_rmse}")

# Test MAE: 0.11803431203695597
# Test RMSE: 0.17109489731409974



















