#Imports
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pytorch_lightning as pl
from pytorch_forecasting import TimeSeriesDataSet
import warnings
warnings.filterwarnings('ignore')

#Reading CSV data and storing it in a dataframe
df = pd.read_csv('T1.csv')
df.head()

#Checking column names and if there are any NAN rows.
df.columns
df.shape
df.isna().any(axis=1).sum()

#Changing format of the 'Date/Time' colmn
df['Date/Time'] = pd.to_datetime(df['Date/Time'], format='%d %m %Y %H:%M')
#Setting the index to the date/time column
df.set_index('Date/Time', inplace=True)

#Creating different columns for the month,day,hour - For seasonality
#Created a time_idx - Incremented by 1 for each time step
df['time_idx'] = range(len(df))
df['month'] = df.index.month
df['day'] = df.index.day
df['hour'] = df.index.hour
df['minute'] = df.index.minute

#I need to create a dummy groupid column (For TimeSeriesDataSet)
df['group_id'] = 'group_1'

#Splitting data into training, validation, and test sets. 70,15,15
total_rows = len(df)
train_ratio = 0.70
validation_ratio = 0.15
end_train = int(train_ratio * total_rows)
end_valid = end_train + int(validation_ratio * total_rows)
train_df = df.iloc[:end_train]
valid_df = df.iloc[end_train:end_valid]
test_df = df.iloc[end_valid:]
len(train_df), len(valid_df), len(test_df)

#Creating a training, validation, and test dataset. 
training = TimeSeriesDataSet(data=train_df,
                             time_idx="time_idx",
                             target="LV ActivePower (kW)",
                             group_ids=["group_id"],
                             max_encoder_length=24,
                             min_encoder_length=12,
                             static_categoricals=["group_id"],
                             time_varying_known_reals=["time_idx", "month", "day", "hour", "minute"],
                             time_varying_unknown_reals=["LV ActivePower (kW)", "Wind Speed (m/s)"])

validation = TimeSeriesDataSet(data=valid_df,
                             time_idx="time_idx",
                             target="LV ActivePower (kW)",
                             group_ids=["group_id"],
                             max_encoder_length=24,
                             min_encoder_length=12,
                             static_categoricals=["group_id"],
                             time_varying_known_reals=["time_idx", "month", "day", "hour", "minute"],
                             time_varying_unknown_reals=["LV ActivePower (kW)", "Wind Speed (m/s)"])

test = TimeSeriesDataSet(data=test_df,
                             time_idx="time_idx",
                             target="LV ActivePower (kW)",
                             group_ids=["group_id"],
                             max_encoder_length=24,
                             min_encoder_length=12,
                             static_categoricals=["group_id"],
                             time_varying_known_reals=["time_idx", "month", "day", "hour", "minute"],
                             time_varying_unknown_reals=["LV ActivePower (kW)", "Wind Speed (m/s)"])


#Creating dataloaders for training, validation, and test
train_dataloader = training.to_dataloader(train=True, batch_size=64, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=64, num_workers=0)
test_dataloader = test.to_dataloader(train=False, batch_size=64, num_workers=0)


#Creating a Baseline Model - Using last observed value in training set to predict future values
def baseline_forecase(train, test, target_col):
    last_observed_value = train[target_col].iloc[-1]
    forecast = np.full(len(test), last_observed_value)
    return forecast

base_preds = baseline_forecase(train_df, test_df, 'LV ActivePower (kW)')


#Making Predictions using baseline model
actuals = test_df['LV ActivePower (kW)'].values

# Calculate evaluation metrics
mae_baseline = mean_absolute_error(actuals, base_preds)
rmse_baseline = np.sqrt(mean_squared_error(actuals, base_preds))

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape_baseline = mean_absolute_percentage_error(actuals, base_preds)

print(f'Baseline MAE: {mae_baseline}')
print(f'Baseline RMSE: {rmse_baseline}')
print(f'Baseline MAPE: {mape_baseline}%')