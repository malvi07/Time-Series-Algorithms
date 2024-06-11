#Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pytorch_lightning as pl
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_lightning import Trainer
import warnings
warnings.filterwarnings('ignore')

#Reading CSV data and storing it in a dataframe
df = pd.read_csv('Kaggle-Dataset-Wind-Turbine-Scada-Dataset.csv')
df.head()

#Checking column names and shape.
df.columns
df.shape
#Checking if there are any rows with any NAN values.
df.isna().any(axis=1).sum()

#Changing format of the 'Date/Time' colmn
df['Date/Time'] = pd.to_datetime(df['Date/Time'], format='%d %m %Y %H:%M')
#Setting the index to the date/time column. Changing the current dataframe
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


#Training the TFT Algorithm
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=16,
    attention_head_size=1,
    dropout=0.1,
    hidden_continuous_size=8,
    output_size=7,  #Default value
    loss=QuantileLoss(),
    log_interval=10,
)

trainer = Trainer(
    max_epochs=30,
    accelerator="gpu",
    devices=1,
    log_every_n_steps=10  
)

trainer.fit(model=tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)


