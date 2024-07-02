df = pd.read_csv('./wind-turbine-scada-dataset/T1.csv')
from utils import *

#Changing format of the 'Date/Time' colmn
df['Date/Time'] = pd.to_datetime(df['Date/Time'], format='%d %m %Y %H:%M')

#Setting index as the date/time column. Changing df type to float32
df.set_index('Date/Time', inplace=True)
df = df.astype(np.float32)

#Creating a dataframe to store columns that contain information about weather
weather_df = df[['Wind Speed (m/s)', 'Wind Direction (°)' ]].copy()
#Creating a dataframe. Gets the hourly mean for data that is related to weather.
weather_hourly_df = weather_df.resample('H').mean()

#Creating a dataframe that stores the target variable (Active Power)
wind_power_df = df[['LV ActivePower (kW)']].copy()
#Creating a dataframe. Gets the hourly mean for power.
wind_power_hourly_df = wind_power_df.resample('H').mean()

#Creating a start index and stop index. 
min_timestamp = wind_power_df.index.min()
max_timestamp = wind_power_df.index.max()
print (f"min timestamp is {min_timestamp} max timestamp is {max_timestamp}")

# Plot the values
plt.figure(figsize=(16, 6))
min_timestamp = wind_power_df.index.min()
max_timestamp = wind_power_df.index.max()
print (min_timestamp, max_timestamp)
temp_df = wind_power_df.loc[min_timestamp: max_timestamp]
plt.plot(temp_df.index, temp_df['LV ActivePower (kW)'], label='Actual wind power')

# Customize the plot
plt.xlabel('Timestamp')
plt.ylabel('Wind power')
plt.title(f'Actual wind power between {min_timestamp} and {max_timestamp}')
plt.legend()
plt.grid(True)
plt.xticks(rotation=75)

# Show plot
plt.show()

#Resetting the index and changing column names for the wind_power_hourly dataframe.
wind_power_hourly_df = wind_power_hourly_df.reset_index()
wind_power_hourly_df = wind_power_hourly_df.rename(columns={'Date/Time': 'time'})
wind_power_hourly_df = wind_power_hourly_df.rename(columns={'LV ActivePower (kW)': 'value_avbl'})

#Resetting the index and changing column names for the weather_hourly dataframe.
weather_hourly_df = weather_hourly_df.reset_index()
weather_hourly_df = weather_hourly_df.rename(columns={'Date/Time':'time'})
weather_hourly_df = weather_hourly_df.rename(columns={'Wind Speed (m/s)':'wind_speed_10m', 'Wind Direction (°)':'wind_direction_10m' })


#Passing in data into function --> preprocess_data. Returns:
'''
1. TimeSeries Training List
2. TimeSeries Validation List
3. TimeSeries List
4. Future Covariates List
5. Past Covariates List
'''
ts_train_list, ts_val_list, ts_list, future_cov_list, past_cov_list = preprocess_data(wind_power_hourly_df,weather_hourly_df, training_split_date = '2018-06-01', val_split_date = '2018-07-01')

#Var for the number of epochs.
NUM_EPOCHS = 15
#Getting hyperparameters from the function get_args().
args = get_args()
# Training the TFT model and storing it in the var model
model = train_TFT_model(args, ts_train_list, past_cov_list, future_cov_list, NUM_EPOCHS)

#Making predictions
all_forecasts = predict_and_evaluate(model, ts_list, future_cov_list, past_cov_list, start_time = '2018-07-02 11:00:00' )

#Renaming columns - Ease of use
all_forecasts = all_forecasts.rename(columns={'value_avbl': 'TFT_value'})
#Creating new column
all_forecasts['lead_time'] = all_forecasts['time'] - all_forecasts['init_time']

#Setting index of the dataframe to the 'time' column
wind_power_hourly_df.set_index('time', inplace=True)

#Setting index of the dataframe to the 'time' column
all_forecasts.set_index('time', inplace=True)

#Implementing a left join merge.
all_forecasts = pd.merge(all_forecasts, wind_power_hourly_df, on='time', how='left')
day_ahead_forecasts = all_forecasts[all_forecasts['lead_time'].between(pd.Timedelta(hours=14), pd.Timedelta(days=1, hours=13))]

# This is for 173 days; so some duplicate times etc; we need to either consider the first and drop the repeat OR
# We need to consider the average over all times 
all_forecasts.shape[0]/240
wind_power_hourly_df.shape

merged_df = day_ahead_forecasts.copy()
print (merged_df.head())

# Identify rows with any NaN values
nan_rows = merged_df[merged_df.isna().any(axis=1)]

len(nan_rows)
#Taking the average over all times and replacing rows that contain NAN values
merged_df.fillna(merged_df.mean(), inplace=True)

