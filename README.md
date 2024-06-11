# Time-Series-Algorithms-Wind-Power-Forecasting

### Introduction
This project is about comparing different advanced algorithms used for predicting future values in time series data.
The aim is to evaluate and compare the performance of several cutting-edge models, including:
* TFT
* TSMixer
* Patch Time Series
* TimeGPT
* TimesNet


### Datasets
This project utilizes two primary datasets:
1. Wind Turbine Scada Dataset: This dataset was sourced from Kaggle. It contains the 2018 Scada Data of a Wind Turbine in Turkey. The dataset can be accessed [here](https://www.kaggle.com/datasets/berkerisen/wind-turbine-scada-dataset/code).
2. Wind Power Forecasting: The dataset was sourced from Kaggle. It contains two and a half years of data for a windmill. The dataset can be accessed [here](https://www.kaggle.com/datasets/theforcecoder/wind-power-forecasting/data)

### Benchmark Metrics ("Numbers to Beat")
Will compare the algorithms against established benchmarks in the field of time series forecasting.

Below are the key performance metrics from studies on the Wind Turbine Scada Dataset [here](https://www.kaggle.com/datasets/berkerisen/wind-turbine-scada-dataset/code) : 

1. Wind Turbine EDA and Modeling 0.986 R2 - Youssif Shaaban Qzamel
   * RMSE (Root Mean Square Error): 149.06
   * R² (Coefficient of Determination): 0.986
   * More details can be found [here](https://www.kaggle.com/code/youssifshaabanqzamel/wind-turbine-eda-and-modeling-0-986-r2)

2. Wind Turbine Modeling R2 0.98 - Ahmad Alghali
   * MAE (Mean Absolute Error): 71.696
   * R² (Coefficient of Determination): 0.982
   * More details can be found [here](https://www.kaggle.com/code/ahmadalghali/wind-turbine-modeling-r2-0-98)

3. Wind Turbine Power Prediction-GBTRegressor PySpark - Melih Akdag
   * R² (Coefficient of Determination): 0.981
   * MAE (Mean Absolute Error): 83.679
   * RMSE (Root Mean Square Error): 179.19
   * More details can be found [here](https://www.kaggle.com/code/akdagmelih/wind-turbine-power-prediction-gbtregressor-pyspark)
  

Below are the key performance metrics from studies on the Wind Turbine Scada Dataset [here](https://www.kaggle.com/datasets/theforcecoder/wind-power-forecasting) : 

1. easy-Wind-power-forecasting - Paridhi Modi
   * MAPE (Mean Absolute Percentage Error): 0.0261
   * RMSE (Root Mean Square Error): 11.626
   * MAE (Mean Absolute Error): 10.543
   * More details can be found [here](https://www.kaggle.com/code/paridhimodi/easy-wind-power-forecasting)

2. Wind Turbine-SARIMA, XGBoost, RandomForest & LSTM - CHRISMAT_10
   * RandomForestRegressor:
     * R² (Coefficient of Determination): 0.915
     * RMSE (Root Mean Square Error): 40.54
     * MAE (Mean Absolute Error): 50.99
     * MAPE (Mean Absolute Percentage Error): 0.064
     * More details can be found [here](https://www.kaggle.com/code/chrismat10/wind-turbine-sarima-xgboost-randomforest-lstm#Results-and-discussion)

