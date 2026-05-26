运行特征报告：
# python featureCorr_Entro.py --input taxi_prediction_hourly_with_weather_may.csv
# python featureCorr_Entro.py --input taxi_prediction_hourly_with_weather_jun.csv

运行xgboost+EMA预测，选前k个特征：
# python xgboostEMA.py --input taxi_prediction_hourly_with_weather_may.csv --k 20