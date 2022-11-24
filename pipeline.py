import os 
import pandas as pd
import cryptocompare

from datetime import datetime
import tensorflow as tf

import inspect
import numpy as np
import matplotlib.pyplot as plt

import hopsworks

# connect to hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

# connect to cryptocompare
cryptocompare.cryptocompare._set_api_key_parameter(os.environ['CRYPTOCOMPARE_API_KEY'])

# get data and update fg
minute_btc = cryptocompare.get_historical_price_minute('BTC', currency='USD', limit=1940)
min_pdf = pd.DataFrame.from_dict(minute_btc) 

min_pdf["month"] = min_pdf["time"].map(lambda x: datetime.fromtimestamp(x))
min_pdf["month"] = min_pdf.month.map(lambda x: datetime(x.year, x.month, 1))
min_pdf["month"] = min_pdf["month"].dt.strftime("%Y-%m-%d")
min_pdf["time"] = min_pdf["time"].map(lambda x: int(x * 1000))

min_pdf = min_pdf[["time","high","low","open","volumefrom","close", "month"]]

min_pdf.columns = ["time","high","low","open","volume","close", "month"]

source_fg = fs.get_or_create_feature_group(
    name="btc_source",
    version=1,
)
source_fg.insert(min_pdf)

# get feature view
feature_view = fs.get_feature_view("minute_with_tf", 1)
feature_view.init_serving(1)
td_transformation_functions = feature_view._single_vector_server._transformation_functions

# fetch batch data
min_pdf.sort_values(["time"], inplace=True)
end_time = int(max(min_pdf["time"].values))
start_time = int(end_time - 3590000)
prev_start_time = int(start_time - 3590000)

dataset = feature_view.get_batch_data(start_time=start_time, end_time=end_time)
for_prev_dataset = feature_view.get_batch_data(start_time=prev_start_time, end_time=start_time)

dataset.sort_values(["time"], inplace=True)
for_prev_dataset.sort_values(["time"], inplace=True)

dataset = dataset[["open", "close", "high", "low", "volume"]]
for_prev_dataset = for_prev_dataset[["open", "close", "high", "low", "volume"]]

# load model
mr = project.get_model_registry()
model = mr.get_model("cnn_lstm_autoreg", version = 3)
model_dir = model.download()
loaded_model = tf.saved_model.load(model_dir)
serving_function = loaded_model.signatures["serving_default"]

# predict
prediction = serving_function(tf.constant(dataset.values.reshape(-1, 60, 5), dtype=tf.float32))["output_1"].numpy()[0][0][0]
prediction_prev = serving_function(tf.constant(for_prev_dataset.values.reshape(-1, 60, 5), dtype=tf.float32))["output_1"].numpy()[0][0][0]

# prepare prediction logs
for feature_name in td_transformation_functions:
    if feature_name == "close":
        td_transformation_function = td_transformation_functions[feature_name]
        sig, foobar_locals = inspect.signature(td_transformation_function.transformation_fn), locals()
        param_dict = dict([(param.name, param.default) for param in sig.parameters.values() if param.default != inspect._empty])
        if td_transformation_function.name == "min_max_scaler":
            prediction = prediction*(param_dict["max_value"]-param_dict["min_value"])+param_dict["min_value"]
            predict_previous = prediction_prev * (param_dict["max_value"]-param_dict["min_value"])+param_dict["min_value"]
            known_previous = dataset["close"].values[59]*(param_dict["max_value"]-param_dict["min_value"])+param_dict["min_value"]


percent_change = ((prediction - predict_previous)/predict_previous)*100
up_or_down = "no change"
if percent_change > 0:
    up_or_down = "up"
elif percent_change < 0:
    up_or_down = "down"

prediction_for_timestamp = end_time + 3600000 
prediction_time = datetime.utcfromtimestamp(end_time // 1000).strftime("%Y-%m-%d %H:%M")
prediction_for_time = datetime.utcfromtimestamp(prediction_for_timestamp // 1000).strftime("%Y-%m-%d %H:%M")

# initialise data of lists.
data_pred_logs = {'pk': [prediction_for_timestamp], 'prediction_time': [prediction_time], 'close_at_prediction':[known_previous], 'predicted_at_prediction':[predict_previous], 'prediction_for_time':[prediction_for_time] , 'predicted_close':[prediction], 'up_or_down': [up_or_down]}
# Create DataFrame
df_pred_logs = pd.DataFrame(data_pred_logs)

# update prediction logs
pred_logs = fs.get_or_create_feature_group(
    name="btc_prediction_logs",
    version=1,
    description="prediction logs",
    primary_key=['pk'],
    event_time='pk',
    online_enabled=True
)
pred_logs.insert(df_pred_logs)

t =  f"Predicted absolute value by {prediction_for_time} is {round(prediction, 4)} $ and its expected to go {up_or_down} by {round(percent_change, 4)} percent"
fig = plt.figure(figsize=(15.5, 2))
plt.text(0.01, 0.5, t, dict(size=15))
plt.savefig('images/model_preds.png')

