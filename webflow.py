import os 
import json
import requests

import pandas as pd

from datetime import datetime, timezone

import inspect
import numpy as np

import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()

btc_source = fs.get_feature_group("btc_source", 1)
btc_prediction_logs = fs.get_feature_group("btc_prediction_logs", 1)

# read feature groups to pdf
min_pdf = btc_source.read() 
pred_pdf = btc_prediction_logs.read()
pred_pdf = pred_pdf.sort_values(by="pk", ascending=[False])

pred_pdf["time"] = pred_pdf.prediction_for_time.map(lambda x: int(datetime.strptime(x, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc).timestamp() * 1000))
pred_pdf = pred_pdf.merge(min_pdf[["time", "close"]], on=["time"], how='left')
pred_pdf["time"] = pred_pdf.prediction_time.map(lambda x: int(datetime.strptime(x, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc).timestamp() * 1000))
#pred_pdf.merge(min_pdf[["time", "close"]], on=["time"])
pred_pdf["what_happened"] = pred_pdf.close > pred_pdf.close_at_prediction 
pred_pdf["predicted_diff"] = pred_pdf.predicted_close - pred_pdf.predicted_at_prediction
pred_pdf["actual_diff"] = pred_pdf.close - pred_pdf.close_at_prediction
pred_pdf["what_happened"] = pred_pdf.what_happened.map(lambda x: "up" if x==True else ("down" if x==False else "missing"))
#pred_pdf.drop(['close', 'time'], axis=1, inplace=True)
pred_pdf = pred_pdf[["pk", "prediction_time", "close_at_prediction", "predicted_at_prediction", "prediction_for_time", "predicted_close", "up_or_down", "close", "what_happened", "predicted_diff", "actual_diff"]]
pred_pdf.columns = ["pk", "prediction_time", "close_at_prediction", "predicted_at_prediction", "prediction_for_time", "predicted_close", "up_or_down", "actual_close", "what_happened", "predicted_diff", "actual_diff"]
pred_pdf

##### code for updating the website 
pred_pdfs = pred_pdf.sort_values(by="pk", ascending=[False])
latest_row = pred_pdf.head(1)
latest_row_2 = latest_row.rename(columns={ 'pk': 'pk-n',
                    'prediction_time': 'prediction-time', 
                   'close_at_prediction': 'close-at-prediction-2', 
                   'predicted_at_prediction': 'predicted-at-prediction-2',
                   'prediction_for_time': 'prediction-for-time-3',
                   'prediction_time': 'prediction-time-2', 
                   'predicted_close': 'predicted-close-2', 
                   'up_or_down': 'up-or-down-2',
                   'what_happened': 'what-happened-2',
                   'predicted_diff': 'predicted-diff-2',
                   'actual_diff': 'actual-diff-2'
                   })

latest_row_2["slug"] = "prediction-last"
latest_row_2['name'] = 'prediction-last'
latest_row_2['_archived'] = False
latest_row_2['_draft'] = False

# convert panda df to json 
latest_row_json = latest_row_2.to_json(orient='records', lines=True)

# prepare the json string exactly as 'requests' API expects
string_result = json.dumps({"fields": json.JSONDecoder().decode(latest_row_json)})

# replace lower case Bolean value with the upper one as it got changed when to_json  
result_json = string_result.replace("false", "False")

#Convert json string to json onject
result_json = json.loads(string_result)

## PUT record in weblow
url1 = os.environ['WEBFLOW_URL_1']

headers = {
    "accept": "application/json",
    "content-type": "application/json",
    "authorization": f"Bearer {os.environ['WEBFLOW_API_KEY']}"
}

response = requests.put(url1, json=result_json, headers=headers)

print(response.text)

#### retrieve the two columns converted them to json and put them into webflow
pred_pdf = pred_pdf.sort_values(by="pk")

predicted_close = pred_pdf["predicted_close"].tolist()
predicted_close_str = str(predicted_close).replace('[', '').replace(']', '')

actual_close = pred_pdf["actual_close"].tolist()
actual_close_str = str(actual_close).replace('[', '').replace(']', '')

result = {"fields": {
        "slug": "prediction-last",
        "name": "prediction-last",
        "_archived": False,
        "_draft": False,
        "all-x": predicted_close_str,
        "all-y": actual_close_str,
        "values": str(list(range(0, len(predicted_close)+1 ))).replace('[', '').replace(']', '')
      }
}

## update the website with all predicted values

######
url2 = os.environ['WEBFLOW_URL_2']
headers = {
    "accept": "application/json",
    "content-type": "application/json",
    "authorization": f"Bearer {os.environ['WEBFLOW_API_KEY']}"
}

response = requests.put(url2, json=result, headers=headers)

print(response.text)
