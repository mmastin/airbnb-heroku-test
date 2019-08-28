import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
import pickle
from joblib import load
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
import sklearn.pipeline

# model
model = pickle.load(open('wednesday_model_1_random_forest.pkl', 'rb))

app = Flask(__name__)

@app.route('/', methods=['POST'])

def make_predict():
    #get data
    data = request.get_json(force=True)

    # testing on tuesday model 6
    predict_request = [data['neighborhood'],
                       data['room_type'],
                       data['accommodates'],
                       data['bedrooms'],
                       data['number_of_reviews'],
                       data['wifi'],
                       data['cable_tv'],
                       data['washer'],
                       data['kitchen']]


    data.update((x, [y]) for x, y in data.items())
    data_df = pd.DataFrame.from_dict(data)
                            
    y_hat = model.predict(data_df)

    # send back to browser
    output = {'y_hat': int(y_hat[0])}
    return jsonify(results=output)



if __name__ == '__main__':
    app.run(port = 9000, debug=True)
