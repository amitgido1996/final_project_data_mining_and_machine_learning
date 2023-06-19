import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os
import pandas as pd

app = Flask(__name__)
rf_model = pickle.load(open('trained_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html', prediction_result=None)

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    city = request.form['City']
    house_type = request.form['type']
    room_number = request.form['room_number']
    area = request.form['Area']
    street = request.form['Street']
    city_area = request.form['city_area']

    # Perform any necessary preprocessing on the input data
    # ...

    # Create the feature DataFrame for prediction
    features = pd.DataFrame([[city, house_type, room_number, area, street, city_area]],
                            columns=['City', 'type', 'room_number', 'Area', 'Street', 'city_area'])

    # Perform the prediction using the loaded model
    prediction = rf_model.predict(features)[0]

    # Render the template with the prediction result
    return render_template('index.html', prediction_result=prediction, city=city, house_type=house_type, room_number=room_number,
                           area=area, street=street, city_area=city_area)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
