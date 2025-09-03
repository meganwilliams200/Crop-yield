from flask import Flask, request, render_template
import numpy as np
import pickle

# Importing model
model = pickle.load(open('model.pkl1', 'rb'))

# Creating Flask app
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    N = int(request.form['Nitrogen'])
    P = int(request.form['Phosporus'])
    K = int(request.form['Potassium'])
    temp = float(request.form['Temperature'])
    humidity = float(request.form['Humidity'])
    ph = float(request.form['ph'])  # Changed to float, assuming ph should be a float
    rainfall = float(request.form['Rainfall'])

    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    # Predict using the loaded model
    prediction = model.predict(single_pred)

    # Mapping from prediction to crop names
    crop_dict = {
        'rice': 1,
        'maize': 2,
        'chickpea': 3,
        'kidneybeans': 4,
        'pigeonpeas': 5,
        'mothbeans': 6,
        'mungbean': 7,
        'blackgram': 8,
        'lentil': 9,
        'pomegranate': 10,
        'banana': 11,
        'mango': 12,
        'grapes': 13,
        'watermelon': 14,
        'muskmelon': 15,
        'apple': 16,
        'orange': 17,
        'papaya': 18,
        'coconut': 19,
        'cotton': 20,
        'jute': 21,
        'coffee': 22
    }

    # Check prediction and output result
    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = "{} is the best crop to be cultivated right there.".format(prediction[0])
    else:
        result = "Sorry, we are not able to predict a crop for the given environment."

    return render_template('index.html', result=result)

# Python main
if __name__ == "__main__":
    app.run(debug=True)
