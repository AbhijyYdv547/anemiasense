import numpy as np 
import pickle
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__, static_url_path='/Flask/static')
model = pickle.load(open('model.pkl','rb'))
print("Model loaded successfully!")

EXPECTED_FEATURES = ['Gender', 'Hemoglobin', 'MCH', 'MCHC', 'MCV']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=["GET","POST"])
def predict():
    if request.method == "POST":
        input_data = {}
        for feature_name in EXPECTED_FEATURES:
            value = request.form.get(feature_name)

            if feature_name == 'Gender':
                if value.lower() == 'male':
                    input_data[feature_name] = 1.0
                elif value.lower() == 'female':
                    input_data[feature_name] = 0.0
                else:
                    input_data[feature_name] = 0.0 
            else:
                input_data[feature_name] = float(value)

        features_df = pd.DataFrame([input_data])
        features_df = features_df[EXPECTED_FEATURES]

        prediction = model.predict(features_df)
        predicted_value = prediction[0]
        print(f"Raw prediction: {predicted_value}")

        if predicted_value == 0:
            result_message = "You don't have any Anemic Disease"
        elif predicted_value == 1:
            result_message = "You have anemic disease"
        else:
            result_message = "Prediction out of expected range."

        text_prefix = "Hence, based on calculations: "
        return render_template("predict.html", prediction_text=text_prefix + result_message)
    else:
        return render_template('predict.html')

if __name__ == "__main__":
    app.run(debug=False, port=5000)




