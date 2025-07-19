from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load model
model = joblib.load('heart_model.pkl')

def get_health_advice(data):
    tips = []
    if data['chol'] > 240:
        tips.append("🧈 High cholesterol — reduce saturated fat.")
    if data['thalach'] < 100:
        tips.append("💓 Low max heart rate — consider cardiology checkup.")
    if data['exang'] == 1:
        tips.append("🏃 Exercise-induced angina — medical attention recommended.")
    if data['age'] > 60:
        tips.append("👴 Age above 60 — regular heart checkups advised.")
    if data['oldpeak'] > 2.0:
        tips.append("📉 Significant ST depression detected — could indicate ischemia.")
    return tips

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

    input_data = [float(request.form[feature]) for feature in features]
    data_dict = dict(zip(features, input_data))

    risk = model.predict_proba([input_data])[0][1]
    risk_percent = int(risk * 100)
    tips = get_health_advice(data_dict)

    return render_template('result.html', risk=risk_percent, tips=tips)

if __name__ == '__main__':
    app.run(debug=True)
