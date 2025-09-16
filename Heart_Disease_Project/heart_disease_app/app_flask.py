from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load your trained model
model = joblib.load('heart_disease_pipeline.pkl')


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the form data
        form = request.form
        input_data = {
            'age': int(form['age']),
            'sex': int(form['sex']),
            'cp': int(form['cp']),
            'trestbps': int(form['trestbps']),
            'chol': int(form['chol']),
            'fbs': int(form['fbs']),
            'restecg': int(form['restecg']),
            'thalach': int(form['thalach']),
            'exang': int(form['exang']),
            'oldpeak': float(form['oldpeak']),
            'slope': int(form['slope']),
            'ca': int(form['ca']),
            'thal': int(form['thal'])
        }

        # Predict
        df = pd.DataFrame([input_data])
        risk = model.predict_proba(df)[0][1]
        risk_percent = round(risk * 100)

        # Decide risk status
        if risk > 0.5:
            status = "High Risk of Heart Disease"
            color = "danger"
        else:
            status = "Low Risk of Heart Disease"
            color = "success"

        return render_template('result.html', risk=risk_percent, status=status, color=color)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
