from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load saved model and scaler
with open("model/model.pkl",  "rb") as f: model  = pickle.load(f)
with open("model/scaler.pkl", "rb") as f: scaler = pickle.load(f)

FEATURE_NAMES = [
    "Pregnancies", "Glucose", "BloodPressure",
    "SkinThickness", "Insulin", "BMI",
    "DiabetesPedigreeFunction", "Age"
]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        values = [float(request.form[f]) for f in FEATURE_NAMES]
        arr    = np.array(values).reshape(1, -1)
        arr_sc = scaler.transform(arr)

        prediction = model.predict(arr_sc)[0]
        probability = model.predict_proba(arr_sc)[0][1] * 100

        result = "Diabetic" if prediction == 1 else "Not Diabetic"
        risk   = round(probability, 1)
        return render_template("result.html",
                               result=result,
                               risk=risk,
                               values=dict(zip(FEATURE_NAMES, values)))
    except Exception as e:
        return render_template("result.html",
                               result="Error",
                               risk=0,
                               error=str(e),
                               values={})

if __name__ == "__main__":
    app.run(debug=True)
