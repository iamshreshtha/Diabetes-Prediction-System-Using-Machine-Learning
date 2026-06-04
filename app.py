from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("diabetes_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    values = [
        float(request.form["pregnancies"]),
        float(request.form["glucose"]),
        float(request.form["bp"]),
        float(request.form["skin"]),
        float(request.form["insulin"]),
        float(request.form["bmi"]),
        float(request.form["dpf"]),
        float(request.form["age"])
    ]

    data = np.array([values])

    data = scaler.transform(data)

    result = model.predict(data)

    if result[0] == 1:
        prediction = "Diabetic"
    else:
        prediction = "Non-Diabetic"

    return render_template(
        "index.html",
        prediction_text=prediction
    )

if __name__ == "__main__":
    app.run(debug=True)