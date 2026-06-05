# Diabetes Prediction System Using Machine Learning

## Project Overview

This project is a Machine Learning-based web application that predicts whether a patient is likely to have diabetes based on medical diagnostic parameters. The model is trained using the Pima Indians Diabetes Dataset and deployed using Flask.

The application allows users to enter patient health metrics through a web interface and receive an instant diabetes risk prediction.

---

## Features

* Diabetes prediction using Machine Learning
* Interactive web interface built with Flask
* Real-time prediction results
* Data preprocessing using StandardScaler
* Trained and deployed using Scikit-Learn
* Easy local deployment

---

## Dataset

**Pima Indians Diabetes Dataset**

Features used:

* Pregnancies
* Glucose
* BloodPressure
* SkinThickness
* Insulin
* BMI
* DiabetesPedigreeFunction
* Age

Target Variable:

* Outcome

  * 0 = Non-Diabetic
  * 1 = Diabetic

---

## Technologies Used

### Frontend

* HTML5
* CSS3

### Backend

* Python
* Flask

### Machine Learning

* Scikit-Learn
* Logistic Regression / Random Forest / Decision Tree
* StandardScaler

### Data Analysis

* Pandas
* NumPy

---

## Project Structure

Diabetes_Prediction_Project/

├── templates/

│   └── index.html

│

├── static/

│   └── style.css

│

├── diabetes_deg_model.pkl

├── scaler.pkl

├── application.py

├── diabetes_prediction.ipynb

├── requirements.txt

└── README.md

---

## Installation

Clone the repository:

git clone https://github.com/yourusername/Diabetes-Prediction-ML.git

Move to project directory:

cd Diabetes-Prediction-ML

Install dependencies:

pip install -r requirements.txt

---

## Running the Application

Start the Flask application:

python application.py

Open your browser and visit:

http://127.0.0.1:5000

---

## Machine Learning Workflow

1. Load the diabetes dataset
2. Perform data cleaning and preprocessing
3. Split data into training and testing sets
4. Scale numerical features using StandardScaler
5. Train the Machine Learning model
6. Save the model using Joblib/Pickle
7. Build Flask web application
8. Deploy locally for user interaction

---

## Sample Input

* Pregnancies: 6
* Glucose: 148
* Blood Pressure: 72
* Skin Thickness: 35
* Insulin: 0
* BMI: 33.6
* Diabetes Pedigree Function: 0.627
* Age: 50

Prediction:

Diabetic

---

## Future Enhancements

* Deploy on Render or Railway
* Add patient history storage
* Integrate database support
* Improve model performance using ensemble learning
* Add graphical health insights dashboard

---

## Results

The trained model predicts diabetes risk using patient medical information and provides quick, user-friendly predictions through a web application.

---

## Author

Shreshtha Singh

Master's Student | Bioinformatics | Machine Learning Enthusiast
