# Diabetes Prediction System

A machine learning web app that takes basic health inputs from a patient and tells whether they are likely to have diabetes or not. I built this as my final year project using Python and Flask.

---

## Why I Built This

Diabetes is one of the most common chronic diseases worldwide, and early detection can make a huge difference in treatment outcomes. Most people don't get tested until symptoms appear. This tool lets anyone enter simple health values and instantly know their risk level — no lab visit needed for a preliminary check.

---

## What It Does

You open the web app, fill in 8 health details like glucose level, BMI, age, and blood pressure, hit the predict button, and it tells you whether you are diabetic or not along with a risk probability percentage.

Behind the scenes, three different ML models were trained on real patient data. The one with the best accuracy was saved and connected to the Flask web app.

---

## Project Structure

```
diabetes_prediction/
│
├── data/
│   └── diabetes.csv
├── model/
│   ├── model.pkl
│   └── scaler.pkl
├── templates/
│   ├── index.html
│   └── result.html
├── static/
│   ├── style.css
│   ├── heatmap.png
│   ├── confusion_matrix.png
│   └── model_comparison.png
├── train_model.py
├── app.py
├── requirements.txt
└── README.md
```

---

## Dataset

I used the PIMA Indians Diabetes Dataset from Kaggle. It has records of 768 female patients with 8 health features each. The dataset is widely used in healthcare ML research and is publicly available.

Source: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

The 8 input features are:

| Feature | What it means |
|---|---|
| Pregnancies | How many times the patient has been pregnant |
| Glucose | Blood sugar level measured in mg/dL |
| BloodPressure | Diastolic blood pressure in mm Hg |
| SkinThickness | Triceps skin fold thickness in mm |
| Insulin | Serum insulin level after 2 hours |
| BMI | Body mass index (weight/height ratio) |
| DiabetesPedigreeFunction | A score based on family history of diabetes |
| Age | Patient age in years |

---

## Models I Trained

I trained three models and compared their accuracy on the test set:

| Model | Accuracy |
|---|---|
| Logistic Regression | ~77% |
| Random Forest | ~79% |
| SVM (RBF Kernel) | ~78% |

Random Forest came out on top with around 79% accuracy, so that's the one saved and used in the app. The training script automatically picks whichever model scores highest, so if you retrain with different data it will still pick the best one.

---

## How the App Was Tested

After training, I tested it with two real patient records from the dataset:

**Test 1 — Known diabetic patient:**
Pregnancies: 6, Glucose: 148, BP: 72, Skin: 35, Insulin: 0, BMI: 33.6, Pedigree: 0.627, Age: 50
Result: Diabetic — 86% probability ✓

**Test 2 — Known non-diabetic patient:**
Pregnancies: 1, Glucose: 85, BP: 66, Skin: 29, Insulin: 0, BMI: 26.6, Pedigree: 0.351, Age: 31
Result: Not Diabetic — 1% probability ✓

Both matched the actual labels in the dataset, which shows the model is working correctly.

---

## Tech Stack

| What | Which tool |
|---|---|
| Programming language | Python 3 |
| Data handling | Pandas, NumPy |
| Machine learning | Scikit-learn |
| Charts and plots | Matplotlib, Seaborn |
| Web framework | Flask |
| Frontend | HTML, CSS |
| Saving the model | Pickle |

---

## How to Run It Yourself

**1. Clone the repo**
```bash
git clone https://github.com/revanth838/diabetes-prediction.git
cd diabetes-prediction
```

**2. Install the required libraries**
```bash
pip install -r requirements.txt
```

**3. Train the model first**
```bash
python train_model.py
```
This will print the accuracy of all three models in the terminal and save the best one along with the scaler inside the model/ folder. It also saves the EDA charts to static/.

**4. Start the web app**
```bash
python app.py
```

**5. Open in browser**
```
http://127.0.0.1:5000
```

---

## Known Limitations

- The dataset only has female patients aged 21 and above, so predictions may not be reliable for males or younger patients
- 79% accuracy means roughly 1 in 5 predictions could be wrong — this is not a replacement for a proper medical test
- Some feature values like insulin and skin thickness had many zero entries which were replaced with median values during preprocessing

---

## What I Plan to Add Next

- Support for predicting heart disease and kidney disease using the same framework
- A patient history page so previous predictions can be tracked
- Better accuracy using XGBoost or a neural network
- Deployment on a cloud platform so anyone can access it without installing anything
- Explanation of why the model made a particular prediction using SHAP values

---

## Disclaimer

This project was made for educational purposes as part of my final year AI curriculum. The predictions it gives are based on a statistical model trained on a limited dataset. Please do not use this as a substitute for actual medical diagnosis. Always consult a qualified doctor.

---

## Author

Revanth
Final Year — Artificial Intelligence
