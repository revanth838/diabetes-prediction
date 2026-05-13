# Diabetes Prediction System 🩺

An AI-powered web application that predicts the likelihood of diabetes in a patient based on health parameters using Machine Learning. Built with Python, Scikit-learn, and Flask.

---

## 📌 Project Overview

This project uses the **PIMA Indians Diabetes Dataset** to train multiple machine learning models and predict whether a patient is diabetic or not. The best-performing model is deployed as a web application where users can enter health values and get an instant prediction with a probability score.

---

## 🎯 Features

- Trained and compared 3 ML models — Logistic Regression, Random Forest, SVM
- Automatically selects the best model based on accuracy
- Clean web interface to enter patient data
- Shows prediction result with risk probability percentage
- Visual risk bar for easy understanding
- EDA charts: correlation heatmap, confusion matrix, model comparison

---

## 🗂️ Project Structure

```
diabetes_prediction/
│
├── data/
│   └── diabetes.csv              # PIMA Indians Diabetes Dataset
│
├── model/
│   ├── model.pkl                 # Saved best ML model
│   └── scaler.pkl                # Saved StandardScaler
│
├── templates/
│   ├── index.html                # Input form page
│   └── result.html               # Prediction result page
│
├── static/
│   ├── style.css                 # Stylesheet
│   ├── heatmap.png               # Correlation heatmap (auto-generated)
│   ├── confusion_matrix.png      # Confusion matrix (auto-generated)
│   └── model_comparison.png      # Model accuracy chart (auto-generated)
│
├── train_model.py                # ML training script
├── app.py                        # Flask web application
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

---

## 🧠 Dataset

- **Name:** PIMA Indians Diabetes Dataset
- **Source:** [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Records:** 768 patients
- **Features:** 8 health parameters
- **Target:** Outcome (1 = Diabetic, 0 = Not Diabetic)

### Input Features

| Feature | Description |
|---|---|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose concentration (mg/dL) |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin fold thickness (mm) |
| Insulin | 2-Hour serum insulin (mu U/ml) |
| BMI | Body mass index (kg/m²) |
| DiabetesPedigreeFunction | Diabetes hereditary score |
| Age | Age of the patient (years) |

---

## 🤖 Models Trained & Compared

| Model | Description |
|---|---|
| Logistic Regression | Simple, interpretable baseline model |
| Random Forest | Ensemble of decision trees, handles non-linearity |
| SVM (RBF Kernel) | Effective for high-dimensional health data |

> The best performing model is automatically saved and used in the web app.

---

## 📊 Results

| Model | Accuracy |
|---|---|
| Logistic Regression | ~77% |
| Random Forest | ~79% |
| SVM | ~78% |

**Best Model: Random Forest** with approximately **79% accuracy**

### Test Case Results

| Test | Input Profile | Prediction | Probability |
|---|---|---|---|
| Case 1 | High glucose, high BMI, age 50 | Diabetic | 86% |
| Case 2 | Low glucose, normal BMI, age 31 | Not Diabetic | 1% |

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.x |
| ML Library | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Web Framework | Flask |
| Frontend | HTML, CSS |
| Model Saving | Pickle |

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/diabetes-prediction.git
cd diabetes-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model
```bash
python train_model.py
```
This will train all models, print accuracy comparison, and save `model.pkl` and `scaler.pkl`.

### 4. Run the web app
```bash
python app.py
```

### 5. Open in browser
```
http://127.0.0.1:5000
```

---

## 📦 Requirements

```
flask
pandas
numpy
scikit-learn
matplotlib
seaborn
```

Install all at once:
```bash
pip install flask pandas numpy scikit-learn matplotlib seaborn
```

---

## 🔮 Future Scope

- Add more diseases (heart disease, kidney disease)
- Improve accuracy using XGBoost or deep learning
- Add user login and patient history tracking
- Deploy on cloud (Render / AWS / Heroku)
- Add SHAP values to explain predictions to patients

---

## ⚠️ Disclaimer

This application is built for educational purposes only. The predictions made by this model are **not a substitute for professional medical diagnosis**. Always consult a qualified doctor for medical advice.

---

## 👨‍💻 Author

**Revanth**
Final Year Student — Artificial Intelligence
