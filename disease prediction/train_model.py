import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

# ── 1. Load dataset ──────────────────────────────────────────────
df = pd.read_csv("data/diabetes.csv")
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nDataset info:")
print(df.info())

# ── 2. Handle missing / zero values ──────────────────────────────
# These columns cannot be 0 in real life — replace with median
cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_with_zeros:
    df[col] = df[col].replace(0, df[col].median())

print("\nMissing values after fix:", df.isnull().sum().sum())

# ── 3. EDA — Correlation heatmap ─────────────────────────────────
os.makedirs("static", exist_ok=True)

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("static/heatmap.png")
plt.close()
print("\nHeatmap saved to static/heatmap.png")

# ── 4. Split features and target ─────────────────────────────────
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 5. Scale features ────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── 6. Train 3 models & compare ──────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM":                 SVC(kernel="rbf", probability=True, random_state=42),
}

results = {}
best_acc  = 0
best_name = ""
best_model = None

print("\n" + "="*50)
print("       MODEL COMPARISON RESULTS")
print("="*50)

for name, model in models.items():
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)
    acc    = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"\n{name}")
    print(f"  Accuracy : {acc*100:.2f}%")
    print(f"  Report   :\n{classification_report(y_test, y_pred, target_names=['No Diabetes','Diabetes'])}")
    if acc > best_acc:
        best_acc   = acc
        best_name  = name
        best_model = model

print("="*50)
print(f"\nBest model  : {best_name}")
print(f"Best accuracy: {best_acc*100:.2f}%")

# ── 7. Confusion matrix for best model ───────────────────────────
y_pred_best = best_model.predict(X_test_sc)
cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Diabetes", "Diabetes"],
            yticklabels=["No Diabetes", "Diabetes"])
plt.title(f"Confusion Matrix — {best_name}")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("static/confusion_matrix.png")
plt.close()
print("Confusion matrix saved to static/confusion_matrix.png")

# ── 8. Model accuracy bar chart ──────────────────────────────────
plt.figure(figsize=(7, 4))
bars = plt.bar(results.keys(), [v*100 for v in results.values()],
               color=["#5DCAA5", "#7F77DD", "#D85A30"], edgecolor="none")
plt.ylim(60, 100)
plt.ylabel("Accuracy (%)")
plt.title("Model Accuracy Comparison")
for bar, val in zip(bars, results.values()):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.5,
             f"{val*100:.1f}%", ha="center", va="bottom", fontsize=11)
plt.tight_layout()
plt.savefig("static/model_comparison.png")
plt.close()
print("Model comparison chart saved to static/model_comparison.png")

# ── 9. Save best model + scaler ──────────────────────────────────
os.makedirs("model", exist_ok=True)
with open("model/model.pkl",  "wb") as f: pickle.dump(best_model, f)
with open("model/scaler.pkl", "wb") as f: pickle.dump(scaler,     f)

print(f"\nModel saved  : model/model.pkl")
print(f"Scaler saved : model/scaler.pkl")
print("\nAll done! Run app.py next.")
