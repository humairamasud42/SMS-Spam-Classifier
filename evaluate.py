"""
evaluate.py

This script:
1. Loads trained models
2. Evaluates them on the test set
3. Saves model comparison results
4. Creates confusion matrix
5. Creates class distribution graph
"""

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from preprocess import load_and_preprocess

# create output folder
os.makedirs("outputs", exist_ok=True)

# load dataset
print("Loading Dataset...")
df = load_and_preprocess("data/spam.csv")

# plot class distribution
class_counts = df["label"].value_counts()
labels = ["Ham", "Spam"]
plt.figure(figsize=(6,4))
plt.bar(labels, class_counts)
plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Number of Messages")
plt.savefig("outputs/class_distribution.png")
plt.close()
print("Class Distribution Saved.")

vectorizer = joblib.load("models/vectorizer.pkl")

# load test data
X_test = joblib.load("models/X_test.pkl")
y_test = joblib.load("models/y_test.pkl")

models = {

    "Naive Bayes":
        joblib.load("models/naive_bayes.pkl"),
    "Logistic Regression":
        joblib.load("models/logistic_regression.pkl"),
    "Linear SVM":
        joblib.load("models/linear_svm.pkl")
}

# evalaute models
results = []
best_model = None
best_accuracy = 0
print("\nEvaluating Models...\n")

for name, model in models.items():

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print("="*40)
    print(name)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    results.append({

        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

# save csv
results_df = pd.DataFrame(results)
results_df.to_csv(
    "outputs/model_comparison.csv",
    index=False
)
print("\nModel Comparison CSV Saved.")

# confusion matrix
predictions = best_model.predict(X_test)
cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Ham", "Spam"]
)

disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig("outputs/confusion_matrix.png")
plt.close()
print("Confusion Matrix Saved.")
print("\nEvaluation Completed Successfully!")