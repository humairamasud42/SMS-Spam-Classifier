"""
train.py

This script:
1. Loads and preprocesses the SMS Spam dataset
2. Converts text into TF-IDF features
3. Splits the dataset into training and testing sets
4. Trains three machine learning models
5. Saves trained models and vectorizer
"""

import os
import joblib

from preprocess import load_and_preprocess
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# create models folder if it doesn't exist
os.makedirs("models", exist_ok=True)
# load dataset

print("Loading dataset...")
df = load_and_preprocess("data/spam.csv")
print("Dataset Loaded Successfully!")
print(df.head())

# TF-IDF vectorization
print("\nCreating TF-IDF features...")
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["clean_message"])
y = df["label"]

# save important TF-IDF words
feature_names = vectorizer.get_feature_names_out()
os.makedirs("outputs", exist_ok=True)
with open("outputs/tfidf_features.txt", "w", encoding="utf-8") as file:
    for word in feature_names:
        file.write(word + "\n")

print("TF-IDF features saved.")

# train-test split
print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# save train/test data
joblib.dump(X_train, "models/X_train.pkl")
joblib.dump(X_test, "models/X_test.pkl")

joblib.dump(y_train, "models/y_train.pkl")
joblib.dump(y_test, "models/y_test.pkl")
print("Training and testing data saved.")

# model initializer
models = {

    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Linear SVM": LinearSVC()

}

# train models and save them
print("\nTraining Models...\n")

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_train)
    accuracy = accuracy_score(y_train, predictions)
    print(f"{name}")
    print(f"Training Accuracy : {accuracy:.4f}")
    print("-" * 40)
    filename = name.lower().replace(" ", "_") + ".pkl"
    joblib.dump(model, f"models/{filename}")

# save vectorizer
joblib.dump(vectorizer, "models/vectorizer.pkl")
print("\nVectorizer Saved!")
print("All Models Saved Successfully!")
print("\nTraining Completed Successfully!")