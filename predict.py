"""
predict.py

This script loads the trained model and predicts
whether a custom SMS message is Spam or Ham.
"""

import joblib
from preprocess import clean_text

print("Loading vectorizer...")
vectorizer = joblib.load("models/vectorizer.pkl")

# load best model (Linear SVM)
print("Loading trained model...")
model = joblib.load("models/linear_svm.pkl")

# prediction function
def predict_message(message):
    """
    Predict whether a message is Spam or Ham.
    """
    # clean message
    cleaned_message = clean_text(message)
    # convert to TF-IDF
    vectorized_message = vectorizer.transform([cleaned_message])
    # predict
    prediction = model.predict(vectorized_message)
    # return 
    if prediction[0] == 1:
        return "SPAM--- "
    else:
        return "HAM--- "

print("\n========== SMS Spam Classifier ==========\n")

while True:
    message = input("Enter an SMS message:\n")
    result = predict_message(message)
    print(f"\nPrediction: {result}\n")
    again = input("Do you want to classify another message? (y/n): ").lower()
    if again != "y":
        break
print("\nThank you for using the SMS Spam Classifier!")