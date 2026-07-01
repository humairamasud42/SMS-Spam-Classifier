# 📩 SMS Spam Classifier using Machine Learning

## Overview

This project is a machine learning-based text classification system that identifies whether an SMS message is **Spam** or **Ham (Not Spam)**.

The project demonstrates the complete workflow of a Natural Language Processing (NLP) classification task, including preprocessing, feature extraction, model training, evaluation, and prediction.

---

## Dataset

**SMS Spam Collection Dataset**

- **Total Messages:** 5,572
- **Classes:** Spam and Ham
- **Task:** Binary Text Classification

---

## Project Structure

```
SMS-Spam-Classifier/
│
├── data/
│   └── spam.csv
│
├── notebooks/
│   └── analysis.ipynb
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── outputs/
│   ├── class_distribution.png
│   ├── confusion_matrix.png
│   ├── model_comparison.csv
│   └── tfidf_features.txt
│
├── models/
│   ├── linear_svm.pkl
│   ├── logistic_regression.pkl
│   ├── naive_bayes.pkl
│   └── vectorizer.pkl
│
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

## Machine Learning Workflow

1. Load Dataset
2. Perform Exploratory Data Analysis (EDA)
3. Clean and preprocess text
4. Convert text into numerical features using TF-IDF
5. Split the dataset into training and testing sets
6. Train multiple machine learning models
7. Evaluate each model
8. Compare model performance
9. Predict custom SMS messages

---

## Text Preprocessing

The following preprocessing steps were applied:

- Convert text to lowercase
- Remove punctuation
- Remove English stopwords
- Remove extra whitespace

---

## Feature Engineering

TF-IDF (Term Frequency–Inverse Document Frequency) was used to convert SMS messages into numerical vectors suitable for machine learning models.

---

## Models Used

- Multinomial Naive Bayes
- Logistic Regression
- Linear Support Vector Machine (Linear SVM)

---

## Evaluation Metrics

The following metrics were used:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

## Results

| Model | Accuracy |
|--------|----------|
| Multinomial Naive Bayes | ~98% |
| Logistic Regression | ~99% |
| Linear SVM | ~99% |

Linear SVM achieved the best overall performance and was selected as the final model.

---

## Why This Dataset?

The SMS Spam Collection dataset is a widely recognized benchmark dataset for text classification. It provides real-world labeled SMS messages and is commonly used for evaluating machine learning algorithms in spam detection tasks.

---

## Why This Approach?

This project uses TF-IDF vectorization because it effectively transforms text into numerical features by emphasizing informative words while reducing the influence of common words.

Multiple classifiers were trained to compare performance and identify the most suitable model for spam detection. This approach demonstrates practical model selection rather than relying on a single algorithm.

---

## How to Run

### Clone the Repository

```bash
git clone https://github.com/yourusername/SMS-Spam-Classifier.git
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train Models

```bash
python src/train.py
```

### Evaluate Models

```bash
python src/evaluate.py
```

### Predict Custom Messages

```bash
python src/predict.py
```

---

## Future Improvements

- Hyperparameter tuning
- Word cloud visualization
- ROC-AUC analysis
- Deep learning models (LSTM, BERT)
- Deploy using Flask or Streamlit

---

BS Computer Science

AI/ML Internship Project
