📩 SMS Spam Classifier using Machine Learning

1. Overview

This project is a machine learning-based text classification system that identifies whether an SMS message is **Spam** or **Ham (Not Spam)**.

The project demonstrates the complete workflow of a Natural Language Processing (NLP) classification task, including preprocessing, feature extraction, model training, evaluation, and prediction.


2. Dataset

**SMS Spam Collection Dataset**

- **Total Messages:** 5,572
- **Classes:** Spam and Ham
- **Task:** Binary Text Classification


3. Project Structure

```
SMS-Spam-Classifier/
│
├── data/
│   └── spam.csv
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
├──test.py
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

4. Machine Learning Workflow

- Load Dataset
- Perform Exploratory Data Analysis (EDA)
- Clean and preprocess text
- Convert text into numerical features using TF-IDF
- Split the dataset into training and testing sets
- Train multiple machine learning models
- Evaluate each model
- Compare model performance
- Predict custom SMS messages


5. Text Preprocessing

The following preprocessing steps were applied:

- Convert text to lowercase
- Remove punctuation
- Remove English stopwords
- Remove extra whitespace


6. Feature Engineering

TF-IDF (Term Frequency–Inverse Document Frequency) was used to convert SMS messages into numerical vectors suitable for machine learning models.


7. Models Used

- Multinomial Naive Bayes
- Logistic Regression
- Linear Support Vector Machine (Linear SVM)


8. Evaluation Metrics

The following metrics were used:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix


9. Results

| Model | Accuracy |
|--------|----------|
| Multinomial Naive Bayes | ~98% |
| Logistic Regression | ~99% |
| Linear SVM | ~99% |

Linear SVM achieved the best overall performance and was selected as the final model.


10. Why This Dataset?

The SMS Spam Collection dataset is a widely recognized benchmark dataset for text classification. It provides real-world labeled SMS messages and is commonly used for evaluating machine learning algorithms in spam detection tasks.


11. Why This Approach?

This project uses TF-IDF vectorization because it effectively transforms text into numerical features by emphasizing informative words while reducing the influence of common words.

Multiple classifiers were trained to compare performance and identify the most suitable model for spam detection. This approach demonstrates practical model selection rather than relying on a single algorithm.


12. How to Run

Clone the Repository

```bash
git clone https://github.com/yourusername/SMS-Spam-Classifier.git
```

Install Dependencies

```bash
pip install -r requirements.txt
```

Train Models

```bash
python src/train.py
```

Evaluate Models

```bash
python src/evaluate.py
```

Predict Custom Messages

```bash
python src/predict.py
```

13. Future Improvements

- Hyperparameter tuning
- Word cloud visualization
- ROC-AUC analysis
- Deep learning models (LSTM, BERT)
- Deploy using Flask or Streamlit
