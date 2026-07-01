import pandas as pd
import string
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def clean_text(text):
    """
    Clean a text message by:
    1. Converting to lowercase
    2. Removing punctuation
    3. Removing stopwords
    4. Removing extra spaces
    """
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))

    # split into words
    words = text.split()
    # remove stopwords
    words = [word for word in words if word not in stop_words]
    # join words back
    return " ".join(words)

def load_and_preprocess(filepath):
    """
    Load dataset and preprocess it.
    """
    df = pd.read_csv(filepath, encoding="latin-1")
    # only required columns
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']
    df['label'] = df['label'].map({
        'ham': 0,
        'spam': 1
    })
    df['clean_message'] = df['message'].apply(clean_text)
    return df