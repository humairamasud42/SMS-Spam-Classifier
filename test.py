# temporary file to test the preprocess.py file
from src.preprocess import load_and_preprocess
df = load_and_preprocess("data/spam.csv")
print(df.head())

