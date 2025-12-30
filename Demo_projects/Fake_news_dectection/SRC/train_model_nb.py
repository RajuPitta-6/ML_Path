# MOdel training

# importing libiries
import os
import pandas as pd
from sklearn.model_selection import  train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

# path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "Data", "Processed", "cleaned_fake_news.csv")

# creating data frame
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=['text', 'label'])

# selecting features
x = df['text']
y = df ['label']

# spliting train and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

# Vectorization
tf = TfidfVectorizer(
    max_features=5000,
            ngram_range=(1, 2),
            stop_words="english"
)

# Model
model = MultinomialNB()

# Pipeline
pipeline = Pipeline(
    steps=[
        ("tfidf", tf),
        ("model", model)
    ]
)

# model training
pipeline.fit(x_train, y_train)


# save model
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "fake_news_model.pkl")
joblib.dump(pipeline, MODEL_PATH)

print("Model trained and saved at :", MODEL_PATH)
