# Data preprocessing

# importing libirires
import os
import pandas as pd
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "Data", "Raw", "train_news.csv")

df = pd.read_csv(DATA_PATH)

# Droping unwanted coloumns
df.drop(columns= ['Unnamed: 0', 'id', 'written_by'], inplace=True)

# Combining headline and news 
df['text'] = df['headline'].astype(str)+ " " + df['news'].astype(str)

# Text cleaning
def text_cleaning(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['text'] = df['text'].apply(text_cleaning)

# droping unnecessry coloumns
df.drop(columns=['news', 'headline'], inplace=True)

# Save processed data
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

processed_path = os.path.join(BASE_DIR, "Data", "Processed")
os.makedirs(processed_path, exist_ok=True)

df.to_csv(os.path.join(processed_path, "cleaned_fake_news.csv"), index=False)
