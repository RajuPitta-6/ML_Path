# importing libirires
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib


# Load Data set
path = 'C:\\Users\\rajup\\Data_sets\\email.csv'
df = pd.read_csv(path)

# print(df.head())

# Text cleaning 
def clean_text(text):
    # convert to lower case
    text = text.lower()

    # Remove special characters but keep space
    text = re.sub(r'[^a-zA-z\s]', '', text)

    # tokenization
    tokens = text.split()

    # join back
    cleaned_text = " ".join(tokens)

    return cleaned_text


df['Message'] = df['Message'].apply(clean_text)

# Label encoding for target column
le = LabelEncoder()
df['Category'] = le.fit_transform(df['Category'])

x = df['Message']
y = df['Category']

# Spliting train and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)

# vectorization
tf = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words="english"
        )
# Model 
model = MultinomialNB()

pipeline = Pipeline(
    steps=[
        ("tfidf", tf),
        ("model", model)
    ]
)
# Model Training
pipeline.fit(x_train, y_train)

# prediction
y_pred = pipeline.predict(x_test)

# Accuracy
print("Accuracy score :", accuracy_score(y_test, y_pred))
print("classification report :", classification_report(y_test, y_pred))

# model saving
joblib.dump(pipeline, "Spam_mail_classifier.pkl")

# loading model
pipeline_ = joblib.load("Spam_mail_classifier.pkl")

# Sample test
new_email = ["Congratulations! You won a free lottery. Claim now"]
prediction = pipeline_.predict(new_email)
print(prediction)
# [1]


'''
output :
Accuracy score : 0.9684361549497847
classification report :               precision    recall  f1-score   support

           0       0.97      1.00      0.98      1208
           1       0.99      0.77      0.87       186

    accuracy                           0.97      1394
   macro avg       0.98      0.88      0.92      1394
weighted avg       0.97      0.97      0.97      1394

[1]
'''
