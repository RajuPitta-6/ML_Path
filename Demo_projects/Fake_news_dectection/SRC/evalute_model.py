import os
import pandas as pd 
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score , classification_report, confusion_matrix

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "Data", "Processed", "cleaned_fake_news.csv")
MODEL_PATH = os.path.join(BASE_DIR, 'models', "fake_news_model.pkl")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
REPORT_PATH = os.path.join(REPORT_DIR, "metrics.txt")

# Creating data frame for data set
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=['text', 'label'])

x = df['text']
y = df['label']
# split
x_train, x_test, y_train, y_test = train_test_split(x , y, test_size=0.2, random_state=42, stratify=y)

# Model loading
model  = joblib.load(MODEL_PATH)


# Predection
y_pred = model.predict(x_test)


# Metrics
accuracy = accuracy_score(y_test,  y_pred)
report  = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
# Save metrics
os.makedirs(REPORT_DIR, exist_ok=True)
with open(REPORT_PATH, "w") as f:
    f.write("MODEL EVALUATION REPORT\n")
    f.write("=======================\n\n")

    f.write(f"Accuracy Score : {accuracy:.4f}\n")
    f.write(f"F1 Score       : {f1:.4f}\n\n")

    f.write("Classification Report:\n")
    f.write(report)
    f.write("\n")

    f.write("Confusion Matrix:\n")
    f.write(str(conf_matrix))

print("✅ Evaluation completed. Metrics saved successfully.")