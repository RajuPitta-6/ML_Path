import os
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "fake_news_model.pkl")

# model load
model = joblib.load(MODEL_PATH)

def predict_news(text):
    predection = model.predict([text])[0]
    return "FAKE NEWS" if predection == 1 else "REAL NEWS"

# Example
if __name__=="__main__":
    
    sample_news = [
    # REAL NEWS
    "The United States Senate passed a bipartisan bill to fund infrastructure projects across several states.",

    # FAKE NEWS
    "US government announced that all citizens will receive $10,000 every month starting next week.",

    # REAL NEWS
    "The Federal Reserve decided to keep interest rates unchanged amid concerns over inflation.",

    # FAKE NEWS
    "NASA confirmed that aliens have signed a peace treaty with the United States military.",

    # REAL NEWS
    "California authorities ordered evacuations as wildfires spread due to extreme weather conditions.",

    # FAKE NEWS
    "The US President secretly resigned last night and handed over power to the army.",

    # REAL NEWS
    "The Supreme Court of the United States heard arguments on a major constitutional case today.",

    # FAKE NEWS
    "Drinking lemon water twice a day has been officially declared as a cure for all cancers by US doctors.",

    # REAL NEWS
    "The US Department of Labor reported an increase in employment rates for the third consecutive month.",

    # FAKE NEWS
    "America will shut down the internet for one week to upgrade its national cyber system."
     ]
    for i in sample_news:
        result = predict_news(i)
        print("Prediction:", result)



'''
Prediction: REAL NEWS 
Prediction: REAL NEWS
Prediction: FAKE NEWS
Prediction: FAKE NEWS
Prediction: REAL NEWS
Prediction: FAKE NEWS
Prediction: REAL NEWS
Prediction: FAKE NEWS
Prediction: REAL NEWS
Prediction: FAKE NEWS

Score : 8/10'''

 

