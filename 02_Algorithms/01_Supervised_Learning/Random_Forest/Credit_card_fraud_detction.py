# Load Data
file_path = 'C:\\Users\\rajup\\Data_sets\\creditcard.csv'

# importing libraries
import pandas as pd 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, recall_score, f1_score
from sklearn.pipeline import Pipeline



# creating Data frame
df = pd.read_csv(file_path)

# Checking for null values
print(df.isnull().sum())

# Slecting featuress and target
x = df.drop(columns=['Class'], axis=1)
y = df['Class']

# Splicting train and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

# ---------------------
# No need of preprocessing 

# -------------------------


# Base model of Random forest
model = RandomForestClassifier(random_state=42)

# Pipelinr
pipline = Pipeline(steps=[('model', model)])

# param_grid
param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [None, 10, 20],
    'model__min_samples_leaf': [1, 5]
}

# GridSearchCV
grid_search = GridSearchCV(estimator=pipline, param_grid= param_grid, cv=5, scoring= 'accuracy', n_jobs= -1)

# Train
grid_search.fit(x_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# predection
y_pred = best_model.predict(x_test)


# ==============================
# Metrics
print("Accuracy score :", accuracy_score(y_test, y_pred))
print("Classification Report :", classification_report(y_test, y_pred))
print("Recall score :", recall_score(y_test, y_pred))
print("f1 score :", f1_score(y_test, y_pred))