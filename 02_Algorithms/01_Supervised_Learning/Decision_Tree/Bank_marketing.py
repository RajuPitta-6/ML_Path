# importing libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# file path & creating data frame
file_path = 'C:\\Users\\rajup\\Data_sets\\bank.csv'
df = pd.read_csv(file_path)

# ------------------------
# checking for null values
print(df.isnull().sum())

# ------------------------
# Droping Noisy columns 
df = df.drop(columns=['duration', 'day', 'month', 'pdays'])

# --------------------------
# Seprates Features and Targeting column
x = df.drop(columns=['deposit'], axis=1)
y = df['deposit']

# -----------------------
# Spliting Train and Test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify= y, random_state=42)

# categorial coloumns
categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome']

# -----------------------
# Preprocessing

preprocess = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown = 'ignore'), categorical_cols)], remainder='passthrough')

# ---------------------------
# Base decision tree model
dt = DecisionTreeClassifier(random_state=42)

# -----------------------------
# pipeline

pipeline = Pipeline(steps=[('preprocess', preprocess), ('model', dt)])


# Grid parameters
param_grid = {
    'model__max_depth': [3, 5, 7, 10, None],
    'model__min_samples_split': [2, 10, 20, 50],
    'model__min_samples_leaf': [1, 5, 10, 20]
}

# --------------------
# GridsearchCV
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)


# Train
grid_search.fit(x_train, y_train)

# Best model selection
best_model = grid_search.best_estimator_

# predection
y_pred =best_model.predict(x_test)

# --------------------------------------
# Result
print("Best Parameters:")
print(grid_search.best_params_)

print("\nTest Accuracy:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

'''
age          0
job          0
marital      0
education    0
default      0
balance      0
housing      0
loan         0
contact      0
day          0
month        0
duration     0
campaign     0
pdays        0
previous     0
poutcome     0
deposit      0
dtype: int64
Best Parameters:
{'model__max_depth': 7, 'model__min_samples_leaf': 1, 'model__min_samples_split': 50}

Test Accuracy:
0.6735333631885356

Classification Report:
              precision    recall  f1-score   support

          no       0.68      0.73      0.70      1175
         yes       0.67      0.61      0.64      1058

    accuracy                           0.67      2233
   macro avg       0.67      0.67      0.67      2233
weighted avg       0.67      0.67      0.67      2233'''