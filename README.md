# ===================== Titanic Survival Prediction 🚢 =====================

# 📌 Project Overview
# Predict whether a passenger survived the Titanic disaster using Machine Learning algorithms.
# Dataset: Titanic Dataset from Kaggle.

# 🎯 Objectives
# - Analyze passenger data and identify survival patterns
# - Implement classification algorithms for prediction
# - Compare model performance

# 📂 Dataset
# Source: Kaggle Titanic Dataset
# Features: Pclass, Sex, Age, Fare, SibSp, Parch, Embarked

# ⚙️ Technologies Used
# Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Jupyter Notebook / Kaggle Notebook

# 🧠 Machine Learning Algorithms
# Logistic Regression
# Random Forest Classifier

# 🔄 Project Workflow
# 1. Data Collection
# 2. Data Cleaning & Preprocessing
# 3. Feature Engineering
# 4. Train-Test Split
# 5. Model Training
# 6. Model Evaluation

# ===================== Import Libraries =====================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ===================== Data Collection =====================
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# ===================== Data Cleaning & Preprocessing =====================
train['Age'].fillna(train['Age'].median(), inplace=True)
test['Age'].fillna(test['Age'].median(), inplace=True)
train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)
test['Fare'].fillna(test['Fare'].median(), inplace=True)
train.drop("Cabin", axis=1, inplace=True)
test.drop("Cabin", axis=1, inplace=True)

# ===================== Feature Engineering =====================
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
test['FamilySize'] = test['SibSp'] + test['Parch'] + 1

train['Title'] = train['Name'].apply(lambda x: x.split(",")[1].split(".")[0].strip())
test['Title'] = test['Name'].apply(lambda x: x.split(",")[1].split(".")[0].strip())

# ===================== Feature Selection =====================
features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize', 'Title']
X = train[features]
y = train['Survived']
X_test = test[features]

# ===================== Preprocessing Pipeline =====================
categorical_features = ['Sex', 'Embarked', 'Title']
numeric_features = ['Pclass', 'Age', 'Fare', 'FamilySize']

preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)],
    remainder='passthrough'
)

# ===================== Model Pipelines =====================
log_reg_pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('model', LogisticRegression(max_iter=1000))
])

rf_pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('model', RandomForestClassifier(n_estimators=200, random_state=42))
])

# ===================== Train-Test Split =====================
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ===================== Model Training & Evaluation =====================
# Logistic Regression
log_reg_pipeline.fit(X_train, y_train)
log_pred = log_reg_pipeline.predict(X_val)
log_acc = accuracy_score(y_val, log_pred)

# Random Forest
rf_pipeline.fit(X_train, y_train)
rf_pred = rf_pipeline.predict(X_val)
rf_acc = accuracy_score(y_val, rf_pred)

# ===================== Results =====================
print(f"📊 Logistic Regression Accuracy: {log_acc*100:.0f}%")
print(f"📊 Random Forest Accuracy: {rf_acc*100:.0f}%")

# ===================== Save Predictions =====================
final_predictions = rf_pipeline.predict(X_test)
submission = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': final_predictions})
submission.to_csv("titanic_predictions.csv", index=False)
print("✅ Saved titanic_predictions.csv!")
