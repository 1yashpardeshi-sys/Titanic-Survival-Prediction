# ===================== Titanic Survival Prediction Project =====================

# 📌 Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Set plot aesthetics to match report style
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12})  # Matches report font size

# ===================== Data Collection =====================
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# ===================== Data Cleaning & Preprocessing =====================
# Fill missing values
train['Age'].fillna(train['Age'].median(), inplace=True)
test['Age'].fillna(test['Age'].median(), inplace=True)

train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)
test['Fare'].fillna(test['Fare'].median(), inplace=True)

# Drop Cabin column (too many missing values)
train.drop("Cabin", axis=1, inplace=True)
test.drop("Cabin", axis=1, inplace=True)

# ===================== Feature Engineering =====================
# Family Size
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
test['FamilySize'] = test['SibSp'] + test['Parch'] + 1

# Extract Title from Name
def extract_title(name):
    return name.split(",")[1].split(".")[0].strip()

train['Title'] = train['Name'].apply(extract_title)
test['Title'] = test['Name'].apply(extract_title)

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

# ===================== Train Models =====================
# Logistic Regression
log_reg_pipeline.fit(X_train, y_train)
log_pred = log_reg_pipeline.predict(X_val)
log_acc = accuracy_score(y_val, log_pred)

# Random Forest
rf_pipeline.fit(X_train, y_train)
rf_pred = rf_pipeline.predict(X_val)
rf_acc = accuracy_score(y_val, rf_pred)

# ===================== Display Results =====================
print(f"📊 Logistic Regression Accuracy: {log_acc*100:.0f}%")
print(f"📊 Random Forest Accuracy: {rf_acc*100:.0f}%\n")

print("=== Logistic Regression Classification Report ===")
print(classification_report(y_val, log_pred))

print("=== Random Forest Classification Report ===")
print(classification_report(y_val, rf_pred))

# ===================== Confusion Matrix =====================
cm = confusion_matrix(y_val, rf_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Random Forest Confusion Matrix", fontsize=14)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ===================== Feature Importance Visualization =====================
# Extract feature names after OneHotEncoding
ohe = log_reg_pipeline.named_steps['preprocess'].named_transformers_['cat']
ohe_features = ohe.get_feature_names_out(categorical_features)
all_features = list(ohe_features) + numeric_features

# Random Forest feature importance
importances = rf_pipeline.named_steps['model'].feature_importances_
feature_importance_df = pd.DataFrame({'Feature': all_features, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8,6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
plt.title("Random Forest Feature Importance", fontsize=14)
plt.show()

# ===================== Final Predictions on Test Set =====================
final_predictions = rf_pipeline.predict(X_test)
submission = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': final_predictions})
submission.to_csv("titanic_predictions.csv", index=False)
print("✅ Saved titanic_predictions.csv!")


