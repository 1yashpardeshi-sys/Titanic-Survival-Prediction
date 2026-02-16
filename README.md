# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

%matplotlib inline

# Load the data
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Preview data
print(train_data.head())
print(test_data.head())

# Info & Missing Values
print(train_data.info())
print(train_data.isnull().sum())

# Visualize missing data
sns.heatmap(train_data.isnull(), cbar=False, cmap="viridis")

# Drop unnecessary columns
train_data.drop(['Ticket','Cabin','Name','PassengerId'], axis=1, inplace=True)
test_data.drop(['Ticket','Cabin','Name'], axis=1, inplace=True)

# Fill missing values
train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)
test_data['Age'].fillna(test_data['Age'].median(), inplace=True)
test_data['Fare'].fillna(test_data['Fare'].median(), inplace=True)

# Convert categorical columns
train_data = pd.get_dummies(train_data, columns=['Sex', 'Embarked'])
test_data = pd.get_dummies(test_data, columns=['Sex', 'Embarked'])

# Separate features and target
X = train_data.drop("Survived", axis=1)
y = train_data["Survived"]

# Train test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Models
models = {
    "Logistic Regression": LogisticRegression(),
    "Support Vector Classifier": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

# Train & Evaluate
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_val_scaled)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"{name} Accuracy: {accuracy:.4f}")
    print(confusion_matrix(y_val, y_pred))
    print(classification_report(y_val, y_pred))
    print("--------------------------------------------------")

# Choose best model (example: RandomForest)
best_model = RandomForestClassifier(n_estimators=200, random_state=42)
best_model.fit(X_train_scaled, y_train)

# Prepare test data
test_scaled = scaler.transform(test_data)

# Predict survival on test
test_predictions = best_model.predict(test_scaled)

# Save submission
submission = pd.DataFrame({
    "PassengerId": pd.read_csv("test.csv")["PassengerId"],
    "Survived": test_predictions
})
submission.to_csv("submission.csv", index=False)
print("Submission file saved!")
