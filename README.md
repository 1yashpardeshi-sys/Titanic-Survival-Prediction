import matplotlib.pyplot as plt

plt.text(0.5, 0.5, 'Titanic Survival Prediction 🚢', fontsize=20, fontweight='bold', ha='center')
plt.axis('off')
plt.show()


📌 Project Overview
The Titanic Survival Prediction project aims to predict whether a passenger survived the Titanic disaster using Machine Learning algorithms. The dataset used is the famous Titanic dataset from Kaggle.

This project demonstrates data preprocessing, feature engineering, classification models, and performance evaluation using Logistic Regression and Random Forest.

# 🎯 Objectives
# - Analyze passenger data and identify survival patterns
# - Implement classification algorithms for prediction
# - Compare model performance

# 📂 Dataset
Dataset Source: Kaggle Titanic Dataset
Features include:

Passenger Class (Pclass)
Gender (Sex)
Age
Fare
Number of siblings/spouses aboard (SibSp)
Number of parents/children aboard (Parch)
Embarked location

# ⚙️ Technologies Used
Python 
Pandas
NumPy 
Matplotlib
Seaborn
Scikit-learn
Kaggle Notebook

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

📊 Results
Logistic Regression Accuracy: 81%
Random Forest Accuracy: 82%
Random Forest performed slightly better than Logistic Regression.

# ===================== Save Predictions =====================
final_predictions = rf_pipeline.predict(X_test)
submission = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': final_predictions})
submission.to_csv("titanic_predictions.csv", index=False)
print("✅ Saved titanic_predictions.csv!")
