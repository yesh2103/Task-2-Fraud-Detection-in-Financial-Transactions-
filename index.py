import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix


file_path = r"D:\3 semister\Exiton\Task 2(Fraud Detection in Financial Transactions)\creditcard.csv"


data = pd.read_csv(file_path)

print(data.head())
print(data.info())



print(data.isnull().sum())

scaler = StandardScaler()
data['Amount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))

X = data.drop(['Class', 'Time'], axis=1)  
y = data['Class']

non_fraud = data[data['Class'] == 0]
fraud = data[data['Class'] == 1]

non_fraud_sample = non_fraud.sample(len(fraud))

balanced_data = pd.concat([non_fraud_sample, fraud])

X_balanced = balanced_data.drop(['Class', 'Time'], axis=1)
y_balanced = balanced_data['Class']

X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

iso_forest = IsolationForest(contamination=len(fraud) / len(data), random_state=42)
iso_forest.fit(X_train)

y_pred_train = iso_forest.predict(X_train)
y_pred_test = iso_forest.predict(X_test)

y_pred_train = [1 if x == -1 else 0 for x in y_pred_train]
y_pred_test = [1 if x == -1 else 0 for x in y_pred_test]

print("Train Classification Report")
print(classification_report(y_train, y_pred_train))

print("Test Classification Report")
print(classification_report(y_test, y_pred_test))


print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_test))


roc_score = roc_auc_score(y_test, y_pred_test)
print(f"AUC-ROC Score: {roc_score}")
