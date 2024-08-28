# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 19:26:47 2024

@author: archi
"""
#Import Librarires
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


############################### About Data ##################################
#Upload Dataset
data = pd.read_csv("ai4i2020.csv")

#Data Overview
print(data.head())
print(data.info())
print(data.isnull().sum())

#Datset Satistics
print(data.describe())

print(data['Machine failure'].value_counts())

################################ EDA ######################################
# Set the style for the plots
sns.set(style='whitegrid')

#Analyzing target Variable
plt.figure(figsize=(8,6))
sns.countplot(data=data, x='Machine failure', palette='Set2')
plt.title('Distribution of Target Variable(Failure)')
plt.show()

# Distribution of numerical features
numerical_features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']

plt.figure(figsize=(12, 8))
for i, feature in enumerate(numerical_features):
    plt.subplot(2, 3, i+1)
    sns.histplot(data[feature], kde=True)
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.show()

# Correlation matrix
df = data.drop(columns = ['Product ID', 'Type'])
corr_matrix = df.corr()

# Plot the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

#Feature Relationship
#Scatterplots to see relationship between the features
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Process temperature [K]', y='Air temperature [K]', hue='Machine failure')
plt.title('Process Temperature vs Air Temprature')
plt.show()

sns.scatterplot(data=df, x='Rotational speed [rpm]', y='Torque [Nm]', hue='Machine failure')
plt.title('Rotational speed  vs Torque')
plt.show()

# Pairplot for a subset of features
sns.pairplot(df[['Air temperature [K]', 'Process temperature [K]', 'Torque [Nm]', 'Tool wear [min]', 'Machine failure']], hue='Machine failure')
plt.show()

#Outlier
plt.figure(figsize=(12,8))
for i, feature in enumerate(numerical_features):
    plt.subplot(2, 3, i+1)
    sns.boxplot(data=df, y=feature)
    plt.title(f'Boxplot of {feature}')
plt.tight_layout()
plt.show()

# Create a new feature for the temperature difference
df['Temp Difference'] = df['Process temperature [K]'] - df['Air temperature [K]']

# Visualize the new feature
sns.histplot(df['Temp Difference'], kde=True)
plt.title('Distribution of Temperature Difference')
plt.show()

# Count the number of rows where 'Temp Difference' is less than 8.6
#count_temp_diff = data[data['Temp Difference'] < 8.6].shape[0]

#print(f"Number of rows where 'Temp Difference' is less than 8.6: {count_temp_diff}")        

############################ Test and Train Dataset ################################
x = df.drop('Machine failure', axis=1)
y = df['Machine failure']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)


print(f"Training data shape: {x_train.shape}")
print(f"Testing data shape: {x_test.shape}")

############################################# Models ################################
# ##############Logistic Regression
#Initilaize model
lr_model = LogisticRegression(random_state=42, max_iter=1000)

# train the model
lr_model.fit(x_train, y_train)

# test the model
y_pred = lr_model.predict(x_test)

# Evaluate the model
print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ROC-AUC Score
lr_auc = roc_auc_score(y_test, lr_model.predict_proba(x_test)[:, 1])
print(f"Logistic Regression ROC-AUC Score: {lr_auc:.4f}")

##################### Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(x_train, y_train)

y_pred = rf_model.predict(x_test)

print("Random Forest Result:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

rf_auc = roc_auc_score(y_test, rf_model.predict_proba(x_test)[:, 1])
print(f"Logistic Regression ROC-AUC Score: {rf_auc:.4f}")

####################################### Feature Importance ############################

fi = rf_model.feature_importances_
indices = np.argsort(fi)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(x.shape[1]):
    print(f"{f + 1}. Feature {x.columns[indices[f]]} ({fi[indices[f]]:.4f})")

# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(x.shape[1]), fi[indices], align="center")
plt.xticks(range(x.shape[1]), x.columns[indices], rotation=90)
plt.tight_layout()
plt.show()
