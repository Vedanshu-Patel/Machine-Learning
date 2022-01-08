#Vedanshu Patel
#20BCE0865
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier
dataset = pd.read_csv("processed.cleveland.data.csv", names=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'output'])

#Data Preprocessing

#Filling missing values Statistics measures
print("*****Before Filling Missing values Row 166, 192, 287, 302*****")
print(dataset.loc[287])
dataset1 = dataset
df1 = pd.DataFrame(dataset1)

print("----- Mean of Column 11 'ca' -----")
print(df1['ca'].mean())
df1.fillna(df1.mean(), inplace=True)
print("*****After Filling Missing values Row 166, 192, 287, 302*****")
print(df1.loc[[166, 192, 287, 302]])

print("----- Mean of Column 12 'thal' -----")
print(df1['thal'].mean())
df1.fillna(df1.mean(), inplace=True)
print("*****After Filling Missing values Row 87, 266*****")
print(df1.loc[[87, 266]])

feature_cols = list(dataset.columns[0:13])

print("Feature columns: \n{}".format(feature_cols))

X = dataset[feature_cols]
y = dataset['output'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=5)
print(X_train)


cl1 = RandomForestClassifier(n_estimators = 50, random_state = 1)
cl1.fit(X_train, y_train)
ypred = cl1.predict(X_test)

cm1 = confusion_matrix(y_test, ypred)
print("Accuracy :", accuracy_score(ypred, y_test))


cl1 = AdaBoostClassifier(n_estimators = 50)
cl1.fit(X_train, y_train)
ypred = cl1.predict(X_test)

cm1 = confusion_matrix(y_test, ypred)
print("Accuracy :", accuracy_score(ypred, y_test))