#https://kaggle.com/merishnasuwal/breast-cancer-prediction-dataset/
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm
from joblib import dump, load

df = pd.read_csv('Breast_cancer_data.csv')
min_max_scaler = preprocessing.MinMaxScaler()


df['target'] = df['diagnosis']
df = df.drop('diagnosis', axis=1)

y = df['target']
x = df.drop('target', axis=1)

min_max_scaler = preprocessing.MinMaxScaler(feature_range = (0, 1))
scaled_x = min_max_scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(scaled_x, y, test_size=0.2, random_state=101)

model = svm.SVC(kernel='linear', C=1)
model.fit(x_train,y_train)

score = model.score(x_test,y_test)

dump(model, 'model.joblib')
print(x.shape)
print(score)
