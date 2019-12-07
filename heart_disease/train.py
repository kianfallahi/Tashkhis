import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import train_test_split
from joblib import dump, load

file = pd.read_csv('heart.csv')

x = file.drop('target',axis=1)
y = file['target']

min_max_scaler = preprocessing.MinMaxScaler()
min_max_scaler = preprocessing.MinMaxScaler(feature_range = (0, 1))
x = min_max_scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=101)

model = svm.SVC(kernel='linear', C=1)
model.fit(x_train,y_train)

score = model.score(x_test,y_test)

dump(model, 'model.joblib')
print(x.shape)
print(score)
