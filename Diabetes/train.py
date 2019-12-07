import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn import preprocessing
from joblib import dump, load

data = pd.read_csv('diabetes.csv')
data_WO = data.drop(['Outcome'],axis=1)

min_max_scaler = preprocessing.MinMaxScaler()
x = data_WO
min_max_scaler = preprocessing.MinMaxScaler(feature_range = (0, 1))
x = min_max_scaler.fit_transform(x)
y = data.Outcome

knn = KNeighborsClassifier(n_neighbors=25,metric ='minkowski',p=1)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =0.3,random_state=42,stratify=y)

knn.fit(x_train,y_train)
score = knn.score(x_test,y_test)

dump(knn, 'model.joblib')
print(x.shape)
print(score)
