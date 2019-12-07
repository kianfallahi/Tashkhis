import numpy as np
from joblib import dump, load

model = load('model.joblib')
x_input = np.array([21,43,55,55,67]).reshape(1,-1)

pre = model.predict(x_input)
print(pre)
