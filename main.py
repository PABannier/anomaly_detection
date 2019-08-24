from anomaly_detection import Detector
import numpy as np
from scipy.io import loadmat

mat = loadmat("ex8data1.mat")
X_train = mat["X"]
X_val = mat["Xval"]
y_val = mat["yval"]

detect = Detector(X_train, X_val, y_val)

detect.fit()

t = detect.predict(np.array([[13.04, 14.07],
                             [23, 58],
                             [13.96, 14.75]]))

print(t)