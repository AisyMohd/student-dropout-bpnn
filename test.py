import numpy as np

X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")
y_train = np.load("y_train.npy", allow_pickle=True)
y_test = np.load("y_test.npy", allow_pickle=True)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
