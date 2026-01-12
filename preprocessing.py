import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

# ===============================
# LOAD DATASET
# ===============================
data = pd.read_excel("Student_dropout.xlsx")

#print("Columns in dataset:")
#print(data.columns)

# ===============================
# DATA PREPROCESSING
# ===============================

# 1. Handle missing values (safe assignment)
for col in data.columns:
    if data[col].dtype != 'object':
        data[col] = data[col].fillna(data[col].mean())
    else:
        data[col] = data[col].fillna(data[col].mode()[0])

# 2. Encode categorical variables
label_encoder = LabelEncoder()
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = label_encoder.fit_transform(data[col])

# ===============================
# DEFINE INPUTS AND OUTPUT
# ===============================
TARGET_COLUMN = "Target"

X = data.drop(TARGET_COLUMN, axis=1)
y = data[TARGET_COLUMN]

print("Input shape:", X.shape)
print("Output shape:", y.shape)

# ===============================
# NORMALIZATION
# ===============================
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# ===============================
# TRAIN-TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# ===============================
# SAVE PROCESSED DATA
# ===============================
np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

joblib.dump(scaler, "scaler.save")

print("✅ Preprocessing completed successfully.")
