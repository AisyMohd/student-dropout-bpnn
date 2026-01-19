# pyright: reportMissingImports=false

# =========================================
# STUDENT DROPOUT PREDICTION USING BPNN
# =========================================

import pandas as pd
import numpy as np

# ML tools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix

# Neural Network
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# =========================================
# 1. LOAD DATASET
# =========================================
data = pd.read_excel("Student_dropout.xlsx")

# =========================================
# 2. TARGET VARIABLE (DOMAIN DECISION)
# Dropout = 1
# Enrolled & Graduate = 0
# =========================================
data['Target'] = data['Target'].map({
    'Dropout': 1,
    'Enrolled': 0,
    'Graduate': 0
})

# =========================================
# 3. PREPROCESSING
# =========================================

# Separate features and target
X = data.drop('Target', axis=1)
y = data['Target']

# Encode categorical features
for col in X.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# =========================================
# 4. TRAIN-TEST SPLIT
# =========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Number of input features:", X.shape[1])  # INPUT NEURONS

# =========================================
# 5. INITIALIZE & DESIGN BPNN
# =========================================
model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(X.shape[1],)))
model.add(Dense(1, activation='sigmoid'))

optimizer = Adam(learning_rate=0.001)

model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# =========================================
# 6. TRAIN NETWORK
# =========================================
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# Save model
model.save("student_dropout_bpnn.h5")

# =========================================
# 7. TEST & EVALUATION
# =========================================
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\nTest Accuracy:", accuracy)
print("\nConfusion Matrix:\n", cm)
