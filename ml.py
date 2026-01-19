# pyright: reportMissingImports=false
# =========================================
# STUDENT DROPOUT PREDICTION USING BPNN
# =========================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# =========================================
# 1. LOAD DATASET
# =========================================
data = pd.read_excel("Student_dropout.xlsx")
print("Q1: Dataset loaded, first 5 rows:\n", data.head())

# =========================================
# 2. TARGET VARIABLE (DOMAIN DECISION)
# =========================================
data['Target'] = data['Target'].map({
    'Dropout': 1,
    'Enrolled': 0,
    'Graduate': 0
})
print("\nQ1: Target variable mapping applied (Dropout=1, Others=0)\n", data['Target'].value_counts())

# =========================================
# 3. PREPROCESSING
# =========================================
X = data.drop('Target', axis=1)
y = data['Target']

# Encode categorical features
for col in X.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nQ3: Preprocessing done. First 5 rows after scaling:\n", X_scaled[:5])

# =========================================
# 4. TRAIN-TEST SPLIT
# =========================================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print("\nQ4: Train/Test split done")
print(" - X_train shape:", X_train.shape)
print(" - X_test shape:", X_test.shape)

# =========================================
# 2. Q2: Input/Output neurons
# =========================================
print("\nQ2: Number of input features (input neurons):", X_scaled.shape[1])
print("Q2: Output neurons = 1 (binary classification: dropout or not)")

# =========================================
# 5. INITIALIZE & DESIGN BPNN (Q5 & Q6)
# =========================================
model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(X_scaled.shape[1],)))  # hidden layer
model.add(Dense(1, activation='sigmoid'))  # output layer

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

print("\nQ5 & Q6: Model designed. Hyperparameters:")
print(" - Hidden layer neurons: 16")
print(" - Activation function (hidden): ReLU")
print(" - Activation function (output): Sigmoid")
print(" - Learning rate:", 0.001)
print(" - Epochs: 100")
print(" - Batch size: 32")

# Model summary
print("\nQ6: Model summary:")
model.summary()

# =========================================
# 6. TRAIN NETWORK
# =========================================
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    verbose=0
)
print("\nQ6: Training complete. Last training accuracy:", history.history['accuracy'][-1])

# Save model
model.save("student_dropout_bpnn.h5")

# =========================================
# 7. TEST & EVALUATION
# =========================================
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\nQ7: Test Accuracy:", accuracy)
print("Q7: Confusion Matrix:\n", cm)
