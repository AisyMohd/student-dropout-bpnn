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
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.callbacks import EarlyStopping

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
print("\nQ1: Target variable mapping applied (Dropout=1, Others=0)")
print(data['Target'].value_counts())

# =========================================
# 3. PREPROCESSING
# =========================================
X = data.drop('Target', axis=1)
y = data['Target']

# Encode categorical features
label_encoders = {}
for col in X.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save preprocessed data
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
preprocessed_data = X_scaled_df.copy()
preprocessed_data['Target'] = y.values
preprocessed_data.to_csv("student_dropout_preprocessed.csv", index=False)

print("\nQ3: Preprocessing completed")
print("✔ Encoded & normalized data saved as 'student_dropout_preprocessed.csv'")

# =========================================
# 4. TRAIN-TEST SPLIT
# =========================================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("\nQ4: Train/Test split completed")
print(" - X_train shape:", X_train.shape)
print(" - X_test shape:", X_test.shape)

# =========================================
# Q2: INPUT / OUTPUT NEURONS
# =========================================
print("\nQ2: Input neurons:", X_scaled.shape[1])
print("Q2: Output neurons: 1 (Binary classification)")

# =========================================
# 5. HYPERPARAMETER SETTINGS (EXPLICIT)
# =========================================
HIDDEN_NODES = 16
LEARNING_RATE = 0.01
BATCH_SIZE = 32
EPOCHS = 100
MAX_ERROR = 0.001  # loss threshold

# =========================================
# 6. INITIALIZE & DESIGN BPNN
# =========================================
model = Sequential()

# Hidden layer
model.add(Dense(
    HIDDEN_NODES,
    activation='relu',
    kernel_initializer=GlorotUniform(),  # Initial weights
    input_shape=(X_scaled.shape[1],)
))

# Output layer
model.add(Dense(
    1,
    activation='sigmoid',
    kernel_initializer=GlorotUniform()
))

# Compile model
optimizer = Adam(learning_rate=LEARNING_RATE)
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\nQ5 & Q6: BPNN Hyperparameters")
print(" - Hidden nodes:", HIDDEN_NODES)
print(" - Activation function (hidden): ReLU")
print(" - Activation function (output): Sigmoid")
print(" - Initial weight initialization: Glorot Uniform (Xavier)")
print(" - Learning rate:", LEARNING_RATE)
print(" - Batch size:", BATCH_SIZE)
print(" - Max epochs:", EPOCHS)
print(" - Max error (loss threshold):", MAX_ERROR)

print("\nQ6: Model Summary")
model.summary()

# =========================================
# 7. TRAIN NETWORK (WITH MAX ERROR)
# =========================================
early_stopping = EarlyStopping(
    monitor='loss',
    min_delta=MAX_ERROR,
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    callbacks=[early_stopping],
    verbose=0
)

print("\nQ6: Training completed")
print(" - Final training accuracy:", history.history['accuracy'][-1])
print(" - Total epochs used:", len(history.history['loss']))

# Save model
model.save("student_dropout_bpnn.h5")

# =========================================
# 8. TEST & EVALUATION
# =========================================
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\nQ7: Test Accuracy:", accuracy)
print("Q7: Confusion Matrix:\n", cm)
