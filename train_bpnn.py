# pyright: reportMissingImports=false

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Load preprocessed training data
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")

# Build BPNN model
model = Sequential()
model.add(Dense(10, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(
    optimizer=Adam(learning_rate=0.01),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train model
history = model.fit(
    X_train,
    y_train,
    epochs=500,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Save trained model
model.save("student_dropout_bpnn_model.h5")

print("Training completed and model saved.")
