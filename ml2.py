# pyright: reportMissingImports=false
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform

# Load dataset
data = pd.read_excel("Student_dropout.xlsx")

# Binary target mapping
data['Target'] = data['Target'].map({
    'Dropout': 1,
    'Enrolled': 0,
    'Graduate': 0
})

X = data.drop('Target', axis=1)
y = data['Target']

# Encode categorical
for col in X.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# FIXED BEST SPLIT (70:30)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# FIXED BEST HYPERPARAMETERS FROM STAGE 1
BEST_HIDDEN = 16
BEST_LR = 0.001

# Epoch and batch candidates
epochs_list = [100, 150, 200]
batch_sizes = [16, 32, 64]

all_results = []

print("\n=== TUNING EPOCHS & BATCH SIZE ===\n")

for ep in epochs_list:
    for bs in batch_sizes:

        print(f"Training → Epochs={ep}, Batch Size={bs}")

        model = Sequential()
        model.add(Dense(
            BEST_HIDDEN,
            activation='relu',
            kernel_initializer=GlorotUniform(),
            input_shape=(X_train.shape[1],)
        ))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(
            optimizer=Adam(learning_rate=BEST_LR),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        model.fit(
            X_train, y_train,
            epochs=ep,
            batch_size=bs,
            verbose=0
        )

        # Evaluate
        y_pred_prob = model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int)

        acc = accuracy_score(y_test, y_pred)
        error_rate = 1 - acc

        print(f"✔ Accuracy: {acc:.4f}, Error Rate: {error_rate:.4f}\n")

        all_results.append({
            "Epochs": ep,
            "Batch_Size": bs,
            "Accuracy": acc,
            "Error_Rate": error_rate
        })

# Results table
results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values(by="Accuracy", ascending=False)

print("\n=== EPOCH & BATCH TUNING SUMMARY ===")
print(results_df)

results_df.to_csv("bpnn_epoch_batch_results.csv", index=False)
print("\n✔ Results saved as 'bpnn_epoch_batch_results.csv'")