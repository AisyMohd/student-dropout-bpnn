# pyright: reportMissingImports=false
# =====================================================
# STUDENT DROPOUT PREDICTION USING BPNN
# =====================================================

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

# =====================================================
# Q1. LOAD DATASET
# =====================================================
data = pd.read_excel("Student_dropout.xlsx")

print("Q1: Dataset loaded successfully")
print("Number of samples:", data.shape[0])
print("Number of features:", data.shape[1])
print(data.head())

# =====================================================
# Q1. DOMAIN-SPECIFIC DECISION (TARGET)
# =====================================================
# Dropout = 1 (at-risk students)
# Enrolled & Graduate = 0 (non-dropout)

data['Target'] = data['Target'].map({
    'Dropout': 1,
    'Enrolled': 0,
    'Graduate': 0
})

print("\nQ1: Target mapping applied")
print(data['Target'].value_counts())

# =====================================================
# Q3. DATA PREPROCESSING
# =====================================================
X = data.drop('Target', axis=1)
y = data['Target']

# Encode categorical attributes
print("\nQ3: Encoding categorical attributes...")
for col in X.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    print(f" - Encoded column: {col}")

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save preprocessed dataset
preprocessed_df = pd.DataFrame(X_scaled, columns=X.columns)
preprocessed_df['Target'] = y.values
preprocessed_df.to_csv("student_dropout_preprocessed.csv", index=False)

print("✔ Standardization applied")
print("✔ Preprocessed data saved")

# =====================================================
# Q2. INPUT / OUTPUT STRUCTURE
# =====================================================
print("\nQ2: Neural Network Structure")
print(" - Input neurons:", X_scaled.shape[1])
print(" - Output neurons: 1 (Binary classification)")

# =====================================================
# Q4. TRAIN–TEST PARTITIONING
# =====================================================
TEST_SIZES = [0.2, 0.3]  # evaluated split ratios
EPOCHS = 100
BATCH_SIZE = 32

all_results = []

print("\nQ4–Q6: Starting experiments...\n")

for test_size in TEST_SIZES:

    print(f"--- Train/Test Split = {int((1-test_size)*100)}:{int(test_size*100)} ---")

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=42
    )

    # =====================================================
    # Q5 & Q6. HYPERPARAMETER INITIALIZATION & TUNING
    # =====================================================
    hidden_nodes_list = [8, 16, 32]
    learning_rates = [0.001, 0.01]

    for hidden_nodes in hidden_nodes_list:
        for lr in learning_rates:

            print(f"Training → Hidden Nodes={hidden_nodes}, LR={lr}")

            model = Sequential()
            model.add(Dense(
                hidden_nodes,
                activation='relu',
                kernel_initializer=GlorotUniform(),
                input_shape=(X_train.shape[1],)
            ))
            model.add(Dense(
                1,
                activation='sigmoid',
                kernel_initializer=GlorotUniform()
            ))

            model.compile(
                optimizer=Adam(learning_rate=lr),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )

            early_stop = EarlyStopping(
                monitor='loss',
                patience=10,
                restore_best_weights=True
            )

            model.fit(
                X_train, y_train,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                callbacks=[early_stop],
                verbose=0
            )

            # =====================================================
            # Q7. TESTING & EVALUATION
            # =====================================================
            y_pred_prob = model.predict(X_test)
            y_pred = (y_pred_prob > 0.5).astype(int)

            acc = accuracy_score(y_test, y_pred)
            error_rate = 1 - acc
            cm = confusion_matrix(y_test, y_pred)

            print(f"✔ Accuracy: {acc:.4f}, Error Rate: {error_rate:.4f}")
            print("Confusion Matrix:\n", cm, "\n")

            all_results.append({
                "Test_Split": f"{int((1-test_size)*100)}:{int(test_size*100)}",
                "Hidden_Nodes": hidden_nodes,
                "Learning_Rate": lr,
                "Accuracy": acc,
                "Error_Rate": error_rate
            })

# =====================================================
# Q7. RESULTS SUMMARY TABLE
# =====================================================
results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values(by="Accuracy", ascending=False)

print("\nQ7: Hyperparameter Tuning Summary")
print(results_df)

results_df.to_csv("bpnn_experiment_results.csv", index=False)
print("\n✔ Results saved as 'bpnn_experiment_results.csv'")

# =====================================================
# FINAL MODEL SELECTION & SAVING
# =====================================================
best = results_df.iloc[0]

BEST_SPLIT = best["Test_Split"]
BEST_HIDDEN = int(best["Hidden_Nodes"])
BEST_LR = float(best["Learning_Rate"])

print("\nFINAL SELECTED CONFIGURATION")
print(" - Train/Test Split:", BEST_SPLIT)
print(" - Hidden Nodes:", BEST_HIDDEN)
print(" - Learning Rate:", BEST_LR)

final_model = Sequential()
final_model.add(Dense(
    BEST_HIDDEN,
    activation='relu',
    kernel_initializer=GlorotUniform(),
    input_shape=(X_scaled.shape[1],)
))
final_model.add(Dense(1, activation='sigmoid'))

final_model.compile(
    optimizer=Adam(learning_rate=BEST_LR),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

final_model.fit(
    X_scaled, y,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=0
)

final_model.save("student_dropout_bpnn_final.h5")
print("\n✔ Final trained model saved as 'student_dropout_bpnn_final.h5'")
