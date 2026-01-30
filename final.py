# pyright: reportMissingImports=false
# =====================================================
# BPNN: FULL TUNING + FINAL EVALUATION (ASSIGNMENT READY)
# =====================================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform

# =====================================================
# 1. LOAD & PREPROCESS DATA
# =====================================================
data = pd.read_excel("Student_dropout.xlsx")

data["Target"] = data["Target"].map({
    "Dropout": 1,
    "Enrolled": 0,
    "Graduate": 0
})

X = data.drop("Target", axis=1)
y = data["Target"]

for col in X.select_dtypes(include="object").columns:
    X[col] = LabelEncoder().fit_transform(X[col])

X = StandardScaler().fit_transform(X)

print("✔ Data preprocessed")

# =====================================================
# 2. HYPERPARAMETER SETTINGS
# =====================================================
SPLITS = [0.3, 0.2]      # 70:30 and 80:20
HIDDEN_NODES = [8, 16, 32]
LEARNING_RATES = [0.001, 0.01]
EPOCHS_LIST = [100, 150, 200]
BATCH_SIZES = [16, 32, 64]

results = []

# =====================================================
# 3. RUN ALL 108 EXPERIMENTS (TUNING)
# =====================================================
for test_size in SPLITS:

    split_name = f"{int((1-test_size)*100)}:{int(test_size*100)}"

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    for h in HIDDEN_NODES:
        for lr in LEARNING_RATES:
            for ep in EPOCHS_LIST:
                for bs in BATCH_SIZES:

                    model = Sequential([
                        Dense(
                            h,
                            activation="relu",
                            kernel_initializer=GlorotUniform(),
                            input_shape=(X_train.shape[1],)
                        ),
                        Dense(1, activation="sigmoid")
                    ])

                    model.compile(
                        optimizer=Adam(learning_rate=lr),
                        loss="binary_crossentropy",
                        metrics=["accuracy"]
                    )

                    model.fit(
                        X_train, y_train,
                        epochs=ep,
                        batch_size=bs,
                        verbose=0
                    )

                    train_acc = accuracy_score(
                        y_train,
                        (model.predict(X_train) > 0.5).astype(int)
                    )

                    test_acc = accuracy_score(
                        y_test,
                        (model.predict(X_test) > 0.5).astype(int)
                    )

                    results.append({
                        "Split": split_name,
                        "Test_Size": test_size,
                        "Hidden_Nodes": h,
                        "Learning_Rate": lr,
                        "Epochs": ep,
                        "Batch_Size": bs,
                        "Train_Accuracy": train_acc,
                        "Test_Accuracy": test_acc
                    })

# =====================================================
# 4. SAVE & DISPLAY TOP 10 MODELS
# =====================================================
results_df = pd.DataFrame(results)
results_df = results_df.sort_values("Test_Accuracy", ascending=False)

print("\n=== TOP 10 BEST MODELS ===")
print(results_df.head(10))

results_df.to_csv("bpnn_all_experiments_results.csv", index=False)
print("\n✔ Results saved to bpnn_all_experiments_results.csv")

# =====================================================
# 5. FINAL EVALUATION USING BEST COMBINATION
# =====================================================
best = results_df.iloc[0]

print("\nBEST CONFIGURATION:")
print(best)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=best["Test_Size"], random_state=42
)

final_model = Sequential([
    Dense(
        int(best["Hidden_Nodes"]),
        activation="relu",
        kernel_initializer=GlorotUniform(),
        input_shape=(X_train.shape[1],)
    ),
    Dense(1, activation="sigmoid")
])

final_model.compile(
    optimizer=Adam(learning_rate=best["Learning_Rate"]),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

final_model.fit(
    X_train, y_train,
    epochs=int(best["Epochs"]),
    batch_size=int(best["Batch_Size"]),
    verbose=0
)

final_train_acc = accuracy_score(
    y_train,
    (final_model.predict(X_train) > 0.5).astype(int)
)

final_test_acc = accuracy_score(
    y_test,
    (final_model.predict(X_test) > 0.5).astype(int)
)

# =====================================================
# 6. FINAL GRAPH – TWO LINES ONLY
# =====================================================
plt.figure()
plt.plot(["Train", "Test"], [final_train_acc, final_test_acc], marker="o")
plt.ylabel("Accuracy")
plt.title("Final BPNN Performance (Best Hyperparameters)")
plt.grid(True)
plt.show()
