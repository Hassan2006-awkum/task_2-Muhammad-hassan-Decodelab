# ============================================================
#   Project 2: Data Classification Using AI
#   Dataset   : Iris Plant Dataset
#   Algorithm : K-Nearest Neighbors (KNN)
#   Author    : Muhammad Hassan
#   Batch     : DecodeLabs 2026
# ============================================================



# ── STEP 1: Import Libraries ────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    accuracy_score,
)

print("=" * 60)
print("   PROJECT 2 — IRIS PLANT CLASSIFICATION USING KNN")
print("=" * 60)


# ── STEP 2: Load Dataset ────────────────────────────────────
iris = load_iris()

X = iris.data            # Features  : shape (150, 4)
y = iris.target          # Labels    : 0=Setosa, 1=Versicolor, 2=Virginica
class_names   = iris.target_names
feature_names = iris.feature_names

# Build a readable DataFrame just for display
df = pd.DataFrame(X, columns=feature_names)
df["Species"] = [class_names[i] for i in y]

print("\n[1] DATASET OVERVIEW")
print(f"    Total Samples  : {X.shape[0]}")
print(f"    Total Features : {X.shape[1]}")
print(f"    Classes        : {list(class_names)}")
print("\n    First 5 rows:")
print(df.head().to_string(index=False))
print(f"\n    Class Distribution:\n{df['Species'].value_counts().to_string()}")


# ── STEP 3: Feature Scaling (StandardScaler) ────────────────
# KNN is distance-based, so all features must be on the same scale.
# StandardScaler transforms each feature to mean=0, std=1.
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n[2] FEATURE SCALING  (StandardScaler)")
print(f"    Mean after scaling  (should be ~0) : {X_scaled.mean(axis=0).round(4)}")
print(f"    Std  after scaling  (should be ~1) : {X_scaled.std(axis=0).round(4)}")


# ── STEP 4: Train-Test Split  80% train / 20% test ──────────
# stratify=y  → keeps the 50/50/50 class balance in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size    = 0.2,
    random_state = 42,
    shuffle      = True,
    stratify     = y,
)

print("\n[3] TRAIN-TEST SPLIT")
print(f"    Training Samples : {X_train.shape[0]}  (80%)")
print(f"    Testing  Samples : {X_test.shape[0]}   (20%)")


# ── STEP 5: Find Optimal K using the Elbow Method ───────────
# We test K = 1 to 30 and pick the K with the lowest error rate.
error_rates = []
k_range     = range(1, 31)

for k in k_range:
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    knn_temp.fit(X_train, y_train)
    preds = knn_temp.predict(X_test)
    error_rates.append(1 - accuracy_score(y_test, preds))

optimal_k = k_range[int(np.argmin(error_rates))]

print("\n[4] OPTIMAL K  (Elbow Method)")
print(f"    Optimal K     : {optimal_k}")
print(f"    Lowest Error  : {min(error_rates):.4f}")


# ── STEP 6: Train Final Model ───────────────────────────────
model = KNeighborsClassifier(n_neighbors=optimal_k)
model.fit(X_train, y_train)

print("\n[5] MODEL TRAINING")
print(f"    Algorithm : K-Nearest Neighbors  (KNN)")
print(f"    K Value   : {optimal_k}")
print( "    Status    : Trained ✓")


# ── STEP 7: Predictions ─────────────────────────────────────
y_pred = model.predict(X_test)


# ── STEP 8: Evaluation ──────────────────────────────────────
acc    = accuracy_score(y_test, y_pred)
f1     = f1_score(y_test, y_pred, average="weighted")
cm     = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=class_names)
report_dict = classification_report(
    y_test, y_pred, target_names=class_names, output_dict=True
)

print("\n[6] EVALUATION RESULTS")
print(f"    Accuracy : {acc * 100:.2f}%")
print(f"    F1 Score : {f1:.4f}")
print("\n    --- Classification Report ---")
print(report)
print("    --- Confusion Matrix ---")
print(pd.DataFrame(cm, index=class_names, columns=class_names).to_string())


# ── STEP 9: Visualisations (saved as PNG) ───────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "Project 2 — Iris Plant Classification (KNN)\n"
    "Muhammad Hassan  |  DecodeLabs 2026",
    fontsize=13, fontweight="bold",
)

# --- Plot 1 : Elbow Curve ---
axes[0, 0].plot(
    k_range, error_rates,
    color="steelblue", marker="o",
    markerfacecolor="orange", markersize=7, linewidth=2,
)
axes[0, 0].axvline(
    x=optimal_k, color="red", linestyle="--",
    label=f"Optimal K = {optimal_k}",
)
axes[0, 0].set_title("Elbow Curve — Choosing Optimal K", fontweight="bold")
axes[0, 0].set_xlabel("K Value")
axes[0, 0].set_ylabel("Error Rate")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# --- Plot 2 : Confusion Matrix Heatmap ---
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=class_names, yticklabels=class_names,
    ax=axes[0, 1], linewidths=0.5,
)
axes[0, 1].set_title("Confusion Matrix", fontweight="bold")
axes[0, 1].set_xlabel("Predicted Label")
axes[0, 1].set_ylabel("True Label")

# --- Plot 3 : Petal Length vs Petal Width Scatter ---
colors = ["#4C72B0", "#DD8452", "#55A868"]
for name, color in zip(class_names, colors):
    subset = df[df["Species"] == name]
    axes[1, 0].scatter(
        subset[feature_names[2]], subset[feature_names[3]],
        label=name, color=color,
        alpha=0.8, edgecolors="k", linewidths=0.4, s=60,
    )
axes[1, 0].set_title("Petal Length vs Petal Width", fontweight="bold")
axes[1, 0].set_xlabel("Petal Length (cm)")
axes[1, 0].set_ylabel("Petal Width  (cm)")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# --- Plot 4 : Performance Metrics Bar Chart ---
metrics = {
    "Accuracy":             acc,
    "F1 Score\n(weighted)": f1,
    "Precision\n(weighted)": report_dict["weighted avg"]["precision"],
    "Recall\n(weighted)":    report_dict["weighted avg"]["recall"],
}
bar_colors = ["#4C72B0", "#55A868", "#DD8452", "#C44E52"]
bars = axes[1, 1].bar(
    metrics.keys(), metrics.values(),
    color=bar_colors, edgecolor="black", linewidth=0.6,
)
for bar, val in zip(bars, metrics.values()):
    axes[1, 1].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.005,
        f"{val:.2%}", ha="center", va="bottom",
        fontsize=10, fontweight="bold",
    )
axes[1, 1].set_ylim(0, 1.15)
axes[1, 1].set_title("Model Performance Metrics", fontweight="bold")
axes[1, 1].set_ylabel("Score")
axes[1, 1].grid(True, axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("iris_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n[7] Chart saved as  iris_results.png  ✓")


# ── STEP 10: Live Prediction Demo ───────────────────────────
# You can change these 4 numbers to predict any iris flower.
# Format: [sepal_length, sepal_width, petal_length, petal_width]  (all in cm)
print("\n[8] LIVE PREDICTION DEMO")

sample        = np.array([[5.1, 3.5, 1.4, 0.2]])   # typical Setosa
sample_scaled = scaler.transform(sample)
prediction    = model.predict(sample_scaled)
probability   = model.predict_proba(sample_scaled)

print(f"    Input  : Sepal = {sample[0][0]}cm x {sample[0][1]}cm | "
      f"Petal = {sample[0][2]}cm x {sample[0][3]}cm")
print(f"    Result : {class_names[prediction[0]].upper()}")
print(f"    Confidence : {probability[0].max() * 100:.1f}%")


# ── complete ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("   PROJECT 2 COMPLETE ✓")
print(f"   Accuracy : {acc * 100:.2f}%   |   F1 Score : {f1:.4f}")
print("=" * 60)
