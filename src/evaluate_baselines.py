import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import os

# Create baseline output dir
os.makedirs("results/baseline", exist_ok=True)

# 1. Load data
print("Loading data...")
train_data = np.load("data/processed/train_data.npy")
test_data = np.load("data/processed/test_data.npy")
test_labels = np.load("data/processed/test_labels.npy") # 0: Normal, 1: Anomaly

# Flatten data for sklearn models (N, 140)
X_train = train_data.reshape(train_data.shape[0], -1)
X_test = test_data.reshape(test_data.shape[0], -1)
y_true = test_labels

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# 2. Train & Evaluate Models
results = []

def evaluate_model(name, y_pred, y_scores):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_scores)
    
    print(f"--- {name} ---")
    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
    
    return {
        "Model": name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4),
        "AUC-ROC": round(auc, 4)
    }

# Isolation Forest
print("\nTraining Isolation Forest...")
iso_forest = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
iso_forest.fit(X_train)
if_pred = iso_forest.predict(X_test)
if_pred_mapped = np.where(if_pred == 1, 0, 1) # 1 -> 0 (Normal), -1 -> 1 (Anomaly)
if_scores = -iso_forest.decision_function(X_test) # Lower score = more anomalous, so we negate
results.append(evaluate_model("Isolation Forest", if_pred_mapped, if_scores))

# One-Class SVM
print("\nTraining One-Class SVM...")
oc_svm = OneClassSVM(gamma='auto', nu=0.05)
oc_svm.fit(X_train)
svm_pred = oc_svm.predict(X_test)
svm_pred_mapped = np.where(svm_pred == 1, 0, 1)
svm_scores = -oc_svm.decision_function(X_test)
results.append(evaluate_model("One-Class SVM", svm_pred_mapped, svm_scores))

# VAE (From draft_v1.md)
vae_result = {
    "Model": "VAE (Proposed)",
    "Accuracy": 0.8600,
    "Precision": 0.9800,
    "Recall": 0.8400,
    "F1-Score": 0.9000,
    "AUC-ROC": 0.9620
}
results.append(vae_result)

# 3. Save baseline_results.csv
df = pd.DataFrame(results)
df.to_csv("results/baseline/baseline_results.csv", index=False)
print("\nSaved results/baseline/baseline_results.csv")

# 4. Generate comparison_table.md
md_content = "# Comparison of Anomaly Detection Models on ECG5000\n\n"
md_content += "| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |\n"
md_content += "|---|---|---|---|---|---|\n"
for _, row in df.iterrows():
    md_content += f"| {row['Model']} | {row['Accuracy']:.4f} | {row['Precision']:.4f} | {row['Recall']:.4f} | {row['F1-Score']:.4f} | {row['AUC-ROC']:.4f} |\n"
with open("results/baseline/comparison_table.md", "w", encoding="utf-8") as f:
    f.write(md_content)
print("Saved results/baseline/comparison_table.md")

# 5. Generate comparison_plot.png
labels = df["Model"].tolist()
acc = df["Accuracy"].tolist()
f1 = df["F1-Score"].tolist()
auc = df["AUC-ROC"].tolist()

x = np.arange(len(labels))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width, acc, width, label='Accuracy', color='#3b82f6')
rects2 = ax.bar(x, f1, width, label='F1-Score (Anomaly)', color='#ef4444')
rects3 = ax.bar(x + width, auc, width, label='AUC-ROC', color='#10b981')

ax.set_ylabel('Scores')
ax.set_title('Performance Comparison: VAE vs Baselines (ECG5000)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 1.1)
ax.legend(loc='lower right')

# Add values on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

fig.tight_layout()
plt.savefig("results/baseline/comparison_plot.png", dpi=150)
print("Saved results/baseline/comparison_plot.png")
