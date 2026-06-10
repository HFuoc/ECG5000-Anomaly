import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, roc_auc_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import ECG_VAE


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LATENT_DIM = 2
THRESHOLD_PERCENTILE = 95
ECG_DURATION_MS = 1000

COLORS = {
    "normal": "#2563EB",
    "anomaly": "#DC2626",
    "reconstruction": "#64748B",
    "threshold": "#111827",
    "highlight": "#FCA5A5",
    "grid_minor": "#FDE2E7",
    "grid_major": "#F5A3AF",
}


def load_data():
    test_data = np.load("data/processed/test_data.npy")
    test_labels = np.load("data/processed/test_labels.npy")
    train_data = np.load("data/processed/train_data.npy")
    return train_data, test_data, test_labels


def load_model():
    model = ECG_VAE(input_dim=140, hidden_dim=64, latent_dim=LATENT_DIM).to(DEVICE)
    model.load_state_dict(torch.load("results/checkpoints/best_model.pth", map_location=DEVICE))
    model.eval()
    return model


def get_errors(model, data):
    tensor = torch.FloatTensor(data).to(DEVICE)
    return model.reconstruction_error(tensor).cpu().numpy()


def clinical_metrics(y_true, scores, threshold):
    predictions = (scores > threshold).astype(int)

    tp = int(np.sum((y_true == 1) & (predictions == 1)))
    tn = int(np.sum((y_true == 0) & (predictions == 0)))
    fp = int(np.sum((y_true == 0) & (predictions == 1)))
    fn = int(np.sum((y_true == 1) & (predictions == 0)))

    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) else 0.0
    balanced_accuracy = (sensitivity + specificity) / 2

    return {
        "threshold": float(threshold),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "predictions": predictions,
    }


def compute_evaluation(train_errors, test_errors, test_labels):
    threshold = np.percentile(train_errors, THRESHOLD_PERCENTILE)
    metrics = clinical_metrics(test_labels, test_errors, threshold)
    metrics["auc_roc"] = roc_auc_score(test_labels, test_errors)
    return metrics


def add_ecg_paper_grid(ax, duration_ms, y_min, y_max, y_unit="normalized"):
    minor_x = 40
    major_x = 200
    minor_y = 0.1 if y_unit == "mV" else 0.05
    major_y = 0.5 if y_unit == "mV" else 0.25

    ax.set_facecolor("#FFF7F8")

    for x in np.arange(0, duration_ms + minor_x, minor_x):
        ax.axvline(x, color=COLORS["grid_minor"], linewidth=0.5, zorder=0)
    for x in np.arange(0, duration_ms + major_x, major_x):
        ax.axvline(x, color=COLORS["grid_major"], linewidth=0.9, zorder=0)

    start_y = np.floor(y_min / minor_y) * minor_y
    end_y = np.ceil(y_max / minor_y) * minor_y

    for y in np.arange(start_y, end_y + minor_y, minor_y):
        ax.axhline(y, color=COLORS["grid_minor"], linewidth=0.5, zorder=0)
    for y in np.arange(start_y, end_y + major_y, major_y):
        ax.axhline(y, color=COLORS["grid_major"], linewidth=0.9, zorder=0)

    ax.set_xlim(0, duration_ms)
    ax.set_ylim(y_min, y_max)


def contiguous_regions(mask):
    regions = []
    start = None

    for index, active in enumerate(mask):
        if active and start is None:
            start = index
        elif not active and start is not None:
            regions.append((start, index - 1))
            start = None

    if start is not None:
        regions.append((start, len(mask) - 1))

    return regions


def plot_ecg_with_highlight(ax, model, test_data, test_labels, duration_ms=ECG_DURATION_MS):
    anomaly_index = np.where(test_labels == 1)[0][0]
    signal = test_data[anomaly_index]
    tensor = torch.FloatTensor(test_data[anomaly_index : anomaly_index + 1]).to(DEVICE)

    with torch.no_grad():
        reconstruction, _, _ = model(tensor)

    reconstruction = reconstruction.cpu().numpy()[0]
    time_ms = np.linspace(0, duration_ms, len(signal))
    step_errors = np.square(signal - reconstruction)
    local_threshold = np.percentile(step_errors, 90)
    abnormal_regions = contiguous_regions(step_errors >= local_threshold)

    y_min = min(signal.min(), reconstruction.min()) - 0.05
    y_max = max(signal.max(), reconstruction.max()) + 0.05
    add_ecg_paper_grid(ax, duration_ms, y_min, y_max)

    for region_index, (start, end) in enumerate(abnormal_regions):
        ax.axvspan(
            time_ms[start],
            time_ms[end],
            color=COLORS["highlight"],
            alpha=0.32,
            linewidth=0,
            label="Vung sai khac cao" if region_index == 0 else None,
        )

    ax.plot(time_ms, signal, color=COLORS["anomaly"], linewidth=1.8, label="ECG dau vao")
    ax.plot(
        time_ms,
        reconstruction,
        color=COLORS["reconstruction"],
        linewidth=1.5,
        linestyle="--",
        label="ECG VAE tai tao",
    )

    ax.set_title("ECG bat thuong: tin hieu va vung sai khac cao")
    ax.set_xlabel("Thoi gian trong nhip ECG (ms)")
    ax.set_ylabel("Bien do ECG da chuan hoa")
    ax.legend(fontsize=8, loc="upper right")


def plot_threshold_distribution(ax, test_errors, test_labels, threshold):
    ax.hist(
        test_errors[test_labels == 0],
        bins=50,
        alpha=0.65,
        color=COLORS["normal"],
        label="Nhip binh thuong",
        density=True,
    )
    ax.hist(
        test_errors[test_labels == 1],
        bins=50,
        alpha=0.55,
        color=COLORS["anomaly"],
        label="Nhip bat thuong",
        density=True,
    )
    ax.axvline(
        threshold,
        color=COLORS["threshold"],
        linestyle="--",
        linewidth=2,
        label=f"Nguong canh bao = {threshold:.5f}",
    )

    ax.set_title("Phan phoi sai khac tai tao va nguong canh bao")
    ax.set_xlabel("Sai khac tai tao so voi nhip binh thuong da hoc")
    ax.set_ylabel("Mat do mau")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)


def plot_threshold_sensitivity_specificity(ax, y_true, scores, active_threshold):
    thresholds = np.quantile(scores, np.linspace(0.01, 0.99, 150))
    sensitivity_values = []
    specificity_values = []

    for threshold in thresholds:
        metrics = clinical_metrics(y_true, scores, threshold)
        sensitivity_values.append(metrics["sensitivity"])
        specificity_values.append(metrics["specificity"])

    ax.plot(
        thresholds,
        sensitivity_values,
        color=COLORS["anomaly"],
        linewidth=2.2,
        label="Do nhay: phat hien bat thuong",
    )
    ax.plot(
        thresholds,
        specificity_values,
        color=COLORS["normal"],
        linewidth=2.2,
        label="Do dac hieu: nhan dung binh thuong",
    )
    ax.axvline(active_threshold, color=COLORS["threshold"], linestyle="--", linewidth=1.8)

    ax.set_title("Danh doi Do nhay / Do dac hieu theo nguong")
    ax.set_xlabel("Nguong sai khac tai tao")
    ax.set_ylabel("Ty le")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)


def plot_evaluation(model, test_data, test_labels, test_errors, metrics):
    os.makedirs("results/figures", exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    plot_threshold_distribution(axes[0], test_errors, test_labels, metrics["threshold"])
    plot_threshold_sensitivity_specificity(axes[1], test_labels, test_errors, metrics["threshold"])
    plot_ecg_with_highlight(axes[2], model, test_data, test_labels)

    plt.tight_layout()
    plt.savefig("results/figures/evaluation.png", dpi=150, bbox_inches="tight")
    plt.show()


def print_metrics(metrics, test_labels):
    print(f"Nguong phat hien (percentile {THRESHOLD_PERCENTILE}): {metrics['threshold']:.6f}")
    print(f"\nAUC-ROC Score: {metrics['auc_roc']:.4f}")
    print("\nClinical Metrics:")
    print(f"  Sensitivity (Do nhay)       : {metrics['sensitivity']:.4f}")
    print(f"  Specificity (Do dac hieu)   : {metrics['specificity']:.4f}")
    print(f"  PPV / Precision             : {metrics['ppv']:.4f}")
    print(f"  NPV                         : {metrics['npv']:.4f}")
    print(f"  Accuracy                    : {metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy           : {metrics['balanced_accuracy']:.4f}")
    print(f"  Confusion Matrix [TN FP FN TP]: {metrics['tn']} {metrics['fp']} {metrics['fn']} {metrics['tp']}")
    print("\nClassification Report:")
    print(classification_report(test_labels, metrics["predictions"], target_names=["Normal", "Anomaly"]))


def main():
    train_data, test_data, test_labels = load_data()
    model = load_model()

    train_errors = get_errors(model, train_data)
    test_errors = get_errors(model, test_data)
    metrics = compute_evaluation(train_errors, test_errors, test_labels)

    print_metrics(metrics, test_labels)
    plot_evaluation(model, test_data, test_labels, test_errors, metrics)
    print("\nDa luu bieu do tai results/figures/evaluation.png")


if __name__ == "__main__":
    main()
