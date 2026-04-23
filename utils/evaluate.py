import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score, confusion_matrix

def evaluate(model, loader, device, num_classes=7):
    model.eval()
    
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # ---------------- OA ----------------
    OA = 100 * np.mean(all_preds == all_labels)

    # ---------------- Confusion Matrix ----------------
    cm = confusion_matrix(all_labels, all_preds, labels=np.arange(num_classes))

    # ---------------- Class-wise Accuracy ----------------
    class_totals = cm.sum(axis=1)

    class_acc = np.divide(
        cm.diagonal(),
        class_totals,
        out=np.zeros_like(class_totals, dtype=float),
        where=class_totals != 0
    )

    class_acc = class_acc * 100  # convert to %

    # ---------------- AA ----------------
    AA = np.mean(class_acc)

    # ---------------- Kappa ----------------
    kappa = cohen_kappa_score(all_labels, all_preds)

    return OA, AA, kappa, class_acc, cm


def print_results(OA, AA, kappa, class_acc):
    print(f"\n🔥 FINAL RESULTS")
    print(f"Overall Accuracy (OA): {OA:.2f}%")
    print(f"Average Accuracy (AA): {AA:.2f}%")
    print(f"Kappa Score: {kappa:.4f}")

    print("\n📊 Class-wise Accuracy:")
    for i, acc in enumerate(class_acc):
        print(f"Class {i}: {acc:.2f}%")