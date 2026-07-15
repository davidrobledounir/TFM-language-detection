from sklearn.metrics import confusion_matrix, f1_score


def score(pred_labels: list[str], true_labels: list[str], labels: list[str]) -> dict:
    return {
        "f1_macro": float(
            f1_score(
                true_labels,
                pred_labels,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "confusion": confusion_matrix(true_labels, pred_labels, labels=labels).tolist(),
        "labels": labels,
    }
