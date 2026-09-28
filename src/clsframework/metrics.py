import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def evaluate_arrays(task, targets, probabilities, names, threshold=0.5):
    y, p = np.asarray(targets), np.asarray(probabilities)
    if len(y) == 0:
        raise ValueError("Cannot evaluate an empty split")
    result = {"samples": len(y)}
    if task == "multiclass":
        prediction = p.argmax(1)
        precision, recall, f1, support = precision_recall_fscore_support(
            y, prediction, labels=np.arange(len(names)), zero_division=0
        )
        result.update(
            accuracy_top1=float(accuracy_score(y, prediction)),
            macro_f1=float(f1.mean()),
            confusion_matrix=confusion_matrix(y, prediction, labels=np.arange(len(names))).tolist(),
        )
        if len(names) >= 5:
            result["accuracy_top5"] = float((np.argsort(p, axis=1)[:, -5:] == y[:, None]).any(axis=1).mean())
        output_names = names
    else:
        prediction = (p >= threshold).astype(int)
        output_names = [names[1]] if task == "binary" else names
        if task == "binary":
            y, p, prediction = y.reshape(-1, 1), p.reshape(-1, 1), prediction.reshape(-1, 1)
        columns = [
            precision_recall_fscore_support(y[:, i], prediction[:, i], labels=[1], zero_division=0)
            for i in range(y.shape[1])
        ]
        precision, recall, f1, support = [np.array([c[j][0] for c in columns]) for j in range(4)]
        result["macro_f1"] = float(f1.mean())
        result["micro_f1"] = float(
            precision_recall_fscore_support(y.ravel(), prediction.ravel(), labels=[1], zero_division=0)[2][0]
        )
        ap, auc = [], []
        for i in range(y.shape[1]):
            ap.append(float(average_precision_score(y[:, i], p[:, i])) if y[:, i].sum() > 0 else None)
            auc.append(float(roc_auc_score(y[:, i], p[:, i])) if len(np.unique(y[:, i])) == 2 else None)
        defined = [v for v in ap if v is not None]
        result.update(
            map=float(np.mean(defined)) if defined else None,
            ap_defined_classes=len(defined),
            per_class_ap=dict(zip(output_names, ap, strict=True)),
            per_class_auc=dict(zip(output_names, auc, strict=True)),
            threshold=threshold,
        )
    result["per_class"] = {
        name: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, name in enumerate(output_names)
    }
    return result
