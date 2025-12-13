import os
import joblib
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


def evaluate_model(model, X, y):
    y_pred = model.predict(X)

    acc = accuracy_score(y, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y, y_pred, average="weighted"
    )

    try:
        if hasattr(model, "predict_proba"):
            y_scores = model.predict_proba(X)[:, 1]
        else:
            y_scores = model.decision_function(X)
        auc = roc_auc_score(y, y_scores)
    except Exception:
        auc = None

    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "ROC_AUC": auc
    }


def save_model(model, model_name, project_root):
    model_dir = os.path.join(project_root, "model")
    os.makedirs(model_dir, exist_ok=True)

    path = os.path.join(
        model_dir,
        f"{model_name.replace(' ', '_').lower()}_best_model.joblib"
    )

    joblib.dump(model, path, compress=3)
    return path