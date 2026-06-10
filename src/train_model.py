from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "dataset.csv"
MODEL_DIR = ROOT / "model"
REPORTS_DIR = ROOT / "reports"
MODEL_PATH = MODEL_DIR / "modelo_final.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"


def load_and_prepare_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna().copy()
    df = df.drop(columns=["customerID"])
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
    df = add_features(df)
    return df


def tenure_group(value: float) -> str:
    if value <= 12:
        return "0-12"
    if value <= 24:
        return "13-24"
    if value <= 36:
        return "25-36"
    return "37+"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["tenure_group"] = df["tenure"].apply(tenure_group).astype("category")
    return df


def split_features_target(df: pd.DataFrame):
    y = df["Churn"]
    X = df.drop(columns=["Churn"])
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    return X, y, num_cols, cat_cols


def build_preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
        ]
    )


def build_models(preprocessor: ColumnTransformer) -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        random_state=42,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        random_state=42,
                        class_weight="balanced",
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "Gradient Boosting": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", GradientBoostingClassifier(random_state=42)),
            ]
        ),
    }


def evaluate_predictions(y_true, y_pred, y_prob) -> dict[str, float]:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
        "AUC-ROC": roc_auc_score(y_true, y_prob),
    }


def evaluate_model(model: Pipeline, X, y, threshold: float = 0.5) -> dict[str, float]:
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    return evaluate_predictions(y, y_pred, y_prob)


def find_threshold(y_true, y_prob) -> dict[str, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )
    valid_thresholds = np.r_[thresholds, 1.0]
    best_idx = int(np.nanargmax(f1))
    return {
        "threshold": float(valid_thresholds[best_idx]),
        "precision": float(precision[best_idx]),
        "recall": float(recall[best_idx]),
        "f1": float(f1[best_idx]),
    }


def feature_importance(model: Pipeline) -> pd.DataFrame:
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    return (
        pd.DataFrame(
            {
                "Feature": preprocessor.get_feature_names_out(),
                "Importance": classifier.feature_importances_,
            }
        )
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )


def logistic_coefficients(model: Pipeline) -> pd.DataFrame:
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    return (
        pd.DataFrame(
            {
                "Feature": preprocessor.get_feature_names_out(),
                "Coefficient": classifier.coef_[0],
                "AbsCoefficient": np.abs(classifier.coef_[0]),
            }
        )
        .sort_values("AbsCoefficient", ascending=False)
        .reset_index(drop=True)
    )


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    df = load_and_prepare_data()
    X, y, num_cols, cat_cols = split_features_target(df)
    preprocessor = build_preprocessor(num_cols, cat_cols)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
    )

    models = build_models(preprocessor)
    validation_rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_val, y_val)
        validation_rows.append({"Modelo": name, **metrics})

    validation_df = (
        pd.DataFrame(validation_rows)
        .sort_values(["AUC-ROC", "F1-Score"], ascending=False)
        .reset_index(drop=True)
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "Accuracy": "accuracy",
        "Precision": "precision",
        "Recall": "recall",
        "F1-Score": "f1",
        "AUC-ROC": "roc_auc",
    }
    cv_rows = []
    for name, model in models.items():
        scores = cross_validate(model, X, y, cv=skf, scoring=scoring, n_jobs=-1)
        cv_rows.append(
            {
                "Modelo": name,
                **{
                    metric: float(scores[f"test_{metric}"].mean())
                    for metric in scoring
                },
            }
        )
    cv_df = (
        pd.DataFrame(cv_rows)
        .sort_values(["AUC-ROC", "F1-Score"], ascending=False)
        .reset_index(drop=True)
    )

    final_model_name = "Gradient Boosting"
    final_model = models[final_model_name]
    val_prob = final_model.predict_proba(X_val)[:, 1]
    threshold_info = find_threshold(y_val, val_prob)
    tuned_threshold = threshold_info["threshold"]

    test_default = evaluate_model(final_model, X_test, y_test, threshold=0.5)
    test_tuned = evaluate_model(final_model, X_test, y_test, threshold=tuned_threshold)
    y_test_prob = final_model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_prob >= tuned_threshold).astype(int)
    cm = confusion_matrix(y_test, y_test_pred)
    fpr, tpr, _ = roc_curve(y_test, y_test_prob)

    validation_df.to_csv(REPORTS_DIR / "comparacao_validacao.csv", index=False)
    cv_df.to_csv(REPORTS_DIR / "validacao_cruzada.csv", index=False)
    pd.DataFrame(
        [
            {"Cenario": "Threshold padrao 0.50", **test_default},
            {f"Cenario": f"Threshold ajustado {tuned_threshold:.3f}", **test_tuned},
        ]
    ).to_csv(REPORTS_DIR / "resultado_teste.csv", index=False)
    feature_importance(final_model).head(20).to_csv(
        REPORTS_DIR / "feature_importance.csv", index=False
    )
    logistic_coefficients(models["Logistic Regression"]).head(20).to_csv(
        REPORTS_DIR / "coeficientes_logistic_regression.csv", index=False
    )

    joblib.dump(final_model, MODEL_PATH)
    metadata = {
        "model_name": final_model_name,
        "threshold": tuned_threshold,
        "threshold_validation": threshold_info,
        "input_columns": X.columns.tolist(),
        "numeric_columns": num_cols,
        "categorical_columns": cat_cols,
        "target": "Churn",
        "class_mapping": {"Nao churn": 0, "Churn": 1},
        "dataset_rows_after_cleaning": int(len(df)),
        "churn_rate": float(y.mean()),
        "test_confusion_matrix": cm.tolist(),
        "test_auc_curve_area": float(auc(fpr, tpr)),
        "test_metrics_default_threshold": test_default,
        "test_metrics_tuned_threshold": test_tuned,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Modelo final:", final_model_name)
    print("Threshold ajustado:", round(tuned_threshold, 3))
    print("Metricas de teste com threshold ajustado:")
    print(pd.Series(test_tuned).round(4))
    print("Modelo salvo em:", MODEL_PATH)


if __name__ == "__main__":
    main()
