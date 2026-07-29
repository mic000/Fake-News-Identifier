"""
Part 4: Baseline and Classical Machine Learning Models

Inputs expected (produced by Part 3):
    <part3-dir>/split_assignment.csv
    <part3-dir>/features/<setting>_<representation>_<split>.npz

Outputs (under --output-dir):
    results_summary.csv
        One row per (input_setting, representation, model) with the
        chosen hyperparameter, validation F1, test Accuracy /
        Precision / Recall / F1, and the SVD size used (if any).
    confusion_matrices.json
        Test-set confusion matrix (TN, FP, FN, TP) for every config.
    predictions/<setting>_<representation>_<model>_test_predictions.csv
        article_id, true label, predicted label for every test row,
        for later error analysis (Part 5).

Example Usage:
    python ming_baseline_modeling.py \
        --part3-dir part3_output \
        --output-dir part4_output \
        --svd-components 150
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

RANDOM_SEED = 123
INPUT_SETTINGS = ["title", "body", "combined"]
REPRESENTATIONS = ["bow_unigram", "tfidf_unigram", "tfidf_uni_bigram", "tfidf_trigram"]
LOGREG_C_VALUES = [0.001, 0.01, 0.1, 1, 10]
NB_ALPHA_VALUES = [0.1, 0.5, 1.0, 2.0, 5.0]
KNN_K_VALUES = [3, 5, 7, 9, 15, 25]

# BASELINE_SETTING = "combined"
# BASELINE_REPRESENTATION = "tfidf_uni_bigram"
BASELINE_MODEL = "logistic_regression"


def load_split_table(split_file):
    split_df = pd.read_csv(split_file)
    tables = {}
    for split_name in ["train", "val", "test"]:
        rows = split_df[split_df["split"] == split_name].sort_values("article_id")
        tables[split_name] = rows.reset_index(drop=True)
    return tables


def load_sparse_features(features_dir, setting_name, representation):
    prefix = f"{setting_name}_{representation}"
    matrices = {}
    for split_name in ["train", "val", "test"]:
        path = features_dir / f"{prefix}_{split_name}.npz"
        matrices[split_name] = sparse.load_npz(path)
    return matrices


def compute_svd_features(tfidf_matrices, n_components):
    X_train = tfidf_matrices["train"]
    safe_n_components = min(n_components, X_train.shape[0] - 1, X_train.shape[1] - 1)
    svd = TruncatedSVD(n_components=safe_n_components, random_state=RANDOM_SEED)
    reduced = {
        "train": svd.fit_transform(X_train),
        "val": svd.transform(tfidf_matrices["val"]),
        "test": svd.transform(tfidf_matrices["test"]),
    }
    return reduced, safe_n_components, svd


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
    }


def compute_confusion_matrix(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "true_negative": int(matrix[0, 0]), "false_positive": int(matrix[0, 1]),
        "false_negative": int(matrix[1, 0]), "true_positive": int(matrix[1, 1]),
    }


def tune_logistic_regression(X_train, y_train, X_val, y_val):
    best_val_f1 = -1
    best_model = None
    best_params = None

    for c_value in LOGREG_C_VALUES:
        model = LogisticRegression(C=c_value, max_iter=1000, random_state=RANDOM_SEED)
        model.fit(X_train, y_train)

        val_pred = model.predict(X_val)
        val_f1 = f1_score(y_val, val_pred, zero_division=0)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model = model
            best_params = {"C": c_value}
    return best_model, best_params, round(float(best_val_f1), 4)


def tune_naive_bayes(X_train, y_train, X_val, y_val):
    best_val_f1 = -1
    best_model = None
    best_params = None

    for alpha_value in NB_ALPHA_VALUES:
        model = MultinomialNB(alpha=alpha_value)
        model.fit(X_train, y_train)

        val_pred = model.predict(X_val)
        val_f1 = f1_score(y_val, val_pred, zero_division=0)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model = model
            best_params = {"alpha": alpha_value}
    return best_model, best_params, round(float(best_val_f1), 4)


def tune_knn(X_train, y_train, X_val, y_val):
    best_val_f1 = -1
    best_model = None
    best_params = None

    for k_value in KNN_K_VALUES:
        model = KNeighborsClassifier(n_neighbors=k_value)
        model.fit(X_train, y_train)

        val_pred = model.predict(X_val)
        val_f1 = f1_score(y_val, val_pred, zero_division=0)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model = model
            best_params = {"n_neighbors": k_value}
    return best_model, best_params, round(float(best_val_f1), 4)


def run_one_config(model_name, tune_function, X_train, y_train, X_val, y_val,
                   X_test, y_test, test_article_ids, config_tag, predictions_dir, models_dir,):
    best_model, best_params, val_f1 = tune_function(X_train, y_train, X_val, y_val)
    test_pred = best_model.predict(X_test)
    test_metrics = compute_metrics(y_test, test_pred)
    test_confusion = compute_confusion_matrix(y_test, test_pred)

    print(
        f"      best params = {best_params} | val F1 = {val_f1} "
        f"| test F1 = {test_metrics['f1']}"
    )

    predictions_dir.mkdir(parents=True, exist_ok=True)
    predictions_df = pd.DataFrame(
        {"article_id": test_article_ids, "true_label": y_test, "predicted_label": test_pred,})
    predictions_df.to_csv(
        predictions_dir / f"{config_tag}_{model_name}_test_predictions.csv",
        index=False,
    )

    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, models_dir / f"{config_tag}_{model_name}.joblib")

    return {
        "model": model_name,
        "best_params": best_params,
        "val_f1": val_f1,
        "accuracy": test_metrics["accuracy"],
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "f1": test_metrics["f1"],
        "confusion matrix": test_confusion,
    }


def run_pipeline(part3_dir, output_dir, svd_n_components):
    split_file = part3_dir / "split_assignment.csv"
    features_dir = part3_dir / "features"
    for required_path in [split_file, features_dir]:
        if not required_path.exists():
            raise FileNotFoundError(f"Expected Part 3 output not found")

    split_tables = load_split_table(split_file)
    y_train = split_tables["train"]["label"].to_numpy()
    y_val = split_tables["val"]["label"].to_numpy()
    y_test = split_tables["test"]["label"].to_numpy()
    test_article_ids = split_tables["test"]["article_id"].to_numpy()

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = output_dir / "predictions"
    models_dir = output_dir / "models"

    all_results = []
    confusion_matrices = {}
    for setting_name in INPUT_SETTINGS:
        for representation in REPRESENTATIONS:
            config_tag = f"{setting_name}_{representation}"
            print(f"\n=== {config_tag} ===")

            matrices = load_sparse_features(features_dir, setting_name, representation)
            X_train = matrices["train"]
            X_val = matrices["val"]
            X_test = matrices["test"]
            model_runs = [
                ("logistic_regression", tune_logistic_regression),
                ("naive_bayes", tune_naive_bayes),
                ("knn_raw", tune_knn),
            ]

            for model_name, tune_function in model_runs:
                result = run_one_config(model_name, tune_function,
                    X_train, y_train, X_val, y_val, X_test, y_test,
                    test_article_ids, config_tag, predictions_dir, models_dir,
                )
                result["input setting"] = setting_name
                result["representation"] = representation
                result["is base model"] = bool(model_name == BASELINE_MODEL)
                confusion_key = f"{config_tag}_{model_name}"
                confusion_matrices[confusion_key] = result.pop("confusion matrix")
                all_results.append(result)

    # SVD is only computed here setting is tfidf_uni_bigram features
    for setting_name in INPUT_SETTINGS:
        config_tag = f"{setting_name}_svd_reduced"
        print(f"\n=== {config_tag} ===")

        tfidf_matrices = load_sparse_features(features_dir, setting_name, "tfidf_uni_bigram")
        svd_matrices, actual_n_components, svd_transformer = compute_svd_features(
            tfidf_matrices, svd_n_components)
        joblib.dump(svd_transformer, models_dir / f"{setting_name}_svd_transformer.joblib")

        print(f"  SVD n_components used: {actual_n_components}")
        result = run_one_config("knn_svd", tune_knn,
            svd_matrices["train"], y_train, svd_matrices["val"], y_val,
            svd_matrices["test"], y_test, test_article_ids, config_tag, predictions_dir, models_dir)
        result["input setting"] = setting_name
        result["representation"] = "tfidf_uni_bigram_svd_reduced"
        result["is base model"] = False
        result["svd n_components"] = actual_n_components
        confusion_key = f"{config_tag}_knn_svd"
        confusion_matrices[confusion_key] = result.pop("confusion matrix")
        all_results.append(result)

    results_df = pd.DataFrame(all_results)
    column_order = ["input setting", "representation", "model", "is base model", "best_params", "val_f1",
                    "accuracy", "precision", "recall", "f1", "svd n_components",]
    results_df = results_df.reindex(columns=column_order)
    results_df = results_df.sort_values(["input setting", "representation", "model"])
    results_path = output_dir / "results_summary.csv"
    results_df.to_csv(results_path, index=False)

    confusion_path = output_dir / "confusion_matrices.json"
    with open(confusion_path, "w", encoding="utf-8") as f:
        json.dump(confusion_matrices, f, indent=2)

    print(results_df.sort_values("f1", ascending=False)
          .head(5)[["input setting", "representation", "model", "f1"]]
          .to_string(index=False))

    return results_df


def main():
    parser = argparse.ArgumentParser(
        description="Part 4: baseline and classical models for WELFake."
    )
    parser.add_argument(
        "--part3-dir",
        required=True,
        help="Path to the Part 3 output folder (contains split_assignment.csv, "
             "features/).",
    )
    parser.add_argument(
        "--output-dir",
        default="part4_output",
        help="Folder where results will be saved.",
    )
    parser.add_argument(
        "--svd-components",
        type=int,
        default=150,
        help="Number of SVD components to reduce tfidf_uni_bigram to, for the KNN+SVD comparison (default: 150).",
    )
    args = parser.parse_args()

    part3_dir = Path(args.part3_dir)
    if not part3_dir.exists():
        raise FileNotFoundError(f"Part 3 output folder not found: {part3_dir.resolve()}")

    output_dir = Path(args.output_dir)
    run_pipeline(part3_dir, output_dir, args.svd_components)


if __name__ == "__main__":
    main()