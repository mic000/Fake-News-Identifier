"""
RQ4: Do different models misclassify the same articles, or do they make
different kinds of errors?

Usage Example:
    python QR4_error.py \
        --part4-predictions-dir part4_output/predictions \
        --part5-predictions-dir part5_output/predictions \
        --output-dir rq4_output

Expects the standard naming convention already used everywhere else:
    combined_tfidf_uni_bigram_<model>_test_predictions.csv
with columns: article_id, true_label, predicted_label
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TEAL = "#028090"
CORAL = "#E76F51"
INK = "#1B262C"
MUTED = "#6B7A80"
GRID = "#E0E6E6"

# Maps the filename fragment -> a clean display name for each model.
MODEL_FILES = {
    "logistic_regression": "Logistic Regression",
    "naive_bayes": "Naive Bayes",
    "knn_raw": "KNN (raw)",
    "knn_svd": "KNN (SVD)",
    "svm": "Linear SVM",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
}

FIXED_SETTING = "combined"
FIXED_REPRESENTATION = "tfidf_uni_bigram"


def load_predictions(predictions_dir):
    if predictions_dir is None:
        return {}

    predictions_dir = Path(predictions_dir)
    found = {}

    for file_key, display_name in MODEL_FILES.items():
        if file_key == "knn_svd":
            candidate = predictions_dir / f"{FIXED_SETTING}_svd_reduced_{file_key}_test_predictions.csv"
        else:
            candidate = predictions_dir / f"{FIXED_SETTING}_{FIXED_REPRESENTATION}_{file_key}_test_predictions.csv"

        if candidate.exists():
            df = pd.read_csv(candidate)
            found[display_name] = df.sort_values("article_id").reset_index(drop=True)

    return found


def build_error_table(all_predictions):
    model_names = list(all_predictions.keys())
    base = None

    for model_name, df in all_predictions.items():
        is_wrong = (df["predicted_label"] != df["true_label"]).astype(int)
        piece = pd.DataFrame({
            "article_id": df["article_id"],
            "true_label": df["true_label"],
            f"wrong__{model_name}": is_wrong.values,
            f"pred__{model_name}": df["predicted_label"].values,
        })
        if base is None:
            base = piece
        else:
            piece = piece.drop(columns=["true_label"])
            base = base.merge(piece, on="article_id", how="inner")

    return base, model_names


def compute_error_type_breakdown(error_table, model_names):
    rows = []
    for model_name in model_names:
        wrong_col = f"wrong__{model_name}"
        pred_col = f"pred__{model_name}"

        is_wrong = error_table[wrong_col] == 1
        false_positive = ((error_table["true_label"] == 0) & (error_table[pred_col] == 1)).sum()
        false_negative = ((error_table["true_label"] == 1) & (error_table[pred_col] == 0)).sum()

        rows.append({
            "model": model_name,
            "total_errors": int(is_wrong.sum()),
            "false_positive (real predicted fake)": int(false_positive),
            "false_negative (fake predicted real)": int(false_negative),
        })

    return pd.DataFrame(rows)


def compute_pairwise_overlap(error_table, model_names):
    n = len(model_names)
    overlap = pd.DataFrame(index=model_names, columns=model_names, dtype=float)

    for a in model_names:
        wrong_a = error_table[f"wrong__{a}"] == 1
        n_wrong_a = wrong_a.sum()
        for b in model_names:
            wrong_b = error_table[f"wrong__{b}"] == 1
            if n_wrong_a == 0:
                overlap.loc[a, b] = np.nan
            else:
                both_wrong = (wrong_a & wrong_b).sum()
                overlap.loc[a, b] = both_wrong / n_wrong_a

    return overlap


def plot_overlap_heatmap(overlap, output_path):
    fig, ax = plt.subplots(figsize=(1.3 * len(overlap) + 2, 1.1 * len(overlap) + 2), dpi=300)
    im = ax.imshow(overlap.values, cmap="Reds", vmin=0, vmax=1)

    ax.set_xticks(range(len(overlap.columns)))
    ax.set_xticklabels(overlap.columns, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(len(overlap.index)))
    ax.set_yticklabels(overlap.index, fontsize=10)

    for i in range(len(overlap.index)):
        for j in range(len(overlap.columns)):
            value = overlap.values[i, j]
            if not np.isnan(value):
                color = "white" if value > 0.6 else INK
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=color, fontsize=9.5)

    ax.set_title(
        "Error Overlap: of the articles ROW got wrong,\nfraction COLUMN also got wrong",
        fontsize=12, fontweight="bold", color=INK, pad=12,
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    output_path = Path(output_path)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path.resolve()}")


def plot_error_type_breakdown(error_type_df, output_path):
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    x = np.arange(len(error_type_df))
    width = 0.38

    ax.bar(x - width / 2, error_type_df["false_positive (real predicted fake)"], width,
           label="False Positive (real -> fake)", color=CORAL, zorder=3)
    ax.bar(x + width / 2, error_type_df["false_negative (fake predicted real)"], width,
           label="False Negative (fake -> real)", color=TEAL, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(error_type_df["model"], rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Number of test articles", fontsize=11, color=INK)
    ax.set_title("Error Type Breakdown by Model", fontsize=13, fontweight="bold", color=INK, pad=12)
    ax.legend(frameon=False, fontsize=10)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    output_path = Path(output_path)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path.resolve()}")


def find_consensus_errors(error_table, model_names, min_models_wrong):
    wrong_cols = [f"wrong__{m}" for m in model_names]
    error_table = error_table.copy()
    error_table["n_models_wrong"] = error_table[wrong_cols].sum(axis=1)

    consensus = error_table[error_table["n_models_wrong"] >= min_models_wrong]
    consensus = consensus.sort_values("n_models_wrong", ascending=False)

    return consensus[["article_id", "true_label", "n_models_wrong"]], error_table["n_models_wrong"]


def main():
    parser = argparse.ArgumentParser(description="RQ4: compare which articles different models misclassify.")
    parser.add_argument("--part4-predictions-dir", default=None, help="Part 4 predictions/ folder")
    parser.add_argument("--part5-predictions-dir", default=None, help="Part 5 predictions/ folder (optional)")
    parser.add_argument("--output-dir", default="rq4_output")
    parser.add_argument("--consensus-threshold", type=int, default=None,
                         help="Flag articles wrong by at least this many models (default: majority, i.e. more than half)")
    args = parser.parse_args()

    all_predictions = {}
    all_predictions.update(load_predictions(args.part4_predictions_dir))
    all_predictions.update(load_predictions(args.part5_predictions_dir))

    if len(all_predictions) < 2:
        raise ValueError(
            f"Only found {len(all_predictions)} model(s) with predictions for "
            f"{FIXED_SETTING}+{FIXED_REPRESENTATION}. Need at least 2 to compare. "
            f"Check --part4-predictions-dir / --part5-predictions-dir."
        )

    print(f"Loaded predictions for {len(all_predictions)} models: {list(all_predictions.keys())}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    error_table, model_names = build_error_table(all_predictions)
    n_test_articles = len(error_table)
    print(f"Test set size (common to all models): {n_test_articles}")

    # --- Error type breakdown ---
    error_type_df = compute_error_type_breakdown(error_table, model_names)
    error_type_df.to_csv(output_dir / "error_type_breakdown.csv", index=False)
    plot_error_type_breakdown(error_type_df, output_dir / "error_type_breakdown.png")
    print("\nError type breakdown:")
    print(error_type_df.to_string(index=False))

    # --- Pairwise overlap ---
    overlap = compute_pairwise_overlap(error_table, model_names)
    overlap.to_csv(output_dir / "error_overlap_matrix.csv")
    plot_overlap_heatmap(overlap, output_dir / "error_overlap_heatmap.png")

    # --- Consensus errors ---
    threshold = args.consensus_threshold or (len(model_names) // 2 + 1)
    consensus_df, n_models_wrong_per_article = find_consensus_errors(error_table, model_names, threshold)
    consensus_df.to_csv(output_dir / "consensus_errors.csv", index=False)

    print(f"\nArticles wrong by >= {threshold} of {len(model_names)} models: {len(consensus_df)} "
          f"out of {n_test_articles} test articles ({len(consensus_df) / n_test_articles:.1%})")

    print("\nDistribution of 'how many models got this article wrong':")
    print(n_models_wrong_per_article.value_counts().sort_index().to_string())

    print(f"\nAll RQ4 outputs saved under: {output_dir.resolve()}")


if __name__ == "__main__":
    main()