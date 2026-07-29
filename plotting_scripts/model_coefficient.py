"""
Show words/phrases actually relies on. This is a different question
than "which words appear most often".

Requires:
    - A trained model .joblib file (saved by Part 4, in models/)
    - The matching vectorizer .joblib file (saved by Part 3, in
      features/, SAME setting + representation as the model)

Example Usage:
    python plotting_scripts/model_coefficient.py \
        --model part4_output/models/combined_tfidf_uni_bigram_logistic_regression.joblib \
        --vectorizer part3_output/features/combined_tfidf_uni_bigram_vectorizer.joblib \
        --top-n 20 \
        --output plotting_scripts/model_coefficients.png
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import matplotlib.pyplot as plt

CORAL = "#E76F51"
TEAL = "#028090"
INK = "#1B262C"
MUTED = "#6B7A80"
GRID = "#E0E6E6"


def load_model_and_vectorizer(model_path, vectorizer_path):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    if not hasattr(model, "coef_"):
        raise TypeError(
            f"The loaded model ({type(model).__name__}) has no .coef_ attribute -- "
            f"this script only supports linear models like Logistic Regression. "
            f"Naive Bayes / KNN / tree models expose feature importance differently."
        )

    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_.ravel()

    if len(feature_names) != len(coefficients):
        raise ValueError(
            f"Vectorizer has {len(feature_names)} features but the model has "
            f"{len(coefficients)} coefficients -- make sure --model and --vectorizer "
            f"come from the same setting + representation."
        )

    return feature_names, coefficients


def plot_top_coefficients(feature_names, coefficients, top_n, output_path):
    order = np.argsort(coefficients)

    most_real = order[:top_n]                    # most negative -> strongest REAL signal
    most_fake = order[::-1][:top_n]               # most positive -> strongest FAKE signal

    real_terms = feature_names[most_real][::-1]
    real_scores = coefficients[most_real][::-1]

    fake_terms = feature_names[most_fake]
    fake_scores = coefficients[most_fake]

    fig, (ax_real, ax_fake) = plt.subplots(1, 2, figsize=(13, max(4, 0.4 * top_n)), dpi=300)

    ax_real.barh(real_terms, real_scores, color=TEAL, zorder=3)
    ax_real.set_title("Pushes toward REAL", fontsize=13, fontweight="bold", color=TEAL, pad=10)
    ax_real.set_xlabel("Coefficient (more negative = stronger)", fontsize=10.5, color=INK)

    ax_fake.barh(fake_terms, fake_scores, color=CORAL, zorder=3)
    ax_fake.set_title("Pushes toward FAKE", fontsize=13, fontweight="bold", color=CORAL, pad=10)
    ax_fake.set_xlabel("Coefficient (more positive = stronger)", fontsize=10.5, color=INK)

    for ax in (ax_real, ax_fake):
        ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color("#B9C4C4")
        ax.tick_params(axis="y", labelsize=10.5, colors=INK)
    plt.tight_layout()

    output_path = Path(output_path)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved: {output_path.resolve()}")
    print(f"\nTop {top_n} words/phrases pushing toward FAKE:")
    for term, score in zip(fake_terms, fake_scores):
        print(f"  {score:+.4f}   {term}")
    print(f"\nTop {top_n} words/phrases pushing toward REAL:")
    for term, score in zip(reversed(real_terms), reversed(real_scores)):
        print(f"  {score:+.4f}   {term}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize which words/phrases a trained Logistic Regression model relies on most."
    )
    parser.add_argument("--model", required=True, help="Path to the trained model .joblib file")
    parser.add_argument("--vectorizer", required=True, help="Path to the matching vectorizer .joblib file")
    parser.add_argument("--top-n", type=int, default=20, help="How many terms to show per side (default: 20)")
    parser.add_argument("--output", default="model_coefficients.png")
    args = parser.parse_args()

    feature_names, coefficients = load_model_and_vectorizer(args.model, args.vectorizer)
    plot_top_coefficients(feature_names, coefficients, args.top_n, args.output)


if __name__ == "__main__":
    main()