"""
Word cloud of the highest-frequency phrases/words in a Part 3 vocabulary.

Example Usage:

    python top_phrase.py
        --vocab part3_outputn/features/combined_tfidf_uni_bigram_vocab.json
        --matrix part3_outputn/features/combined_tfidf_uni_bigram_train.npz
        --phrases-only
        --max-words 60
        --output top_phrases_wordcloud.png

Drop --phrases-only to include single words too.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from wordcloud import WordCloud
import matplotlib.pyplot as plt

PALETTE = ["#028090", "#00A896", "#02C39A", "#015159", "#6B7A80", "#E76F51"]
INK = "#1B262C"


def load_vocab(vocab_path):
    with open(vocab_path, encoding="utf-8") as f:
        return json.load(f)


def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    idx = abs(hash(word)) % len(PALETTE)
    return PALETTE[idx]


def build_frequencies(vocab, matrix_path, phrases_only):
    matrix = sparse.load_npz(matrix_path)
    if len(vocab) != matrix.shape[1]:
        raise ValueError(
            f"Vocab has {len(vocab)} terms but the matrix has {matrix.shape[1]} columns -- "
            f"these two files don't match. Make sure --vocab and --matrix come from the "
            f"same run (same setting + representation)."
        )

    totals = np.asarray(matrix.sum(axis=0)).ravel()
    pairs = list(zip(vocab, totals))

    if phrases_only:
        pairs = [(term, score) for term, score in pairs if " " in term]
        if not pairs:
            raise ValueError("No multi-word terms found in this vocabulary -- try without --phrases-only.")

    freq_dict = {term: float(score) for term, score in pairs}

    return freq_dict


def main():
    parser = argparse.ArgumentParser(description="Word cloud of high-frequency phrases from a Part 3 vocabulary.")
    parser.add_argument("--vocab", required=True, help="Path to the *_vocab.json file")
    parser.add_argument("--matrix", required=True, help="Path to the matching *_train.npz file")
    parser.add_argument("--phrases-only", action="store_true",
                         help="Only include multi-word terms (bigrams/trigrams), not single words")
    parser.add_argument("--max-words", type=int, default=60, help="Max number of words/phrases in the cloud")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--output", default="phrase_wordcloud.png")
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    vocab = load_vocab(args.vocab)
    freq_dict = build_frequencies(vocab, args.matrix, args.phrases_only)

    wc = WordCloud(
        width=args.width,
        height=args.height,
        background_color="white",
        max_words=args.max_words,
        color_func=color_func,
        prefer_horizontal=0.92,
        relative_scaling=0.55,   # bigger gap between high- and low-frequency word sizes
        collocations=False,
        margin=6,
    ).generate_from_frequencies(freq_dict)

    fig, ax = plt.subplots(figsize=(args.width / 150, args.height / 150), dpi=300)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")

    title = args.title or ("Highest-Frequency Phrases" if args.phrases_only else "Highest-Frequency Terms")
    ax.set_title(title, fontsize=16, fontweight="bold", color=INK, pad=14)

    plt.tight_layout()
    output_path = Path(args.output)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved: {output_path.resolve()}")
    top10 = sorted(freq_dict.items(), key=lambda p: p[1], reverse=True)[:10]
    print("\nTop 10 by frequency (word cloud sizes are driven by these):")
    for term, score in top10:
        print(f"  {score:>10,.1f}   {term}")


if __name__ == "__main__":
    main()