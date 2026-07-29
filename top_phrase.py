"""
Word cloud of the highest-frequency phrases/words in a Part 3 vocabulary.

Example Usage:
    python top_phrase.py \
        --vocab part3_outputn/features/combined_tfidf_uni_bigram_vocab.json \
        --matrix part3_outputn/features/combined_tfidf_uni_bigram_train.npz \
        --phrases-only \
        --remove-stopwords \
        --max-words 60 \
        --output top_phrases_wordcloud.png

Drop --phrases-only to include single words too.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
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


def is_all_stopwords(term, stopwords):
    words = term.split()
    return all(w in stopwords for w in words)


def build_frequencies(vocab, matrix_path, phrases_only, remove_stopwords):
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

    if remove_stopwords:
        before_count = len(pairs)
        pairs = [(term, score) for term, score in pairs if not is_all_stopwords(term, ENGLISH_STOP_WORDS)]
        removed = before_count - len(pairs)
        print(f"Removed {removed} all-stopword terms out of {before_count} "
              f"({len(pairs)} remaining).")
        if not pairs:
            raise ValueError("Every term was filtered out as a stopword-only term -- try without --remove-stopwords.")

    freq_dict = {term: float(score) for term, score in pairs}

    return freq_dict


def main():
    parser = argparse.ArgumentParser(description="Word cloud of high-frequency phrases from a Part 3 vocabulary.")
    parser.add_argument("--vocab", required=True, help="Path to the *_vocab.json file")
    parser.add_argument("--matrix", required=True, help="Path to the matching *_train.npz file")
    parser.add_argument("--phrases-only", action="store_true",
                        help="Only include multi-word terms (bigrams/trigrams), not single words")
    parser.add_argument("--remove-stopwords", action="store_true",
                        help="Drop terms made ENTIRELY of stopwords (e.g. 'in the', 'to be') so the "
                             "cloud surfaces content words instead. Phrases with at least one "
                             "content word (e.g. 'donald trump') are kept even if partly stopwords.")
    parser.add_argument("--max-words", type=int, default=60, help="Max number of words/phrases in the cloud")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--output", default="phrase_wordcloud.png")
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    vocab = load_vocab(args.vocab)
    freq_dict = build_frequencies(vocab, args.matrix, args.phrases_only, args.remove_stopwords)

    wc = WordCloud(
        width=args.width,
        height=args.height,
        background_color="white",
        max_words=args.max_words,
        color_func=color_func,
        prefer_horizontal=0.92,
        relative_scaling=0.55,  # bigger gap between high- and low-frequency word sizes
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