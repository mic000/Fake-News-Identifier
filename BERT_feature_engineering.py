"""
BERT embeddings as an alternative feature representation.
Starts from the Part 2 output (WELFake_part2_preprocessed.csv.gz)
and reuses the EXISTING split_assignment.csv

WHY THIS IS DIFFERENT FROM PART 3'S PIPELINE
    TF-IDF/BoW *learn* a vocabulary and weights FROM your training data.
    BERT here is used as a FROZEN, already-pretrained model. Every article
    is just passed through the same fixed. The trade-off: BERT embeddings
    are DENSE  or "look at which words matter" do NOT translate directly
    to a BERT-based model.

COMPATIBILITY WITH YOUR EXISTING PART 4 SCRIPT
    Dense embeddings are saved wrapped in scipy's sparse format anyway,
    purely so your existing load_sparse_features() function (which calls
    sparse.load_npz) can load them completely unchanged -- no need to
    modify Part 4's loading code. To actually USE this representation in
    Part 4, just add "bert" to the REPRESENTATIONS list there (same
    one-line change used for adding bow_unigram / tfidf_trigram before).

    IMPORTANT CAVEAT: Naive Bayes (MultinomialNB) requires non-negative
    input and will error out on BERT embeddings (they can be negative).
    Skip "bert" + "naive_bayes" in Part 4, or guard it with a try/except.
    Logistic Regression, KNN, and Linear SVM are all fine with dense,
    signed features.

REQUIRES INTERNET ACCESS to huggingface.co to download the pretrained
model the first time (subsequent runs use the local cache). Run this on
your own machine, not in a network-restricted sandbox.

Usage:

    python bert_feature_extraction.py \
        --input processed/WELFake_part2_preprocessed.csv.gz \
        --split-file part3_output/split_assignment.csv \
        --output-dir part3_output \
        --model-name bert-base-uncased \
        --max-length 256 \
        --batch-size 16
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from transformers import AutoModel, AutoTokenizer

INPUT_SETTINGS = {
    "title": "clean_title",
    "body": "clean_body",
    "combined": "clean_title_body",
}


def load_split_table(split_file):
    split_df = pd.read_csv(split_file)
    tables = {}
    for split_name in ["train", "val", "test"]:
        rows = split_df[split_df["split"] == split_name].sort_values("article_id")
        tables[split_name] = rows.reset_index(drop=True)
    return tables


def mean_pool(token_embeddings, attention_mask):
    """
    Averages a document's token embeddings into one fixed-size vector,
    ignoring padding tokens (attention_mask == 0). This is the standard,
    widely-used way to turn BERT's per-token output into one embedding
    per document when the model itself was not fine-tuned for
    classification (no [CLS]-based classification head trained).
    """
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.shape).float()
    summed = torch.sum(token_embeddings * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def embed_texts(texts, tokenizer, model, device, max_length, batch_size):
    """
    Runs texts through BERT in batches and returns one mean-pooled
    embedding per text, as a single (n_texts, hidden_size) numpy array.
    """
    all_embeddings = []
    model.eval()

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)

            output = model(**encoded)
            pooled = mean_pool(output.last_hidden_state, encoded["attention_mask"])
            all_embeddings.append(pooled.cpu().numpy())

            done = min(start + batch_size, len(texts))
            print(f"    embedded {done}/{len(texts)}", end="\r")

    print()
    return np.vstack(all_embeddings)


def run_pipeline(input_path, split_file, output_dir, model_name, max_length, batch_size, device_name):
    print("Loading Part 2 preprocessed dataset...")
    df = pd.read_csv(input_path)
    for col in ["clean_title", "clean_body", "clean_title_body"]:
        df[col] = df[col].fillna("")

    print("Loading existing split assignment (reusing the same rows as Part 3)...")
    split_tables = load_split_table(split_file)
    for split_name, table in split_tables.items():
        print(f"  {split_name}: {len(table)} rows")

    df_by_id = df.set_index("article_id")
    ids_by_split = {name: split_tables[name]["article_id"] for name in ["train", "val", "test"]}

    device = torch.device(device_name if device_name else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    print(f"Loading pretrained model: {model_name} (requires internet access on first run)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    features_dir = output_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    for setting_name, column in INPUT_SETTINGS.items():
        print(f"\n=== Input setting: {setting_name} ({column}) ===")

        for split_name in ["train", "val", "test"]:
            texts = df_by_id.loc[ids_by_split[split_name], column].tolist()
            print(f"  Embedding {split_name} ({len(texts)} docs)...")

            embeddings = embed_texts(texts, tokenizer, model, device, max_length, batch_size)

            # Wrapped in sparse format purely so Part 4's existing
            # sparse.load_npz-based loader works unchanged -- these
            # embeddings are actually dense.
            sparse_wrapped = sparse.csr_matrix(embeddings)
            out_path = features_dir / f"{setting_name}_bert_{split_name}.npz"
            sparse.save_npz(out_path, sparse_wrapped)
            print(f"    saved: {out_path} (shape {embeddings.shape})")

    print("\nDone. To use this representation in Part 4, add \"bert\" to REPRESENTATIONS "
          "there (skip pairing it with naive_bayes -- see the module docstring for why).")


def main():
    parser = argparse.ArgumentParser(description="Part 3b: BERT embedding feature extraction for WELFake.")
    parser.add_argument("--input", required=True, help="Path to WELFake_part2_preprocessed.csv or .csv.gz")
    parser.add_argument("--split-file", required=True, help="Path to the existing split_assignment.csv to reuse")
    parser.add_argument("--output-dir", required=True, help="Part 3 output folder (features/ saved alongside existing files)")
    parser.add_argument("--model-name", default="bert-base-uncased", help="HuggingFace model name (default: bert-base-uncased)")
    parser.add_argument("--max-length", type=int, default=256, help="Max tokens per document (default: 256; body text will be truncated)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for embedding (default: 16)")
    parser.add_argument("--device", default=None, help="'cuda' or 'cpu' (default: auto-detect)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file was not found: {input_path.resolve()}")

    split_file = Path(args.split_file)
    if not split_file.exists():
        raise FileNotFoundError(f"Split assignment file was not found: {split_file.resolve()}")

    output_dir = Path(args.output_dir)

    run_pipeline(input_path, split_file, output_dir, args.model_name, args.max_length, args.batch_size, args.device)


if __name__ == "__main__":
    main()