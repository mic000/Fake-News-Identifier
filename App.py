"""
A simple web UI: paste in a title + body, pick which trained model
configuration to use, and get a REAL / FAKE prediction with confidence.

This ties together everything already produced by the group's scripts:
    - ming_feature_engineering.py   (Part 3: vectorizers, saved as
                                      features/*_vectorizer.joblib)
    - ming_baseline_modeling.py     (Part 4: models, saved as
                                      models/*.joblib, plus
                                      models/*_svd_transformer.joblib)
    - etta_advanced_models.py       (Part 5: SVM / Decision Tree /
                                      Random Forest, saved as
                                      part5_output/models/*.joblib)

It does NOT retrain anything -- it only loads whatever .joblib files
already exist on disk and lets you pick one from a dropdown.

CRITICAL: a brand-new article must be cleaned the exact same way every
training article was in Part 2 (Arunkumar's pipeline), or the words
won't match the vectorizer's vocabulary at all. The clean_text() /
remove_source_markers() functions below are copied verbatim from
arunkumar_preprocessing_final_corrected.py for that reason -- don't
edit one without checking the other stays in sync.

Usage Examples:
    pip install gradio joblib scikit-learn scipy pandas

    python app.py \
        --part3-dir part3_output \
        --part4-dir part4_output \
        --part5-dir part5_output

Then open the local URL Gradio prints in your browser.
"""

import argparse
import html
import re
from pathlib import Path

import joblib
import gradio as gr

# -------------------------------------------------------------------
# Cleaning logic -- copied from Arunkumar's Part 2 pipeline so a new
# article is cleaned exactly the same way every training article was.
# -------------------------------------------------------------------

URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+)", flags=re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", flags=re.IGNORECASE)
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
EXTRA_SPACE_PATTERN = re.compile(r"\s+")
NON_LETTER_PATTERN = re.compile(r"[^a-zA-Z\s']")
REUTERS_DATELINE_PATTERN = re.compile(
    r"^\s*[A-Z][A-Z .'-]{1,40}\s*(?:\([A-Za-z]+\))?\s*[-\u2013\u2014]\s*"
)
SOURCE_MARKERS = ["reuters", "new york times", "breitbart", "cnn", "fox news"]


def remove_source_markers(text):
    cleaned = str(text)
    cleaned = re.sub(r"\(\s*reuters\s*\)", " ", cleaned, flags=re.IGNORECASE)
    cleaned = REUTERS_DATELINE_PATTERN.sub(" ", cleaned)
    for marker in sorted(SOURCE_MARKERS, key=len, reverse=True):
        cleaned = re.sub(rf"\b{re.escape(marker)}\b", " ", cleaned, flags=re.IGNORECASE)
    return cleaned


def clean_text(text, remove_markers=True):
    if text is None:
        return ""
    cleaned = html.unescape(str(text))
    cleaned = HTML_TAG_PATTERN.sub(" ", cleaned)
    cleaned = URL_PATTERN.sub(" ", cleaned)
    cleaned = EMAIL_PATTERN.sub(" ", cleaned)
    if remove_markers:
        cleaned = remove_source_markers(cleaned)
    cleaned = cleaned.lower()
    cleaned = NON_LETTER_PATTERN.sub(" ", cleaned)
    cleaned = re.sub(r"'{2,}", "'", cleaned)
    cleaned = EXTRA_SPACE_PATTERN.sub(" ", cleaned).strip()
    return cleaned


def build_input_text(setting, raw_title, raw_body):
    clean_title = clean_text(raw_title)
    clean_body = clean_text(raw_body)
    if setting == "title":
        return clean_title
    if setting == "body":
        return clean_body
    return (clean_title + " " + clean_body).strip()


# -------------------------------------------------------------------
# Scanning disk for which (setting, representation, model) combos are
# actually available, based on which .joblib files exist. This is
# enumerated against the known, fixed set of settings/representations/
# models each script produces, rather than parsed from filenames
# (settings/representations/model names all contain underscores, so
# parsing them back out of a combined filename is fragile).
# -------------------------------------------------------------------

SETTINGS = ["title", "body", "combined"]
REPRESENTATIONS = ["bow_unigram", "tfidf_unigram", "tfidf_uni_bigram", "tfidf_trigram"]

# Fixed to the group's established best configuration (highest overall F1
# across the project, combined + tfidf_uni_bigram). This keeps the UI to
# one dropdown -- "which model" -- instead of also asking about setting
# and representation, which were already decided during Part 3/4.
FIXED_SETTING = "combined"
FIXED_REPRESENTATION = "tfidf_uni_bigram"

PART4_MODELS = [
    ("logistic_regression", "Logistic Regression"),
    ("naive_bayes", "Naive Bayes"),
    ("knn_raw", "KNN (raw features)"),
]
PART5_MODELS = [
    ("svm", "Linear SVM"),
    ("decision_tree", "Decision Tree"),
    ("random_forest", "Random Forest"),
]


def scan_available_configs(part3_dir, part4_dir, part5_dir):
    """
    Returns a dict: {display_label: config_dict}, one entry per MODEL --
    setting and representation are fixed to FIXED_SETTING /
    FIXED_REPRESENTATION (the group's established best combination), so
    the only real choice left is which model to use.
    """
    features_dir = part3_dir / "features"
    configs = {}
    vectorizer_path = features_dir / f"{FIXED_SETTING}_{FIXED_REPRESENTATION}_vectorizer.joblib"

    # Part 4: Logistic Regression / Naive Bayes / KNN(raw)
    if part4_dir is not None:
        models_dir = part4_dir / "models"
        for model_key, model_label in PART4_MODELS:
            model_path = models_dir / f"{FIXED_SETTING}_{FIXED_REPRESENTATION}_{model_key}.joblib"
            if vectorizer_path.exists() and model_path.exists():
                configs[model_label] = {
                    "setting": FIXED_SETTING,
                    "vectorizer_path": vectorizer_path,
                    "model_path": model_path,
                    "svd_transformer_path": None,
                }

        # KNN + SVD: same tfidf_uni_bigram vectorizer, plus a saved SVD
        # transformer, feeding a separately-trained KNN model.
        svd_path = models_dir / f"{FIXED_SETTING}_svd_transformer.joblib"
        knn_svd_model_path = models_dir / f"{FIXED_SETTING}_svd_reduced_knn_svd.joblib"
        if vectorizer_path.exists() and svd_path.exists() and knn_svd_model_path.exists():
            configs["KNN (SVD-reduced)"] = {
                "setting": FIXED_SETTING,
                "vectorizer_path": vectorizer_path,
                "model_path": knn_svd_model_path,
                "svd_transformer_path": svd_path,
            }

    # Part 5: SVM / Decision Tree / Random Forest -- Etta's script only
    # ever trains on the "combined" + tfidf_uni_bigram configuration,
    # which is the same one everything else here is fixed to.
    if part5_dir is not None and FIXED_SETTING == "combined" and FIXED_REPRESENTATION == "tfidf_uni_bigram":
        models_dir = part5_dir / "models"
        for model_key, model_label in PART5_MODELS:
            model_path = models_dir / f"combined_tfidf_uni_bigram_{model_key}.joblib"
            if vectorizer_path.exists() and model_path.exists():
                configs[model_label] = {
                    "setting": FIXED_SETTING,
                    "vectorizer_path": vectorizer_path,
                    "model_path": model_path,
                    "svd_transformer_path": None,
                }

    return configs


# -------------------------------------------------------------------
# Prediction
# -------------------------------------------------------------------

def predict(title, body, config_label, configs):
    if not config_label:
        return "Please choose a model configuration first.", ""

    config = configs[config_label]
    setting = config["setting"]

    if setting in ("title", "combined") and not (title or "").strip():
        return "This configuration needs a title -- please fill it in.", ""
    if setting in ("body", "combined") and not (body or "").strip():
        return "This configuration needs a body -- please fill it in.", ""

    cleaned_text = build_input_text(setting, title, body)

    vectorizer = joblib.load(config["vectorizer_path"])
    X = vectorizer.transform([cleaned_text])

    if config["svd_transformer_path"] is not None:
        svd = joblib.load(config["svd_transformer_path"])
        X = svd.transform(X)

    model = joblib.load(config["model_path"])
    prediction = model.predict(X)[0]
    label_text = "FAKE" if prediction == 1 else "REAL"

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[0]
        confidence_text = f"Confidence -- real: {probabilities[0]:.1%} | fake: {probabilities[1]:.1%}"
    elif hasattr(model, "decision_function"):
        margin = model.decision_function(X)[0]
        confidence_text = (
            f"(This model has no probability output; raw decision margin = {margin:+.3f}, "
            f"positive leans FAKE, negative leans REAL)"
        )
    else:
        confidence_text = "(No confidence score available for this model.)"

    cleaned_preview = cleaned_text[:200] + ("..." if len(cleaned_text) > 200 else "")
    result_text = f"## Prediction: {label_text}"
    detail_text = f"{confidence_text}\n\nCleaned text used (first 200 chars):\n> {cleaned_preview}"

    return result_text, detail_text


# -------------------------------------------------------------------
# UI
# -------------------------------------------------------------------

def build_app(configs):
    labels = sorted(configs.keys())
    default_label = labels[0] if labels else None

    with gr.Blocks(title="WELFake: Real vs. Fake News Classifier") as demo:
        gr.Markdown("# Real vs. Fake News Classifier")
        gr.Markdown(
            "Paste in a news title and/or body, choose a trained model configuration, "
            "and see the prediction. This loads your already-trained models -- it does "
            "not retrain anything."
        )

        gr.Markdown(
            f"Fixed to the group's established best configuration: "
            f"**{FIXED_SETTING} + {FIXED_REPRESENTATION}**. Pick any trained model below."
        )

        if not labels:
            gr.Markdown(
                "**No trained models were found for this configuration.** Make sure you've "
                "run Part 3 (ming_feature_engineering.py) and Part 4 (ming_baseline_modeling.py) "
                "-- and optionally Part 5 (etta_advanced_models.py) -- and that you pointed "
                "--part3-dir / --part4-dir / --part5-dir at the right folders."
            )

        with gr.Row():
            title_box = gr.Textbox(label="Article Title", placeholder="e.g. Breaking: Scientists Discover Shocking Truth")
        with gr.Row():
            body_box = gr.Textbox(label="Article Body", lines=8, placeholder="Paste the article's body text here...")

        config_dropdown = gr.Dropdown(
            choices=labels,
            value=default_label,
            label="Model",
        )

        predict_button = gr.Button("Predict", variant="primary")

        result_output = gr.Markdown()
        detail_output = gr.Markdown()

        predict_button.click(
            fn=lambda t, b, c: predict(t, b, c, configs),
            inputs=[title_box, body_box, config_dropdown],
            outputs=[result_output, detail_output],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="UI for predicting real/fake news with trained WELFake models.")
    parser.add_argument("--part3-dir", required=True, help="Part 3 output folder (has features/*.joblib)")
    parser.add_argument("--part4-dir", default=None, help="Part 4 output folder (has models/*.joblib)")
    parser.add_argument("--part5-dir", default=None, help="Part 5 output folder (has models/*.joblib), optional")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    args = parser.parse_args()

    part3_dir = Path(args.part3_dir)
    part4_dir = Path(args.part4_dir) if args.part4_dir else None
    part5_dir = Path(args.part5_dir) if args.part5_dir else None

    if not part3_dir.exists():
        raise FileNotFoundError(f"Part 3 output folder not found: {part3_dir.resolve()}")

    configs = scan_available_configs(part3_dir, part4_dir, part5_dir)
    print(f"Found {len(configs)} usable model configuration(s):")
    for label in sorted(configs.keys()):
        print(f"  - {label}")

    demo = build_app(configs)
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()