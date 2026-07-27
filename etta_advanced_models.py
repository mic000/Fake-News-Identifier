from scipy import sparse

# load Data
# Load the features from the .npz files
feature_dir = "part3_output/features"

X_train = sparse.load_npz(
    f"{feature_dir}/combined_tfidf_uni_bigram_train.npz"
)

X_val = sparse.load_npz(
    f"{feature_dir}/combined_tfidf_uni_bigram_val.npz"
)

X_test = sparse.load_npz(
    f"{feature_dir}/combined_tfidf_uni_bigram_test.npz"
)

# Load the labels from the split_assignment.csv file
# train/val/test (80/10/10)
import pandas as pd

split_df = pd.read_csv("part3_output/split_assignment.csv")

train_df = split_df[split_df.split=="train"]
val_df = split_df[split_df.split=="val"]
test_df = split_df[split_df.split=="test"]

y_train = train_df["label"].values
y_val = val_df["label"].values
y_test = test_df["label"].values


# ==========================
# Tune Hyperparameters
# ==========================

# Tune Linear SVM
from sklearn.svm import LinearSVC

def tune_svm(X_train, y_train, X_val, y_val):

    best_model = None
    best_C = None
    best_f1 = -1

    c_values = [0.1, 1, 10]

    for c in c_values:

        model = LinearSVC(C=c, random_state=123)
        model.fit(X_train, y_train)

        pred = model.predict(X_val)
        f1 = f1_score(y_val, pred)

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_C = c

    return best_model, best_C, best_f1


# Tune Decision Tree
from sklearn.tree import DecisionTreeClassifier

def tune_decision_tree(X_train, y_train, X_val, y_val):

    best_model = None
    best_depth = None
    best_f1 = -1

    depth_values = [10, 20, None]

    for depth in depth_values:

        model = DecisionTreeClassifier(max_depth=depth, random_state=123)
        model.fit(X_train, y_train)

        pred = model.predict(X_val)
        f1 = f1_score(y_val, pred, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_depth = depth

    return best_model, best_depth, best_f1

# Tune Random Forest
from sklearn.ensemble import RandomForestClassifier

def tune_random_forest(X_train, y_train, X_val, y_val):

    best_model = None
    best_n_estimators = None
    best_f1 = -1

    n_values = [50, 100, 200]

    for n in n_values:

        model = RandomForestClassifier(n_estimators=n, random_state=123, n_jobs=-1)
        model.fit(X_train, y_train)

        pred = model.predict(X_val)
        f1 = f1_score(y_val, pred)

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_n_estimators = n

    return best_model, best_n_estimators, best_f1




# Evaluate the model
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,confusion_matrix)

def evaluate_model(model_name, model, X_test, y_test):

    pred = model.predict(X_test)

    result = {
        "Model": model_name,
        "Accuracy": round(accuracy_score(y_test, pred), 4),
        "Precision": round(precision_score(y_test, pred, zero_division=0), 4),
        "Recall": round(recall_score(y_test, pred, zero_division=0), 4),
        "F1": round(f1_score(y_test, pred, zero_division=0), 4),
    }

    cm = confusion_matrix(y_test, pred)

    return result, cm

# ============================
# Run the hyperparameter tuning for each model
# ============================

svm_model, svm_C, svm_f1 = tune_svm(X_train, y_train, X_val, y_val)
print(f"Best SVM Model: C={svm_C}, F1={svm_f1}")

dt_model, dt_depth, dt_f1 = tune_decision_tree(X_train, y_train, X_val, y_val)
print(f"Best Decision Tree Model: max_depth={dt_depth}, F1={dt_f1}")

rf_model, rf_n_estimators, rf_f1 = tune_random_forest(X_train, y_train, X_val, y_val)
print(f"Best Random Forest Model: n_estimators={rf_n_estimators}, F1={rf_f1}")

# Evaluate the best models on the test set
svm_result, svm_cm = evaluate_model("SVM", svm_model, X_test, y_test)
dt_result, dt_cm = evaluate_model("Decision Tree", dt_model, X_test, y_test)
rf_result, rf_cm = evaluate_model("Random Forest", rf_model, X_test, y_test)

results_df = pd.DataFrame([
    svm_result,
    dt_result,
    rf_result
])

print(results_df)
print("SVM Confusion Matrix:\n", svm_cm)
print("Decision Tree Confusion Matrix:\n", dt_cm)
print("Random Forest Confusion Matrix:\n", rf_cm)



