"""
Train_Model.py  (Part 1 of 2 - see Load_Model.py for Part 2)
=============================================================
Direct Python translation of Train_Model.js. Every function below
implements exactly the same steps, in the same order, using the same
formulas as the JavaScript version - nothing was changed, renamed
(beyond JS camelCase -> Python snake_case), optimized, or rewritten
with a different algorithm.

Only Python's own standard library is used (sys, os, re, json, math,
random) -- no NumPy, no pandas, no scikit-learn, no external ML library.

The model is saved with Python's built-in `json` module (matching the
JS version's `fs.writeFileSync(..., JSON.stringify(...))`) to
"linear_regression_model.json".

Run:
    python Train_Model.py <path-to-csv>
    (defaults to "house_price_100000.csv" in the same folder if no path
    is given)

You will be prompted for the target column name at runtime, unless you
set TARGET_COLUMN below.
"""

import sys
import os
import re
import json
import math
import random

# ------------------------------------------------------------------
# CONFIG - the only things that ever change between datasets
# ------------------------------------------------------------------
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()


def resolve_csv_path():
    # Mirrors JS: process.argv[2] || path.join(__dirname, "house_price_100000.csv")
    if len(sys.argv) > 1:
        return sys.argv[1]
    return os.path.join(SCRIPT_DIR, "house_price_100000.csv")


CSV_PATH = resolve_csv_path()
TARGET_COLUMN = None  # e.g. "Price" / "Salary" / "Marks" - or leave
                       # None to be prompted for it interactively.

# Degree of polynomial expansion applied to NUMERIC features only
# (1 = plain linear regression). Raising this adds x^2, x^3, ... and
# pairwise interaction terms (x1*x2).
POLY_DEGREE = 2

# The exact filename requested for the saved model.
MODEL_SAVE_PATH = os.path.join(SCRIPT_DIR, "linear_regression_model.json")


# -----------------------------------------------------
# Step 1: Load and parse the CSV file (no csv library)
# -----------------------------------------------------
def load_csv(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    lines = re.split(r"\r?\n", raw)
    headers = [h.strip() for h in lines[0].split(",")]

    rows = []
    for line in lines[1:]:
        values = line.split(",")
        row = {}
        for i, h in enumerate(headers):
            v = "" if i >= len(values) else values[i].strip()
            row[h] = None if v == "" else v
        rows.append(row)

    return headers, rows


# -----------------------------------------------------
# Step 2: Auto-detect which columns are numeric vs categorical
# -----------------------------------------------------
def detect_column_types(rows, headers):
    types = {}
    for h in headers:
        all_numeric = True
        saw_value = False
        for row in rows:
            v = row[h]
            if v is None:
                continue
            saw_value = True
            try:
                float(v)
            except (TypeError, ValueError):
                all_numeric = False
        types[h] = "numeric" if (saw_value and all_numeric) else "categorical"
    return types


def cast_columns(rows, headers, types):
    for row in rows:
        for h in headers:
            if row[h] is None:
                continue
            if types[h] == "numeric":
                row[h] = float(row[h])


# -----------------------------------------------------
# Step 3: Check for missing values
# -----------------------------------------------------
def count_missing(rows, headers):
    missing = {h: 0 for h in headers}
    for row in rows:
        for h in headers:
            v = row[h]
            if v is None or (isinstance(v, float) and math.isnan(v)):
                missing[h] += 1
    return missing


# -----------------------------------------------------
# Step 3b: Fill in missing values
# -----------------------------------------------------
def impute_missing(rows, headers, types):
    for h in headers:
        if types[h] == "numeric":
            present = [r[h] for r in rows if r[h] is not None and not (isinstance(r[h], float) and math.isnan(r[h]))]
            if len(present) == 0:
                continue
            mean = sum(present) / len(present)
            for r in rows:
                if r[h] is None or (isinstance(r[h], float) and math.isnan(r[h])):
                    r[h] = mean
        else:
            counts = {}
            for r in rows:
                if r[h] is not None:
                    counts[r[h]] = counts.get(r[h], 0) + 1
            mode_list = sorted(counts.keys(), key=lambda k: -counts[k])
            if mode_list:
                mode = mode_list[0]
                for r in rows:
                    if r[h] is None:
                        r[h] = mode


# -----------------------------------------------------
# Step 4: Drop duplicate rows
# -----------------------------------------------------
def drop_duplicates(rows):
    seen = set()
    unique = []
    duplicate_count = 0

    for row in rows:
        key = json.dumps(row)
        if key in seen:
            duplicate_count += 1
        else:
            seen.add(key)
            unique.append(row)

    return unique, duplicate_count


# -----------------------------------------------------
# Step 5: Standard Scaler -> x' = (x - mean) / std
# -----------------------------------------------------
def scaler_fit(matrix):
    n_rows = len(matrix)
    n_cols = len(matrix[0])
    mean = [0.0] * n_cols
    std = [0.0] * n_cols

    for j in range(n_cols):
        total = 0.0
        for i in range(n_rows):
            total += matrix[i][j]
        mean[j] = total / n_rows

    for j in range(n_cols):
        sum_sq = 0.0
        for i in range(n_rows):
            sum_sq += (matrix[i][j] - mean[j]) ** 2
        std[j] = math.sqrt(sum_sq / n_rows)
        if std[j] == 0:
            std[j] = 1

    return {"mean": mean, "std": std}


def scaler_transform(matrix, scaler):
    return [[(val - scaler["mean"][j]) / scaler["std"][j] for j, val in enumerate(row)] for row in matrix]


def scaler_inverse_transform(matrix, scaler):
    return [[val * scaler["std"][j] + scaler["mean"][j] for j, val in enumerate(row)] for row in matrix]


def scaler_fit_1d(arr):
    s = scaler_fit([[v] for v in arr])
    return {"mean": s["mean"][0], "std": s["std"][0]}


def scaler_transform_1d(arr, scaler):
    return [(v - scaler["mean"]) / scaler["std"] for v in arr]


def scaler_inverse_transform_1d(arr, scaler):
    return [v * scaler["std"] + scaler["mean"] for v in arr]


# -----------------------------------------------------
# Step 5b: One-hot encoding for categorical feature columns
# -----------------------------------------------------
def build_one_hot_map(rows, categorical_feature_names):
    one_hot_map = {}
    for h in categorical_feature_names:
        one_hot_map[h] = sorted(set(r[h] for r in rows))
    return one_hot_map


def one_hot_encode_row(row, categorical_feature_names, one_hot_map):
    encoded = []
    for h in categorical_feature_names:
        categories = one_hot_map[h]
        for k in range(1, len(categories)):
            encoded.append(1 if row[h] == categories[k] else 0)
    return encoded


# -----------------------------------------------------
# Step 5c: Polynomial feature expansion for numeric columns
# -----------------------------------------------------
def expand_polynomial_features(numeric_values, degree):
    expanded = list(numeric_values)

    for d in range(2, degree + 1):
        for v in numeric_values:
            expanded.append(v ** d)
    if degree >= 2:
        for i in range(len(numeric_values)):
            for j in range(i + 1, len(numeric_values)):
                expanded.append(numeric_values[i] * numeric_values[j])
    return expanded


def polynomial_feature_names(numeric_feature_names, degree):
    names = list(numeric_feature_names)
    for d in range(2, degree + 1):
        for n in numeric_feature_names:
            names.append(f"{n}^{d}")
    if degree >= 2:
        for i in range(len(numeric_feature_names)):
            for j in range(i + 1, len(numeric_feature_names)):
                names.append(f"{numeric_feature_names[i]}*{numeric_feature_names[j]}")
    return names


# -----------------------------------------------------
# Step 6: Build X and y automatically
# -----------------------------------------------------
def build_features_and_target(rows, headers, target_column, types):
    feature_names = [h for h in headers if h != target_column]
    numeric_feature_names = [h for h in feature_names if types[h] == "numeric"]
    categorical_feature_names = [h for h in feature_names if types[h] == "categorical"]

    one_hot_map = build_one_hot_map(rows, categorical_feature_names)

    expanded_feature_names = polynomial_feature_names(numeric_feature_names, POLY_DEGREE) + [
        f"{h}={cat}" for h in categorical_feature_names for cat in one_hot_map[h][1:]
    ]

    X = []
    for row in rows:
        numeric_part = expand_polynomial_features([row[f] for f in numeric_feature_names], POLY_DEGREE)
        categorical_part = one_hot_encode_row(row, categorical_feature_names, one_hot_map)
        X.append(numeric_part + categorical_part)
    y = [row[target_column] for row in rows]

    return {
        "X": X,
        "y": y,
        "numeric_feature_names": numeric_feature_names,
        "categorical_feature_names": categorical_feature_names,
        "one_hot_map": one_hot_map,
        "expanded_feature_names": expanded_feature_names,
    }


# -----------------------------------------------------
# Step 7: Gradient Descent with Adam updates
# -----------------------------------------------------
# y_pred = X . weights + bias ; Cost (MSE) = (1/n) * sum((y_pred - y)^2)
def gradient_descent(
    X, y,
    learning_rate=0.03, epochs=5000, tolerance=1e-12,
    beta1=0.9, beta2=0.999, epsilon=1e-8, log_every=0,
):
    n = len(X)
    n_features = len(X[0])

    weights = [0.0] * n_features
    bias = 0.0
    m_w = [0.0] * n_features  # Adam: running mean of gradient
    v_w = [0.0] * n_features  # Adam: running mean of gradient^2
    m_b = 0.0
    v_b = 0.0
    cost_history = []
    prev_cost = float("inf")

    for epoch in range(1, epochs + 1):
        y_pred = [sum(val * weights[j] for j, val in enumerate(row)) + bias for row in X]
        errors = [pred - y[i] for i, pred in enumerate(y_pred)]

        grad_weights = [0.0] * n_features
        for j in range(n_features):
            grad = 0.0
            for i in range(n):
                grad += errors[i] * X[i][j]
            grad_weights[j] = (2 / n) * grad
        grad_bias = (2 / n) * sum(errors)

        for j in range(n_features):
            m_w[j] = beta1 * m_w[j] + (1 - beta1) * grad_weights[j]
            v_w[j] = beta2 * v_w[j] + (1 - beta2) * grad_weights[j] ** 2
            m_hat = m_w[j] / (1 - beta1 ** epoch)
            v_hat = v_w[j] / (1 - beta2 ** epoch)
            weights[j] -= (learning_rate * m_hat) / (math.sqrt(v_hat) + epsilon)

        m_b = beta1 * m_b + (1 - beta1) * grad_bias
        v_b = beta2 * v_b + (1 - beta2) * grad_bias ** 2
        m_hat_b = m_b / (1 - beta1 ** epoch)
        v_hat_b = v_b / (1 - beta2 ** epoch)
        bias -= (learning_rate * m_hat_b) / (math.sqrt(v_hat_b) + epsilon)

        cost = sum(e * e for e in errors) / n
        cost_history.append(cost)

        if log_every and (epoch % log_every == 0 or epoch == epochs):
            print(f"  epoch {epoch}: cost (scaled MSE) = {cost:.8f}")

        if abs(prev_cost - cost) < tolerance:
            break  # converged early
        prev_cost = cost

    return {"weights": weights, "bias": bias, "cost_history": cost_history}


def predict_linear(X, weights, bias):
    return [sum(val * weights[j] for j, val in enumerate(row)) + bias for row in X]


# -----------------------------------------------------
# Step 8: Normal Equation (closed-form) - kept as a cross-check
# -----------------------------------------------------
def transpose(matrix):
    return [[row[col_index] for row in matrix] for col_index in range(len(matrix[0]))]


def mat_mul(a, b):
    result = []
    for i in range(len(a)):
        result_row = []
        for j in range(len(b[0])):
            s = 0.0
            for k in range(len(b)):
                s += a[i][k] * b[k][j]
            result_row.append(s)
        result.append(result_row)
    return result


def invert_matrix(matrix):
    n = len(matrix)
    identity = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    augmented = [list(matrix[i]) + identity[i] for i in range(n)]

    for col in range(n):
        pivot_row = col
        for row in range(col + 1, n):
            if abs(augmented[row][col]) > abs(augmented[pivot_row][col]):
                pivot_row = row
        augmented[col], augmented[pivot_row] = augmented[pivot_row], augmented[col]

        pivot = augmented[col][col]
        if abs(pivot) < 1e-12:
            raise ValueError("Matrix is singular and cannot be inverted.")

        for j in range(2 * n):
            augmented[col][j] /= pivot

        for row in range(n):
            if row != col:
                factor = augmented[row][col]
                for j in range(2 * n):
                    augmented[row][j] -= factor * augmented[col][j]

    return [row[n:] for row in augmented]


def add_bias_column(X):
    return [[1.0] + list(row) for row in X]


def fit_linear_regression_normal_equation(X, y):
    Xb = add_bias_column(X)
    Xt = transpose(Xb)
    xtx_inv = invert_matrix(mat_mul(Xt, Xb))
    xty = mat_mul(Xt, [[v] for v in y])
    return [row[0] for row in mat_mul(xtx_inv, xty)]


def predict_normal_equation(X, theta):
    Xb = add_bias_column(X)
    return [sum(val * theta[i] for i, val in enumerate(row)) for row in Xb]


# -----------------------------------------------------
# Step 9: Evaluation metrics
# -----------------------------------------------------
def mean_absolute_error(y_true, y_pred):
    n = len(y_true)
    return sum(abs(val - y_pred[i]) for i, val in enumerate(y_true)) / n


def mean_squared_error(y_true, y_pred):
    n = len(y_true)
    return sum((val - y_pred[i]) ** 2 for i, val in enumerate(y_true)) / n


def r2_score(y_true, y_pred):
    mean_y = sum(y_true) / len(y_true)
    ss_res = sum((val - y_pred[i]) ** 2 for i, val in enumerate(y_true))
    ss_tot = sum((val - mean_y) ** 2 for val in y_true)
    return 1 - ss_res / ss_tot


def mean_absolute_percentage_error(y_true, y_pred):
    diffs = [abs((val - y_pred[i]) / val) for i, val in enumerate(y_true) if val != 0]
    if not diffs:
        return float("nan")
    return (sum(diffs) / len(diffs)) * 100


# -----------------------------------------------------
# Step 10: Leave-One-Out Cross-Validation
# -----------------------------------------------------
def leave_one_out_cv_gd(X, y, gd_options):
    n = len(X)
    preds = [None] * n

    for i in range(n):
        X_train = [row for idx, row in enumerate(X) if idx != i]
        y_train = [val for idx, val in enumerate(y) if idx != i]

        x_scaler_fold = scaler_fit(X_train)
        y_scaler_fold = scaler_fit_1d(y_train)
        X_train_scaled = scaler_transform(X_train, x_scaler_fold)
        y_train_scaled = scaler_transform_1d(y_train, y_scaler_fold)

        result = gradient_descent(X_train_scaled, y_train_scaled, **gd_options)
        weights, bias = result["weights"], result["bias"]

        X_test_scaled = scaler_transform([X[i]], x_scaler_fold)
        pred_scaled = predict_linear(X_test_scaled, weights, bias)[0]
        pred_value = scaler_inverse_transform_1d([pred_scaled], y_scaler_fold)[0]

        preds[i] = pred_value

    return {
        "mae": mean_absolute_error(y, preds),
        "rmse": math.sqrt(mean_squared_error(y, preds)),
        "r2": r2_score(y, preds),
        "preds": preds,
    }


# -----------------------------------------------------
# Step 10b: 80/20 Train/Test Split evaluation
# -----------------------------------------------------
def train_test_split(X, y, test_ratio=0.2):
    n = len(X)
    indices = list(range(n))
    for i in range(len(indices) - 1, 0, -1):
        j = random.randint(0, i)
        indices[i], indices[j] = indices[j], indices[i]
    test_count = max(1, round(n * test_ratio))
    test_idx = set(indices[:test_count])

    X_train, y_train, X_test, y_test = [], [], [], []
    for i, row in enumerate(X):
        if i in test_idx:
            X_test.append(row)
            y_test.append(y[i])
        else:
            X_train.append(row)
            y_train.append(y[i])
    return X_train, y_train, X_test, y_test


def evaluate_train_test_gd(X, y, gd_options):
    X_train, y_train, X_test, y_test = train_test_split(X, y)

    x_scaler = scaler_fit(X_train)
    y_scaler = scaler_fit_1d(y_train)
    X_train_scaled = scaler_transform(X_train, x_scaler)
    y_train_scaled = scaler_transform_1d(y_train, y_scaler)

    result = gradient_descent(X_train_scaled, y_train_scaled, **gd_options)
    weights, bias = result["weights"], result["bias"]

    X_test_scaled = scaler_transform(X_test, x_scaler)
    pred_scaled = predict_linear(X_test_scaled, weights, bias)
    preds = scaler_inverse_transform_1d(pred_scaled, y_scaler)

    return {
        "mae": mean_absolute_error(y_test, preds),
        "rmse": math.sqrt(mean_squared_error(y_test, preds)),
        "r2": r2_score(y_test, preds),
        "preds": preds,
        "y_test": y_test,
    }


# -----------------------------------------------------
# Step 11: Interactive helper (training-time only - target column)
# -----------------------------------------------------
def resolve_target_column(headers, types):
    if TARGET_COLUMN and TARGET_COLUMN in headers:
        return TARGET_COLUMN

    numeric_headers = [h for h in headers if types[h] == "numeric"]
    target = ""
    while target not in headers or types.get(target) != "numeric":
        target = input(
            f"\nColumns found: {', '.join(headers)}\n"
            f"Numeric columns (valid targets): {', '.join(numeric_headers)}\n"
            f"Enter target column name: "
        ).strip()
    return target


# -----------------------------------------------------
# Step 12: Save the trained model to a JSON file
# -----------------------------------------------------
# Every parameter required to reproduce predictions later (weights,
# bias, scalers, feature metadata, etc.) is collected into a plain
# dict and written with json.dump - the Python equivalent of the JS
# version's fs.writeFileSync(path, JSON.stringify(model, null, 4)).
def save_model(model, file_path=MODEL_SAVE_PATH):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=4)
    print(f"\nModel saved to: {file_path}")


# -----------------------------------------------------
# Main training pipeline
# -----------------------------------------------------
def main():
    headers, rows = load_csv(CSV_PATH)
    print("Shape:", len(rows), "rows,", len(headers), "columns")
    print("Columns:", headers)

    types = detect_column_types(rows, headers)
    cast_columns(rows, headers, types)
    print("\nDetected column types:", types)

    print("\nMissing values per column (before imputation):", count_missing(rows, headers))
    impute_missing(rows, headers, types)

    unique, duplicate_count = drop_duplicates(rows)
    print("\nDuplicate rows found:", duplicate_count)

    target_column = resolve_target_column(headers, types)
    print(f"\nUsing \"{target_column}\" as the target column.")

    built = build_features_and_target(unique, headers, target_column, types)
    X, y = built["X"], built["y"]
    numeric_feature_names = built["numeric_feature_names"]
    categorical_feature_names = built["categorical_feature_names"]
    one_hot_map = built["one_hot_map"]
    expanded_feature_names = built["expanded_feature_names"]
    print("Feature columns used (after encoding):", expanded_feature_names)

    # ---- Standardize features and target ----
    x_scaler = scaler_fit(X)
    y_scaler = scaler_fit_1d(y)
    X_scaled = scaler_transform(X, x_scaler)
    y_scaled = scaler_transform_1d(y, y_scaler)

    # ---- Train with Gradient Descent (on scaled data) ----
    print("\n--- Training with Gradient Descent (Adam) ---")
    gd_options = {"learning_rate": 0.03, "epochs": 5000, "tolerance": 1e-12, "log_every": 1000}
    result = gradient_descent(X_scaled, y_scaled, **gd_options)
    weights, bias, cost_history = result["weights"], result["bias"], result["cost_history"]
    print("\nLearned weights (scaled space):", [f"{w:.4f}" for w in weights])
    print("Learned bias (scaled space):", f"{bias:.4f}")

    pred_scaled_all = predict_linear(X_scaled, weights, bias)
    pred_all = scaler_inverse_transform_1d(pred_scaled_all, y_scaler)
    print("\nFit on full training data (the model HAS seen these rows):")
    print("  mean_absolute_error:", mean_absolute_error(y, pred_all))
    print("  root_mean_squared_error:", math.sqrt(mean_squared_error(y, pred_all)))
    print("  r2_score:", r2_score(y, pred_all))
    print("  mean_absolute_percentage_error:", f"{mean_absolute_percentage_error(y, pred_all):.3f}%")

    # ---- Honest evaluation: LOOCV for small data, Train/Test split for larger ----
    print("\n--- Evaluation on data the model has NOT seen ---")
    if len(unique) <= 30:
        print("(Small dataset -> using Leave-One-Out Cross-Validation)")
        cv = leave_one_out_cv_gd(X, y, gd_options)
        print("mean_absolute_error:", cv["mae"])
        print("root_mean_squared_error:", cv["rmse"])
        print("r2_score:", cv["r2"])
        print("mean_absolute_percentage_error:", f"{mean_absolute_percentage_error(y, cv['preds']):.3f}%")
    else:
        print("(Larger dataset -> using an 80/20 Train/Test Split)")
        tt = evaluate_train_test_gd(X, y, gd_options)
        print("mean_absolute_error:", tt["mae"])
        print("root_mean_squared_error:", tt["rmse"])
        print("r2_score:", tt["r2"])
        print("mean_absolute_percentage_error:", f"{mean_absolute_percentage_error(tt['y_test'], tt['preds']):.3f}%")
    print(
        "\nCompare the two blocks above: the SECOND one is the honest measure of how good this model is. "
        "If it's much worse than the first, the model is overfitting - raising POLY_DEGREE further will "
        "widen that gap, not close it."
    )

    # ---- Cross-check vs. closed-form Normal Equation ----
    print("\n--- Cross-check vs. Normal Equation (closed-form, unscaled) ---")
    try:
        theta_ne = fit_linear_regression_normal_equation(X, y)
        pred_ne = predict_normal_equation(X, theta_ne)
        print("Normal equation coefficients [intercept, " + ", ".join(expanded_feature_names) + "]:")
        print(theta_ne)
        print("\nRow # | GD value | Normal-Eq value | Actual value")
        for i, row in enumerate(unique[:5]):
            print(f"{i + 1}\t{pred_all[i]:.2f}\t{pred_ne[i]:.2f}\t{row[target_column]}")
    except ValueError as err:
        print("Skipped (matrix not invertible for this dataset):", err)

    # -----------------------------------------------------
    # Build the dict that Load_Model.py needs, then save it as JSON.
    # Nothing above this point was changed; this is purely "package up
    # what already exists and write it to disk".
    # -----------------------------------------------------
    model = {
        "weights": weights,
        "bias": bias,
        "x_scaler": x_scaler,                            # {"mean": [...], "std": [...]} for X columns (post-encoding)
        "y_scaler": y_scaler,                             # {"mean": float, "std": float} for target column
        "numeric_feature_names": numeric_feature_names,
        "categorical_feature_names": categorical_feature_names,
        "one_hot_map": one_hot_map,                       # category lists per categorical column
        "expanded_feature_names": expanded_feature_names, # column order after poly expansion + one-hot
        "target_column": target_column,
        "poly_degree": POLY_DEGREE,
        "gd_options": gd_options,                         # learning_rate, epochs, tolerance, etc. (for reference)
        "cost_history": cost_history,
    }
    save_model(model, MODEL_SAVE_PATH)


if __name__ == "__main__":
    main()
