"""
Load_Model.py  (Part 2 of 2 - see Train_Model.py for Part 1)
=============================================================
Direct Python translation of Load_Model.js. This file loads
"linear_regression_model.json" (created by Train_Model.py), restores
every saved parameter, asks the user for a new input row, applies the
EXACT SAME preprocessing used during training (polynomial expansion,
one-hot encoding, StandardScaler transform), and predicts using the
SAME `predict_linear` equation from the original script:

        y_pred = X . weights + bias

No mathematical equation or prediction logic was changed. The helper
functions below (`expand_polynomial_features`, `one_hot_encode_row`,
`scaler_transform`, `scaler_inverse_transform_1d`, `predict_linear`) are
translated line-for-line from the original JS, because prediction needs
them too. `predict_custom_house(...)` is also unchanged in its internal
logic - it was simply moved here from the training file (matching
Load_Model.js) and reads its parameters from the loaded JSON dict
instead of live training variables.

Run:
    python Load_Model.py
"""

import os
import json
import math

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()

MODEL_SAVE_PATH = os.path.join(SCRIPT_DIR, "linear_regression_model.json")


# -----------------------------------------------------
# Load the saved JSON file back into memory
# -----------------------------------------------------
def load_model(file_path=MODEL_SAVE_PATH):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            f'Model file not found at "{file_path}". Run Train_Model.py first to create it.'
        )
    with open(file_path, "r", encoding="utf-8") as f:
        model = json.load(f)
    print(f"\nModel loaded from: {file_path}")
    return model


# -----------------------------------------------------
# Preprocessing helpers - translated UNCHANGED from the original script
# (prediction needs the exact same feature expansion/scaling used
#  during training)
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


def one_hot_encode_row(row, categorical_feature_names, one_hot_map):
    encoded = []
    for h in categorical_feature_names:
        categories = one_hot_map[h]
        for k in range(1, len(categories)):
            encoded.append(1 if row[h] == categories[k] else 0)
    return encoded


def scaler_transform(matrix, scaler):
    return [[(val - scaler["mean"][j]) / scaler["std"][j] for j, val in enumerate(row)] for row in matrix]


def scaler_inverse_transform_1d(arr, scaler):
    return [v * scaler["std"] + scaler["mean"] for v in arr]


# -----------------------------------------------------
# Prediction equation - translated UNCHANGED from the original script
# y_pred = X . weights + bias
# -----------------------------------------------------
def predict_linear(X, weights, bias):
    return [sum(val * weights[j] for j, val in enumerate(row)) + bias for row in X]


# -----------------------------------------------------
# Interactive prediction on a user-typed row.
# Logic is UNCHANGED from the original `predictCustomHouse` - it now
# simply reads weights/bias/scalers/feature metadata from the loaded
# `model` dict instead of from freshly-trained local variables.
# -----------------------------------------------------
def predict_custom_house(model):
    weights = model["weights"]
    bias = model["bias"]
    x_scaler = model["x_scaler"]
    y_scaler = model["y_scaler"]
    numeric_feature_names = model["numeric_feature_names"]
    categorical_feature_names = model["categorical_feature_names"]
    one_hot_map = model["one_hot_map"]
    poly_degree = model["poly_degree"]
    target_column = model["target_column"]

    print("\n========= Predict Your Own Value =========")
    prompts = [f"{f} (number)" for f in numeric_feature_names] + [
        f"{f} (one of: {'/'.join(one_hot_map[f])})" for f in categorical_feature_names
    ]
    answer = input(f"Enter {', '.join(prompts)} (comma-separated): ")

    raw_values = [v.strip() for v in answer.split(",")]
    expected_count = len(numeric_feature_names) + len(categorical_feature_names)
    if len(raw_values) != expected_count:
        print(f"\nExpected {expected_count} values, got {len(raw_values)}. Got: \"{answer}\"")
        return

    numeric_part = []
    for v in raw_values[: len(numeric_feature_names)]:
        try:
            numeric_part.append(float(v))
        except ValueError:
            print("\nAll numeric fields must be valid numbers.")
            return

    categorical_row = {}
    for i, f in enumerate(categorical_feature_names):
        categorical_row[f] = raw_values[len(numeric_feature_names) + i]
    categorical_part = one_hot_encode_row(categorical_row, categorical_feature_names, one_hot_map)

    # same polynomial expansion used at training time, so this vector
    # lines up with the weights the model actually learned
    input_vector = expand_polynomial_features(numeric_part, poly_degree) + categorical_part
    scaled_input = scaler_transform([input_vector], x_scaler)
    pred_scaled = predict_linear(scaled_input, weights, bias)[0]
    pred_value = scaler_inverse_transform_1d([pred_scaled], y_scaler)[0]

    summary = ", ".join(f"{p.split(' ')[0]}={raw_values[i]}" for i, p in enumerate(prompts))
    print(f"\nInput -> {summary}")
    print(f"Predicted value ({target_column}) = {pred_value:,.2f}")


# -----------------------------------------------------
# Main loading + prediction pipeline
# -----------------------------------------------------
def main():
    model = load_model(MODEL_SAVE_PATH)

    print("\nRestored model summary:")
    print("  Target column:", model["target_column"])
    print("  Numeric features:", model["numeric_feature_names"])
    print("  Categorical features:", model["categorical_feature_names"])
    print("  Polynomial degree:", model["poly_degree"])
    print("  Learned bias (scaled space):", f"{float(model['bias']):.4f}")

    predict_custom_house(model)


if __name__ == "__main__":
    main()
