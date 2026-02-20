"""
eval.py
Evaluates trained EfficientNet-B0 models on the test set.
Records test accuracy, loss, inference time, and per-class metrics.
"""

import os
import json
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tf_keras as keras
import tensorflow_hub as hub

from src.dataset import generate_dataset, which_dataset
from src.utils import get_class_names
#--------------------------------------------------------------#
# Constants                                                    #
#--------------------------------------------------------------#
CLASSES      = [7, 35]
RESOLUTIONS  = [32, 128]
AUG_LEVELS   = [0.0, 0.1, 0.2, 0.3, 0.4]
MODELS_DIR   = "../models"
METRICS_PATH = "../artifacts/metrics.json"
PLOTS_DIR    = "../artifacts/plots"


#--------------------------------------------------------------#
# Helpers                                                      #
#--------------------------------------------------------------#

def get_model_name(C, R, L):
    """ 
    Looks up the model name for a given C, R, L from metrics.json.

    Args:
        C: Number of classes (7 or 35)
        R: Resolution (32 or 128)
        L: RWDA Level

    Raises:
        ValueError: If no matching C, R, and L

    Returns:
        Name of the model
    """
    with open(METRICS_PATH, "r") as f:
        all_results = json.load(f)

    for entry in all_results:
        if entry["C"] == C and entry["R"] == R and entry["L"] == L:
            return entry["model_name"]

    raise ValueError(f"No model found in metrics.json for C={C}, R={R}, L={L}")


def load_model(C, R, L):
    """
    Loads model for given C, R, and L
    
    :param C: Number of classes (7 or 35)
    :param R: Resolution (32 or 128)
    :param L: RWDA level/Augmentation level
    """
    model_name = get_model_name(C, R, L)
    model_dir = os.path.join(MODELS_DIR, model_name)
    model = keras.models.load_model(
        os.path.join(model_dir, 'model.keras'),
        custom_objects={'KerasLayer': hub.KerasLayer}
    )
    print(f"Loading model from {model_dir}")
    return model


def evaluate(model, test_data):
    """
    Evaluates the model on the test set and records inference time.

    Args:
        model: The model to be evaluated
        test_data: The test data to evaluate the model on

    Returns:
        A dictionary with test accuracy, test loss, and evaluation time in seconds
    """

    eval_start = datetime.datetime.now()
    test_loss, test_accuracy = model.evaluate(test_data, steps = len(test_data))
    eval_end = datetime.datetime.now()
    eval_time = (eval_end-eval_start).total_seconds()
    print(f"Test accuracy = {test_accuracy:.4f}")
    print(f"Test loss = {test_loss:.4f}")
    print(f"Evaluation time = {eval_time:.1f}")

    return {
        "test_accuracy" : round(test_accuracy,4),
        "test_loss"     : round(test_loss, 4),
        "evaluation time": round(eval_time, 1)
    }

def get_predictions(model, test_data):
    """
    Predicts the held out test set using the input model.

    Returns the true classes and predicted classes as numpy arrays
    
    :param model: The model using which prediction is done
    :param test_data: The test data to predict
    """

    y_true = []
    y_pred = []

    for i in range(len(test_data)):
        X, y = test_data[i]
        y_t = np.argmax(y)
        y_true.extend(y_t)

        preds = model.predict(X)
        y_p = np.argmax(preds)
        y_pred.extend(y_p)
    
    return np.array(y_true), np.array(y_pred)

def plot_confusion_matrix(y_true, y_pred, class_names, C, R, L):
    """
    Plots and saves the confusion matrix heatmap.

    :param y_true: True labels
    :param y_pred: Predicted labels
    :param class_names: Class names
    :param C: Number of classes (7 or 35)
    :param R: Resolution (32 or 128)
    :param L: RWDA level/Augmentation level
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)
    confusion_matrix = confusion_matrix(y_true, y_pred)
    fig_size = 10 if C==7 else 20

    plt.figure(figsize=(fig_size, fig_size))
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt="d",
        cmap="Blues"
    )
    plt.xticks(np.arange(0, C, 1), labels=class_names, rotation=90)
    plt.yticks(np.arange(0, C, 1), labels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix | {C} Classes | {R}x{R} Res | {L} RWDA Level")
    plt.tight_layout()

    plot_path = os.path.join(PLOTS_DIR, f"CM-{C}_R-{R}_L-{L}_I-T.png")
    plt.savefig(plot_path)
    plt.show()
    print(f"Confusion matrix saved to {plot_path}")

def get_classification_report(y_true, y_pred, class_names):
    """
    Generates the classification report and returns it as a dictionary

    :param y_true: True labels
    :param y_pred: Predicted labels
    :param class_names: Class names
    """

    report = classification_report(y_true, y_pred, class_names, output_dict=True)
    print(classification_report(y_true, y_pred, class_names))
    return report

def update_metrics(C, R, L, new_results):
    """
    Updates the metrics.json file to include test results

    :param C: Number of classes (7 or 35)
    :param R: Resolution (32 or 128)
    :param L: RWDA level/Augmentation level
    :param new_results: Dictionary of results to add
    """
    with open(METRICS_PATH, 'r') as f:
        all_results = json.load(f)
    
    for entry in all_results:
        if entry["C"]==C and entry["R"]==R and entry["L"]==L:
            entry.update(new_results)
            break
    
    with open(METRICS_PATH, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    print(f"Metrics updated for C={C}, R={R}, and L={L}")

#--------------------------------------------------------------#
# Main evaluation loop                                         #
#--------------------------------------------------------------#

def main_eval(C, R, L, class_names, test_data):
    """
    Runs the full evaluation pipeline for one dataset variant.
    Automatically looks up the correct model name from metrics.json.
    
    :param C: Number of classes (7 or 35)
    :param R: Resolution (32 or 128)
    :param L: RWDA level/Augmentation level
    :param class_names: Class names
    :param test_data: Test data
    """

    print(f"\n{'='*55}")
    print(f"Evaluating | C={C} | R={R} | L={L}")
    print(f"{'='*55}")
    model_name = get_model_name(C, R, L)
    model = load_model(C, R, L)
    y_true, y_pred = get_predictions(model, test_data)
    test_results = evaluate(model, test_data)

    plot_confusion_matrix(y_true, y_pred, class_names)

    report = get_classification_report(y_true, y_pred, class_names)

    update_metrics(C, R, L,
                   {**test_results,
                    "per_class_metric":report})

def main():
    for C in CLASSES:
        for R in RESOLUTIONS:
            for L in AUG_LEVELS:
                target_dir = which_dataset(C, R, L)
                target_dir = os.path.join(target_dir, "train")
                class_names = get_class_names(target_dir=target_dir)
                _, _, test_data = generate_dataset(C, R, L)
                main_eval(C, R, L, class_names=class_names, test_data=test_data)
    
    print(f"\n{'='*55}")
    print(f"All 20 datasets evaluated.")
    print(f"Results saved to {METRICS_PATH}")
    print(f"{'='*55}")