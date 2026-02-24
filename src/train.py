"""
train.py
Trains EfficientNet-B0 across all 20 dataset variants and logs results.
"""

import os
import json
import datetime
import tf_keras as keras
from src.dataset import generate_dataset
from src.model import create_model

#--------------------------------------------------------------#
# Constants                                                    #
#--------------------------------------------------------------#

CLASSES      = [7, 35]
RESOLUTIONS  = [128]
AUG_LEVELS   = [0.0, 0.1, 0.2, 0.3, 0.4]
BATCH_SIZE   = 64
MAX_EPOCHS   = 20
PATIENCE     = 3
MODELS_DIR   = "../models"
METRICS_PATH = "../artifacts/metrics.json"

#--------------------------------------------------------------#
# Helpers                                                      #
#--------------------------------------------------------------#

def compile_model(model):
    """
    compile_model
    Compiles the model with Adam optimizer and categorical crossentropy loss.

    Args:
        model: Uncompiled Keras model from create_model()

    Returns:
        Compiled Keras model
    """
    model.compile(
        loss = 'categorical_crossentropy',
        optimizer = keras.optimizers.Adam(),
        metrics = ["accuracy"]
    )
    return model

def model_train(C, R, L):
    """
    Trains the model on a dataset with the specified congfiguration C, R, and L.
    Validates using corresponding validation set, and  implements the early stopping
    callback with the following details:
       
        monitor = "val_loss",
        patience = PATIENCE,
        verbose = 1,
        restore_best_weights = True
    
    Logs the time it takes to train in seconds.

    Args:
        C (int): Number of classes (7 or 35)
        R (int): Resolution (32 or 128)
        L (float): Augmentation level

    Returns:
        Dictionary: A dictionary for metrics containing the following:
            Number of classes
            Resolution of the input images
            Augmentation Level
            Name of the model
            Best validation accuracy
            Best validatin loss
            Number of epochs run
            Total training time in seconds
    """
    train_data, valid_data = generate_dataset(C, R, L)
    model, model_name = create_model(C, R, L, trainable=True)
    model = compile_model(model)
    model_dir = os.path.join(MODELS_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"Training | C={C} | R={R} | L={L}")
    print(f"{'='*55}")

    model.summary()
    train_start = datetime.datetime.now()

    early_stopping = keras.callbacks.EarlyStopping(
    monitor = "val_loss",
    patience = PATIENCE,
    verbose = 1,
    restore_best_weights = True
    )

    history = model.fit(
        train_data,
        epochs = MAX_EPOCHS,
        steps_per_epoch = len(train_data),
        validation_data = valid_data,
        validation_steps = len(valid_data),
        callbacks = [early_stopping],
        verbose = 2
    )

    train_end = datetime.datetime.now()
    train_time = (train_end - train_start).total_seconds()

    model.save(os.path.join(model_dir, 'model.keras'))

    val_accuracy = max(history.history["val_accuracy"])
    val_loss = min(history.history["val_loss"])
    epochs_run = len(history.history["val_loss"])

    print(f"\nTraining complete in {train_time:.1f}s")
    print(f"Best val accuracy: {val_accuracy:.4f}")

    return {
        "C"                 : C,
        "R"                 : R,
        "L"                 : L,
        "model_name"        : model_name,
        "best_val_accuracy" : round(val_accuracy, 4),
        "best_val_loss"     : round(val_loss, 4),
        "epochs_run"        : epochs_run,
        "train_time_s"      : round(train_time, 1),
    }

def main():
    """
    The main loop to train on all the dataset configurations.
    Makes the required directories for storing saved models and metrics.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)

    all_results = []

    for C in CLASSES:
        for R in RESOLUTIONS:
            for L in AUG_LEVELS:
                result = model_train(C, R, L)
                all_results.append(result)

                # Save after every run so progress isn't lost if something crashes
                with open(METRICS_PATH, "w") as f:
                    json.dump(all_results, f, indent=4)
                print(f"Results saved to {METRICS_PATH}")

    print(f"\n{'='*55}")
    print(f"All 20 datasets trained.")
    print(f"Results saved to {METRICS_PATH}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()