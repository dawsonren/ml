# config.py
import torch

CONFIG = {
    "learning_rate": 0.001,
    "batch_size": 100,
    "num_epochs": 10,
    "num_workers": 2,
    "num_folds": 5,  # For k-fold cross-validation
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

HYPERPARAM_LIST = [
    {
        "learning_rate": 0.001,
        "batch_size": 100,
    },
    {
        "learning_rate": 0.001,
        "batch_size": 50,
    },
    {
        "learning_rate": 0.001,
        "batch_size": 200,
    },
    {
        "learning_rate": 0.01,
        "batch_size": 100,
    },
    {
        "learning_rate": 0.01,
        "batch_size": 200,
    },
    {
        "learning_rate": 0.01,
        "batch_size": 500,
    },
]