# config.py
import torch

CONFIG = {
    "batch_size": 100,
    "num_workers": 2,
    "num_epochs": 10,
    "learning_rate": 0.001,
    "num_folds": 5,  # For k-fold cross-validation
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}
