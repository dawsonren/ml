# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model import SimpleCNN
from config import CONFIG

def train_model(dataset, train_indices, hyperparams, verbose=True):
    """
    Train the model on the given training indices.
    Returns the trained model.
    """
    # Create a DataLoader for the given subset of data
    train_subset = Subset(dataset, train_indices)
    train_loader = DataLoader(train_subset, batch_size=hyperparams["batch_size"],
                              shuffle=True, num_workers=CONFIG["num_workers"])

    # Instantiate the model and move it to the specified device (CPU/GPU)
    model = SimpleCNN().to(hyperparams["device"])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])

    # Training loop
    for epoch in range(hyperparams["num_epochs"]):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader, 0):
            inputs, labels = inputs.to(hyperparams["device"]), labels.to(hyperparams["device"])

            optimizer.zero_grad()          # Zero the gradients
            outputs = model(inputs)          # Forward pass
            loss = criterion(outputs, labels)
            loss.backward()                # Backpropagation
            optimizer.step()               # Update weights

            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        if verbose:
            print(f'Epoch {epoch+1}/{hyperparams["num_epochs"]} - Loss: {avg_loss:.3f}')
    return model

def evaluate_model(model, dataset, val_indices, hyperparams):
    """
    Evaluate the model on the validation set defined by val_indices.
    Returns the accuracy on the validation set.
    """
    val_subset = Subset(dataset, val_indices)
    val_loader = DataLoader(val_subset, batch_size=hyperparams["batch_size"],
                            shuffle=False, num_workers=hyperparams["num_workers"])

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(hyperparams["device"]), labels.to(hyperparams["device"])
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy
