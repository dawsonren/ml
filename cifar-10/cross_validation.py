# cross_validation.py
from sklearn.model_selection import KFold
from torchvision import datasets, transforms
from train import train_model, evaluate_model

def cross_validate(hyperparams, verbose=True):
    # Define the transform and load the CIFAR-10 training dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    num_samples = len(dataset)
    indices = list(range(num_samples))

    # Initialize KFold with the desired number of splits
    kfold = KFold(n_splits=hyperparams["num_folds"], shuffle=True, random_state=42)
    
    fold_accuracies = []
    for fold, (train_indices, val_indices) in enumerate(kfold.split(indices)):
        if verbose:
            print(f"\nStarting Fold {fold+1}")
        # Train model on the training split of this fold
        model = train_model(dataset, train_indices, hyperparams, verbose)
        # Evaluate model on the validation split of this fold
        accuracy = evaluate_model(model, dataset, val_indices, hyperparams)
        fold_accuracies.append(accuracy)
        if verbose:
            print(f"Validation Accuracy for Fold {fold+1}: {accuracy:.2f}%")
    
    avg_accuracy = sum(fold_accuracies) / len(fold_accuracies)
    if verbose:
        print(f"\nAverage Cross-Validation Accuracy: {avg_accuracy:.2f}%")

    return avg_accuracy
