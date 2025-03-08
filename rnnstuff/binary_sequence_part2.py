import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define the RNN model
class BinarySequenceRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(BinarySequenceRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # RNN layer
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Sigmoid activation for binary output
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, hidden=None):
        # Initialize hidden state if none is provided
        if hidden is None:
            hidden = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward pass through RNN
        out, hidden = self.rnn(x, hidden)
        
        # Pass through the output layer
        out = self.fc(out)
        
        # Apply sigmoid to get binary probability
        out = self.sigmoid(out)
        
        return out, hidden
    
    def init_hidden(self, batch_size, device):
        # Initialize hidden state
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

# Function to generate binary sequence patterns
def generate_pattern(pattern_type='repeating', seq_length=100):
    if pattern_type == 'repeating':
        # Simple repeating pattern (e.g., [0,0,0,1,1,1])
        base_pattern = [0, 0, 0, 1, 1, 1]
        repetitions = seq_length // len(base_pattern) + 1
        return np.array(base_pattern * repetitions)[:seq_length]
    
    elif pattern_type == 'fibonacci':
        # Fibonacci-inspired binary pattern
        sequence = [0, 1]
        while len(sequence) < seq_length:
            next_bit = (sequence[-1] + sequence[-2]) % 2
            sequence.append(next_bit)
        return np.array(sequence)
    
    elif pattern_type == 'alternating_blocks':
        # Alternating blocks of 0s and 1s with increasing lengths
        sequence = []
        block_size = 1
        bit = 0
        while len(sequence) < seq_length:
            sequence.extend([bit] * block_size)
            bit = 1 - bit
            if bit == 0:  # After every 0-1 cycle, increase block size
                block_size += 1
        return np.array(sequence[:seq_length])
    
    else:  # Default to random with some structure
        sequence = []
        p = 0.2  # Probability of flipping
        current = 0
        for _ in range(seq_length):
            if np.random.random() < p:
                current = 1 - current
            sequence.append(current)
        return np.array(sequence)

# Function to create training data
def create_training_data(sequence, seq_length, pred_length):
    X, y = [], []
    for i in range(len(sequence) - seq_length - pred_length):
        # Input is a chunk of the sequence
        X.append(sequence[i:i+seq_length])
        # Target is the next pred_length elements
        y.append(sequence[i+seq_length:i+seq_length+pred_length])
    
    return np.array(X), np.array(y)

# Function to train the model
def train_model(model, X_train, y_train, epochs=100, lr=0.01, clip_value=1.0):
    # Convert data to torch tensors and reshape for RNN input
    X_train = torch.FloatTensor(X_train).unsqueeze(2)  # Shape: [batch_size, seq_length, input_size]
    y_train = torch.FloatTensor(y_train).unsqueeze(2)  # Shape: [batch_size, pred_length, output_size]
    
    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    losses = []
    
    for epoch in range(epochs):
        # Initialize hidden state
        hidden = model.init_hidden(X_train.size(0), X_train.device)
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs, _ = model(X_train, hidden)
        
        # Calculate loss
        loss = criterion(outputs, y_train)
        
        # Backward pass
        loss.backward()
        
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        
        # Update weights
        optimizer.step()
        
        # Store loss
        losses.append(loss.item())
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    return losses

# Function to predict future sequence
def predict_sequence(model, initial_sequence, pred_length, device='cpu'):
    model.eval()
    
    # Convert initial sequence to tensor
    sequence = torch.FloatTensor(initial_sequence).unsqueeze(0).unsqueeze(2).to(device)
    
    # Initialize predictions list with initial sequence
    predictions = initial_sequence.copy()
    
    # Initialize hidden state
    hidden = model.init_hidden(1, device)
    
    with torch.no_grad():
        # Get prediction for entire sequence at once
        outputs, hidden = model(sequence, hidden)
        
        # Convert outputs to binary (0 or 1)
        binary_preds = (outputs.squeeze() > 0.5).int().cpu().numpy()
        
        # Add predictions to the list
        for pred in binary_preds[0]:
            predictions = np.append(predictions, pred)
    
    return predictions

# Main function to run the entire demo
def run_demo():
    # Parameters
    input_size = 1
    hidden_size = 64
    output_size = 1
    num_layers = 2
    seq_length = 20
    pred_length = 10
    epochs = 100
    lr = 0.001
    
    # Generate sequence data
    pattern_type = 'alternating_blocks'  # Try different patterns: 'repeating', 'fibonacci', 'alternating_blocks'
    full_sequence = generate_pattern(pattern_type, 1000)
    
    print(f"Generated {pattern_type} pattern. First 50 elements:")
    print(full_sequence[:50])
    
    # Create training data
    X_train, y_train = create_training_data(full_sequence, seq_length, pred_length)
    print(f"Training data shape: X: {X_train.shape}, y: {y_train.shape}")
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BinarySequenceRNN(input_size, hidden_size, output_size, num_layers).to(device)
    
    # Train model
    print("Training model...")
    losses = train_model(model, X_train, y_train, epochs, lr)
    
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()
    
    # Test prediction
    test_index = 500  # Index to start prediction from
    test_length = 100  # Length of sequence to predict
    
    initial_sequence = full_sequence[test_index:test_index+seq_length]
    predicted_sequence = predict_sequence(model, initial_sequence, test_length)
    
    # Plot actual vs predicted sequence
    actual_sequence = full_sequence[test_index:test_index+seq_length+test_length]
    
    plt.figure(figsize=(15, 5))
    plt.plot(range(len(actual_sequence)), actual_sequence, 'b-', label='Actual')
    plt.plot(range(len(predicted_sequence)), predicted_sequence, 'r--', label='Predicted')
    plt.axvline(x=seq_length, color='g', linestyle='--', label='Prediction Start')
    plt.title('Actual vs Predicted Binary Sequence')
    plt.xlabel('Sequence Index')
    plt.ylabel('Value (0 or 1)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Calculate accuracy for the predicted part
    predicted_part = predicted_sequence[seq_length:]
    actual_part = actual_sequence[seq_length:]
    accuracy = np.mean(predicted_part == actual_part)
    print(f"Prediction accuracy: {accuracy:.4f}")
    
    # Demonstrate long-term prediction
    long_pred_length = 200
    initial_sequence = full_sequence[:seq_length]
    long_prediction = predict_sequence(model, initial_sequence, long_pred_length)
    
    plt.figure(figsize=(15, 5))
    plt.plot(range(len(initial_sequence) + long_pred_length), np.append(initial_sequence, full_sequence[seq_length:seq_length+long_pred_length]), 'b-', label='Actual')
    plt.plot(range(len(long_prediction)), long_prediction, 'r--', label='Predicted')
    plt.axvline(x=seq_length, color='g', linestyle='--', label='Prediction Start')
    plt.title('Long-term Prediction')
    plt.xlabel('Sequence Index')
    plt.ylabel('Value (0 or 1)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    run_demo()