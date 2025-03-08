import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define the RNN model
class BinaryRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BinaryRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, hidden=None):
        # Initialize hidden state with zeros if not provided
        if hidden is None:
            hidden = torch.zeros(1, x.size(0), self.hidden_size, device=x.device)
        
        # Forward pass through RNN
        out, hidden = self.rnn(x, hidden)
        
        # Pass through linear layer and apply sigmoid for binary prediction
        out = self.fc(out)
        out = self.sigmoid(out)
        
        return out, hidden

# Generate synthetic binary sequences
def generate_binary_sequences(num_sequences, seq_length):
    # Generate random binary sequences
    sequences = np.array(
        [[0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1],
         [1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1]]
    )
    
    # Prepare input and target sequences
    X = sequences[:, :-1]  # Input: all but last element
    y = sequences[:, 1:]   # Target: all but first element
    
    return X, y

# Convert numpy arrays to PyTorch tensors
def prepare_data(X, y):
    X_tensor = torch.FloatTensor(X).unsqueeze(-1)  # Add feature dimension
    y_tensor = torch.FloatTensor(y).unsqueeze(-1)  # Add feature dimension
    return X_tensor, y_tensor

# Training function
def train_model(model, X, y, epochs=100, learning_rate=0.01):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    losses = []
    
    for epoch in range(epochs):
        # Forward pass
        outputs, _ = model(X)
        loss = criterion(outputs, y)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        # Print training progress
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    return losses

# Evaluate model
def evaluate_model(model, X, y):
    model.eval()
    with torch.no_grad():
        outputs, _ = model(X)
        predictions = (outputs > 0.5).float()
        accuracy = (predictions == y).float().mean()
        print(f'Accuracy: {accuracy:.4f}')
    return accuracy

# Generate some example binary sequences
num_sequences = 1000
seq_length = 10
X, y = generate_binary_sequences(num_sequences, seq_length)
X_tensor, y_tensor = prepare_data(X, y)

# Define model parameters
input_size = 1  # Binary input (0 or 1)
hidden_size = 16
output_size = 1  # Binary output (0 or 1)

# Create and train the model
model = BinaryRNN(input_size, hidden_size, output_size)
losses = train_model(model, X_tensor, y_tensor, epochs=100)

# Evaluate the model
accuracy = evaluate_model(model, X_tensor, y_tensor)

# Plot loss curve
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Test with a few examples
def predict_sequence(model, input_seq, steps=5):
    model.eval()
    print(torch.FloatTensor(input_seq))
    input_tensor = torch.FloatTensor(input_seq).view(1, -1, 1)
    print(input_tensor)
    
    # Initial prediction
    with torch.no_grad():
        output_seq = input_seq.copy()
        curr_input = input_tensor
        
        # Generate 'steps' new elements
        for _ in range(steps):
            prediction, _ = model(curr_input)
            last_pred = 1 if prediction[0, -1, 0].item() > 0.5 else 0
            output_seq = np.append(output_seq, last_pred)
            
            # Update input for next prediction
            curr_input = torch.FloatTensor([[last_pred]]).view(1, 1, 1)
    
    return output_seq

# Example usage
test_seq = np.array([1, 1])
predicted = predict_sequence(model, test_seq)
print(f"Input sequence: {test_seq}")
print(f"Predicted sequence (with 5 additional steps): {predicted}")