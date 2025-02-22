import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# 1. Data Preprocessing and Loading
# -----------------------------------
# CIFAR-10 images are 32x32 RGB images. We apply two main transforms:
#   - ToTensor(): Converts images to PyTorch tensors and scales pixel values to [0,1]
#   - Normalize(): Normalizes the tensor image with mean and std for each channel.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Mean & std for R, G, B
])

# Download and load the training dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
# DataLoader wraps the dataset and provides mini-batches.
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                          shuffle=True, num_workers=2)

# Download and load the test dataset
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=False, num_workers=2)

# Define the class labels for CIFAR-10
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 2. Define the CNN Model
# -------------------------
# We subclass nn.Module to create our custom network.
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # First convolutional layer: 
        #   - Input channels = 3 (RGB)
        #   - Output channels = 16
        #   - Kernel size = 3
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        # Second convolutional layer: 16 -> 32 channels.
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # Max pooling layer: reduces spatial dimensions by taking the maximum value.
        self.pool = nn.MaxPool2d(2, 2)
        # Fully connected layers to map features to 10 classes.
        self.fc1 = nn.Linear(32 * 8 * 8, 120)  # 32 channels, 8x8 feature map after pooling twice.
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        # Activation function: ReLU introduces non-linearity.
        self.relu = nn.ReLU()

    def forward(self, x):
        # Pass input through the first conv layer, then apply ReLU and pooling.
        x = self.pool(self.relu(self.conv1(x)))
        # Pass through the second conv layer, then apply ReLU and pooling.
        x = self.pool(self.relu(self.conv2(x)))
        # Flatten the tensor from (batch_size, channels, height, width) to (batch_size, -1)
        x = x.view(-1, 32 * 8 * 8)
        # Fully connected layers with ReLU activations in between.
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        # Output layer: raw scores for each class.
        x = self.fc3(x)
        return x

# Instantiate the network.
net = SimpleCNN()

# 3. Define Loss Function and Optimizer
# ---------------------------------------
# CrossEntropyLoss combines softmax and negative log-likelihood loss.
criterion = nn.CrossEntropyLoss()
# Use the Adam optimizer for updating the network parameters.
optimizer = optim.Adam(net.parameters(), lr=0.001)

# 4. Training the Network
# --------------------------
num_epochs = 10  # You can adjust the number of epochs.
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # 'data' is a tuple of inputs and labels.
        inputs, labels = data

        # Zero the parameter gradients. This is important to clear out gradients from previous iterations.
        optimizer.zero_grad()

        # Forward pass: compute predicted outputs by passing inputs to the network.
        outputs = net(inputs)
        # Compute the loss: how far is the prediction from the actual label?
        loss = criterion(outputs, labels)
        # Backward pass: compute gradient of the loss with respect to all model parameters.
        loss.backward()
        # Perform a single optimization step (parameter update).
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # Print every 100 mini-batches
            print(f'Epoch {epoch + 1}, Batch {i + 1}: Loss = {running_loss / 100:.3f}')
            running_loss = 0.0

print('Finished Training')

# 5. Testing the Network on the Test Data
# ------------------------------------------
correct = 0
total = 0
with torch.no_grad():  # Turn off gradients for validation to speed up computations.
    for data in testloader:
        images, labels = data
        outputs = net(images)
        # Get the predicted class by selecting the index with the highest score.
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on the 10000 test images: {100 * correct / total:.2f}%')
