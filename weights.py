import torch
import torch.nn as nn

# Define the SimpleNN class here
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)  # Adjusted to match the saved model
        self.fc2 = nn.Linear(16, 8)           # Adjusted to match the saved model
        self.fc3 = nn.Linear(8, 1)            # Adjusted to match the saved model

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


# Make sure input_size is defined based on your model
input_size = 11  # For example, adjust this based on your data

# Initialize the model
model = SimpleNN(input_size)

# Load the saved model weights
model.load_state_dict(torch.load('trained_model.pth'))

# Print the weights
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"Layer: {name}")
        print(param.data)
with open("model_weights.txt", "w") as f:
    for name, param in model.named_parameters():
        if param.requires_grad:
            f.write(f"Layer: {name}\n")
            f.write(f"{param.data}\n\n")
