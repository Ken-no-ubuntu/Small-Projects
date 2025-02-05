import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from randomized_data import random_data
# Define the model class (Same architecture as used during training)
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)  # Match architecture during training
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(16, 8)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out

# Load the trained model
input_size = 11  # Input features count must match the one during training
model = SimpleNN(input_size)
model.load_state_dict(torch.load('trained_model.pth'))
model.eval()  # Set to evaluation mode

# Load and process new data
#new_data = pd.read_csv('new_data.csv')  # Replace with your actual file
X_new = random_data[['X1_age', 'X2_job', 'X3_marital', 'X4_salary', 'X5_default',
                  'X6_balance', 'X7_housing', 'X8_loan', 'X9_contact',
                  'X13_campaign', 'X15_previous']].values

# Standardize new data
scaler = StandardScaler()
X_new_scaled = scaler.fit_transform(X_new)

# Convert to tensor
X_new_tensor = torch.tensor(X_new_scaled, dtype=torch.float32)

# Make predictions
with torch.no_grad():
    predictions = model(X_new_tensor)

# Output predictions
predicted_classes = (predictions > 0.5).int()  # For binary classification
print("Predicted classes:", predicted_classes)

