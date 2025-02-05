import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from justmodel import SimpleNN  # Import your model
from randomized_data import random_data
# Load the model
input_size = 11  # Make sure this matches your training
model = SimpleNN(input_size)
model.load_state_dict(torch.load('trained_model.pth'))
model.eval()  # Set model to evaluation mode

# Load and process new data
#new_data = pd.read_csv('new_data.csv')  # Your new data
random_data = random_data.sample(n=10)
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
predicted_classes = (predictions > 0.5).int()  # Binary classification
print("Predicted classes:", predicted_classes)
