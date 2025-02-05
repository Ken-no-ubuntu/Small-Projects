import shap
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from numericonlydata import data_num
# Load your saved model architecture
class SimpleNN(torch.nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 16)
        self.fc2 = torch.nn.Linear(16, 8)
        self.fc3 = torch.nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

# Initialize the model and load the trained weights
input_size = 11  # Number of features in your data
model = SimpleNN(input_size)
model.load_state_dict(torch.load('trained_model.pth', weights_only=True))
model.eval()  # Set the model to evaluation mode

# Load the data (ensure it's the same preprocessing as your training data)
#data_num = pd.read_csv('your_data.csv')  # Replace with the actual data file
# Define the columns you want to convert to numeric
cols = ['X1_age', 'X2_job', 'X3_marital', 'X4_salary', 'X5_default',
        'X6_balance', 'X7_housing', 'X8_loan', 'X9_contact', 
        'X13_campaign', 'X15_previous']

# Apply pd.to_numeric safely using .loc to avoid the 
# Ensure you're working with a proper DataFrame copy
data_num = data_num.copy(deep=True)

# Use .loc properly for applying numeric conversion
for col in cols:
    data_num.loc[:, col] = pd.to_numeric(data_num[col], errors='coerce')



# Extract the relevant features for your model
X = data_num[cols].values


# Feature scaling (reuse the scaler from your training step)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert to PyTorch tensor
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

# Wrap the model with SHAP's deep explainer
explainer = shap.DeepExplainer(model, X_tensor)

# Calculate SHAP values for a subset of your data (to avoid large memory usage)
shap_values = explainer.shap_values(X_tensor[:100], check_additivity=False)  # Adjust the slice as needed

# Plot summary of feature importance
shap.summary_plot(np.array(shap_values[0]), X, feature_names=[
    'age', 'job', 'marital', 'salary', 'default', 'balance', 'housing', 
    'loan', 'contact', 'campaign', 'previous'
])
