import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from numericonlydata import data_num

# Import the feature importance methods from the new file
from gradients import compute_gradients, integrated_gradients

# Separate features and target
X = data_num[['X1_age', 'X2_job', 'X3_marital', 'X4_salary', 'X5_default',
       'X6_balance', 'X7_housing', 'X8_loan', 'X9_contact', 'X13_campaign',
       'X15_previous']].values  # input features
y = data_num['X17_ioutcome'].values  # target values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)  # Reshaped to be a column vector
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Define the model class
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)  # First fully connected layer
        self.fc2 = nn.Linear(16, 8)  # Second fully connected layer
        self.fc3 = nn.Linear(8, 1)   # Output layer for binary classification
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Apply ReLU activation function
        x = torch.relu(self.fc2(x))  # Apply ReLU activation function
        x = torch.sigmoid(self.fc3(x))  # Sigmoid for binary classification
        return x

# Initialize the model
input_size = X_train.shape[1]  # Number of input features
model = SimpleNN(input_size)

# Define the loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for classification
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    # Backward pass and optimization
    optimizer.zero_grad()  # Clear the gradients
    loss.backward()        # Backpropagation
    optimizer.step()       # Update weights
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate on test data
with torch.no_grad():  # Disable gradient tracking for inference
    test_outputs = model(X_test_tensor)
    predicted = test_outputs.round()  # Round the sigmoid output to get 0 or 1

# Calculate accuracy
accuracy = (predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)
print(f'Accuracy on test set: {accuracy:.4f}')

# Save the model
torch.save(model.state_dict(), 'trained_model.pth')  # Saves only the state_dict (recommended)

### Compute Feature Importance using the imported methods

# Feature names
feature_names = ['X1_age', 'X2_job', 'X3_marital', 'X4_salary', 'X5_default', 
                 'X6_balance', 'X7_housing', 'X8_loan', 'X9_contact', 'X13_campaign', 'X15_previous']

# Gradient-based feature importance
grad_importance = compute_gradients(model, X_train_tensor, y_train_tensor)
print("Gradient-based Feature Importance:")
for i, feature in enumerate(feature_names):
    print(f"{feature}: {grad_importance[i]:.4f}")

# Integrated Gradients
integrated_grad_importance = integrated_gradients(model, X_train_tensor, y_train_tensor)
print("\nIntegrated Gradients Feature Importance:")
for i, feature in enumerate(feature_names):
    print(f"{feature}: {integrated_grad_importance[i]:.4f}")
