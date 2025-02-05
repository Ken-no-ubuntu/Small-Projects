# train_model.py

# 1. Import necessary libraries
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.nn as nn

# 2. Load CSV data
data = pd.read_csv('basic_data.csv', encoding='cp932')

# 3. Prepare features (X) and target (y)
X = data.drop('target', axis=1).values  # Replace 'target' with your column name
y = data['target'].values

# 4. Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 5. Convert data to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # Shape as (N, 1)

# 6. Create PyTorch dataset and data loaders
dataset = TensorDataset(X_tensor, y_tensor)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# 7. Define the neural network model
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# 8. Initialize the model, loss function, and optimizer
input_size = X_tensor.shape[1]
model = SimpleNN(input_size)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 9. Train the model
def train(model, loader, criterion, optimizer, epochs=50):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in loader:
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}')

train(model, train_loader, criterion, optimizer)

# 10. Evaluate the model
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            predictions = model(X_batch)
            predicted_labels = (predictions > 0.5).float()
            correct += (predicted_labels == y_batch).sum().item()
            total += y_batch.size(0)

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')

evaluate(model, test_loader)

# 11. Save the model
torch.save(model.state_dict(), 'model.pth')

# 12. Example prediction with new data
new_data = torch.tensor([[0.5, 1.2, -0.3, 4.1]], dtype=torch.float32)  # Example
model.eval()
with torch.no_grad():
    prediction = model(new_data)
    print(f'Prediction: {prediction.item():.4f}')  # Probability of being 1
