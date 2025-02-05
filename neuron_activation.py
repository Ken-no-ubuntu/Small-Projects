import matplotlib.pyplot as plt
import torch
from prossecing import model
# Example input (adjust based on your model's input size)
sample_input = torch.tensor([[0.5, 0.1, -0.2, 0.4, 0.3, 0.7, -0.5, 0.9, 0.1, 0.2, 0.8, 1, 1, 0, 1, 0]])

# Ensure you load the model correctly
model.eval()  # Set the model to evaluation mode

# Forward pass to get activations
with torch.no_grad():
    activations = model.fc2(sample_input).numpy()

# Plot the activations
plt.bar(range(len(activations[0])), activations[0])
plt.xlabel('Neuron')
plt.ylabel('Activation Value')
plt.title('Neuron Activations for a Sample Input')
plt.show()

