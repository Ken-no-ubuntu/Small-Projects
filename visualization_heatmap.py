import matplotlib.pyplot as plt
import numpy as np
from weights import model
# Example: Visualizing weights for each input feature in the first layer (fc1)
weights = model.fc1.weight.detach().numpy()

# Create a heatmap of weights
plt.figure(figsize=(8, 6))
plt.imshow(weights, cmap='coolwarm', aspect='auto')
plt.colorbar(label='Weight Value')
plt.xlabel('Input Feature')
plt.ylabel('Neuron')
plt.title('Weights Heatmap for Input Features (Layer 1)')
plt.show()

