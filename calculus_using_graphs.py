
import numpy as np
import matplotlib.pyplot as plt

# Define the function
def f(x):
    return x**2

# Define the range
x = np.linspace(0, 2, 1000)
y = f(x)

# Create the plot
plt.figure(figsize=(8, 6))

# Plot the function
plt.plot(x, y, label='f(x) = x^2')

# Fill the area under the curve
plt.fill_between(x, y, where=[(0 <= xi <= 2) for xi in x], color='skyblue', alpha=0.4)

# Add labels and title
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Area Under the Curve of f(x) = x^2')
plt.legend()

# Show the plot
plt.show()

# Calculate the area under the curve
area = np.trapz(y, x)
print("The area under the curve is:", area)