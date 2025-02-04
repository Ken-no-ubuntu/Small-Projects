import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
plt.rcParams['font.family'] = 'MS Gothic'
# Your original data
x = np.array([2014, 2016, 2018, 2020])
y = np.array([15.2, 17, 22.1, 23.9])

# Generate a smoother set of x-values
x_smooth = np.linspace(x.min(), x.max(), 300)

# Apply cubic interpolation to get smooth y-values
spl = make_interp_spline(x, y, k=3)
y_smooth = spl(x_smooth)

# Plot the smoothed data
plt.plot(x_smooth, y_smooth, color='blue', linewidth=2)  # Change color and thickness
plt.title('高齢者貧困率推移')
plt.xlabel('年次')
plt.ylabel('貧困率')
plt.show()