import pandas as pd
import matplotlib.pyplot as plt

# Simple bolier plate code for making linear graphs
file_path = 'C:\\Users\\jyoji\\Downloads\\Students Score for Experimentation.xlsx'  # Replace with your file path
df = pd.read_excel(file_path)

#
x_values = df.iloc[:3, 0] 
y_values = df.iloc[:3, 1]  

plt.plot(x_values, y_values, marker='o')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('Linear Graph from Excel Data')
plt.show()
