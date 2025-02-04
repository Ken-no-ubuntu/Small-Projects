import pandas as pn
import numpy as np
import matplotlib.pyplot as plt
df = pn.read_excel(r"C:\Users\jyoji\Downloads\FEH_00200502_240813125814.xlsx", usecols="B:AK", skiprows=8, nrows=46)
# Example data
prefectures = df['AREA']
expenditures = df['#D0330703_Health expenditure per capita (Prefecture + Municipality)[thousand yen]']
size = df['#D0320201_Percentage of allowances (Prefecture)[%]']
plt.figure(figsize=(10, 6))
plt.scatter(prefectures, expenditures, alpha=0.5)

# Add labels and title
plt.xlabel("Prefectures")
plt.ylabel("Healthcare Expenditure per Capita (Yen)")
plt.title("Healthcare Expenditure per Capita by Prefecture in Japan")

# Show the chart
plt.show()
