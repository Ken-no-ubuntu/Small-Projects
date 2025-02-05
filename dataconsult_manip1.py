import pandas as pd
from dataconsult1df import df_sort_age
import matplotlib.pyplot as plt
data = df_sort_age
plt.figure(figsize=(6,6))
plt.pie(data["X1_age"], labels=data["X4_salary"])
plt.axis("equal")
plt.show
print(data.head())