import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.family'] = 'MS Gothic'
x = [2014,2016,2018,2020]
y = [15.2,17,22.1,23.9]
plt.plot(x,y, color='red', linewidth=2)
plt.title('高齢者貧困率推移')
plt.xlabel('年次')
plt.ylabel('貧困率')
plt.show()
print('貧困率')