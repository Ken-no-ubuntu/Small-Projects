import numpy as py
import geopandas as gp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sea
plt.rcParams["font.family"]="MS Gothic"
old_df = pd.read_excel(r"C:\Users\jyoji\Downloads\a01200_2.xlsx", sheet_name= "第12表", usecols="L,T", skiprows=8, nrows=47).rename(columns={15.5:"Old Pop"}).dropna()
graph = sea.kdeplot(data=old_df["Old Pop"], color="red", fill=bool)
plt.ylabel("都道府県割合")
plt.xlabel("75歳以上齢者割合密度（KDE）")
plt.plot()
plt.show()
