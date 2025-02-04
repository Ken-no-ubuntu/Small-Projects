import numpy as py
import geopandas as gp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sea
plt.rcParams["font.family"]="MS Gothic"
old_df = pd.read_excel(r"C:\Users\jyoji\Downloads\a01200_2.xlsx", sheet_name= "第12表", usecols="L,R", skiprows=8, nrows=47).rename(columns={29.0:"Old Pop"}).dropna()
bins1 = [20, 25, 30, 35, 40]
labels1 = ["20-25%","25-30%","30-35%", "35-40%"]
binned_val = pd.cut(x=old_df["Old Pop"], bins= bins1, labels = labels1)
counts = binned_val.value_counts()
explode = [0.05] * len(counts)
print(counts)
print(old_df.max())
print(old_df.min())
print(old_df["Old Pop"].mean())
plt.figure(figsize=(8, 6))
plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140, explode= explode, shadow= True)
plt.title('5%範囲のビンで高齢者(65歳以上)割合別の都道府県数')
plt.show()