import numpy as py
import geopandas as gp
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams['font.family'] = 'MS Gothic'
df = pd.read_excel(r"C:\Users\jyoji\Downloads\FEH_00200502_240813125814.xlsx", usecols= "B, AK",skiprows=8).rename(columns={"#D0330703_Health expenditure per capita (Prefecture + Municipality)[thousand yen]":"Expend"}).dropna()
df["AREA"] = df["AREA"].str.replace("-ken","").str.replace("-to","").str.replace("-fu","")
#df = df.rename(columns={'AREA': 'NAME_1'})
geoval = gp.read_file(r"C:\Users\jyoji\Downloads\%E5%B9%B3%E6%88%90_27_%E5%B9%B4%E5%9B%BD%E5%8B%A2%E8%AA%BF%E6%9F%BB_%E9%83%BD%E9%81%93%E5%BA%9C%E7%9C%8C%E7%95%8C_-_Japan_Prefecture_Boundaries_ECM.zip")
#geoval = geoval.drop(columns= "NAME_1")
geoval = pd.concat([geoval, df], axis = 1)
#geoval = pd.merge(geoval,df, on='NAME_1')
# Plotting the choropleth map
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
geoval.plot(figsize=(5,5),column='Expend', cmap='OrRd', legend=True, linewidth=0.2, edgecolor="black", ax=ax)
for x, y, label in zip(geoval.geometry.centroid.x, geoval.geometry.centroid.y, geoval['KEN_NAME']):
    plt.text(x, y, label, fontsize=7, ha='center', va='center')
cb = ax.get_figure().get_axes()[1]  # Get the colorbar axis
cb.text(1.1, 0.5, '濃度（赤＝高、白＝低）', 
        rotation=0, ha='left', va='center', fontsize=10, transform=cb.transAxes)

# Add title and axis off
plt.title('都道府県別一人当たりの医療費')
plt.axis('off')
# Show the plot
plt.show()
A=df["NAME_1"]!=geoval["NAME_1"]
