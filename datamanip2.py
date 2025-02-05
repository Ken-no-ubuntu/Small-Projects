import pandas as pd
from dataconsult1df import df_sort_age
data = df_sort_age
countage = data["X1_age"].value_counts()
df1 = countage.reset_index()
countsal = data["X4_salary"].value_counts()
df2 = countsal.reset_index()
#'0_20歳未満' '1_20代' '2_30代' '3_40代' '4_50代' '5_60歳以上'
gr1 = data[data["X1_age"]== "5_60歳以上"]
profgr1 = gr1["X4_salary"].value_counts(), gr1["X6_balance"].value_counts(), gr1["X13_campaign"].value_counts()
print(profgr1)
