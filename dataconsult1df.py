import pandas as pd
df = pd.read_csv(r"C:\Users\jyoji\Downloads\Bank_marketing_training_001 (1).csv", encoding='cp932')
df["agenum"] = df['X1_age'].str[0].astype(int)
#here 1 to 5 in age code means 20代、30代、40代、50代、60歳以上
#'0_20歳未満' '1_20代' '2_30代' '3_40代' '4_50代' '5_60歳以上'
df_sort_age = df.sort_values(by ="agenum", ascending=True)
#salary types ['2_中' 'X_不明' '1_低' '3_高']
df_sort_age["X4_salary"] = df_sort_age["X4_salary"].replace("X_不明","0")
df_sort_age["rev"] = df_sort_age["X4_salary"].str[0].astype(int) 
df_sort_age = df_sort_age.sort_values(by =["agenum", "rev"], ascending=[True, True])
#df_sort_age_rev = df_sort_age.sort_values()
#print(df_sort_age["X1_age"], df_sort_age["X4_salary"])
print(df.head)