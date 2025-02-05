import pandas as pd
from sklearn.preprocessing import LabelEncoder
data = pd.read_csv('basic_data.csv', encoding='cp932')
data["X4_salary"] = data["X4_salary"].replace("X_不明", '0')
data["X9_contact"] = data["X9_contact"].replace("X_不明", '0')
data_extract= ['X1_age', 'X2_job', 'X3_marital', 'X4_salary', 'X5_default', 'X6_balance', 'X7_housing', 'X8_loan', 'X9_contact', 'X13_campaign', 'X15_previous', 'X16_toutcome', 'X17_ioutcome']
for col in data_extract:
    data[col] = data[col].str.extract(r'(\d+)') 
data_num = data.dropna()
cols =data_num.columns.drop("ID")
data_num[cols] = data_num[cols].apply(pd.to_numeric, errors='coerce')



print(data_num)