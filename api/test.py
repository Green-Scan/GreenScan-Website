import pandas as pd
data_info = pd.read_csv("/Users/aviral/miniforge3/Green/GreenScan-Website/api/disease_info.csv",encoding='cp1252')
# print(data_info.iloc[[0],[1]])
print(str(data_info.loc[[1], 'disease_name']))

print(data_info['disease_name'].values[2])