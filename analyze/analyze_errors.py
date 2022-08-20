import pandas as pd
import glob

df = pd.read_pickle("../errors.pkl")
pd.set_option('display.max_columns', None)
print(df.sort_values('MAPE', ascending=True))

print(len(df))


df.sort_values('MAPE', ascending=True).to_csv('vkz_errors.csv',index=False)
