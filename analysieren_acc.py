import pandas as pd
import glob

df = pd.read_pickle("acc.pkl")

df = df.round(2)
pd.set_option('display.max_columns', None)
df.sort_values('acc', ascending=False).to_csv('classification_acc.csv',index=False)

df = pd.read_pickle("acc2.pkl")

df = df.round(2)
pd.set_option('display.max_columns', None)
df.sort_values('acc', ascending=False).to_csv('classification_acc2.csv',index=False)
