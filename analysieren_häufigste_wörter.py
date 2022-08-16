import pandas as pd
import glob
from nltk.stem.snowball import SnowballStemmer

path = 'csvs/label/*'
all_files = glob.glob(path + "/*top_seller.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0, sep=';', encoding='latin-1')
    li.append(df)

df = pd.concat(li, axis=0, ignore_index=True)

df.to_pickle("classification_count.pkl")

path = 'csvs/mean_revenue/*'
all_files = glob.glob(path + "/*.csv")
path = 'csvs/median_revenue/*'
all_files = all_files + glob.glob(path + "/*.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0, sep=';', encoding='latin-1')
    li.append(df)

df = pd.concat(li, axis=0, ignore_index=True)

df.to_pickle("mean_count.pkl")

#df = pd.read_pickle("test.pkl")
df1 = pd.read_pickle("classification_count.pkl")
df2 = pd.read_pickle("mean_count.pkl")

df = df1.append(df2)

df['Wort'] = df['Wort'].apply(lambda x: x.lower())
stemmer = SnowballStemmer('german')
df['Wort'] = df['Wort'].apply(lambda x: stemmer.stem(x) )

df_new = df['Wort'].value_counts()[:50]

print(df_new)