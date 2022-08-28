import pandas as pd
import glob
from nltk.stem.snowball import SnowballStemmer

path = '../csvs/label/*'
all_files = glob.glob(path + "/ohne_confound_top_seller.csv")

li = []

for filename in all_files:
    if any([x in filename for x in ['lower', 'stopped']]):
        print("found")
        print(filename)
    else:
        continue
    df = pd.read_csv(filename, index_col=None, header=0, sep=';', encoding='latin-1')
    li.append(df)

df = pd.concat(li, axis=0, ignore_index=True)

df.to_pickle("no_c_classification_count.pkl")


path = '../csvs/label_2/*'
all_files = glob.glob(path + "/ohne_confound_top_seller.csv")

li = []

for filename in all_files:
    if any([x in filename for x in ['lower', 'stopped']]):
        print("found")
    else:
        continue
    df = pd.read_csv(filename, index_col=None, header=0, sep=';', encoding='latin-1')
    li.append(df)

df = pd.concat(li, axis=0, ignore_index=True)

df.to_pickle("no_c_classification2_count.pkl")

df = pd.concat(li, axis=0, ignore_index=True)

df.to_pickle("mean_count.pkl")

df1 = pd.read_pickle("no_c_classification_count.pkl")
df2 = pd.read_pickle("no_c_classification2_count.pkl")

df = df1.append(df2)

print(df)
col = df.columns[1]
df_new = df.groupby(['Wort'])[col].agg('sum').nlargest(59)

print(df_new)
