import pandas as pd
import glob
from nltk.stem.snowball import SnowballStemmer

path = 'csvs/label/*'


# brand
# price brand
# price brand category

file_names = ['brand price', 'brand price category_id']
li = []

for f in file_names:
    all_files = glob.glob(path + "/"+f+"_top_seller.csv")

    for filename in all_files:
        if any([x in filename for x in ['lower','stopped']]):
            print(filename)
        else:
            continue

        df = pd.read_csv(filename, index_col=None, header=0, sep=';', encoding='latin-1')
        li.append(df)

df = pd.concat(li, axis=0, ignore_index=True)

df.to_pickle("classification_count.pkl")


path = 'csvs/label_2/*'

# price brand
# category preis
# price brand category

file_names = ['brand price', 'brand price category_id']

li = []

for f in file_names:
    all_files = glob.glob(path + "/"+f+"_top_seller.csv")

    for filename in all_files:
        if any([x in filename for x in ['lower','stopped']]):
           print(filename)
        else:
            continue

        df = pd.read_csv(filename, index_col=None, header=0, sep=';', encoding='latin-1')
        li.append(df)

df = pd.concat(li, axis=0, ignore_index=True)

df.to_pickle("classification2_count.pkl")



path = 'csvs/mean_revenue/*'
all_files = glob.glob(path + "/brand price.csv")
path = 'csvs/median_revenue/*'
all_files = all_files + glob.glob(path + "/brand price category_id.csv")

li = []

for filename in all_files:
    if any([x in filename for x in ['stopped', 'lower']]):
        print("found")
    else:
        continue

    if 'ohne' in filename:
        continue

    print(filename)
    df = pd.read_csv(filename, index_col=None, header=0, sep=';', encoding='latin-1')
    li.append(df)

df = pd.concat(li, axis=0, ignore_index=True)

df.to_pickle("mean_count.pkl")

df1 = pd.read_pickle("classification_count.pkl")
df2 = pd.read_pickle("classification2_count.pkl")
#df3 = pd.read_pickle("mean_count.pkl")

df = df1.append(df2)
#df = df.append(df3)

col = df.columns[1]
df_new = df.groupby(['Wort'])[col].agg('sum').nlargest(65)
pd.set_option('display.max_rows', None)
print(df_new)
