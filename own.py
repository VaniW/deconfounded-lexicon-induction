""" Example for causal_selection. """
import causal_attribution as selection
import random
import pandas as pd
import sys
from bs4 import BeautifulSoup

import nltk
from collections import Counter

csv_path = r'../../data/data.csv'

def build_vocab(text, n=1000):
	all_words = [w for s in text for w in s]
	c = Counter(all_words)
	return [w for w, _ in c.most_common(n)]

print('Reading data...')
df = pd.read_pickle('../../data/data.pkl').head(30)

df['mean_revenue'] = (df['mean_revenue']-df['mean_revenue'].min())/(df['mean_revenue'].max()-df['mean_revenue'].min())

text = []
description_wo_html = []
for index, row in df.iterrows():
	description = row['description']
	wo_html = BeautifulSoup(description, "lxml").text
	description_wo_html.append(wo_html)
	text += [nltk.word_tokenize(wo_html.lower())]

# generelles Preprocessing

df['description'] = description_wo_html

#print(df['mean_revenue'])
#sys.exit()
#df.to_csv(csv_path, sep='\t')

# Use a variety of variables (categorical and continuous) 
#  to score a vocab.
print('Scoring vocab...')

vocab = build_vocab(text)

n2t = {
		'description': 'input',
		'brand': 'control',
		'mean_revenue': 'predict',
		'price': 'control',
}	

# Run the residualization model through its paces...
scores = selection.score_vocab(
	vocab=vocab,
	df=df,
	name_to_type=n2t,
	scoring_model='residualization',
 	batch_size=2,
 	train_steps=500)

# And then the adversarial one...
scores = selection.score_vocab(
	vocab=vocab,
	df=df,
	name_to_type=n2t,
	scoring_model='adversarial',
 	batch_size=2,
 	train_steps=500)

print('a')
print(scores)
print('b')

print('Evaluating vocab...')
# Now evaluate 2 vocabs, and ensure that the larger
#  vocab is more informative.
full_scores = selection.evaluate_vocab(
	vocab=vocab[:100],
	df=df,
	name_to_type=n2t)
partial_scores = selection.evaluate_vocab(
	vocab=[vocab[-1]],
	df=df,
	name_to_type=n2t)

assert full_scores['mean_revenue'] > partial_scores['mean_revenue']


# And just for good measure have a continuous outcome.
partial_scores = selection.evaluate_vocab(
	vocab=[],
	df=df,
	name_to_type=n2t)


print('Tests passed!')
