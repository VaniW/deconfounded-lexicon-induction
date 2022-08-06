""" Example for causal_selection. """
import causal_attribution as selection
import random
import pandas as pd
import sys
from bs4 import BeautifulSoup
import csv
import os

import nltk
from collections import Counter

def build_vocab(text, n=1000):
	all_words = [w for s in text for w in s]
	c = Counter(all_words)
	return [w for w, _ in c.most_common(n)]

def print_test():
	'''print('Evaluating vocab...')
		# Now evaluate 2 vocabs, and ensure that the larger
		#  vocab is more informative.
		full_scores = selection.evaluate_vocab(
			vocab=vocab,
			df=df,
			name_to_type=n2t)
		partial_scores = selection.evaluate_vocab(
			vocab=[vocab[-1]],
			df=df,
			name_to_type=n2t)

		print(full_scores[measure])
		print(partial_scores[measure])

		assert full_scores[measure] > partial_scores[measure]


		# And just for good measure have a continuous outcome.
		partial_scores = selection.evaluate_vocab(
			vocab=[],
			df=df,
			name_to_type=n2t)


		print('Tests passed!')'''

print('Reading data...')
df = pd.read_pickle('../../data/preprocessed_classified.pkl')

measures = ['label_num','label']
measures = ['label']
columns = ['stemmed', 'stopped', 'lower', 'no_punct', 'tokens']
confounds = [
	{'brand': 'control'},
	{'price': 'control'},
	{'category_id': 'control'},
	{'brand': 'control', 'price': 'control'},
	{'brand': 'control', 'category_id': 'control'},
	{'price': 'control', 'category_id': 'control'},
	{'brand': 'control', 'price': 'control'},
	{'brand': 'control', 'price': 'control', 'category_id': 'control'},
]
times = 20
top = 50

for confound in confounds:
	for col in columns:
		for measure in measures:
			print(measure)
			df['description'] = df[col].apply(lambda x: ' '.join(x))

			#sys.exit()

			# Use a variety of variables (categorical and continuous)
			#  to score a vocab.
			print('Scoring vocab...')

			vocab = build_vocab(df[col]) # todo einige stoppwörter hinzufügen beim etc., ist stemmed das richtige??

			n2t = {
					'description': 'input',
					measure: 'predict'
			}

			n2t = {**n2t, **confound}

			scores_tot = {}

			for i in range(times):
				# run the adversarial one...
				scores, targets, preds = selection.score_vocab(
					vocab=vocab,
					df=df,
					name_to_type=n2t,
					scoring_model='adversarial',
					batch_size=16,
					train_steps=255,
					max_seq_len=700)

				full_scores = selection.evaluate_vocab(
					vocab=vocab,
					df=df,
					name_to_type=n2t)

				print(f"Scores für {i}")
				for key, score_list in scores[measure].items():
					scores_tot[key] = {}
					for j in range(top):
						if score_list[j][0] in scores_tot[key]:
							val = scores_tot[key][score_list[j][0]]
							val_new = (score_list[j][1]+val)
							scores_tot[key][score_list[j][0]] = val_new
						else:
							scores_tot[key][score_list[j][0]] = score_list[j][1]

				# nur zum Testen
				print_test()

			for key, val in scores_tot.items():
				for k in val:
					scores_tot[key][k] = scores_tot[key][k]/times

				scores_tot[key] = dict(sorted(scores_tot[key].items(), key=lambda item: item[1], reverse=True))

				print(scores_tot[key])

				path = 'csvs/'+measure+'/'+col

				isExist = os.path.exists(path)

				if not isExist:
					os.makedirs(path)

				file = path +'/'+ '_'.join(list(confound.keys())) + '_' + key +'.csv'
				with open(file, 'w') as f:
					f.write("%s; %s\n" % ('Wort', 'Durchschnittlicher Wert'))
					for k in scores_tot[key].keys():
						f.write("%s; %s\n" % (k, scores_tot[key][k]))
