""" Example for causal_selection. """
import causal_attribution as selection
import random
import pandas as pd
import sys
from bs4 import BeautifulSoup
import csv
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, \
    mean_absolute_percentage_error

import nltk
from collections import Counter

def plot_diagram(targets, preds, file):
    keys_t = sorted(range(len(targets)), key=lambda k: targets[k])
    targets = np.array([targets[i] for i in keys_t])
    preds = np.array([preds[i] for i in keys_t])

    #figure(figsize=(10,10), dpi=80)
    plt.plot(range(len(targets)), targets, '+', fillstyle='none')
    plt.plot(range(len(targets)), preds, 'x', fillstyle='none')

    x = range(1,len(targets)+1)

    m1, b1 = np.polyfit(x, targets, 1)
    m2, b2 = np.polyfit(x, preds, 1)

    plt.xlabel('Produkt')
    plt.ylabel('Durchschnittliche Verkaufszahlen')

    plt.legend(('Tatsächlich', 'Vorhersage', 'Tat. Reg.', 'Vorh. Reg.'))
    plt.yscale('log')
    #plt.ylim(0, 1700)
    #plt.xlim(0, 1700)
    #plt.axis('scaled')

    plt.savefig(file+'.png')#, bbox_inches="tight")

    plt.plot(x, m1*targets + b1)
    plt.plot(x, m2*targets + b2)
    plt.savefig(file + '_reg.png', bbox_inches="tight")
    plt.close('all')
    #sys.exit()

def build_vocab(text, n=4000):
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

# für ohne confounds
df['ohne_confounds'] = 0

measures = ['mean_revenue', 'median_revenue']
columns = ['stemmed', 'stopped', 'lower', 'no_punct', 'tokens']
confounds = [
    {},
    {'brand': 'control'},
    {'price': 'control'},
    {'category_id': 'control'},
    {'brand': 'control', 'price': 'control'},
    {'brand': 'control', 'category_id': 'control'},
    {'price': 'control', 'category_id': 'control'},
    {'brand': 'control', 'price': 'control'},
    {'brand': 'control', 'price': 'control', 'category_id': 'control'}
]
col = columns[0]
times = 1#20
top = 50

df_errors = pd.DataFrame([], columns=['metric', 'confounds', 'preprocessing', 'MAE', 'MSE', 'MSLE', 'MAPE'])

for confound in confounds:
    for col in columns:
        for measure in measures:

            confound_names = "ohne_confound"
            if len(confound) > 0:
                confound_names = ' '.join(list(confound.keys()))


            print(measure)
            print(col)
            print(confound_names)

            df['description'] = df[col].apply(lambda x: ' '.join(x))

            # sys.exit()

            # Use a variety of variables (categorical and continuous)
            #  to score a vocab.
            print('Scoring vocab...')

            vocab = build_vocab(df[col])

            n2t = {
                'description': 'input',
                measure: 'predict'
            }

            if len(confound) > 0:
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

                print(f"Scores für {i}")
                score_list = scores[measure]['N/A']
                for j in range(top):
                    if score_list[j][0] in scores_tot:
                        val = scores_tot[score_list[j][0]]
                        val_new = (score_list[j][1] + val)
                        scores_tot[score_list[j][0]] = val_new
                    else:
                        scores_tot[score_list[j][0]] = score_list[j][1]

                # nur zum Testen
                print_test()

                if i == times - 1:

                    for j, element in enumerate(preds):
                        preds[j] = element[0]

                    path = 'errors/' + measure + '/' + col

                    isExist = os.path.exists(path)

                    if not isExist:
                        os.makedirs(path)

                    file = path + '/' + confound_names

                    with open(file+ '.txt', 'w') as f:
                        f.writelines("MAE: " + str(mean_absolute_error(targets, preds)))
                        f.writelines(" MSE: " + str(mean_squared_error(targets, preds)))
                        f.writelines(" MSLE: " + str(mean_squared_log_error(targets, preds)))
                        f.writelines(" MAPE: " + str(mean_absolute_percentage_error(targets, preds)))

                    if measure == 'mean_revenue':
                        df_errors.loc[len(df_errors.index)] = [measure, confound_names, col,
                                                               str(mean_absolute_error(targets, preds)),
                                                               str(mean_squared_error(targets, preds)),
                                                               str(mean_squared_log_error(targets, preds)),
                                                               str(mean_absolute_percentage_error(targets, preds))]

                    plot_diagram(targets, preds, file)

            for k in scores_tot:
                scores_tot[k] = scores_tot[k] / times

            scores_tot = dict(sorted(scores_tot.items(), key=lambda item: item[1], reverse=True))

            #print(scores_tot)

            path = 'csvs/' + measure + '/' + col

            isExist = os.path.exists(path)

            if not isExist:
                os.makedirs(path)

            file = path + '/' + confound_names + '.csv'
            with open(file, 'w') as f:
                f.write("%s; %s\n" % ('Wort', 'Durchschnittlicher Wert'))
                for key in scores_tot.keys():
                    f.write("%s; %s\n" % (key, scores_tot[key]))

df_errors.to_pickle("errors.pkl")
