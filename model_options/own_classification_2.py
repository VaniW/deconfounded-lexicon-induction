""" Example for causal_selection. """
import causal_attribution as selection
import random
import pandas as pd
import sys
from bs4 import BeautifulSoup
import csv
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import nltk
from collections import Counter
import itertools


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    fig = plt.figure()

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm, decimals=3, out=None)

    thresh = cm.max() / 2.

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    plt.ylabel('Echtes Label')
    plt.xlabel('Vorausgesagtes Label')

    path = 'cms/' + measure + '_2/' + col

    isExist = os.path.exists(path)

    if not isExist:
        os.makedirs(path)

    file = path + '/' + confound_names + '.jpg'
    fig.savefig(file, bbox_inches='tight')
    plt.close('all')


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
df = pd.read_pickle('../../../data/preprocessed_classified.pkl')

df = df[df['label'] != 'normal']

# für ohne confounds
df['ohne_confounds'] = 0

measures = ['label']
columns = ['stemmed', 'stopped', 'lower', 'no_punct', 'tokens']
confounds = [
    {},
    {'brand': 'control'},
    {'price': 'control'},
    {'category_id': 'control'},
    {'brand': 'control', 'price': 'control'},
    {'brand': 'control', 'category_id': 'control'},
    {'price': 'control', 'category_id': 'control'},
    {'brand': 'control', 'price': 'control', 'category_id': 'control'},
]
times = 5
top = 50

amount_top = df.groupby('label').count()['id']['top_seller']
amount_bottom = df.groupby('label').count()['id']['non_seller']

df_acc = pd.DataFrame([], columns=['metric', 'confounds', 'preprocessing', 'acc', 'bottom', 'top'])

for confound in confounds:
    for col in columns:
        for measure in measures:

            confound_names = "ohne_confound"
            if len(confound) > 0:
                confound_names = str(' '.join(list(confound.keys())))

            path = 'cms/' + measure + '_2/' + col

            file = path + '/' + confound_names + '.jpg'
            #if os.path.isfile(file):
            #    continue

            print(measure)
            print(col)
            print(confound_names)

            df['description'] = df[col].apply(lambda x: ' '.join(x))

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
                    max_seq_len=640)

                print(f"Scores für {i}")
                for key, score_list in scores[measure].items():
                    scores_tot[key] = {}
                    for j in range(top):
                        if score_list[j][0] in scores_tot[key]:
                            val = scores_tot[key][score_list[j][0]]
                            val_new = (score_list[j][1] + val)
                            scores_tot[key][score_list[j][0]] = val_new
                        else:
                            scores_tot[key][score_list[j][0]] = score_list[j][1]

                # nur zum Testen
                print_test()

                if i == times - 1:

                    for j, element in enumerate(preds):
                        temp = np.argmax(element)
                        if temp == 0:
                            element[temp] = 0
                            if np.argmax(element) == 0:
                                preds[j] = 2
                            else:
                                preds[j] = np.argmax(element)
                        else:
                            preds[j] = temp


                    cm = confusion_matrix(targets, preds)
                    class_names = ['Ladenhüter', 'Top-Seller']

                    print(cm)
                    plot_confusion_matrix(cm, class_names,
                                          normalize=True,
                                          title='Normalized Confusion matrix',
                                          cmap=plt.cm.Blues)

                    df_acc.loc[len(df_acc.index)] = [measure, confound_names , col,
                                                            accuracy_score(targets, preds),
                                                            np.diag(cm)[0]/amount_bottom,
                                                            np.diag(cm)[1]/amount_top
                                                     ]


            for key, val in scores_tot.items():
                if key == "N/A":
                    continue
                for k in val:
                    scores_tot[key][k] = scores_tot[key][k] / times

                scores_tot[key] = dict(sorted(scores_tot[key].items(), key=lambda item: item[1], reverse=True))

                #print(scores_tot[key])

                path = 'csvs/' + measure + '_2/' + col

                isExist = os.path.exists(path)

                if not isExist:
                    os.makedirs(path)

                file = path + '/' + confound_names + '_' + key + '.csv'
                with open(file, 'w', encoding="utf-8") as f:
                    f.write("%s; %s\n" % ('Wort', 'Durchschnittlicher Wert'))
                    for k in scores_tot[key].keys():
                        f.write("%s; %s\n" % (k, scores_tot[key][k]))

df_acc.to_pickle("acc2.pkl")