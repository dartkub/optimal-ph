from sklearn import tree
import numpy as np
import pickle

import pandas as pd
from itertools import product
from collections import Counter
import sklearn
from scipy.stats import spearmanr
from sklearn.model_selection import ShuffleSplit, train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

one2all ={    'A': ('A', 'ALA', 'alanine'),
              'R': ('R', 'ARG', 'arginine'),
              'N': ('N', 'ASN', 'asparagine'),
              'D': ('D', 'ASP', 'aspartic acid'),
              'C': ('C', 'CYS', 'cysteine'),
              'Q': ('Q', 'GLN', 'glutamine'),
              'E': ('E', 'GLU', 'glutamic acid'),
              'G': ('G', 'GLY', 'glycine'),
              'H': ('H', 'HIS', 'histidine'),
              'I': ('I', 'ILE', 'isoleucine'),
              'L': ('L', 'LEU', 'leucine'),
              'K': ('K', 'LYS', 'lysine'),
              'M': ('M', 'MET', 'methionine'),
              'F': ('F', 'PHE', 'phenylalanine'),
              'P': ('P', 'PRO', 'proline'),
              'S': ('S', 'SER', 'serine'),
              'T': ('T', 'THR', 'threonine'),
              'W': ('W', 'TRP', 'tryptophan'),
              'Y': ('Y', 'TYR', 'tyrosine'),
              'V': ('V', 'VAL', 'valine'),
              'X': ('X', 'GLX', 'glutaminx'),
              'Z': ('Z', 'GLI', 'glycine'),
              'J': ('J', 'NLE', 'norleucine'),
              'U': ('U', 'CYC', 'cysteinc'),
              'B':()}

aa2charges = {'K':"+",
              'R':"+",
              'H':"+",
              'D':"-",
              'E':"-",
              'A':"0",
              'N':"0",
              'C':"0",
              'Q':"0",
              'G':"0",
              'I':"0",
              'L':"0",
              'M':"0",
              'F':"0",
              'P':"0",
              'S':"0",
              'T':"0",
              'W':"0",
              'Y':"0",
              'V':"0",
              'X':"0",
              'Z':"0",
              'J':"0",
              'U':"0",
              'B':"0"}
  
aa_ls = sorted(one2all.keys())

def get_n_grams(seq, n=3):
    return list(zip(*[seq[i:] for i in range(n)]))

def get_frequencies(n_grams, pos_dict):
    freqs = np.zeros(len(pos_dict))
    value_count = Counter(n_grams)
    for v, c in value_count.most_common():
        i = pos_dict[v]
        freqs[i] = c
    
    return freqs


class BaselineModel:
    def __init__(self, model_file_path):
        self.model_file_path = model_file_path

    def vectorize_sequences(self, sequence_array):
        vectorize_on_length = np.vectorize(len)
        return np.reshape(vectorize_on_length(sequence_array), (-1, 1))

    def train(self, df_train):
        X = self.vectorize_sequences(df_train['sequence'].to_numpy())
        y = df_train['mean_growth_PH'].to_numpy()

        model = tree.DecisionTreeRegressor()
        model.fit(X, y)

        with open(self.model_file_path, 'wb') as model_file:
            pickle.dump(model, model_file)

    def predict(self, df_test):
        with open(self.model_file_path, 'rb') as model_file:
            model: tree.DecisionTreeRegressor = pickle.load(model_file)

        X = df_test['sequence'].to_numpy()
        X_vectorized = self.vectorize_sequences(X)
        return model.predict(X_vectorized)


class TreeDecisionOnCharged:
    def __init__(self, model_file_path):
        self.model_file_path = model_file_path

    def predict(self, df_test):
        
        with open(self.model_file_path, 'rb') as model_file:
            model: tree.DecisionTreeRegressor = pickle.load(model_file)

        ls_charged_3grams = [p for p in product(sorted(set(aa2charges.values())), repeat=3)]
        charged_positions = {g:i for i, g in enumerate(ls_charged_3grams)}

        df_test['charged_sequence'] = df_test['sequence'].apply(lambda seq: [aa2charges[i] for i in seq])
        df_test['charged_3_grams'] = df_test['charged_sequence'].apply(get_n_grams)
        df_test['charged_freqs'] = df_test['charged_3_grams'].apply(lambda x: get_frequencies(x, pos_dict=charged_positions))
        
        X = df_test['charged_freqs']
        X = np.array([row for row in X])
        
        return model.predict(X)


