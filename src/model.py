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


from tqdm import tqdm
from codes import *

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


def compute_hand_features(seq):

    header = ""
    features = []

    pi_dict = {
        "A" : [6.00,  2.34, 9.69, 0.0],
        "R" : [10.76, 2.17, 9.04, 12.48],
        "N" : [5.41, 2.02,  8.80, 0.0],
        "D" : [2.77, 1.88,  9.60, 3.65],
        "C" : [5.07, 1.96, 10.28, 8.18],
        "Q" : [5.65, 2.17, 9.13, 0.0],
        "E" : [3.22, 2.19, 9.67, 4.25],
        "G" : [5.97, 2.34, 9.60, 0.0],
        "H" : [7.59, 1.82, 9.17, 6.00],
        "I" : [6.02, 2.36, 9.60, 0.0],
        "L" : [5.98, 2.36, 9.60, 0.0],
        "K" : [9.74, 2.18, 8.95, 10.53],
        "M" : [5.74, 2.28, 9.21, 0.0],
        "F" : [5.48, 1.83, 9.13, 0.0],
        "P" : [6.3, 1.99, 10.60, 0.0],
        "S" : [5.68, 2.21, 9.15, 0.0],
        "T" : [5.60, 2.09, 9.10, 0.0],
        "W" : [5.89, 2.83, 9.39, 0.0],
        "Y" : [5.66, 2.20, 9.11, 10.07],
        "V" : [5.96, 2.32, 9.62, 0.0],
    }    


    # n_measurements, mean, std, ph
    pka_exp_dict = {        
        "D" : [376, 3.4892553191489357, 1.0401442098888558, 5.146728723404252],
        "E" : [410, 4.144268292682927, 0.7552699896641738, 5.235024390243901],
        "H" : [201, 6.676616915422887, 1.0302772159974134, 6.462288557213929],
        "K" : [135, 10.657481481481469, 0.7216644531849207, 7.957777777777777],
        }



    # length
    N = len(seq)

    # net charge
    n_R = seq.count("R")
    n_K = seq.count("K")
    n_H = seq.count("H")
    n_D = seq.count("D")
    n_E = seq.count("E")
    charge = n_R + n_K + n_H - n_E - n_D
    charge_noHis = charge - n_H

    # avg pKa, PI, PI+mpKa
    pi = 0.0
    pka= 0.0
    pimpka= 0.0
    ls_pi = []
    ls_pka1 = []
    ls_pka2 = []
    ls_pka3 = []
    ls_pka_exp_mean = []
    ls_pka_exp_std = []
    for aa in seq:

        if aa not in pi_dict.keys(): continue

        ls_pi.append(pi_dict[aa][0])
        ls_pka1.append(pi_dict[aa][1])
        ls_pka2.append(pi_dict[aa][2])
        ls_pka3.append(pi_dict[aa][3])
        
        if aa in pka_exp_dict.keys():
            ls_pka_exp_mean.append(pka_exp_dict[aa][1])
            ls_pka_exp_std.append(pka_exp_dict[aa][2])


    ls_pi = np.array(ls_pi)
    ls_pka1= np.array(ls_pka1)
    ls_pka2= np.array(ls_pka2)
    ls_pka3= np.array(ls_pka3)
    ls_pka_exp_mean = np.array(ls_pka_exp_mean)
    ls_pka_exp_std = np.array(ls_pka_exp_std)

    # ASP+GLU ratio
    ratio_DE = (n_D+n_E)/N

    # LYS+ARG ratio
    ratio_RK = (n_R+n_K)/N
    ratio_RKH = (n_R+n_K+n_H)/N

    return np.array([N, charge, charge*np.power(N, 0.4), charge_noHis, charge_noHis*np.power(N, 0.4), ratio_DE, ratio_RK, ratio_RKH, np.mean(ls_pi), np.mean(ls_pka1), np.mean(ls_pka2), np.mean(ls_pka3), np.mean(ls_pka_exp_mean), np.mean(ls_pka_exp_std)])


def compute_i_features(fn_fasta):

    #print(fn_fasta)

    kw = {'path': fn_fasta, 'order': 'ACDEFGHIKLMNPQRSTVWY'}
    ls = ['AAC', 'CKSAAP', 'DPC', 'DDE', #'TPC',
			  'GAAC','CKSAAGP', 'GDPC', 'GTPC',
			  #'NMBroto', 'Moran', 'Geary',
			  'CTDC', 'CTDT', 'CTDD',
			  'CTriad', 
			  #'SOCNumber', 'QSOrder',
			  #'PAAC',
			  'APAAC',
			  #'SSEC', 'DisorderC'
			]
    fastas = readFasta.readFasta(fn_fasta)
    ls_features = []
    for feature in ls:
        myFun = feature + '.' + feature + '(fastas, **kw)'
        encodings = eval(myFun)
        ls_features+=encodings[1][1:]
        

    return np.array(ls_features)

def calculate_features(df, n_features=4263):

    n_features=4235 # no ss

    print("CALCULATING FEATURE MATRIX")
    counter = 0

    for ind, row in  tqdm(df.iterrows(), total=df.shape[0]):
        seq = row['sequence']

        feature_vector = np.zeros(shape=(n_features))

        fn_fasta = make_fasta(seq, counter)

        # HAND FEATURES
        hand_features = compute_hand_features(seq)
        #print(hand_features)

        # SS FEEATURES
        #ss_features = compute_ss_features(fn_fasta)
        #print(ss_features)

        # iFeatures
        i_features = compute_i_features(fn_fasta)
        #print(i_features)

        feature_vector = np.concatenate([hand_features, i_features])
        if not os.path.exists("./features/"):
            os.system('mkdir ./features/')
        np.save("./features/seq" + str(counter+1) + ".npy", feature_vector)
        counter+=1

        os.system('rm -r ' + fn_fasta)    

    print("DONE")
    return

def make_fasta(seq, counter):
    fn_out = "./sequences/seq" + str(counter) + ".fasta"
    f = open(fn_out, "w")
    header = ">sequence" + str(counter) + '\n'
    f.write(header)
    f.write(seq+'\n')
    return fn_out

def pAA(aaL,a):
    if len(aaL) != 0:
        return np.abs(aaL-a).mean()
    else:
        return 0

def aspDistSeq(seq):
    aspP = np.array([idx for idx, item in enumerate(seq) if 'D' in item])
    gluP = np.array([idx for idx, item in enumerate(seq) if 'E' in item])
    hisP = np.array([idx for idx, item in enumerate(seq) if 'H' in item])
    lysP = np.array([idx for idx, item in enumerate(seq) if 'K' in item])
    zP = np.array([idx for idx, item in enumerate(seq) if 'Z' in item])
    bP = np.array([idx for idx, item in enumerate(seq) if 'B' in item])
    aspF = np.zeros(aspP.shape[0])

    if len(aspP) == 0:
        return 0

    for i,a in enumerate(aspP):
        ad = np.abs(aspP-a).mean()
        gd = pAA(gluP,a)
        hd = -pAA(hisP,a)
        ld = -pAA(lysP,a)
        zd = 0.5*pAA(zP,a)
        bd = 0.5*pAA(bP,a)
        aspF[i] = ad+gd+hd+ld+zd+bd

    return aspF.mean()
    


def get_n_grams(seq, n=3):
    return list(zip(*[seq[i:] for i in range(n)]))

def get_frequencies(n_grams, pos_dict):
    freqs = np.zeros(len(pos_dict))
    value_count = Counter(n_grams)
    for v, c in value_count.most_common():
        i = pos_dict[v]
        freqs[i] = c
    
    return freqs

def count_charged_amino_acids(seq):
    return [seq.count(aa) for aa in ['R', 'H', 'K', 'D', 'E']]


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
        df_test['charged_count'] = df_test['sequence'].apply(count_charged_amino_acids)


        df_test['aspF'] = df_test['sequence'].apply(aspDistSeq)

        calculate_features(df_test)

        # HERE SHOULD BE CONCATENATION CODE



        X_freq = df_test['charged_freqs']
        X_freq = np.array([row for row in X_freq])

        X_counts = df_test['charged_count'].values
        X_counts = np.array([row for row in X_counts])
        
        X = np.hstack([X_freq, X_counts])

        return model.predict(X)


