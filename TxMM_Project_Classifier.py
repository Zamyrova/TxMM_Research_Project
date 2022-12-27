#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 11:12:37 2022

@author: mariiazamyrova
"""
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize, sent_tokenize
import nltk
import sklearn.datasets
import sklearn.metrics as skmr
import sklearn.model_selection as skm
from sklearn.svm import SVC
from nltk import pos_tag, ngrams, FreqDist
from statistics import mean
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import string
from empath import Empath
lexicon = Empath()
from TxMM_Project_Load_Data import get_manual_labels 
nltk.download('words')
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))
POS_TAGS = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS',
 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP',
 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'TO', 'UH', 'VB', 'VBG', 
 'VBD', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WRB']

def txt_to_feature_vec(txt_raw):
    feat_vec = []
    bag_of_words = [x for x in wordpunct_tokenize(txt_raw)]
    sentences = sent_tokenize(txt_raw)
    
    chars = [c for w in bag_of_words for c in list(w)]
    char_freqs = dict.fromkeys(list('abcdefghijklmnopqrstuvwxyz'), 0)
    for c in chars:
        if c.lower() in char_freqs.keys():
            char_freqs[c.lower()] += 1
            
    punct_freqs = dict.fromkeys(string.punctuation, 0)
    for c in chars:
        if c.lower() in punct_freqs.keys():
            punct_freqs[c.lower()] += 1
            
    pos_tags = [x[1] for x in pos_tag(wordpunct_tokenize(txt_raw))]
    pos_tag_freqs = dict.fromkeys(POS_TAGS, 0)
    for p in pos_tags:
        if p in pos_tag_freqs.keys():
            pos_tag_freqs[p] += 1
    
    #length of text in words
    feat_vec.append(len(bag_of_words))
    
    #length of text in sentences
    feat_vec.append(len(sentences))
    
    #character frequencies (normalized)
    for c in char_freqs:
        feat_vec.append(char_freqs[c]/len(chars))
        
    #punctuation frequencies (normalized)
    for c in punct_freqs:
        feat_vec.append(punct_freqs[c]/len(chars))
    
    #POS tag frequencies (normalized)
    for p in pos_tag_freqs:
        feat_vec.append(pos_tag_freqs[p]/len(bag_of_words))
    
    #word emotional categories
    cats = lexicon.analyze(txt_raw, normalize=True)
    for cat in cats:
        feat_vec.append(cats[cat])
        
    return feat_vec
    #digit frequencies
    #character frequancies
    #punctuation frequency 
    #POS frequency
    #female noun frequency
    #female adjs frequency (LIWC)
    #female verb frequency
    #male noun frequency
    #male adjs frequency (LIWC)
    #male verb frequency
    
    

def load_tr_ts_data(file, df_path, df_with_dates):
    label_set = get_manual_labels(file)
    inds = [it[0] for it in label_set]
    y = [it[2] for it in label_set]
    toy_df = pd.read_csv(df_path)
    toy_df_dates = pd.read_csv(df_with_dates)
    toy_df_filt = toy_df.loc[[it not in toy_df_dates['Unnamed: 0'].to_list() for it in toy_df['Unnamed: 0'].to_list()]]['description'].to_list()
    toys_clean = [toy_df_filt[i] for i in inds]
    #toy_df_filt.loc[[it in inds for it in toy_df_filt['Unnamed: 0'].to_list()]]['description'].to_list()
    X = list(map(txt_to_feature_vec, toys_clean))
    X_train, X_test, y_train, y_test = skm.train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def classify(X_train, y_train, X_test):
    clf = SVC(kernel='linear')
    clf.fit(X_train, y_train)
    return clf.predict(X_test)

def validate(X, y):
    skf = skm.StratifiedKFold(n_splits=10)
    scores = []
    for fold_id, (train_inds, val_inds) in enumerate(skf.split(X, y)):

        # Collect the data for this train/validation split
        X_train = [X[tr_ind] for tr_ind in train_inds]
        y_train = [y[tr_ind] for tr_ind in train_inds]
        X_val = [X[val_ind] for val_ind in val_inds]
        y_val = [y[val_ind] for val_ind in val_inds]

        # Classify and add the scores to be able to average later
        y_pred = classify(X_train, y_train, X_val)
        scores.append(list(skmr.precision_recall_fscore_support(y_val, y_pred)[:3]))
    return np.mean(np.array(scores), axis=0)

def info_extractor_eval(info_pred, info_true):
    TP, TN, FP, FN = 0
    for pred, true in list(zip(info_pred, info_true)):
        if len(true)==0:
            if len(pred)==0: 
                TN += 1
            else: 
                FP += 1
        else:
            if set(true).issubset(set(pred)):
                TP += 1
            else: 
                FN += 1
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    f1 = 2*precision*recall/(precision+recall)
    return precision, recall, f1
            
    
def main(): 
    X_train, X_test, y_train, y_test = load_tr_ts_data('/Users/mariiazamyrova/Downloads/Project_manual_labels3.txt',
                                                       '/Users/mariiazamyrova/Downloads/toys_for_class.csv',
                                                       '/Users/mariiazamyrova/Downloads/toys_with_dates.csv')
    validation = validate(X_train, y_train)
    
    print(validation)
    
    
if __name__ == '__main__':
    main()