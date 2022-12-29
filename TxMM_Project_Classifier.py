#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 11:12:37 2022

@author: mariiazamyrova
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize, sent_tokenize
import nltk
import sklearn.datasets
import sklearn.metrics as skmr
import sklearn.model_selection as skm
from sklearn.svm import SVC, LinearSVC
from sklearn.feature_selection import SelectFromModel
from nltk import pos_tag, ngrams, FreqDist
from statistics import mean
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import string
from empath import Empath
import re
from sklearn import tree, ensemble
lexicon = Empath()
from TxMM_Project_Load_Data import get_manual_labels, get_gender_label, get_gender_labels
nltk.download('words')
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))
POS_TAGS = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS',
 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP',
 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'TO', 'UH', 'VB', 'VBG', 
 'VBD', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WRB']

def txt_to_feature_vec(txt_raw):
    txt_raw = re.sub(r'(?<=[\w|\W|\s])\\n(?=[\w|\W|\s])|\s\s+', ' ', txt_raw)
    txt_raw = re.sub(r'\\\'s', r'\'s', txt_raw)
    female_labels, male_labels, neutral_labels = get_gender_labels(txt_raw)
    female_labels = set([re.sub(r'\s', '', lab) for lab in female_labels])
    male_labels = set([re.sub(r'\s', '', lab) for lab in male_labels])
    neutral_labels = set([re.sub(r'\s', '', lab) for lab in neutral_labels])
    feat_vec = []
    bag_of_words = [x for x in wordpunct_tokenize(txt_raw)]
    words_no_stop = [x for x in bag_of_words if x.lower() not in stop_words]
    words_stop = [x for x in bag_of_words if x.lower() in stop_words]
    sentences = sent_tokenize(txt_raw)
    
    chars = [c for w in bag_of_words for c in list(w)]
    char_freqs = dict.fromkeys(list('abcdefghijklmnopqrstuvwxyz'), 0)
    for c in chars:
        if c.lower() in char_freqs.keys():
            char_freqs[c.lower()] += 1
          
    digit_freqs = dict.fromkeys(list('0123456789'), 0)
    for c in chars:
        if c.lower() in digit_freqs.keys():
            digit_freqs[c.lower()] += 1
            
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
    
    #length of text in words without stopwords
    feat_vec.append(len(words_no_stop))
    
    #ratio of stop words
    #feat_vec.append(len(words_stop)/len(bag_of_words))
    
    #number of unique words/ vocab richness
    feat_vec.append(len(list(set(bag_of_words))))
    
    #number of unique words no stopwords 
    #feat_vec.append(len(list(set(words_no_stop))))
    
    #length of text in sentences
    #feat_vec.append(len(sentences))
    
    #average word length per whole text with no stop words 
    avg_word_len_no_stop = mean([len(w) for w in words_no_stop])
    feat_vec.append(avg_word_len_no_stop)
    
    #average word length per whole text with stop words 
    avg_word_len = mean([len(w) for w in bag_of_words])
    feat_vec.append(avg_word_len)
    
    #number of upper characters
    feat_vec.append(len([c for c in chars if c.isupper()]))
    
    #character frequencies (normalized)
    for c in char_freqs:
        feat_vec.append(char_freqs[c]/len(chars))
        
    #digit frequencies (normalized)
    
    for c in digit_freqs:
        feat_vec.append(digit_freqs[c]/len(chars))
    
        
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
        
    #number of female category indicators
    feat_vec.append(len(female_labels))
    
    #number of male category indicators
    feat_vec.append(len(male_labels))
    
    #number of neutral category indicators
    #feat_vec.append(len(neutral_labels))
        
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
    #label_set = [it for it in label_set if it[2]!=2]
    for it in range(len(label_set)):
        if label_set[it][2] == 2:
            label_set[it] = (label_set[it][0], label_set[it][1], 0)
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

def classify(X_train, y_train, X_test, clf):
    #clf = MLPClassifier()
    #clf = KNeighborsClassifier(n_neighbors=3)
    #clf = SVC(kernel='linear')
    clf.fit(X_train, y_train)
    return clf.predict(X_test)

def compare_classifiers(X, y, clfs):
    skf = skm.StratifiedKFold(n_splits=10)
    data_split = list(enumerate(skf.split(X, y)))
    for clf in clfs:
        scores = []
        for fold_id, (train_inds, val_inds) in data_split.copy():
    
            # Collect the data for this train/validation split
            X_train = [X[tr_ind] for tr_ind in train_inds]
            y_train = [y[tr_ind] for tr_ind in train_inds]
            X_val = [X[val_ind] for val_ind in val_inds]
            y_val = [y[val_ind] for val_ind in val_inds]
    
            # Classify and add the scores to be able to average later
            y_pred = classify(X_train, y_train, X_val, clf)
            scores.append(np.array(list(skmr.precision_recall_fscore_support(y_val, y_pred)))[:3]) 
        print(np.mean(np.array(scores), axis=0))
    
    

def validate(X, y, clf = SVC(kernel='linear')):
    skf = skm.StratifiedKFold(n_splits=10)
    scores = []
    for fold_id, (train_inds, val_inds) in enumerate(skf.split(X, y)):

        # Collect the data for this train/validation split
        X_train = [X[tr_ind] for tr_ind in train_inds]
        y_train = [y[tr_ind] for tr_ind in train_inds]
        X_val = [X[val_ind] for val_ind in val_inds]
        y_val = [y[val_ind] for val_ind in val_inds]

        # Classify and add the scores to be able to average later
        y_pred = classify(X_train, y_train, X_val, clf)
        scores.append(np.array(list(skmr.precision_recall_fscore_support(y_val, y_pred)))[:3])
    return np.mean(np.array(scores), axis=0)

def info_extractor_eval(file, df_path, df_with_dates):
    label_set = get_manual_labels(file)
    inds = [it[0] for it in label_set]
    info_true = [it[1] for it in label_set]
    
    toy_df = pd.read_csv(df_path)
    toy_df_dates = pd.read_csv(df_with_dates)
    toy_df_filt = toy_df.loc[[it not in toy_df_dates['Unnamed: 0'].to_list() for it in toy_df['Unnamed: 0'].to_list()]]['description'].to_list()
    toys_clean = [toy_df_filt[i] for i in inds]
    info_pred = get_gender_label(toys_clean)
    print(info_pred)
    TP, TN, FP, FN = np.zeros((4, ))
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
    validation = validate(X_train, y_train, tree.DecisionTreeClassifier())#, clf = Pipeline([
  #('feature_selection', SelectFromModel(LinearSVC())),
  #('classification', SVC(kernel='linear'))
#]))
    print(validation)
    
    #compare_classifiers(X_train, y_train, [MLPClassifier(), KNeighborsClassifier(n_neighbors=3), SVC(kernel='linear')])
    
    #pred_test = classify(X_train, y_train, X_test, MLPClassifier())
    
    #precision_test, recall_test, f1_test = np.array(list(skmr.precision_recall_fscore_support(y_test, pred_test)))[:3]
    
    #print(precision_test, recall_test, f1_test)
    
    #prec, rec, f1 = info_extractor_eval('/Users/mariiazamyrova/Downloads/Project_manual_labels3.txt',
    #                                                   '/Users/mariiazamyrova/Downloads/toys_for_class.csv',
    #                                                   '/Users/mariiazamyrova/Downloads/toys_with_dates.csv')
    
    #print(prec, rec, f1)
    
    
if __name__ == '__main__':
    main()