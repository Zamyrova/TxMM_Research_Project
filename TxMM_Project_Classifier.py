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
    feat_vec.append(len([c for c in chars if c.isupper()])/len(chars))
    
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
        scores.append(list(skmr.precision_recall_fscore_support(y_val, y_pred))[:3])
    return np.mean(np.array(scores), axis=0)

def ablation_analysis(X, y, clf):
    groups = [[], # all features
              list(range(5)), # word features
              list(range(5, 32)),# char features
              list(range(32, 42)), # digit features
              list(range(42, 74)), # punct features
              list(range(74, 108)), # POS frequencies
              list(range(108, 302)), # emotional category frequencies
              list(range(-2, 0)) # gender ngrams
              ]
    skf = skm.StratifiedKFold(n_splits=10)
    data_split = list(enumerate(skf.split(X, y)))
    data_split = list(enumerate(skf.split(X, y)))
    overall_scores = []
    for group in groups:
        scores = []
        feature_subset = []
        for item in range(len(X)):
            feats = [X[item][i] for i in range(len(X[0])) if i not in group]
            feature_subset.append(feats)
            
        for fold_id, (train_inds, val_inds) in data_split.copy():
    
            X_train = [X[tr_ind] for tr_ind in train_inds]
            y_train = [y[tr_ind] for tr_ind in train_inds]
            X_val = [X[val_ind] for val_ind in val_inds]
            y_val = [y[val_ind] for val_ind in val_inds]
    
            # Classify and add the scores to be able to average later
            y_pred = classify(X_train, y_train, X_val, clf)
            scores.append(np.array(list(skmr.precision_recall_fscore_support(y_val, y_pred, average='micro')))[2])
    
        # Compute the averaged score
        f_score = sum([x for x in scores]) / len(scores)
        overall_scores.append(f_score)
    
    figure = plt.figure(figsize=(10,5), dpi=200)
    fig = plt.subplot()
    plt.bar(np.arange(len(overall_scores)), overall_scores, width=.5, color='blue')
    labels = ["A: None", "B: Words", "C: Characters", "D: Digits", "E: Puncuation", 
                "F: POS frequencies", "G: Emotional category frequencies", "H: Gender labels"]
    for lab in labels:
        plt.bar([0], [0], width=0, label=lab, color='blue')
    fig.set_xticks(np.arange(len(overall_scores)))
    fig.set_xticklabels(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])
    box = fig.get_position()
    fig.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    fig.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig.set_title("Ablation analysis")
    fig.set_ylabel("F-scores")
    fig.set_xlabel("Features")
    plt.savefig('Project_ablation2.png')
    
def classify_dates(X_train, y_train, X_test_file, clf):
    toy_df_dates = pd.read_csv(X_test_file)
    toy_df_descrs = toy_df_dates['description'].to_list()
    features = [txt_to_feature_vec(descr) for descr in toy_df_descrs]
    feature_analysis_dict = {}
    feature_analysis_dict['ID'] = toy_df_dates['Unnamed: 0'].to_list()
    feature_analysis_dict['Release Date'] = toy_df_dates['release_date'].to_list()
    feature_analysis_dict['Gender label'] = classify(X_train, y_train, features, clf)
    feature_analysis_dict['Text length'] = [item[0] for item in features]
    feature_analysis_dict['Text length (no stopwords)'] = [item[1] for item in features]
    feature_analysis_dict['Number of unique words'] = [item[2] for item in features]
    feature_analysis_dict['Avg word length'] = [item[4] for item in features]
    feature_analysis_dict['Avg word length (no stopwords)'] = [item[3] for item in features]
    feature_analysis_dict['Upper case characters'] = [item[5] for item in features]
    feature_analysis_dict['Digit number'] = sum([feat for item in features for i, feat in enumerate(item) if i in list(range(32, 42))])
    feature_analysis_dict['Vowels'] = sum([feat for item in features for i, feat in enumerate(item) if i in [6, 10, 14, 20, 26, 30]])
    feature_analysis_dict['Consonants'] = sum([feat for item in features for i, feat in enumerate(item) if i in [7, 8, 9, 11, 12, 13, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 27, 28, 29, 31]])
    feature_analysis_dict['Exclamation marks'] = [item[42] for item in features]
    feature_analysis_dict['Question marks'] = [item[63] for item in features]
    feature_analysis_dict['Other punctuation'] = sum([feat for item in features for i, feat in enumerate(item) if (i in list(range(42, 74)) and i!=42 and i!=63)])
    feature_analysis_dict['Adjectives'] = [item[80] for item in features]
    feature_analysis_dict['Adjectives (comparative)'] = [item[81] for item in features]
    feature_analysis_dict['Adjectives (superlative)'] = [item[82] for item in features]
    feature_analysis_dict['Nouns'] = sum([feat for item in features for i, feat in enumerate(item) if i in [85, 86]])
    feature_analysis_dict['Proper nouns'] = sum([feat for item in features for i, feat in enumerate(item) if i in [87, 88]])
    feature_analysis_dict['Posessive ending'] = [item[90] for item in features]
    feature_analysis_dict['Personal pronouns'] = [item[91] for item in features]
    feature_analysis_dict['Posessive pronouns'] = [item[92] for item in features]
    feature_analysis_dict['Verbs'] = [item[99] for item in features]
    
    for i, cat in enumerate(lexicon.analyze(toy_df_descrs[0], normalize=True).keys()):
        feature_analysis_dict[cat] = [item[108+i] for item in features]
        
    feature_analysis_dict['Female regex label'] = [item[-2] for item in features]
    feature_analysis_dict['Male regex label'] = [item[-1] for item in features]
    
    feature_df = pd.DataFrame(feature_analysis_dict)
    feature_df = feature_df.sort_values(by=['Release Date'])
    feature_df.to_csv('toys_with_dates_features.csv')
    
    average_date_feature_df = feature_df.group_by(['Release Date','Gender label']).mean()
    average_date_feature_df.to_csv('toys_with_dates_avg_features.csv')
    
    
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
    
    classify_dates(X_train, y_train, '/Users/mariiazamyrova/Downloads/toys_with_dates.csv', MLPClassifier())
    #ablation_analysis(X_train, y_train, MLPClassifier())
    
    #validation = validate(X_train, y_train, MLPClassifier())
    #print(validation)
    
    #compare_classifiers(X_train, y_train, [MLPClassifier(), MLPClassifier(activation='logistic', solver='lbfgs')])
    
    #compare_classifiers(X_train, y_train, [MLPClassifier(), KNeighborsClassifier(), SVC(), tree.DecisionTreeClassifier(), ensemble.RandomForestClassifier()])
    
    #pred_test = classify(X_train, y_train, X_test, MLPClassifier())
    
    #precision_recall_f1_test = list(skmr.precision_recall_fscore_support(y_test, pred_test))[:3]
    #print(precision_recall_f1_test)
    
    #prec, rec, f1 = info_extractor_eval('/Users/mariiazamyrova/Downloads/Project_manual_labels3.txt',
    #                                                   '/Users/mariiazamyrova/Downloads/toys_for_class.csv',
    #                                                   '/Users/mariiazamyrova/Downloads/toys_with_dates.csv')
    
    #print(prec, rec, f1)
    
    
if __name__ == '__main__':
    main()