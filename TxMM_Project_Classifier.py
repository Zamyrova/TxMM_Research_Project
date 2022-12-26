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
import sklearn.metrics
import sklearn.model_selection as skm
from sklearn.svm import SVC
from nltk import pos_tag, ngrams, FreqDist
from statistics import mean
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import liwc
from TxMM_Project_Load_Data import get_manual_labels 
nltk.download('words')
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))
POS_TAGS = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS',
 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP',
 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'TO', 'UH', 'VB', 'VBG', 
 'VBD', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WRB']

def get_feature_vecs(X_raw):
    
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
    
    

def load_tr_ts_data(file, df_path):
    label_set = get_manual_labels(file)
    inds = [it[0] for it in label_set]
    y = [it[2] for it in label_set]
    toy_df = pd.read_csv(df_path)
    toys_filt = toy_df.loc[[it in inds for it in toy_df['Unnamed: 0'].to_list()]]['description'].to_list()
    X = get_feature_vecs(toys_filt)
    X_train, y_train, X_test, y_test = skm.train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, y_train, X_test, y_test
    

def main(): 
    #X_train, y_train, X_test, y_test = load_tr_ts_data('/Users/mariiazamyrova/Downloads/Project_manual_labels3.txt', '/Users/mariiazamyrova/Downloads/toys_for_class.csv')
    s = ['ant', 'aunt', 'auntie']
    parse, category_names = liwc.load_token_parser('LIWC2007_English100131.dic')
    
if __name__ == '__main__':
    main()