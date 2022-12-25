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
import sklearn.model_selection
from sklearn.svm import SVC
from nltk import pos_tag, ngrams, FreqDist
from statistics import mean
import matplotlib.pyplot as plt
import numpy as np
from TxMM_Project_Load_Data.py import get_manual_labels 
nltk.download('words')
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))
POS_TAGS = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS',
 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP',
 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'TO', 'UH', 'VB', 'VBG', 
 'VBD', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WRB']

