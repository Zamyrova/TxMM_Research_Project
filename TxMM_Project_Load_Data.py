#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 11:40:17 2022

@author: mariiazamyrova
"""
import pandas as pd
import numpy as np
from html.parser import HTMLParser
from nltk.corpus import wordnet as wn
import regex as re
from multiprocessing import Pool
import inflect
import random
inflecter = inflect.engine()

kid_nouns = ['kid', 'child', 'youngster', 'baby', 'teen', 'infant', 'toddler',
             'little one', 'year olds', 'young']
boy_nouns = list(map(lambda x: x.name()[:-5], wn.synsets('boy')[:-1]))
girl_nouns = list(map(lambda x: x.name()[:-5],wn.synsets('girl')[:-1]))

female_nouns = ['female', 'woman', 'princess', 'mother', 'mom', 'her',
               'aunt', 'grandmother', 'grandmom', 'sister', 'lady',
               'mommy', 'mum', 'mama', 'wife', 'niece', 'miss', 'gal']+girl_nouns
female_nouns = [w.replace('_', ' ') for w in female_nouns]
female_nouns += [inflecter.plural(w) for w in female_nouns]
male_nouns = ['male', 'man', 'prince', 'father', 'dad', 'boyfriend', 'him',
             'uncle', 'grandfather', 'granddad', 'brother', 'bro', 'gentleman',
             'daddy', 'papa', 'husband', 'nephew', 'mister', 'guy']+boy_nouns
male_nouns = [w.replace('_', ' ') for w in male_nouns]
male_nouns += [inflecter.plural(w) for w in male_nouns]
neutral_nouns = kid_nouns+['person', 'adult', 'cousin', 'age', 'family',
                          'parent', 'grandparent', 'sibling',
                          'spouse', 'friend', 'fan', 'fiance', 'everyone']
neutral_nouns = [w.replace('_', ' ') for w in neutral_nouns]
neutral_nouns += [inflecter.plural(w) for w in neutral_nouns]

female_adjs = ['feminine', 'womanly', 'girly']
male_adjs = ['masculine', 'manly']
neutral_adjs = ['unisex', 'gender-neutral']

all_nouns = female_nouns+male_nouns+neutral_nouns
all_adjs = female_adjs+male_adjs+neutral_adjs

# Based on an example from python documentation
# https://docs.python.org/3/library/html.parser.html#html.parser.HTMLParser
class MyHTMLParser(HTMLParser):
    
    text_parts = []
    
    def handle_starttag(self, tag, attrs):
        return tag, attrs

    def handle_endtag(self, tag):
        return tag

    def handle_data(self, data):
        self.text_parts.append(data)
        return data

    def handle_comment(self, data):
        return data

    def handle_entityref(self, name):
        c = chr(name2codepoint[name])
        return c

    def handle_charref(self, name):
        if name.startswith('x'):
            c = chr(int(name[1:], 16))
        else:
            c = chr(int(name))
        return c

    def handle_decl(self, data):
        return data
    
    def get_text(self, data):
        self.text_parts = []
        self.feed(data)
        return self.text_parts

parser = MyHTMLParser()

# Load data from json to DataFrame
def load_data(reader):
    data = [item for item in reader]
    return pd.concat(data)
 
# Extract date from detail column
def date_from_detail(detail):
    if pd.notna(detail):
        if ' Date first listed on Amazon:' in detail.keys():
            return detail[' Date first listed on Amazon:'][-4:]
        else: return np.nan
    else:
        return np.nan

# Extract date from Date column
def date_from_date(date):
    months = r'(?:\bAugust|September|October|November|December|January|February|March|April|May|June|July\b)'
    pattern = ''.join([months, r'\s[0-9]{1,2}\,\s[0-9]{4}'])
    if pd.notna(date):
        if re.fullmatch(pattern, date): return date[-4:]
        else: return np.nan
    else:
        return np.nan

# Combine dates found in the detail and date columns into a single column   
def date_merger(col1, col2):
    if pd.isna(col2): return col1
    else: return col2

# Convert HTML markup to raw text with no markup tags    
def html_to_text(val, as_ar = False, sep = ' '):
    if as_ar:
        return [' '.join(parser.get_text(v)) for v in val]
    else:
        return sep.join([' '.join(parser.get_text(v)) for v in val])

# Clean the data and format the DataFrame    
def preprocess_data(df):
    # set all empty details values to NaN for later filtering
    df['details'].iloc[[not bool(det) for det in df['details'].values]] = np.nan
    # same for description
    df['description'].iloc[[not bool(descr) for descr in df['description'].values]] = np.nan
    # same for feature
    df['feature'].iloc[[not bool(feat) for feat in df['feature'].values]] = np.nan
    # same for title
    df['title'].iloc[[not bool(title) for title in df['title'].values]] = np.nan
    # same for category
    df['category'].iloc[[not bool(cat) for cat in df['category'].values]] = np.nan

    df_filt = df[pd.notna(df['description']) & pd.notna(df['category'])]
    df_filt = df_filt[pd.notna(df_filt['feature']) & pd.notna(df_filt['title'])]
    df_filt = df_filt.drop(['tech1', 'tech2', 'also_buy', 'also_view', 'main_cat', 'fit'], axis=1)
    df_filt = df_filt.drop(['imageURL', 'imageURLHighRes', 'rank', 'similar_item'], axis=1)

    df_final = df_filt.drop(['price'], axis=1)
    df_descr = df_final.description.to_list()
    split_length = round(len(df_descr)/4)
    inds = list(range(0, len(df_descr), split_length))+[len(df_descr)]
    
    df_descr_split = [df_descr[inds[i]:inds[i+1]] for i in range(len(inds[:-1]))]
    df_final_descr = []
    for split in df_descr_split:
        pool = Pool()
        df_final_descr += pool.map(html_to_text, split)
        pool.close()
    
    df_final['description'] = df_final_descr
    df_final = df_final.loc[[bool(re.search(r'\w+', d)) for d in df_final['description'].to_list()]]
    df_final = df_final.drop_duplicates(keep='first', subset=['brand', 'description', 'title'])
    
    df_with_dates = df_final
    df_with_dates['date2'] = df_with_dates.details.map(date_from_detail)
    df_with_dates['date'] = df_with_dates.date.map(date_from_date)
    df_with_dates['release_date'] = list(map(date_merger, df_with_dates['date'], df_with_dates['date2']))
    df_with_dates = df_with_dates.drop(['date', 'date2', 'details'], axis=1)
    df_with_dates = df_with_dates[pd.notna(df_with_dates['release_date'])]
    df_with_dates = df_with_dates.sort_values(by=['release_date'])
    
    df_final = df_final.drop(['date', 'date2', 'details', 'release_date'], axis=1)
    
    return df_final, df_with_dates

# Extract gender labels from text
def get_gender_labels(text):
    # 0 neutral
    # 1 female
    # -1 male
    
    f_pat = r'(?:\b'+'|'.join(female_nouns)+'\b)'
    m_pat = r'(?:\b'+'|'.join(male_nouns)+'\b)'
    n_pat = r'(?:\b'+'|'.join(neutral_nouns)+'\b)'
    
    f_and_m = r'(?:'+f_pat+r's?(?![^s])\s*(?:(?:\band\s|or\s|\&\s\b)|\s*\W)\s*'+m_pat+r's?(?![^s])\s?[\W|\s])'
    m_and_f = r'(?:'+m_pat+r's?(?![^s])\s*(?:(?:\band\s|or\s|\&\s\b)|\s*\W)\s*'+f_pat+r's?(?![^s])\s?[\W|\s])'
    
    filter_pattern_f = r'(?:\s*(?:(?<!\b\snot\s\b)\bfor\b)\s(?:\w*\s)*(?:\s*'+f_pat+r's?(?![^s])\s?[\\|&]?[\W|\s]*)+)+'
    filter_pattern_m = r'(?:\s*(?:(?<!\b\snot\s\b)\bfor\b)\s(?:\w*\s)*(?:\s*'+m_pat+r's?(?![^s])\s?[\\|&]?[\W|\s]*)+)+'
    filter_pattern_n = r'\s*(?:\bfor\b)\s(?:\w*\s)*(?:'+f_and_m+r'|'+m_and_f+r')' 
    patterns = [f_pat, m_pat, n_pat]
    
    return (re.findall(f_pat, text, flags=re.IGNORECASE, overlapped=True), 
            re.findall(m_pat, text, flags=re.IGNORECASE, overlapped=True),
            re.findall(n_pat, text, flags=re.IGNORECASE, overlapped=True))
    
# Create the set of data points for manual labeling
def get_label_set(df, df_dates, size=50):
    label_set = get_manual_labels('Project_manual_labels.txt')
    indecies = [it[0] for it in label_set]
    def check_non_neutral(matches, gender = 1):
        words = []
        if gender==1:
            words = female_nouns
        else: 
            words = male_nouns
        for match in matches:
            if (set(re.sub(r'\,|\.', '', match).split(' ')) & set(words)):
                return False
        return True
    
    df_filt = df.loc[[it not in df_dates['Unnamed: 0'].to_list() for it in df['Unnamed: 0'].to_list()]]
    
    f_pat = r'(?:\b'+'|'.join(female_nouns)+'\b)'
    m_pat = r'(?:\b'+'|'.join(male_nouns)+'\b)'
    n_pat = r'(?:\b'+'|'.join(neutral_nouns)+'\b)'
    
    f_and_m = r'(?:'+f_pat+r'\s*(?:(?:\band\s|or\s|\&\s\b)|\s*\W)\s*'+m_pat+r'[\W|\s])'
    m_and_f = r'(?:'+m_pat+r'\s*(?:(?:\band\s|or\s|\&\s\b)|\s*\W)\s*'+f_pat+r'[\W|\s])'
    
    filter_pattern_f = r'(?:\s*(?:(?<!\b\snot\s\b)\bfor\b)\s(?:\w*\s)*(?:\s*'+f_pat+r's?\s?[\\|&]?[\W|\s])+)+'
    filter_pattern_m = r'(?:\s*(?:(?<!\b\snot\s\b)\bfor\b)\s(?:\w*\s)*(?:\s*'+m_pat+r's?\s?[\\|&]?[\W|\s])+)+'
    filter_pattern_n = r'\s*(?:\bfor\b)\s(?:\w*\s)*(?:'+f_and_m+r'|'+m_and_f+r')' 
    patterns = [filter_pattern_m, filter_pattern_f, filter_pattern_n]

    df_descrs_shuffle = list(enumerate([descr for descr in df_filt['description'].to_list() if type(descr)==str]))
    random.shuffle(df_descrs_shuffle)
    
    df_descrs_to_label = []
    i=0
    for ind, pat in enumerate(patterns, start=1):
        i=0
        while len(df_descrs_to_label)/size!=ind and i<len(df_descrs_shuffle):
            out=[]
            try:
                out = re.findall(pat, df_descrs_shuffle[i][1], flags=re.IGNORECASE, overlapped=True)
            except TypeError:
                print(df_descrs_shuffle[i][1])
            if df_descrs_shuffle[i] not in df_descrs_to_label and df_descrs_shuffle[i][0] not in indecies:
                if ind < 3:
                    out_neut = re.findall(filter_pattern_n, df_descrs_shuffle[i][1], flags=re.IGNORECASE, overlapped=True)
                    if len(out_neut)==0 and len(out)!=0: 
                        df_descrs_to_label.append(df_descrs_shuffle[i])
                        if i%5==0 and len(df_descrs_to_label)>0:
                            print(df_descrs_to_label[-1][0], out)
                elif ind==3:
                    out_f = re.findall(filter_pattern_f, df_descrs_shuffle[i][1], flags=re.IGNORECASE, overlapped=True)
                    out_m = re.findall(filter_pattern_m, df_descrs_shuffle[i][1], flags=re.IGNORECASE, overlapped=True)
                    if (len(out_f)!=0 and len(out_m)!=0) or len(out)!=0:
                        df_descrs_to_label.append(df_descrs_shuffle[i])
                    if i%5==0 and len(df_descrs_to_label)>0:
                        print(df_descrs_to_label[-1][0], out)
                
            i+=1
    return df_descrs_to_label

# Extract the manual gender labels from the txt file as a list
def get_manual_labels(file):
    with open(file, 'r') as f:
        labeled_data = f.read().split('\n')
    def string_to_tuple(item):
        no_brackets_txt = (item[1:])[:-1]
        ind, remainder = re.split(r'\,\s', no_brackets_txt, maxsplit=1)
        ind = int(ind)
        class_label = None
        if remainder[-2:]=='-1':
            class_label, remainder =  [int(remainder[-2:]), remainder[:-6]]
        else:
            class_label, remainder =  [int(remainder[-1]), remainder[:-5]]
        regex_label_and_remainder = re.split(r'(?:\[\')', remainder)
        if len(regex_label_and_remainder)>1:
            regex_label, remainder = regex_label_and_remainder[1], regex_label_and_remainder[0]
            regex_label = re.split(r'\'\,\s\'', regex_label)
        else:
            regex_label, remainder = [], regex_label_and_remainder[0]
        
        return (ind, regex_label, class_label)
    
    return list(map(string_to_tuple, labeled_data))

def main():

    # Read the downloaded json dataset into a Pandas DataFrame
    file_meta="/Users/mariiazamyrova/Downloads/meta_Toys_and_Games.json"
    meta_reader = pd.read_json(file_meta, chunksize=10000, lines=True)
    
    toy_meta_df = load_data(meta_reader)
    
    # Clean the data and distinguish the data to be used for classifier training data 
    # and data to be used for analyzing how language changes through the years
    toys_for_class, toys_with_dates = preprocess_data(toy_meta_df)
    
    # Store the obtained DataFrames as CSV files
    toys_for_class.to_csv('/Users/mariiazamyrova/Downloads/toys_for_class.csv')
    toys_with_dates.to_csv('/Users/mariiazamyrova/Downloads/toys_with_dates.csv')

    # Read the DataFrames from CSV
    toys_for_class = pd.read_csv('/Users/mariiazamyrova/Downloads/toys_for_class.csv')
    toys_with_dates = pd.read_csv('/Users/mariiazamyrova/Downloads/toys_with_dates.csv')
    
    # Extract the set of data points to be manually labeled
    label_set = get_label_set(toys_for_class, toys_with_dates, size=100)
    
    # Write the data item to a text file to label them by hand
    with open('Project_manual_labels.txt', 'w') as f:
        for l in label_set:
            f.write(str(l)+'\n')
     
if __name__ == '__main__':
    main()