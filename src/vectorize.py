import numpy as np
import pandas as pd
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

import spacy
from spacy.language import Language
from spacy.lang.en.stop_words import STOP_WORDS
nlp = spacy.load('en_core_web_md')

import os
cwd = os.getcwd()


def clean_html(raw_html: str) -> (str):
    """Accepts a string of html and applies the following
    steps to remove the html tags:
    1. Applies re.sub to the string: clean_text, pattern
    2. Checks each word for tags or extra characters/space with 
    re.sub, str.replace, strip, then adds them to a list: clean_words
    3. Joins clean_words and returns it as a string.

    Args:
        raw_html (str): a string of html.

    Returns:
        (str): a string of clean text with no html tags.
    """    
    pattern = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    clean_text = re.sub(pattern, '', raw_html)
    
    clean_words = []
    for word in clean_text.split(' '):
        w = re.sub(r'[^\w\s]',' ',word)
        w = w.replace(r'.*</p>\.', '.* ')
        clean_words.append(w.strip())

    return ' '.join(clean_words)


def filter_lem_vecs(lem_vecs: list[tuple], thresh: int) -> (list[str]):
    """Accepts a list of lemma/vector pairs and returns a list
    of lemmas for the pairs that appear more frequently than 
    indicated by thresh.

    Args:
        lem_vecs (list[tuple]): A list of tuples containing
        lemmas and their vectors.
        
        thresh (int): The minimum frequency of the lemma to
        filter the results by.

    Returns:
        (list): A list of lemmas with a frequency greater
        than thresh.
    """    
    lv = [lm[0] for lem in lem_vecs 
          for lm in lem]
    return list(set([lem.lower() 
                for lem in lv 
                if lv.count(lem) > 2]))


def extract_vecs(lem_vecs: list[tuple], target: str) -> (list[float]):
    """Accepts a list of lemma/word_vector pairs and returns
    the word vector for the pairs by which the lemma matches
    the target lemma.

    Args:
        lem_vecs (list): a list of lemma/word vector pairs.
        target (str): a lemma.

    Returns:
        list: a list of word vectors.
    """    
    return [n[1] for n in lem_vecs 
            if n[0].lower() == target]


def vectorize_df(df:pd.DataFrame, vocab:list[str]) -> (pd.DataFrame):
    """Accepts a Pandas dataframe of Facebook ad data and a list
    of filtered vocabulary and creates a feature in the dataframe
    for each word in the vocabulary.

    Args:
        df (pd.DataFrame): a Pandas dataframe.
        vocab (list[str]): a list of lemmas as strings.

    Returns:
        (pd.DataFrame): a modified Pandas dataframe containing 
        word vectors as features.
    """
    for v in vocab:
        df[v] = df['lem_vecs'].apply(lambda x: extract_vecs(x, v))
        df[v] = df[v].apply(lambda x: x[0] if len(x) > 0 else 0)
    return df


def vectorize(data):
    """Accepts a pd.DataFrame() of Facebook ad data and applies
    the following preprocessing steps:
    1. Drops unneccessary columns.
    2. Applies labels to target column: data['label']
    3. Removed unncessary HTML tags: clean_html, data['nlp_data']
    4. Calculates and creates Vader SIA column: data['pol']
    5. Creates SpaCy nlp object column: data['nlp']
    6. Lemmatizes and pre-processes corpus: data['lems']
    7. Pairs lemmas with the mean of their vectors: data['lem_vecs']
    8. Filtering the lemmas by frequency: fltd_vocab, fltr_lem_vecs
    9. Passes the data and fltd_vocab to vectorize_df.
    10.Returns a modified pd.DataFrame that includes word vectors for 
    fltd_vocab as features.
    
    Args:
        data (pd.DataFrame): a Pandas dataframe containing Facebook ad data.

    Returns:
        (pd.DataFrame): modified Pandas dataframe that includes word vectors 
                        for fltd_vocab as features.
    """
    # drop the columns not being used
    data = data.drop([col for col in data.columns 
                  if col not in ['message', 'label']], axis=1)
    
    # applying binary label
    data['label'] = data['label'].apply(lambda x: 1 if x=='lib' else 0)
    
    # removing html tags and extra characters
    data['nlp_data'] = data['message'].apply(lambda x: clean_html(x))
    
    # calculating polarity score
    data['pol'] = data['nlp_data'].apply(lambda x: analyzer.polarity_scores(x))
    data['pol'] = data['pol'].apply(lambda x: x['compound'])
    
    # creating SpaCy nlp object
    data['docs'] = [nlp(doc) for doc in data['nlp_data']]
    
    # cleaning up lemmas
    data['lems'] = [[n for n in doc 
                     if n.is_digit == False 
                     and n.is_punct == False 
                     and n.is_stop == False 
                     and len(str(n)) >= 3] 
                    for doc in data['docs']]
    
    # pairing lemmas with the mean of their vectors
    data['lem_vecs'] = [[(n.lemma_, (n.vector).mean()) 
                         for n in doc] 
                         for doc in data['lems']]
    
    # applying fltr_lem_vecs
    fltd_vocab = filter_lem_vecs(data['lem_vecs'], 2)
    
    return vectorize_df(data, fltd_vocab)