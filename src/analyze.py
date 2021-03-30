import numpy as np
import pandas as pd

import spacy
from spacy.language import Language
from spacy.lang.en.stop_words import STOP_WORDS

import warnings
warnings.filterwarnings('ignore')

from collections import Counter

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

from src.vectorize import *


def count_data(docs: list, thresh: int) -> (list):
    '''Accepts a list of docs, and frequency threshold
    and returns only the items in the data that occur
    more frequently than the threshold.
    '''
    docs_flat = [d.lower() for doc in docs for d in doc]
    docs_counted = Counter(docs_flat).most_common()
    return [d for d in docs_counted if d[1] >= thresh]


def analyze_data(docs: list) -> (dict):
    '''Accepts a list of docs and removes any html tags
    from the text. Then, a spaCy pretrained model is loaded
    and each doc is transformed into a spaCy nlp object. 
    The lemmas for each token are extracted and the list is
    cleaned and filtered. Features for polarity, words-per-doc,
    trim vocab, named entities, named entity labels, and
    pos tags and retuns a dictionary with all of those features.'''
  
    nlp_data = data.apply(lambda x: clean_html(x))
    nlp = spacy.load('en_core_web_md')
    docs = [nlp(doc) for doc in nlp_data]
    lemmas = [[n.lemma_ for n in doc] for doc in docs]
    vocab = [n.lemma_ for doc in docs 
        for n in doc 
        if n.is_digit == False
        and n.is_punct == False
        and n.is_stop ==False
        and len(str(n)) >= 3]

    trim_vocab = list(set([f.lower() 
                            for f in filter_vocab(vocab, 10)]))
    
    ne = [[ent.text for ent in doc.ents] for doc in docs]
    ne_labels = [[ent.label_ for ent in doc.ents] for doc in docs]
    tags = [[n.pos_ for n in doc] for doc in docs]
    
    feature_dict = {}
    feature_dict['pol'] = nlp_data.apply(lambda x: analyzer.polarity_scores(x))
    feature_dict['wpd'] = nlp_data.apply(lambda x: len(x))
    feature_dict['trim_vocab'] = trim_vocab
    features = [(ne, 'named_entities', 10),
                (ne_labels, 'entity_labels', 0), 
                (tags, 'pos_tags', 0)]
    
    for f in features:
        feature_dict[f[1]] = count_data(f[0], f[2])
    
    return feature_dict
