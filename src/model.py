import numpy as np
import pandas as pd

import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

import spacy
from spacy.language import Language
from spacy.lang.en.stop_words import STOP_WORDS

from collections import Counter

from src.vectorize import *

from sklearn.model_selection import *
from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


def run_tts_model(X: np.array, y: np.array, clfs: list, clsfs: list, 
                  rs: int) -> (pd.DataFrame):
    '''Accepts X, y data as np.arrays, a list of sklearn classifiers, a 
    list of strings for the name of each classifier, and a random state.
    A model is created for each classifier by using train_test_split for
    the test sizes of 0.2, 0.4, and 0.6. The scores for each classifier
    and test size are returned in a pd.DataFrame.
    '''
    
    sizes = [0.2, 0.4, 0.6]
    clf_score_df = pd.DataFrame()
    num = 0
    for clf in clfs:
        clsf = clsfs[num]
        print(clsf)
        
        clf_df = pd.DataFrame()
        score_card = pd.DataFrame()
        for sz in sizes:
            X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=sz, random_state=rs)
        
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            accuracy = accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred)
        
            scores = [
              (clsf, 'clf'),
              (sz, 'size'),
              (f1, 'f1'), 
              (precision, 'precision'), 
              (recall, 'recall'), 
              (accuracy, 'accuracy'),
              (roc_auc, 'roc_auc'),
              (y_test, 'y_test'),
              (y_pred, 'y_pred')]
            
            for score in scores:
                score_card[score[1]] = [score[0]]
            clf_df = pd.concat([clf_df, score_card], axis=0)          
        clf_score_df = pd.concat([clf_score_df, clf_df], axis=0)
        num += 1

    return clf_score_df


def run_models(X: np.array, y: np.array, platform: str) -> (None):
    '''Accepts X, y data as np.arrays, and either 'facebook' or 'reddit'
    as a platform. Then, three sklearn classifiers are initialized:
    LogisticRegression, GaussianNB, annd SVC(probability=True). Next,
    two sets of models using train_test_split and cross-validation are 
    executed using the provided X, y data. The results of each are stored
    in the results directory.
    '''

    lr = LogisticRegression()
    nb = GaussianNB()
    svc = SVC(probability=True)
    
    clfs = [lr, nb, svc]
    clsfs = ['logistic regression', 'gaussian nb', 
             'support vector classifier']
    rs = 42
    
    tts_results = run_tts_model(X, y, clfs, clsfs, rs)
    tts_results.to_csv('results/' + platform + '/tts_results.csv', index=False)

    cv_results = run_cv_model(X, y, clfs, clsfs, rs)
    cv_results.to_csv('results/' + platform + '/cv_results.csv', index=False)
    
    return


def load_fb_data() -> (pd.DataFrame):
    '''
    '''
    lib = pd.read_csv('/Users/christineegan/spacy_workspace/facebook/facebook/fb_data/processed/lib_5k_data.csv')
    con = pd.read_csv('/Users/christineegan/spacy_workspace/facebook/facebook/fb_data/processed/con_5k_data.csv')
    data = pd.concat([lib, con], axis=0)
    drop = [col for col in data.columns if col not in ['label', 'message']]
    data = data.drop(drop, axis=1)
    nlp = spacy.load('en_core_web_md')
    data['nlp_data'] = data['message'].apply(lambda x: clean_html(x))
    data['pol'] = data['nlp_data'].apply(lambda x: analyzer.polarity_scores(x))
    data['docs'] = [nlp(doc) for doc in data['nlp_data']]
    
    return data


def execute_fb_model() -> (None):
    '''Retrieves a sample of Facebook data for liberal and conservative 
    ads. Applies vectorize data to each, then concatenates the modified
    dataframes. Unnecessary columns are dropped, and the data is divided
    into np.arrays X and y. Then, run models is applied to the resuling 
    dataframe.
    '''
    
    lib = pd.read_csv('fb_data/processed/lib_5k_data.csv')
    con = pd.read_csv('fb_data/processed/con_5k_data.csv')

    lib_vectorized_data = vectorize_data(lib)
    con_vectorized_data = vectorize_data(con)

    data = pd.concat([lib_vectorized_data, con_vectorized_data],
                     axis=0).fillna(0)

    drop = ['html', 'political', 'not_political', 'thumbnail', 
            'created_at', 'updated_at', 'lang', 'images', 
            'impressions', 'political_probability', 'targeting', 
            'suppressed', 'advertiser', 'entities',  'page',
            'lower_page', 'targetings', 'targetedness', 
            'listbuilding_fundraising_proba', 'page_id','in_payers',
            'targets', 'paid_for_by','id', 'title']

    data = data.drop(drop, axis=1)
    data.to_csv('fb_data/processed/vectorized/model_data.csv', index=False)

    data = pd.read_csv('fb_data/processed/vectorized/model_data.csv')
    y = data['label'].to_numpy()
    X = data.drop('label', axis=1).to_numpy()
    
    run_models(X, y, 'facebook')
    
    return