import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import *
from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import os
import os.path
cwd = os.getcwd()
from typing import Any

from vectorize2 import *


def eval_clf(clf: Any, clsf: str, sz: float, 
             X_train: np.array, X_test: np.array, 
             y_train: np.array, y_test: np.array) -> (list[tuple]):
    """Accepts training and validation sets of X, y data, 
    an Sci-Kit Learn classifier, the name of the classifier,
    and the test size then:
    1. Fits the training data to the classifier.
    2. Generates predictions for the validation set: y_pred.
    3. Returns a list of tuples with the name of the score paired
    with the score. 

    Args:
        clf (Any): an Sci-Kit Learn classifier
        clsf (str): the name of the classifier
        sz (float): the test size
        X_train (np.array): feature data training set
        X_test (np.array): feature data training set
        y_train (np.array): target data training set
        y_test (np.array): target data validation set

    Returns:
        (list[tuple]): 
    """    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    return [(clsf, 'clf'),
            (sz, 'size'),
            (f1_score(y_test, y_pred), 'f1'), 
            (precision_score(y_test, y_pred), 'precision'), 
            (recall_score(y_test, y_pred), 'recall'), 
            (accuracy_score(y_test, y_pred), 'accuracy'),
            (roc_auc_score(y_test, y_pred), 'roc_auc'),
            (y_pred, 'y_pred'),
            (y_test, 'y_test')]


def run_clf(clf, clsf, X, y) -> (pd.DataFrame):
    """Accepts X, y data, a list of Sci-Kit Learn classifiers,
    and a list of classifier names. Then:

    Args:
        X (np.array): feature data
        y (np.array): target data ['label']
        clf (Any): Sci-Kit Learn classifiers
        clsf (list[str]): a list of names of classifiers

    Returns:
        (pd.DataFrame): A Pandas dataframe of model evaluation
        scores for the models named in clfs. 
    """
    sizes = [0.2, 0.3, 0.4]
    clf_df = pd.DataFrame()
    for sz in sizes:
        X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=sz, random_state=42)

        scores = eval_clf(clf, clsf, sz,
                 X_train, X_test, y_train, y_test)
        
        score_card = pd.DataFrame()
        for score in scores:
            score_card[score[1]] = [score[0]]      
        clf_df = pd.concat([clf_df, score_card], axis=0)  
        
    return clf_df


def run_tts_models(X: np.array, y: np.array, clfs: list, 
                      clsfs: list) -> (pd.DataFrame):
    """Accepts X, y data, a list of Sci-Kit Learn classifiers,
    and a list of classifier names. Then:
    1. Creates a dataframe for the eval scores: clf_score_df
    2. Initializing a number to index the names in classifier: num
    3. Loops through each classifier in clfs, and:
        1. Prints the name of the classifier.
        2. Passes the clf, name, and X, y data 
        to run_clf to obtain the scores for that clf: clf_df
        3. Concats the scores for each clf clf_score_df.
        4. Adds 1 to num.
    Returns a Pandas dataframe with the scores for each clf.

    Args:
        X (np.array): feature data
        y (np.array): target data ['label']
        clfs (list): a list of Sci-Kit Learn classifiers
        clsfs (list[str]): a list of names of classifiers

    Returns:
        (pd.DataFrame): A Pandas dataframe of model evaluation
        scores for the models named in clfs. 
    """    
    
    clf_score_df = pd.DataFrame()
    num = 0
    for clf in clfs:
        print('[*]', clsfs[num], 'in progress...')
        clf_df = run_clf(clf, clsfs[num], X, y)
        clf_score_df = pd.concat([clf_score_df, clf_df])
        num += 1
        
    return clf_score_df


def run_models(X: np.array, y: np.array) -> (None):
    '''Accepts X, y data. Then:
    1. Three Sci-Kit Learn classifiers are initialized:
        1. LogisticRegression: lr 
        2. GaussianNB: gb
        3. SVC(probability=True): svc
    2. The classifiers and their names are are stored
    in a list:
        1. clfs (list[classifiers]): Sci-Kit Learn classifiers
        2. clsfs (list[str]): Names of classifiers
    3. Passes X, y, clfs, and clsfs to run_tts_models: tts_results
    4. Passes tts_results to save_results.
    '''
    lr = LogisticRegression()
    nb = GaussianNB()
    svc = SVC(probability=True)
    
    clfs = [lr, nb, svc]
    clsfs = ['Logistic Regression', 'Gaussian NB', 
             'SVC']
    
    tts_results = run_tts_models(X, y, clfs, clsfs)
    
    save_results(tts_results)
    
    return
    
    
def save_results(tts_results: pd.DataFrame) -> (None):
    """Accepts a Pandas dataframe of model results, then:
    1. Obtains the string for cwd and joins it to "results/"
    to indicate a results folder: results
    2. Concats folder to a string for the current date: date
    3. Checks to see if a results directory is present, and 
    creates one if there is not. 
    4. Checks the results directory for a date folder and
    creates one if there is not.
    5. Generates a string for the file name using the current
    time: file_name
    6. Generates a string of the file_path by concatenating
    date and file_name: file_path
    7. Saves tts_results to file_path.
    
    Args:
        tts_results (pd.DataFrame): a Pandas dataframe of 
        results from models.
    """
    results = os.path.join(cwd, "results/")
    date = results + str(datetime.now())[0:10] + '/'
    
    if os.path.isdir(results) == False:
        os.mkdir(results)
    if os.path.isdir(date) == False:
        os.mkdir(date)
        
    file_name = str(datetime.now())[11:19].replace(':', '_')  
    file_path = date +  file_name + ".csv"
    tts_results.to_csv(file_path, index=False)
    print('[*] Results saved to', file_path)
    
    return
    
    
def execute_fb_model() -> (None):
    """Loads a sample of Facebook ad data for with either liberal
    or conservative labels. A sample is taken from each class, 
    vectorized, then passed to run_models.
    """
    
    print('[*] Importing data...')
    lib = pd.read_csv(os.path.join(cwd, 
            "data/processed/fb_lib_5k_data.csv")).sample(500)
    con = pd.read_csv(os.path.join(cwd, 
            "data/processed/fb_con_5k_data.csv")).sample(500)
    
    print('[*] Vectorizing data...')
    lib_vec = vectorize(lib)
    con_vec = vectorize(con)

    print('[*] Preparing model...')
    data = pd.concat([lib_vec, con_vec], axis=0).fillna(0)
    
    drop = ['message', 'nlp_data', 'docs', 'lems', 'lem_vecs']
    data = data.drop(drop, axis=1)
    
    print('[*] Executing models...')
    y = data['label'].to_numpy()
    X = data.drop('label', axis=1).to_numpy()
    
    run_models(X, y)
    
    return  

if __name__ == '__main__':
    execute_fb_model()