import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import *
from sklearn.metrics import *
from sklearn.naive_bayes import GaussianNB
from sklearn.inspection import permutation_importance

import os
import os.path


def calculate_importances(data: pd.DataFrame) -> (dict):
    """Accepts a pd.DataFrame as input, then:
    1. Divides the data into X, y
    2. Performs 70/30 train_test_split
    3. Fits training data to Gaussian NB classifier
    4. Calculates permutation importance given the
    classifier and text data.
    
    Args:
        data (pd.DataFrame): a Pandas dataframe containing 
        Facebook ad data.
    Returns:
        (dict): 
    """
    y = data['_label_'].to_numpy()
    X = data.drop('_label_', axis=1).to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                          test_size=0.3, random_state=42)
    clf =  GaussianNB()
    clf.fit(X_train, y_train)
    
    r = permutation_importance(clf, X_test, y_test, 
                       n_repeats=5, random_state=0)
    return r


def extract_feat_imps(i_feats: list, 
             r: dict) -> (list), (list), (list):
    """Accepts a list of word vector features, and
    a dictionary of importances, then:
    1. Applies an index to the importances to map to 
    the index of list of features
    2. Appends the feature to a list of features.
    3. Appends the importance mean for that feature
    to the list of importance means
    4. Appends the standard deviation of the importances
    for that feature to the list of standard deviations

    Args:
        data (pd.DataFrame): a Pandas dataframe containing 
        Facebook ad data.
    Returns:
        (list): list of words as features
        (list): list of importance means
        (list): list of importance STDs
    """
    features = []
    i_means = []
    i_stds = []
    for i in r.importances_mean.argsort()[::-1]:
        features.append(i_feats[i])
        i_means.append(r.importances_mean[i])
        i_stds.append(r.importances_std[i])
    return features, i_means, i_stds


def create_imp_df(features, i_means, i_stds):
    """Accepts a list of word vector features, list of 
    importance means, and list of standard deviations,
    then:
    1. Creates a pd.DataFrame
    2. Adds a column for each list
    3. Sorts the dataframe by mean

    Args:
        features (list): list of words as features
        i_means (list): list of importance means
        i_stds (list): list of importance STDs
    Returns:
        (pd.DataFrame): A Pandas dataframe of importances
        sorted in descending order by mean
    """
    imp_df = pd.DataFrame()
    imp_df['features'] = features
    imp_df['i_means'] = i_means
    imp_df['i_stds'] = i_stds
    return imp_df.sort_values(by='i_means',
                            ascending=False)


def importance_results(data: pd.DataFrame) -> (pd.DataFrame):
    """Accepts a pd.DataFrame as input, then:
    1. Calculates feature importance for data with
    calculate_importances
    2. Creates a list of features
    3. Extracts mean/std for importance of each feature
    4. Creates a pd.DataFrame of all the features and 
    importances with create_imp_df

    Args:
        data (pd.DataFrame): A Pandas dataframe of importances
        sorted in descending order by mean
    Returns:
        (pd.DataFrame)  
    """
    r = calculate_importances(data)
    i_feats = [col for col in data.columns 
               if col != "_label_"]
    features, i_means, i_stds = extract_feat_imps(i_feats, r)
    return create_imp_df(features, i_means, i_stds)


def eval_importances(data, samp_sz, num_iters):
    """Accepts a pd.DataFrame, sample size, and 
    number of iterations, then:
    
    1. Loads data from each class (libs, cons)
    2. For each specified iteration:
        -concats samples of size indicated from each class
        -calculates feature importances for that iteration
        -saves the results in the imp_results directory
    3. Extracts mean/std for importance of each feature
    4. Creates a pd.DataFrame of all the features and 
    importances with create_imp_df

    Args:
        data (pd.DataFrame): A Pandas dataframe of importances
        sorted in descending order by mean
    Returns:
        (pd.DataFrame)  
    """
    libs = data.loc[data['_label_'] == 1]
    cons = data.loc[data['_label_'] == 0]
    for i in range(num_iters):
        print("[*] Performing test: ", i)
        df = pd.concat([libs.sample(samp_sz),
                    cons.sample(samp_sz)], axis=0)
        results = importance_results(df)
    
        root_dir = "/Users/christineegan/flatiron-capstone/"
        save_results(root_dir + "importances/imp_results/", results        
    return
                     
if __name__ == "__main__":
    data = pd.read_csv('/Users/christineegan/flatiron-capstone/data/processed/fb_5k/vectorized/fb_model_vec.csv')
    eval_importances(data, samp_sz, num_iters)