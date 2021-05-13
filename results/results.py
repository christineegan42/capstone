import numpy as np
import pandas as pd
import os
from os import listdir
from datetime import datetime, date


def load_master_results(source_dir):
    """Accepts a list of files to ignore and the name of
    a source directory, then:
    1. Loads all of the result csv files from the source
    directory with os.listdir().
    2. Ignores files from the ignore list.
    3. Returns concatenated dataframe with all of the results.
    """
    file_names = [f for f in listdir(source_dir)]
    file_names = [f for f in file_names
                  if f not in [f for f in file_names
                              if not f.endswith(".csv")]]
    master_results = pd.DataFrame()
    for f in file_names:
        df = pd.read_csv(source_dir + f)
        master_results = pd.concat([master_results, df], axis=0)
    master_results = master_results.sort_values(by=['clf', 'size'])
    master_results = master_results.groupby(['clf', 'size']).sum()
    return master_results


def calculate_mean_scores(source_dir):
    """Accepts a list of files to ignore and the name of
    a source directory, then:
    1. Loads all of the result csv files from the source
    directory with load_master_results().
    2. Creates a column to calculate the mean of all thes
    scores in the row for that model and test size.
    3. Resets the index and renames the columns to be more
    descriptive.
    4. Returns concatenated dataframe with all of the 
    results and the mean of all evaluation metrics.
    """
    results = load_master_results(source_dir)
    results['mean_score'] = results.mean(axis=1)
    results = results.reset_index()
    results = results.rename(columns={'clf': 'model', 
                                  'size': 'test size'})
    return results


def save_results(target_dir, results):
    """Accepts the path of a target directory as a string and
    a Pandas dataframe of model results, then:
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
        results (pd.DataFrame): a Pandas dataframe of 
        results from models.
    """
    date = (target_dir + str(datetime.now())[0:10] + '/')
    print(date)
    
    if os.path.isdir(target_dir) == False:
        os.mkdir(target_dir)
    if os.path.isdir(date) == False:
        os.mkdir(date)
        
    file_name = str(datetime.now())[11:19].replace(':', '_')  
    file_path = date + file_name + ".csv"
    results.to_csv(file_path, index=False)
    print('[*] Results saved to', file_path)
    return file_path