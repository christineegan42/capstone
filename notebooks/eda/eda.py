import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

import spacy
from spacy.language import Language
from spacy.lang.en.stop_words import STOP_WORDS
nlp = spacy.load("en_core_web_md")


def load_data(f_path: str, f1: str, 
              f2: str) -> (pd.DataFrame):
    """Accepts 1 filepath and 2 file names for liberal and 
    conservative Facebook ad data, then: 
    1. Reads in data using pd.read_csv as lib and con
    2. Concatenates lib and con to create data
    3. Drops irrelevant columns
    Args:
        f_path (str): the path to the directory
        f1 (str): file name of liberal data
        f2 (str): file name of conservative data
    Returns:
        (str): a string of clean text with no html tags
    """
    lib = pd.read_csv(f_path + f1)
    con = pd.read_csv(f_path + f2)
    data = pd.concat([lib, con], axis=0)
    drop_cols = [col for col in data.columns if col not in ["label", "message"]]
    data = data.drop(drop_cols, axis=1)
    return data


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
    pattern = re.compile("<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")
    clean_text = re.sub(pattern, "", raw_html)
    
    clean_words = []
    for word in clean_text.split(" "):
        w = re.sub(r"[^\w\s]"," ",word)
        w = w.replace(r".*</p>\.", ".* ")
        clean_words.append(w.strip())

    return " ".join(clean_words)


def process_data(data):
    """Accepts a pd.DataFrame() of Facebook ad data and applies
    the following preprocessing steps:
    1. Drops unneccessary columns.
    2. Applies labels to target column: data["label"]
    3. Removed unncessary HTML tags: clean_html, data["nlp_data"]
    4. Calculates and creates Vader SIA column: data["pol"]
    5. Creates SpaCy nlp object column: data["nlp"]
    6. Lemmatizes and pre-processes corpus: data["lems"]
    7. Pairs lemmas with the mean of their vectors: data["lem_vecs"]
    8. Filtering the lemmas by frequency: fltd_vocab, fltr_lem_vecs
    9. Passes the data and fltd_vocab to vectorize_df.
    10.Returns a modified pd.DataFrame that includes ["tags"] (POS tags),
    ["nes"] (named entities), and ["lens"] (message lengths) as features.
    Args:
        data (pd.DataFrame): a Pandas dataframe containing Facebook ad data.
    Returns:
        (pd.DataFrame): modified Pandas dataframe that includes word vectors 
                        for fltd_vocab as features.
    """
    # drop the columns not being used
    data = data.drop([col for col in data.columns 
                  if col not in ["message", "label"]], axis=1)
    
    # applying binary label
    data["label"] = data["label"].apply(lambda x: "Liberal" if x=="lib" 
                                        else "Conservative")
    
    # removing html tags and extra characters
    data["nlp_data"] = data["message"].apply(lambda x: clean_html(x))
    
    # calculating polarity score
    data["pol"] = data["nlp_data"].apply(lambda x: analyzer.polarity_scores(x))
    data["pol"] = data["pol"].apply(lambda x: x["compound"])
    
    # creating SpaCy nlp object
    data["docs"] = [nlp(doc) for doc in data["nlp_data"]]
    
    # cleaning up lemmas
    data["lems"] = [[n for n in doc 
                     if n.is_digit == False 
                     and n.is_punct == False 
                     and n.is_stop == False 
                     and len(str(n)) >= 3] 
                    for doc in data["docs"]]
    
    # counting pos tags
    data["tags"] = [[token.tag_ for token in doc] for doc in data["docs"]]
    
    # counting named entities
    data["nes"] = [[ent.label_ for ent in doc.ents] for doc in data["docs"]]
    
    # obtaining word count
    data["lens"] = data["nlp_data"].apply(lambda x: len(list(x)))

    return data


def ad_length_stats(data: pd.DataFrame, 
        thresh: int) -> (pd.DataFrame, pd.DataFrame):
    """Accepts a pd.DataFrame of Facebook politic ads,
    and threshold for message length as an interger then:
    1. Calculates and prints the percentage of ads with 
    lengths above (long) and below (short) the threshold.
    2. Calcualates the number of conservative and liberal
    ads that are short
    3. Calcualates the number of conservative and liberal
    ads that are long
    Args:
        data (pd.DataFrame): a pd.DataFrame of Facebook ads
        thresh (int): a threshold for ad length
    Returns:
        (None)
    """
    print('Facebook Ad Length\n', '='*40, sep='')
    short = data.loc[data['lens'] < thresh]
    long = data.loc[data['lens'] >= thresh]
    print(str(len(short)*100 / len(data)) + '% of ads are short')
    print(str(len(long)*100 / len(data)) + '% of ads are long\n')
    
    short_cons = len(short.loc[short['label'] == 'Conservative'])
    short_libs = len(short.loc[short['label'] == 'Liberal'])
    print(str(round((short_cons*100 / len(short)),2)) + '% of short ads are conservative')
    print(str(round((short_libs*100 / len(short)),2)) + '% of short ads are liberal\n')
    
    long_cons = len(long.loc[long['label'] == 'Conservative'])
    long_libs = len(long.loc[long['label'] == 'Liberal'])
    print(str(round((long_cons*100 / len(long)),2)) + '% of long ads are conservative')
    print(str(round((long_libs*100 / len(long)),2)) + '% of long ads are liberal\n')
    
    return short, long


def pol_dist_by_label(data: pd.DataFrame) -> (None):
    """Accepts a pd.DataFrame of Facebook politic ads,
    then:
    1. Applies a Seaborn theme
    2. Plots a catplot of polarity by label
    Args:
        data (pd.DataFrame): a pd.DataFrame of Facebook ads
    Returns:
        (None)
    """    
    sns.set(style="whitegrid", context="notebook", 
            rc={"grid.linewidth": 1, "font.size": 1})
    ax = sns.catplot(x="label", y="pol", kind="box", 
        data=data)
    ax.set(xlabel="Political Label", ylabel="Polarity", 
           title="Polarity by Political Label")
    
    return 


def pol_len_by_label(data: pd.DataFrame) -> (None):
    """Accepts a pd.DataFrame of Facebook politic ads,
    then:
    1. Applies a Seaborn theme
    2. Plots a catplot of polarity by message length and
    political label
    Args:
        data (pd.DataFrame): a pd.DataFrame of Facebook ads
    Returns:
        (None)
    """ 
    sns.set(style="whitegrid", context="notebook")
    ax = sns.relplot(x="pol", y="lens", hue="label",
            sizes=(40, 400), alpha=.5, palette="muted",
            height=6, data=data);
    ax.set_axis_labels("Polarity", "Message Length");
    ax.fig.suptitle("Polarity vs Message Length");
    ax.legend.set_title("Political Label")
    
    return 


def pol_by_len_dbl(short: pd.DataFrame,
                   long: pd.DataFrame) -> (None):
    """Accepts a pd.DataFrame of Facebook politic ads,
    then:
    1. Applies a Seaborn theme
    2. Plots two scatterplots as subplots for long 
    ad data and short ad data
    Args:
        data (pd.DataFrame): a pd.DataFrame of Facebook ads
    Returns:
        (None)
    """ 
    
    f, axs = plt.subplots(1, 2, figsize=(8, 4), 
            gridspec_kw=dict(width_ratios=[4, 4]), 
                      sharey=True, sharex=True)
    sns.scatterplot(x=short['lens'], y=short['pol'], hue=short['label'],
                ax=axs[0]).set_title('Polarity Distribution of Short Ads')
    axs[0].set(xlabel='Message Length', ylabel='Polarity')
    sns.scatterplot(x=long['lens'], y=long['pol'], hue=long['label'],
                ax=axs[1]).set_title('Polarity Distribution of Long Ads')
    axs[1].set(xlabel='Message Length', ylabel='Polarity')
    f.tight_layout()
    
    return