import numpy as np
import pandas as pd

import re

import spacy
from spacy.language import Language
from spacy.lang.en.stop_words import STOP_WORDS

from collections import Counter

from src.analyze import *
from src.vectorize import *
from src.model import *

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

import gensim
import gensim.corpora as corpora

def split_data(data: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    '''A helper function to divide a pd.DataFrame by label, returning
    two pd.DataFrames divided by label.
    '''
    lib_data = data[data['label'] == 'lib']
    con_data = data[data['label'] == 'con']
    
    return lib_data, con_data


def run_lda_model(docs: list, n_topics: int, n_passes: int, n_words: int, 
                  model_name: str) -> (list):
    '''Accepts a list of spacy docs and constructs a gensim dictionary and
    corpus for an LDA (topic) model. The parameters for number of topics,
    passes and words are applied. Model is returned. 
    '''
    dictionary = corpora.Dictionary(docs)
    corpus = [dictionary.doc2bow(text) for text in docs]

    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=n_topics, 
                                               id2word=dictionary, passes=15)
    return  ldamodel


def plot_polarity_hist(lib_features: pd.DataFrame, 
                       con_features: pd.DataFrame) -> (None):
    '''Accepts liberal features and conservative features as pd.DataFrame
    and plots a histogram to visualize differences in polarity between
    the groups.
    '''
    
    lib_pol = pd.DataFrame(lib_features['pol']).rename(columns={'message': 'pol'})
    lib_pol['label'] = lib_pol['pol'].apply(lambda x: re.sub(str(x), 'lib', str(x)))
    con_pol = pd.DataFrame(con_features['pol']).rename(columns={'message': 'pol'})
    con_pol['label'] = con_pol['pol'].apply(lambda x: re.sub(str(x), 'con', str(x)))
    pol_df = pd.concat([lib_pol, con_pol], axis=0)
    pol_df['pol'] = pol_df['pol'].apply(lambda x: x['compound'])
    fig = px.histogram(pol_df, x='pol', color='label', template='plotly_white')
    fig.show()
    
    return


def get_wpd(lib_features: pd.DataFrame, 
                       con_features: pd.DataFrame) -> (None):
    '''Accepts liberal features and conservative features as pd.DataFrame
    and returns the average number of words per document.
    '''
    lib_wpd_mean = lib_features['wpd'].mean()
    print('')
    print('liberal words per doc:', round(lib_wpd_mean, 1))
    con_wpd_mean = con_features['wpd'].mean()
    print('conservative words per doc:', round(con_wpd_mean, 1))
    
    return


def plot_wpd_hist(lib_features: pd.DataFrame, 
                       con_features: pd.DataFrame) -> (None):
    '''Accepts liberal features and conservative features as pd.DataFrame
    and returns a histogram to visualize the number of words per document
    for each group.
    '''
    lib_wpd = pd.DataFrame(lib_features['wpd']).rename(columns={'message': 'wpd'})
    lib_wpd['label'] = lib_wpd['wpd'].apply(lambda x: re.sub('\d.*', 'lib', str(x)))
    con_wpd= pd.DataFrame(con_features['wpd']).rename(columns={'message': 'wpd'})
    con_wpd['label'] = con_wpd['wpd'].apply(lambda x: re.sub('\d.*', 'con', str(x)))
    wpd_df = pd.concat([lib_wpd, con_wpd], axis=0)
    wpd_df = wpd_df[wpd_df['wpd'] <= 600]
    fig = px.histogram(wpd_df, x='wpd', color='label', template='plotly_white')
    fig.show()

    return wpd_df['wpd'].describe()


def get_ents(lib_features: pd.DataFrame, 
                       con_features: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    '''Accepts liberal features and conservative features as pd.DataFrame
    and returns a pd.DataFrame for each containing the words with frequences between
    150 and 250.
    '''
    lib_ents = pd.DataFrame(lib_features['named_entities'], columns=['ents', 'freq'])
    lib_ents = lib_ents[lib_ents['freq'] >= 150]
    lib_ents = lib_ents[lib_ents['freq'] <= 250].sort_values(by='freq')

    con_ents = pd.DataFrame(con_features['named_entities'], columns=['ents', 'freq'])
    con_ents = con_ents[con_ents['freq'] >= 150]
    con_ents = con_ents[con_ents['freq'] <= 250].sort_values(by='freq')
    
    return lib_ents, con_ents


def plot_ent_bar(lib_ents: pd.DataFrame, 
                       con_ents: pd.DataFrame) -> (None):
    '''Accepts liberal features and conservative features as pd.DataFrame
    and returns a bar plot to visualize the distribution of named entities 
    passed by get_ents for each group.
    '''
    fig = go.Figure()
    fig.add_trace(go.Bar(y=lib_ents['ents'],
                x=lib_ents['freq'],
                name='Liberal',
                orientation='h'
                ))
    fig.add_trace(go.Bar(y=con_ents['ents'],
                x=con_ents['freq'],
                name='Conservative',
                orientation='h'
                ))

    fig.update_layout(
    title='Entity Distribution',
    xaxis_tickfont_size=14,
    yaxis=dict(
        title='Frequency',
        titlefont_size=16,
        tickfont_size=14,
    ),
    legend=dict(
        xanchor='left',
        x=1,
        y=1.0,
        yanchor='top',

        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    ),
    template='plotly',
    barmode='group',
    bargap=0.15,
    bargroupgap=0.1
    )
    fig.show()
    

def get_els(lib_features: pd.DataFrame, 
                       con_features: pd.DataFrame) -> (None):
    '''Accepts liberal features and conservative features as pd.DataFrame
    and returns a pd.DataFrame for each containing the entity labels with 
    frequencies between 150 and 250.
    '''
    lib_els = pd.DataFrame(lib_features['entity_labels'], columns=['label', 'freq'])
    lib_els = lib_els[lib_els['freq'] >= 500]

    con_els = pd.DataFrame(con_features['entity_labels'], columns=['label', 'freq'])
    con_els = con_els[con_els['freq'] >= 500]
    
    return lib_els, con_els


def plot_els_bar(lib_els: pd.DataFrame, con_els: pd.DataFrame) -> (None):
        '''Accepts liberal features and conservative features as pd.DataFrame
    and returns a bar plot to visualize the distribution of named entity labels 
    passed by get_els for each group.
    '''
    fig = go.Figure()
    fig.add_trace(go.Bar(x=lib_els['label'],
                y=lib_els['freq'],
                name='Liberal'
                ))
    fig.add_trace(go.Bar(x=con_els['label'],
                y=con_els['freq'],
                name='Conservative'
                ))

    fig.update_layout(
        title='Entity Label Distribution',
        xaxis_tickfont_size=14,
        yaxis=dict(
            title='Frequency',
            titlefont_size=16,
            tickfont_size=14,
        ),
    legend=dict(
        xanchor='left',
        x=1,
        y=1.0,
        yanchor='top',

        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
        ),
        template='plotly_white',
        barmode='group',
        bargap=0.15, # gap between bars of adjacent location coordinates.
        bargroupgap=0.1 # gap between bars of the same location coordinate.
    )
    fig.show()
    
    return

    
def get_tags(lib_features: pd.DataFrame, 
                  con_features: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    '''Accepts liberal features and conservative features as pd.DataFrame
    and returns a pd.DataFrame for each containing the pos_tags with 
    frequencies greater than 500.
    '''
    lib_tags = pd.DataFrame(lib_features['pos_tags'], columns=['tag', 'freq'])
    lib_tags = lib_tags[lib_tags['freq'] >= 500]

    con_tags = pd.DataFrame(con_features['pos_tags'], columns=['tag', 'freq'])
    con_tags = con_tags[con_tags['freq'] >= 500]
    
    return lib_tags, con_tags


def plot_tag_bar(lib_tags: pd.DataFrame, 
                 con_tags: pd.DataFrame) -> (None):
    '''Accepts liberal features and conservative features as pd.DataFrame
    and returns a bar plot to visualize the distribution of pos tags 
    passed by get_tags for each group.'''
    fig = go.Figure()
    fig.add_trace(go.Bar(x=lib_tags['tag'],
                y=lib_tags['freq'],
                name='Liberal'
                ))
    fig.add_trace(go.Bar(x=con_tags['tag'],
                y=con_tags['freq'],
                name='Conservative'
                ))

    fig.update_layout(
        title='POS Tag Distribution',
        xaxis_tickfont_size=14,
        yaxis=dict(
            title='Frequency',
            titlefont_size=16,
            tickfont_size=14,
        ),
        legend=dict(
            xanchor='left',
            x=1,
            y=1.0,
            yanchor='top',

            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(255, 255, 255, 0)'
        ),
        template='plotly_white',
        barmode='group',
        bargap=0.15, # gap between bars of adjacent location coordinates.
        bargroupgap=0.1 # gap between bars of the same location coordinate.
    )
    fig.show()
    
    return


def create_wordcloud(vocab: list, cmap: str) -> (WordCloud):
    '''Accepts a pd.DataFrame of word counts and displays a word cloud 
    visualization.
    '''
    
    counter = Counter(vocab)
    counts = pd.DataFrame.from_dict(counter, orient='index',
                        columns=['frequency']).reset_index()
    counts = counts.rename(columns={'index': 'word'}).sort_values(by='frequency',
                                                      ascending=False)
    
    counts['lens'] = counts['word'].apply(lambda x: len(x))
    counts = counts[counts['lens'] > 3]
    
    counts = counts.reset_index()

    all_words = list(counts['word'])
    allwords = all_words[25:300]
    all_words = ' '.join(all_words)
        
    cloud = WordCloud(width=800, height=400, colormap=cmap,
                        random_state=21, max_font_size=110,
                        collocations=True).generate(all_words)
    
    plt.imshow(cloud, interpolation='bilinear')
    plt.axis('off')
    plt.show();
    
    return 


def plot_word_bar(vocab: pd.DataFrame) -> (None):
        '''Accepts liberal features and conservative features as pd.DataFrame
    and returns a bar plot to visualize the distribution of the 25-45 most
    common words in the corpus.'''
    counter = Counter(vocab)
    counts = pd.DataFrame.from_dict(counter, orient='index',
                        columns=['frequency']).reset_index()
    counts = counts.rename(columns={'index': 'word'}).sort_values(by='frequency',
                                                      ascending=False)
    
    counts['lens'] = counts['word'].apply(lambda x: len(x))
    counts = counts[counts['lens'] > 3]
    
    counts = counts.reset_index()
    counts = counts[25:45]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=counts['word'],
                y=counts['frequency'],
                name='Liberal'
                ))

    fig.update_layout(
        title='Most Frequent Words',
        xaxis_tickfont_size=14,
        yaxis=dict(
            title='Frequency',
            titlefont_size=16,
            tickfont_size=14,
        ),
    legend=dict(
        xanchor='left',
        x=1,
        y=1.0,
        yanchor='top',

        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
        ),
        template='plotly_white',
        barmode='group',
        bargap=0.15, # gap between bars of adjacent location coordinates.
        bargroupgap=0.1 # gap between bars of the same location coordinate.
    )
    
    fig.show()
    
    
def run_lda_model(docs: list, n_topics: int, n_passes: int, n_words: int, 
                  model_name: str) -> (list):
    '''Accepts a list of spacy docs and constructs a gensim dictionary and
    corpus for an LDA (topic) model. The parameters for number of topics,
    passes and words are applied. Model is returned. 
    '''
    dictionary = corpora.Dictionary(docs)
    corpus = [dictionary.doc2bow(text) for text in docs]

    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=n_topics, 
                                               id2word=dictionary, passes=15)
    ldamodel.save(model_name + '.gensim')
    return  ldamodel

