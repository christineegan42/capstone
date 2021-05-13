# Facebook Ad Data Text Classification

## I. Introduction
In this [blog](https://github.com/christineegan42/flatiron-capstone/blob/main/PoliticalPolarization.md) , I describe my thoughts on why politics in the United States have become so polarized. While I suggested different reasons why this might be the case and how that relates to how we use technology to influence others, I decided that there was one factor that I wanted to investigate the relationship between language and political persuasion, and how liberals and conservatives use language to influence voters.

#### Navigate this Repository 

```
flatiron-capstone
│ README.md
└─notebooks
│  └─labeling
│  │   fb_labels.ipynb
│  └─eda
│  │   eda_polarity_and_length.ipynb
│  │   eda.py
│  └─results
│  │   master_results.ipynb
│  │   results.py
│  └─feature_importances
│      feature_importance.ipynb
│      importances.py
└─src
│  model.py
│  vectorize.py
└─appendix
│  resources.txt
│  setup_instructions.txt
│  images

```

## II. Methodology
### 1. Hypothesis
Language in political ad data will be distinct among liberals and conservatives.

### 2. Data
I collected my data according to a guiding principle discussed in the paper [A Nation under Joint Custody:How Conflicting Family Models divide US Politics](https://digitalassets.lib.berkeley.edu/etd/ucb/text/Wehling_berkeley_0028E_13309.pdf) . In the paper, Eva Wheling discusses ‘elite-to-citizen’ discourse which she describes as language used politicians and elites that is directed toward citizens. 

The source I used to obtain examples of elite-to-citizen discourse was a collection of over 200,000 Facebook political ads from 2019-2021 available as a CSV from [ProPublica Data Store](https://www.propublica.org/datastore/dataset/political-advertisements-from-facebook).

### 3. Processing
#### 	I. Labeling (fb_labels.ipynb)
The original data had over 200,000 observations. In order to extract a smaller sample I set some parameters as to how I would filter the data.
1. Data is labeled as liberal or conservative based on the affiliation of the organization that paid for the ad, as represented by the 'paid-by' column. 
2. Data is limited to only those rows whose payer appears more than one hundred times.
This resulted in a data set of approximately 20,000 rows with approximately 200 unique payers.

Next, I reviewed the list of payers and used the following combination of manual and automated techniques to label each ad.
1. Named entity recognition (NER) with SpaCy was used in order to extract payers with names that included words similar to "Republican" and "Democrat".
2. Named entity recognition was used to extract the names of any politicans, and label the politicans whose affiliation was known.
3. Politicans and organizations that could not be labeled using NER were manually researched using [OpenSecrets](https://www.opensecrets.org/) to determine the political affiliation of all of the payers.
 
When the affiliation of all the payers was determined and the correct label was applied to each row, I checked the values for a class imbalance and noticed that there were 14,000 rows that were labeled liberal and only approximately 5,000 rows that were labeled conservative. To address the class imbalance I under-sampled and used 5,000 observations from each class for my analysis.

#### 	II. Feature Engineering (src/vectorize.py)
The original ad data from ProPublica had 27 features. Since this analysis was focused on text data from ad messages, after the labeling process all columns except for 'message' and '_label_' were eliminated. 

The following steps were performed on the liberal set and the conservative set separately, and combined later during the modeling phase.
##### Word Embeddings
1. Data from the ‘message’ column was used for the linguistic analysis. HTML and superfluous characters were removed. Then each observation was transformed into a SpaCy document. 
2. The new list of documents was then lemmatized, and filtered for digits, punctuation, stop words, and words that are three characters or less.
3. Each word was assigned a word-vector using the SpaCy pre-trained model ‘en-web-code-md’.
4. The vocabulary was trimmed to words that appear at least ten times. Then, a column was created for each word in the vocabulary. The value of each column was the mean vector for the word as it appears in that row. 
##### Sentiment/Polarity
1. The compound polarity for each observation was calculated using Vader Sentiment Intensity Analyzer.
2. Messages were divded into long (length had a z-score > 1) and short (length had a z-score < 1). 
3. 98.38% of ads are short, 62% of ads are long. 
4. Of short ads, 50.75% are conservative while 49.25% are liberal. 
5. Of long ads, 4.32% are conservative and 95.68% are liberal
6. Longer ads were overwhelmingly liberal and positive.         
![scatterplot of polarity and length by label](https://github.com/christineegan42/flatiron-capstone/blob/main/appendix/images/pol_len_label.png)        

### 3. Models
I fit and tested three different models (Logistic Regression, Niave Bayes, and Support Vector Classifier) to compare their performance, using the following process for each classifier:
1. A sample of 1,000 observations was loaded from each vectorized dataframe (liberal and conservative) and concatenated to create one dataframe of 2,000 observations for the model.
2. The data was divded into X and y data, then split into a training and validation set.
3. The training data was then fit that particular classifier, and evaluated using three different test sizes: 0.2, 0.3, 0.4.

Results for each classifier were combined and saved to the /results directory. 

The process is repeated 10 times, and the results from each trial are combined into a masters results data frame, which is then grouped by classifier and text size. The mean score for all of the evaluation metrics is extracted from each and compared.

#### Results
| model               |   test size |      f1 |   precision |   recall |   accuracy |   roc_auc |   mean_score |
|:--------------------|------------:|--------:|------------:|---------:|-----------:|----------:|-------------:|
| Gaussian NB         |         0.2 | 9.65259 |     9.94671 |  9.37688 |    9.665   |   9.66357 |      9.66095 |
| Gaussian NB         |         0.3 | 9.56674 |     9.82702 |  9.32119 |    9.575   |   9.5767  |      9.57333 |
| Gaussian NB         |         0.4 | 9.47513 |     9.69629 |  9.26551 |    9.4825  |   9.48414 |      9.48071 |
| Logistic Regression |         0.2 | 6.4213  |     5.9322  |  7.0201  |    6.15    |   6.15433 |      6.33559 |
| Logistic Regression |         0.3 | 6.39771 |     5.95296 |  6.93709 |    6.11167 |   6.10613 |      6.30111 |
| Logistic Regression |         0.4 | 6.3345  |     5.91879 |  6.83623 |    6.05875 |   6.05287 |      6.24023 |
| SVC                 |         0.2 | 6.13127 |     6.12301 |  6.17085 |    6.1825  |   6.18244 |      6.15801 |
| SVC                 |         0.3 | 6.04041 |     6.15965 |  5.98013 |    6.13167 |   6.13268 |      6.08891 |
| SVC                 |         0.4 | 5.99584 |     6.11838 |  5.93052 |    6.09125 |   6.09246 |      6.04569 |

The model that had the highest mean score was Gaussian Niave Bayes using an 80/20 train/test split. However, special consideration was given with respect to precision because a lot of political language is domain specific and there was bound to be a large vocabulary shared between the classes and there was a high likelyhood that many features might be equally as prevelant among either class. 


#### Most Important Features
To retrieve the most important features in this model, I use Sci-Kit Learn Permutation Importance. 
1. Obtained the results of three models (Logistic Regression, Niave Bayes, and Support Vector Classifier) and selected Niave Bayes because it had the best performance.
2. Loaded the vectorized data used to create the models. Divided the data by class, and then sampled 1000 observations from each class. 
3. The data frames were then concatenated and split into X, y data, then into a training and validation set, and fit the data to Sci-Kit Learn Gaussian NB.
4. Then, the permutation importance was calculated for the selected sample of observations and stored.
5. This process was repeated 10 times, and the results of each test were combined into a master dataframe, and then grouped by feature by calculating the mean of the mean importance for each feature. 

The features with the top-ten highest mean of means:
| features   |   mean_importance |
|:-----------|------------------:|
| debate     |            0.0088 |
| stage      |            0.0078 |
| corporate  |            0.0052 |
| democracy  |            0.005  |
| qualify    |            0.0047 |
| mainstream |            0.0044 |
| media      |            0.0042 |
| dnc        |            0.0041 |
| pac        |            0.004  |
| organize   |            0.0034 |


#### Analysis of Top 10 Most Important Features
![barplot of mean feature importances](https://github.com/christineegan42/flatiron-capstone/blob/main/appendix/images/importances_viz.png)       

**(coming soon)**
An analysis of how often the most important features occur among each class and in what context.

## IV. Future Work
* The ProPublica Facebook ad data set is very rich. There are still many valuable insights that can be gleaned from further examination of different features in the original data, especially the relationship between the ad target demographics and the content of the ad.
* Reimagining the model to be more sensitive to framing in language by using Named Entity Recognition and POS tags to identify key nouns and verbs used in the discourse to evoke certain frames to investigate who is evoking which frames. 
* Extending this model to analyze Citizen discourse using Reddit data and comparing Elite-to-Citizen and Citizen political discourse. In addition, the model could be extended to decipher between Elite-to-Citizen and Citizen discourse.
