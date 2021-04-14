# Politics, Polarization, & Pragmatics
**An investigation into the effect of language on discourse.**

In this [blog](https://github.com/christineegan42/capstone/blob/main/PoliticalPolarization.md) , I describe my thoughts on why politics in the United States have become so polarized. While I suggested different reasons why this might be the case and how that relates to how we use technology to influence others, I decided that there was one factor that was being ignored that I wanted to investigate. This factor was the relationship between language and political persuasion, and how liberals and conservatives use language to influence voters. I wanted to take a closer look at the intersection of language, political polarization, and social media.

## I. Methodology
### 1. Hypothesis
Liberals and conservatives will attempt to use different frames to persuade voters on social media. When we communicate with others using language, we evoke frames that help others understand our message. One way to test this is to examine the kind of language that is being used, and that if the language between liberal and conservative messaging is distinct.

### 2. Data
I collected my data according to a guiding principle discussed in the paper  [A Nation under Joint Custody:How Conflicting Family Models divide US Politics](https://digitalassets.lib.berkeley.edu/etd/ucb/text/Wehling_berkeley_0028E_13309.pdf) . In the paper, Eva Wheling discusses ‘Elite-to-Citizen’ discourse and ‘Citizen-Discourse’.

*For the purposes of this project, I’m going to examine Elite-to-Citizen discourse.*

#### Elite-to-Citizen: 
[ProPublica Data Store](https://www.propublica.org/datastore/dataset/political-advertisements-from-facebook)      
This is discourse from politicians aimed at the public. For this, I chose a collection of over 200,000 Facebook political ads from 2019-2021 available as a CSV. 

#### Citizen
This is discourse among citizens. For this, I chose posts and comments on  [Reddit](https://www.reddit.com/)  political forums from 2020-2021. I obtained this data using the Reddit API. For information about how I did that, you can view the code on my  [Github](https://github.com/christineegan42/reddit-calls) . You can also view **this** to see how I processed the data.

### 3. Processing
##### 	I. Labeling (facebook_labels.ipynb)
			1. The original data had over 200,000 rows. For the purposes of this analysis, I could get by with a smaller sample. I decided that an ad would be labeled as liberal or conservative based on the political affiliation of the organization named in the ‘paid-by’ column.
			2. The first filter was to limit the data to only those rows whose payer appears more than one hundred times. This resulted in a data set of approximately 20,000 rows with 200 unique payers.
			3. I reviewed the list of unique payers and used I used techniques such as Named Entity Recognition to extract organization names/affiliations from the list payers. When I could not use automated techniques to apply labels, conducted research using [OpenSecrets](https://www.opensecrets.org/) to determine the political affiliation of all of the payers. 
			4. When the affiliation of all the payers was determined, the correct label was applied to each row. I checked the values for a class imbalance and noticed that there were 14,000 rows that were labeled liberal and only approximately 5,000 rows that were labeled conservative.
			5. To address the class imbalance I under-sampled and used 5,000 observations from each class for my analysis.
##### 	II. Feature Engineering (src/vectorize.py)
	The following steps were performed on the liberal set and the conservative set separately, and combined later during the modeling phase.
			1. Word Embeddings
				1. Data from the ‘message’ column was used for the linguistic analysis. HTML and superfluous characters were removed. Then each observation was transformed into a SpaCy document. 
				2. The new list of documents was then lemmatized, and filtered for digits, punctuation, stop words, and words that are three characters or less.
				3. Each word was assigned a word-vector using the SpaCy pre-trained model ‘en-web-code-md’.
				4. The vocabulary was trimmed to words that appear at least ten times. Then, a column was created for each word in the vocabulary. The value of each column was the vector for the word as it appears in that row. 
			2. Sentiment/Polarity
				1. The compound polarity for each observation was calculated using Vader Sentiment Intensity Analyzer.

### 4. Model & Results (model.py)
If the language that liberals and conservatives use in political ads is distinct, a model should be able to classify a political message as liberal or conservative.
		1. The liberal and conservative data frames are individually processed and vectorized.
		2. The two data frames are combined and the resulting data set is fit three separate models: Logistic Regression, Naive Bayes, and Support Vector Classifier. Then, each model is evaluated using three different test sizes: 0.2, 0.3, 0.4.
	

## II. Findings

### Most Common Words
*(coming soon)*

### Most Common Topics
*(coming soon)*

### Most Important Words
*(coming soon)*


## IV. Future Work
* The ProPublica Facebook ad data set is very rich. There are still many valuable insights that can be gleaned from further examination of different features in the original data, especially the relationship between the ad target demographics and the content of the ad.
* Reimagining the model to be more sensitive to framing in language by using Named Entity Recognition and POS tags to identify key nouns and verbs used in the discourse to evoke certain frames to investigate who is evoking which frames. 
* Extending this model to analyze Citizen discourse using Reddit data and comparing Elite-to-Citizen and Citizen political discourse. In addition, the model could be extended to decipher between Elite-to-Citizen and Citizen discourse.

