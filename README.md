# # Politics, Polarization, and Pragmatics
## An investigation into the effect of language on discourse.

In this [blog](https://github.com/christineegan42/capstone/blob/main/PoliticalPolarization.md), I describe my thoughts on why politics in the United States have become so polarized. While I suggested different reasons why this might be the case and how that relates to how we use technology to influence others, I decided that there was one factor that was being ignored that I wanted to investigate. This factor was the relationship between language and political persusion, and how liberals and conservatives use language to influence voters. I wanted to take a closer look at the intersection of language, political polarization, and social media. 

### Hypothesis
Liberals and conservatives will attempt to use different frames to persuade voters on social media. When we communicate with others using language, we evoke frames that help others understand our message. One way to test this is to examine the kind of language that is being used, and that if the language between liberal and conservative messaging is distinct.

###  Data:
I collected my data according to a guiding principle discussed in the paper [A Nation under Joint Custody:How Conflicting Family Models divide US Politics](https://digitalassets.lib.berkeley.edu/etd/ucb/text/Wehling_berkeley_0028E_13309.pdf). In the paper, Eva Wheling discusses 'Elite to Citizen' discourse and 'Citizen Discourse'.
##### Elite to Citizen
This is discourse from politicians aimed at the public. For this, I chose a collection of Facebook political ads from 2019-2020. I downloaded this as CSV from the Propublica. You can also view **this** to see how I processed the data.

##### Citizen
This is discourse among citizens. For this, I chose posts and comments on [Reddit](https://www.reddit.com/) political forums from 2020-2021. I obtained this data using the Reddit API. For information about how I did that, you can view the code on my [Github](https://github.com/christineegan42/reddit-calls). You can also view **this** to see how I processed the data.

##### Facebook
1. LDA
2. Basic EDA

##### Reddit
1. LDA
2. Basic EDA

#### Model Results
##### Experiment 1 - Facebook Data: Liberal or Conservative?
**Data:** [Propublica Facebook Political Ad Data](https://www.propublica.org/datastore/dataset/political-advertisements-from-facebook)          
**Methodology:**    
(Label Notebook, Model Script)          
**Summary**: The original data from Propublica had over 200k rows. I filtered the data to include only those rows with a 'paid_for_by' value occurred over 100 times, which resulted in about 200 unique payers. The research that I outline in the **Label Notebook** describes the process that I used to research and label the political affiliation of the payers. I created a feature called 'label' that reflected the political affiliation, then took a sample of 5,000 from each label (liberal and conservative). I processed these observations, then I fit three different models, Logistic Regression, Naive Bayes, and Support Vector Classifier and performed a test for three different training sizes on each (0.2, 0.4, 0.6).       
**Results** Of the three different models, Niave Bayes performed the best overall. 


##### Experiment 2 - Reddit Data: Liberal or Conservative?
**Data:** I collected [Reddit](https://www.reddit.com/) posts and comments using [Reddit API](https://www.reddit.com/wiki/api) and a CLI app I created, [reddit_calls](https://github.com/christineegan42/reddit-calls). Data was collected from the following subreddits:     


| Liberal    | Conservative    |   
| :--------- | :-------------- |    
| Progressive | Libertarian   |   
| Democrats   | Conservative  |   
| Liberal     | Republican    |   
| DemocraticSocialism   | Republicans   |          

**Methodology:**    
(Label Notebook, Model Script)   
**Results** *Pending*

##### Experiment 3 - Facebook/Reddit Data: Elite-to-Citizen  or Citizen?
**Data**: I combined the data from the sources in Experiment 1 and Experiment 2. A post was labeled as Elite-to-Citizen if it was a Facebook ad and Citizen if it was a Reddit post. 

#### Results
*pending*

#### Future Work
* Developing a model that focuses more precisely on language that lends itself to certain frames. I plan to achieve this using Name Entity Recognition and POS tagging to identify patterns nouns and verbs and how that intersects lexical items and their associated frames.
* Applying this experiment to social media data from other platforms.
* Investigating the relationship between the ad target demographics and the content of the ad. 
