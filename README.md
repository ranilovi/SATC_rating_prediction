
# Sex and The City Script Text Analysis

The purpose of this project is the analysis of the entire script of the TV Show Sex and The City. Sex and the City is an American romantic comedy-drama television series created by Darren Star and produced by HBO. 

![Image of SATC](https://ichef.bbci.co.uk/news/660/cpsprodpb/15BD8/production/_101884098_satcfour_getty.jpg)

# Table of Content
0. [Downloads and Libraries](#lud)  
1. [Exploratory Data Analyis](#eda)  
2. [Text Preprocessing](#text-prepro)
   * [Tokenization](#tokenization)
3. [Natural Language Processing](#nlp)
    * [Lemmatization](#lemmatization)
    * [TF-IDF](#tfidf)
4. [Predictive Modeling](#prediction)
    * [Dimensionality Reduction](#dimensionality)   
        * [Preselection of Features](#pre-feat)   
        * [Principal Component Analysis](#pca) 
    * [Linear Regression](#linreg)
    * [Decision Tree Regressor](#regtrees) 
    * [Bagging Model](#bagging) 
    * [XGBoost](#xgb) 
    * [Prediction Including the Lines per Character ](#lpc)
    * [Model Comparison](#modelcomp)
5. [BONUS: Episode Lookup](#keyword)
    
   



## Downloads and Libraries <a class="anchor" id="lud"></a>


```python
import pandas as pd
import math
import numpy as np

import seaborn as sns 
import matplotlib.pyplot as plt

#types
from collections import Counter

# NLP library imports
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.tokenize import word_tokenize

from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize
from nltk.tokenize import sent_tokenize

#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('stopwords')

#Skelarn for TfIdf 
from sklearn.feature_extraction.text import TfidfVectorizer

#Sklearn model building and evaluation
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics 

#Sklearn for linear regression 
from sklearn import linear_model
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression

#DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import tree, export_graphviz
from sklearn.ensemble import RandomForestRegressor

import graphviz
from IPython.display import Image, display

# run this if grphviz throws error, after installing graphviz
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'

# PCA 
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings('ignore')

#xgb
import xgboost as xgb

#Plot Libraries
from  matplotlib import pyplot
import seaborn

#graphing
%matplotlib inline

#more vectorization
import gensim
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors

#IMBd API
#!pip install git+https://github.com/alberanid/imdbpy
from imdb import IMDb
```

    C:\Users\dorar_000\Anaconda3\lib\site-packages\sklearn\ensemble\weight_boosting.py:29: DeprecationWarning: numpy.core.umath_tests is an internal NumPy module and should not be imported. It will be removed in a future NumPy release.
      from numpy.core.umath_tests import inner1d


## Exploratory Data Analyis <a class="anchor" id="eda"></a>

The present data set consists of 39208 lines of text with an average reating of 7.52 on IMDB. The poorest rating is 7.0, the highest rating is 8.8. The data set contains 94 unique episodes.

We first load the kaggle dataset ("satc_all_lines.csv"), which contains all lines of dialogue:


```python
satc_all_lines = pd.read_csv("https://raw.githubusercontent.com/ranilovi/SATC_rating_prediction/master/SATC_all_lines.csv").drop(["Unnamed: 0"],axis=1)
```

Then we need the IMDb ratings (and possibly other information) for the episodes. 
We used ParseHub to generate "imdb_eps.csv" in order to collect all episode IDs, which are the unique identifiers used by the IMBd API to run requests. 


```python
# get the imdb scraped information
eps = pd.read_csv("https://raw.githubusercontent.com/ranilovi/SATC_rating_prediction/master/imdb_eps.csv")
ep_ids = eps['ID']
# create an instance of the IMDb class
ia = IMDb()
# get the series
episodes =list(map(lambda x:ia.get_movie(x), ep_ids))
ratings = [x['rating'] for x in episodes]
eps['Rating'] = ratings
```


```python
# merge the two datasets to add targets:
satc_all_lines_w_rating = pd.merge(satc_all_lines, eps, how='left', 
                                   left_on=['Season', 'Episode'],right_on=['Season', 'Episode'])
satc_all_lines_w_rating['Episode'] = satc_all_lines_w_rating['Episode'].apply(lambda x: str(x).zfill(2))
```


```python
#satc_all_lines_w_rating = pd.read_csv("satc_all_lines_w_rating.csv").drop(["Unnamed: 0"],axis=1)
#create col with season and episode
satc_all_lines_w_rating['S_E'] = satc_all_lines_w_rating['Season'].astype(int).astype(str)+"_"+satc_all_lines_w_rating['Episode']
satc_all_lines_w_rating.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Season</th>
      <th>Episode</th>
      <th>Speaker</th>
      <th>Line</th>
      <th>date_job</th>
      <th>ep_data_name</th>
      <th>ep_data_url</th>
      <th>ID</th>
      <th>Rating</th>
      <th>S_E</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>01</td>
      <td>Carrie</td>
      <td>Once upon a time, an English journalist came t...</td>
      <td>NaN</td>
      <td>Sex and the City</td>
      <td>https://www.imdb.com/title/tt0698663/?ref_=tte...</td>
      <td>698663</td>
      <td>7.4</td>
      <td>1_01</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>01</td>
      <td>Carrie</td>
      <td>Elizabeth was attractive and bright.</td>
      <td>NaN</td>
      <td>Sex and the City</td>
      <td>https://www.imdb.com/title/tt0698663/?ref_=tte...</td>
      <td>698663</td>
      <td>7.4</td>
      <td>1_01</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>01</td>
      <td>Carrie</td>
      <td>Right away she hooked up with one of the city'...</td>
      <td>NaN</td>
      <td>Sex and the City</td>
      <td>https://www.imdb.com/title/tt0698663/?ref_=tte...</td>
      <td>698663</td>
      <td>7.4</td>
      <td>1_01</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>01</td>
      <td>Tim</td>
      <td>The question remains Is this really a company ...</td>
      <td>NaN</td>
      <td>Sex and the City</td>
      <td>https://www.imdb.com/title/tt0698663/?ref_=tte...</td>
      <td>698663</td>
      <td>7.4</td>
      <td>1_01</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>01</td>
      <td>Carrie</td>
      <td>Tim was 42, a wellliked and respected investme...</td>
      <td>NaN</td>
      <td>Sex and the City</td>
      <td>https://www.imdb.com/title/tt0698663/?ref_=tte...</td>
      <td>698663</td>
      <td>7.4</td>
      <td>1_01</td>
    </tr>
  </tbody>
</table>
</div>




```python
satc_all_lines_w_rating.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Season</th>
      <th>ID</th>
      <th>Rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>39208.000000</td>
      <td>39208.000000</td>
      <td>39208.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.657289</td>
      <td>698653.514232</td>
      <td>7.531769</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.692661</td>
      <td>26.903364</td>
      <td>0.281950</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>698608.000000</td>
      <td>7.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.000000</td>
      <td>698631.000000</td>
      <td>7.300000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.000000</td>
      <td>698653.000000</td>
      <td>7.500000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>5.000000</td>
      <td>698676.000000</td>
      <td>7.700000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>6.000000</td>
      <td>698701.000000</td>
      <td>8.800000</td>
    </tr>
  </tbody>
</table>
</div>




```python
len(satc_all_lines_w_rating['S_E'].unique())
```




    93




```python
# Plotting rating distribution
X_ratings = satc_all_lines_w_rating.groupby(['S_E']).min().Rating.value_counts()
x = sns.barplot(X_ratings.index,X_ratings)
x.set(xlabel='Ratings',ylabel='Frequencies',title='Frequencies of ratings'.format(satc_all_lines_w_rating.shape[0]))
plt.show()
#X_ratings
```


![png](output_files/output_12_0.png)


## Text Preprocessing <a class="anchor" id="#text-prepro"></a>

In order to perform an analysis of the present text, we pre-process the text (represented by seperate lines in our data set). We bring all words to lowercase to be able to compare words across their position in a sentence ("Hello" and "hello" should be the same word). We furthermore strip all text of special characters. 


```python
#text transformation
satc_all_lines_w_rating["cleaned"] = satc_all_lines_w_rating.Line.tolist()
#all to lowercase 
satc_all_lines_w_rating.cleaned = [str(line).lower() for line in satc_all_lines_w_rating.cleaned]

#remove special chars
chars_remove = ["@", "/", "#", ".", ",", "!", "?", "(", ")", "-", "_","’","'", "\"", ":"]
trans_dict = {initial:" " for initial in chars_remove}
satc_all_lines_w_rating.cleaned = [line.translate(str.maketrans(trans_dict)) for line in satc_all_lines_w_rating.cleaned]
satc_all_lines_w_rating.head()

print("Before transformation:",satc_all_lines_w_rating.Line[0],"\n After transformation:", satc_all_lines_w_rating.cleaned[0])
satc_all_lines_w_rating.head()
```

    Before transformation: Once upon a time, an English journalist came to New York. 
     After transformation: once upon a time  an english journalist came to new york 





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Season</th>
      <th>Episode</th>
      <th>Speaker</th>
      <th>Line</th>
      <th>date_job</th>
      <th>ep_data_name</th>
      <th>ep_data_url</th>
      <th>ID</th>
      <th>Rating</th>
      <th>S_E</th>
      <th>cleaned</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>01</td>
      <td>Carrie</td>
      <td>Once upon a time, an English journalist came t...</td>
      <td>NaN</td>
      <td>Sex and the City</td>
      <td>https://www.imdb.com/title/tt0698663/?ref_=tte...</td>
      <td>698663</td>
      <td>7.4</td>
      <td>1_01</td>
      <td>once upon a time  an english journalist came t...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>01</td>
      <td>Carrie</td>
      <td>Elizabeth was attractive and bright.</td>
      <td>NaN</td>
      <td>Sex and the City</td>
      <td>https://www.imdb.com/title/tt0698663/?ref_=tte...</td>
      <td>698663</td>
      <td>7.4</td>
      <td>1_01</td>
      <td>elizabeth was attractive and bright</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>01</td>
      <td>Carrie</td>
      <td>Right away she hooked up with one of the city'...</td>
      <td>NaN</td>
      <td>Sex and the City</td>
      <td>https://www.imdb.com/title/tt0698663/?ref_=tte...</td>
      <td>698663</td>
      <td>7.4</td>
      <td>1_01</td>
      <td>right away she hooked up with one of the city ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>01</td>
      <td>Tim</td>
      <td>The question remains Is this really a company ...</td>
      <td>NaN</td>
      <td>Sex and the City</td>
      <td>https://www.imdb.com/title/tt0698663/?ref_=tte...</td>
      <td>698663</td>
      <td>7.4</td>
      <td>1_01</td>
      <td>the question remains is this really a company ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>01</td>
      <td>Carrie</td>
      <td>Tim was 42, a wellliked and respected investme...</td>
      <td>NaN</td>
      <td>Sex and the City</td>
      <td>https://www.imdb.com/title/tt0698663/?ref_=tte...</td>
      <td>698663</td>
      <td>7.4</td>
      <td>1_01</td>
      <td>tim was 42  a wellliked and respected investme...</td>
    </tr>
  </tbody>
</table>
</div>



### Tokenization <a class="anchor" id="tokenization"></a>
We then tokenize the text, this means, we break up the text into invidiual tokens. We are using the NLTK tokenize package and are splitting sentences up into single tokens using word_tokenize. The method invoked by word_tokenize is using the "Treebank tokenizer", a tokenizer that uses regular expressions to tokenize text.

After tokenization, we remove stopwords from the text, using the english stopwords from NLTK. Stopwords are language specific word that carry no meaning for the purpose of a text analyis.


```python
print(str(len(stopwords.words("english")) ) +" stopwords are included in the stopwords corpus for the English language, containing words such as" + str(stopwords.words("english")[20:34]))
```

    179 stopwords are included in the stopwords corpus for the English language, containing words such as['himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs']



```python
#We are using NLKT tokenizer to split all text up into individual tokens
satc_all_lines_w_rating["tokenized"] = [word_tokenize(line) for line in satc_all_lines_w_rating.cleaned]

#Removing stopwords for topic extraction using nltk stopwords library
stopw = stopwords.words('english')
satc_all_lines_w_rating["w_o_stopwords"] = [[token for token in line if token not in stopw if len(token)>2] for line in satc_all_lines_w_rating.tokenized]
satc_all_lines_w_rating.head()

print("Before transformation:",satc_all_lines_w_rating.Line[0],"\n After transformation:", satc_all_lines_w_rating.w_o_stopwords[0])
```

    Before transformation: Once upon a time, an English journalist came to New York. 
     After transformation: ['upon', 'time', 'english', 'journalist', 'came', 'new', 'york']



```python
all_words = []
for line in satc_all_lines_w_rating["w_o_stopwords"]:
    for word in line:
        all_words.append(word)

dist = nltk.FreqDist(all_words)
X = [nb[1] for nb in dist.most_common(20)]
y = [nb[0] for nb in dist.most_common(20)]
x = sns.barplot(np.array(X),np.array(y))
x.set(xlabel='Word frequencies',ylabel='Words',title='Most common words in the Sex and The City Scripts')
plt.show()
```


![png](output_files/output_18_0.png)



```python
#saving tokenized data set
#satc_all_lines_w_rating.to_csv("satc_all_lines_w_rating_tokenized.csv")
```

## Natural Language Processing <a class="anchor" id="nlp"></a>
### Lemmatization <a class="anchor" id="lemmatization"></a>

In order to combine words that represent the same meaning, we are using the WordNetLemmatizer() to transform our corpus. "WordNet® is a large lexical database of English. Nouns, verbs, adjectives and adverbs are grouped into sets of cognitive synonyms (synsets), each expressing a distinct concept. Synsets are interlinked by means of conceptual-semantic and lexical relations." [Source: https://wordnet.princeton.edu]


```python
# apply lemmatization from wordnet in order to merge words that come from the same meaning .- for example, "friend" and "friends" should be counted as the same word

def lemmatize(tokens):
    tokens = [WordNetLemmatizer().lemmatize(WordNetLemmatizer().lemmatize(WordNetLemmatizer().lemmatize(token,pos='a'),pos='v'),pos='n') for token in tokens]
    return tokens  

satc_all_lines_w_rating["lemmatized"] = [lemmatize(line) for line in satc_all_lines_w_rating.w_o_stopwords]
```


```python
#look at the difference in words 
print(satc_all_lines_w_rating.w_o_stopwords[0:5])
print(satc_all_lines_w_rating.lemmatized[0:5])
```

    0    [upon, time, english, journalist, came, new, y...
    1                      [elizabeth, attractive, bright]
    2    [right, away, hooked, one, city, typically, el...
    3           [question, remains, really, company, want]
    4    [tim, wellliked, respected, investment, banker...
    Name: w_o_stopwords, dtype: object
    0    [upon, time, english, journalist, come, new, y...
    1                      [elizabeth, attractive, bright]
    2    [right, away, hook, one, city, typically, elig...
    3            [question, remain, really, company, want]
    4    [tim, wellliked, respect, investment, banker, ...
    Name: lemmatized, dtype: object



```python
all_words = []
for line in satc_all_lines_w_rating["lemmatized"]:
    for word in line:
        all_words.append(word)

dist = nltk.FreqDist(all_words)
X = [nb[1] for nb in dist.most_common(20)]
y = [nb[0] for nb in dist.most_common(20)]
x = sns.barplot(np.array(X),np.array(y))
x.set(xlabel='Word frequencies',ylabel='Words',title='Most common words in the Sex and The City Scripts, Lemmatized')
plt.show()
```


![png](output_files/output_23_0.png)


### TF-IDF <a class="anchor" id="tfidf"></a>


Tf-Idf is a widely used statistic in the field of information retreival, used to quantify the relevance of a query or key word to a document inside a corpus.  It is commonly used in search engine optimization and text mining. The statistic is designed to assigns higher relevance to a term if it occurs often in the document, but penalize it if it occus in many different documents, i.e. it is not unique or specific to one or few documents. The general formula to compute the Tf-Idf score for term _t_ and documet _d_ in corpus _D_ is:

<center> 
$tf.idf(t,d,D)=tf(t,d)*idf(t,D)$ <br><br>
$=\frac{n(t,d)}{N(d)}*log\frac{|D|}{d(t)}$<br><br>
</center>
where:<br><br>
$tf(t,d)$ is the term frequency function<br><br>
$idf(t,D)$ is the inverse document frequency function<br><br>
$n(t,d)$ is the number of times t occurs in d<br><br>
$N(d)$ is the number of unique terms in d<br><br>
$d(t)$ is the number of documents in D which contain t<br><br>

We are using a TFIDF statistic across our data set in order to understand, which words are significant in describing the content per episode. In our dataset, a document are all the lines of text spoken in one episode, and the corpus are all episodes together. 


```python
#group by episode to define one episode as one document for tf-idf

#initizalize new df
satc_text_per_episode = pd.DataFrame(
    columns=list(satc_all_lines_w_rating)[5:10])

#init
current_s_e = "1_01"
all_lines = []
#collect all speakers per episode 
all_speakers = []
carrie = 0
samantha = 0
miranda = 0
charlotte = 0
stanford = 0

for index, row in satc_all_lines_w_rating.iterrows():
        if(row.S_E == current_s_e):
            #we use the lemmatized version of the lines, change here for different choice of text preprocessing
            all_lines.append(row.lemmatized)
            all_speakers.append(row.Speaker)
            
        if row.S_E != current_s_e:
            #flatten all tokens for previous episode and add them to the dataframe
            flat_all_lines = [word for line in all_lines for word in line]
            
            #count lines per main character 
            for name in all_speakers:
                if name == "Carrie":
                    carrie = carrie + 1
                if name == "Samantha":
                    samantha = samantha + 1
                if name == "Miranda":
                    miranda = miranda + 1
                if name == "Charlotte":
                    charlotte = charlotte + 1   
        
            
            
            #add a new row to new df, taking the information from the index-1 (the last s_e)
            satc_text_per_episode = satc_text_per_episode.append({'ep_data_name': satc_all_lines_w_rating.ep_data_name[index-1],
                                                                  'ep_data_url': satc_all_lines_w_rating.ep_data_url[index-1],
                                                                  'ID': satc_all_lines_w_rating.ID[index-1],
                                                                  'Rating': satc_all_lines_w_rating.Rating[index-1],
                                                                  'S_E': satc_all_lines_w_rating.S_E[index-1],
                                                                  'all_text': flat_all_lines,
                                                                  'all_speakers': all_speakers,
                                                                  'lines_carrie': carrie,
                                                                  'lines_samantha': samantha,
                                                                  'lines_charlotte':charlotte,
                                                                  'lines_miranda': miranda,
                                                                
                                                                 
                                                                 }, ignore_index=True)
        
            #update vars
            all_lines = []
            all_speakers = []
            carrie = 0
            samantha = 0
            miranda = 0
            charlotte = 0
            stanford = 0
            current_s_e = row.S_E


satc_text_per_episode.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ep_data_name</th>
      <th>ep_data_url</th>
      <th>ID</th>
      <th>Rating</th>
      <th>S_E</th>
      <th>all_speakers</th>
      <th>all_text</th>
      <th>lines_carrie</th>
      <th>lines_charlotte</th>
      <th>lines_miranda</th>
      <th>lines_samantha</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Sex and the City</td>
      <td>https://www.imdb.com/title/tt0698663/?ref_=tte...</td>
      <td>698663</td>
      <td>7.4</td>
      <td>1_01</td>
      <td>[Carrie, Carrie, Carrie, Tim, Carrie, Carrie, ...</td>
      <td>[upon, time, english, journalist, come, new, y...</td>
      <td>150.0</td>
      <td>33.0</td>
      <td>29.0</td>
      <td>26.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Models and Mortals</td>
      <td>https://www.imdb.com/title/tt0698649/?ref_=tte...</td>
      <td>698649</td>
      <td>7.4</td>
      <td>1_02</td>
      <td>[Carrie, Nick, Nick , Miranda , Nick , Nick , ...</td>
      <td>[date, nick, fairly, successful, sport, agent,...</td>
      <td>109.0</td>
      <td>6.0</td>
      <td>9.0</td>
      <td>25.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Bay of Married Pigs</td>
      <td>https://www.imdb.com/title/tt0698618/?ref_=tte...</td>
      <td>698618</td>
      <td>7.4</td>
      <td>1_03</td>
      <td>[Carrie, Carrie, Carrie, Carrie, Carrie, Carri...</td>
      <td>[friend, patience, husband, invite, hampton, w...</td>
      <td>137.0</td>
      <td>19.0</td>
      <td>39.0</td>
      <td>27.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Valley of the Twenty-Something Guys</td>
      <td>https://www.imdb.com/title/tt0698697/?ref_=tte...</td>
      <td>698697</td>
      <td>7.5</td>
      <td>1_04</td>
      <td>[Carrie, Carrie, Carrie, Carrie, Carrie, Carri...</td>
      <td>[seem, meet, everywhere, street, corner, party...</td>
      <td>174.0</td>
      <td>41.0</td>
      <td>24.0</td>
      <td>28.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>The Power of Female Sex</td>
      <td>https://www.imdb.com/title/tt0698688/?ref_=tte...</td>
      <td>698688</td>
      <td>7.3</td>
      <td>1_05</td>
      <td>[Carrie, Carrie, Samantha, Samantha, Carrie, S...</td>
      <td>[host, balzac, overnight, become, restaurant, ...</td>
      <td>138.0</td>
      <td>18.0</td>
      <td>30.0</td>
      <td>37.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
##create the tfidf matrix
vect = TfidfVectorizer(analyzer ='word',ngram_range=(1,1),encoding='latin1')
vect_transformed = vect.fit_transform([text for text in satc_text_per_episode['all_text'].astype(str)])

feature_names = np.array(vect.get_feature_names())

satc_tfidf = pd.concat([satc_text_per_episode[['S_E','ep_data_name','ID','Rating']],
                        pd.DataFrame(vect_transformed.todense(), columns = feature_names)],axis=1)

#create a second dataframe containing the lines per character: 
satc_tfidf_LPC = pd.concat([satc_text_per_episode[['S_E','ep_data_name','ID','Rating','lines_carrie','lines_samantha','lines_charlotte','lines_miranda']],
                        pd.DataFrame(vect_transformed.todense(), columns = feature_names)],axis=1)

satc_tfidf.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>S_E</th>
      <th>ep_data_name</th>
      <th>ID</th>
      <th>Rating</th>
      <th>000</th>
      <th>00am</th>
      <th>00amer</th>
      <th>00pm</th>
      <th>100</th>
      <th>1000</th>
      <th>...</th>
      <th>zoie</th>
      <th>zone</th>
      <th>zoo</th>
      <th>zooey</th>
      <th>zorro</th>
      <th>zsa</th>
      <th>zsu</th>
      <th>zucchini</th>
      <th>zygote</th>
      <th>éclairs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1_01</td>
      <td>Sex and the City</td>
      <td>698663</td>
      <td>7.4</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1_02</td>
      <td>Models and Mortals</td>
      <td>698649</td>
      <td>7.4</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1_03</td>
      <td>Bay of Married Pigs</td>
      <td>698618</td>
      <td>7.4</td>
      <td>0.016817</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1_04</td>
      <td>Valley of the Twenty-Something Guys</td>
      <td>698697</td>
      <td>7.5</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1_05</td>
      <td>The Power of Female Sex</td>
      <td>698688</td>
      <td>7.3</td>
      <td>0.068272</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 9692 columns</p>
</div>



Exploring the tf-idf matrix by checking some of the highest and lowest scores:


```python
#some words with smallest and largest tfids
sorted_tfidf_index = vect_transformed.max(0).toarray()[0].argsort()
print('Smallest tfidf:\n{}\n'.format(feature_names[sorted_tfidf_index[:10]]))
print('Largest tfidf: \n{}'.format(feature_names[sorted_tfidf_index[:-11:-1]]))
```

    Smallest tfidf:
    ['tiny' 'whenever' 'track' 'round' 'perform' 'impossible' 'obvious'
     'exact' 'ahead' 'replace']
    
    Largest tfidf: 
    ['laney' 'threesome' 'pattern' 'phil' 'fake' 'model' 'yankee' 'zsa' 'soul'
     'javier']


We want to explore if there are any interesting correlations between the ratings and tf-idf scores of individual tokens (features). Since there are more than 9,000 features, we'll look at the correlation matrix of the 10 most correlated words:


```python
cols = satc_tfidf.ix[:,3:].corr().abs().sort_values('Rating', ascending = False).index[:11]

sns.heatmap(satc_tfidf[cols].corr(),cmap="YlGnBu", annot=True, fmt=".1g", linewidths = 0.5)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x19be0b7c4a8>




![png](output_files/output_32_1.png)


## Predictive Modeling <a class="anchor" id="prediction"></a>

### Model Comparison - Note

In order to compare the performance of different models, using different kinds of preprocessing and features, we designed the function evaluate_model in the process in comparing them. We will present different models and show their predictive capability by comparing error metrics, as well as looking at prediction precision with plots and other visual aids (such as a modeling of regression trees later on).

As the key metric to compare models with we are using the mean squared error regression loss. 


```python
# evaluate_model takes as argument
# modelname a string - to create an easy overview for final comparison 
# a model as provided by the sklearn library
# number of features: as selected with the function get_features based on their tfdif score 
# pca_dim: number of dimensions kept in pca, can be set to None if no PCA
# df: should be satc_tfidf or satc_tfidf_LPC
# LPC: boolean with default falue False, set to True if you want to include the lines per character as a feature 

# returns 
# results - dataframe containing results from 10 iterations
# model_details_and_averages - description of the input and average over ten iterations 
# target_test,target_pred_test - y* and y_pred for the test set

def evaluate_model(modelname,model,number_of_features,pca_dim,df,LPC=False):
    
    #get the words that we will look at in the analysis
    selected_words = get_features('tfidf',number_of_features)
    
    #define the features and targets 
    if LPC == False:
        features = df[df.columns[4:]].filter(selected_words,axis=1).values
        target= df['Rating'].values
    
    #if we also use lines per character, we need to select different cols
    if LPC == True:
        features = df[df.columns[8:]].filter(selected_words,axis=1).values
        target= df['Rating'].values
    

    #if pca_dim != None, create the pca model 
    if pca_dim: 
        pca = PCA(n_components=pca_dim)

    
    results={}
    #labels = satc_tfidf['Rating'].unique()
    num_run = 20

    #get ten results to account for randomness 
    for i in range (num_run):

        # separate datasets into training and test datasets once, no folding
        features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.3)
        
        #if pca, apply the dim reduction 
        if pca_dim: 
                features_train = pca.fit_transform(features_train)
                features_test = pca.transform(features_test)

        # train the features and target datasets and fit to a model
        trained_model = model.fit(features_train, target_train)


        # predict target with feature test set using trained model
        target_pred_train = list(trained_model.predict(features_train))
        target_pred_test = list(trained_model.predict(features_test))

        results[i]=[metrics.mean_squared_error(target_test, target_pred_test),
                metrics.mean_absolute_error (target_test, target_pred_test),
                metrics.explained_variance_score(target_test, target_pred_test),
                metrics.r2_score(target_test, target_pred_test),
                   ]

    results = pd.DataFrame.from_dict(data=results,orient='index',columns=['MSE','MAbsE', 'Explained_variance_score','R_squared']) 
    model_details_and_averages = [
    str(modelname),
    number_of_features,pca_dim,
    results.MSE.mean(),
    results.MAbsE.mean(),
    results.Explained_variance_score.mean(),
    results.R_squared.mean(),
    ]

    
    return results,model_details_and_averages, target_test,target_pred_test

#evaluate model without lines per char
#evaluate_model("Lasso",lasso_model,50,50,satc_tfidf)[1]
#evaluate model with lines per char
#evaluate_model("Lasso",lasso_model,50,50,satc_tfidf_LPC,True)[1]
```

### Dimensionality Reduction  <a class="anchor" id="dimensionality"></a> 

Calculating the TF-IDF scores for each word that occurs in the data set, we end up with a matrix of the shape 93x8926, 93x8930 if we want to includge the lines spoken per character per episode in our data set. We are dealing with p>>n, a data set where the number of then the number of variables is much higher then the number of observations. We furthermore have an extremely sparse matrix because of the nature of the TF-IDF matrix. 

We are going to look at different ways to reduce the dimensionality: 
 + Preselection of features based on their TF-IDF scores
 + Dimensionality Reduction based on Principal Component Analysis



```python
satc_tfidf.shape
```




    (92, 9692)




```python
satc_tfidf_LPC.shape
```




    (92, 9696)



#### Preselection of Features  <a class="anchor" id="pre-feat"></a> 


```python
def get_features(method ='tf', k=100):
    k=k
    if (method == 'tfidf'):
        #getting topK highest tfidf words
        top_k = feature_names[sorted_tfidf_index[:-(k+1):-1]]
    elif (method == 'tf'):
        #topK most common words
        counter = Counter([item for sublist in satc_text_per_episode['all_text'] for item in sublist])
        top_k = counter.most_common(k)
    else:
        top_k = None
        print("Bad input!! Choose tf or tfidf as first arg")
    
    return list(top_k)
```


```python
features_words = get_features('tfidf',50)
print(features_words)
```

    ['laney', 'threesome', 'pattern', 'phil', 'fake', 'model', 'yankee', 'zsa', 'soul', 'javier', 'church', 'ghost', 'sam', 'shrink', 'honeymoon', 'jim', 'postit', 'sandwich', 'cheat', 'rabbit', 'jeremy', 'whore', 'fetish', 'karma', 'weight', 'gay', 'raw', 'myth', 'bitsy', 'paris', 'atlantic', 'madeline', 'stewardess', 'shoe', 'dildo', 'berger', 'mao', 'sailor', 'change', 'ray', 'lsland', 'vogue', 'game', 'ball', 'money', 'email', 'freak', 'nina', 'marry', 'ovary']


 #### Principal Component Analysis  <a class="anchor" id="pca"></a> 
 
Principal component analysis (PCA) is a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables into a set of values of linearly uncorrelated variables called principal components. The resulting vectors (each being a linear combination of the variables and containing n observations) are an uncorrelated orthogonal basis set.

We are dealing with 9000 different dimensions on the variable side, of which we can probably disregard most: most tf-idf entries in the matrix carry little information. We are therefore interested in reducing the dimensionality of our predictor variables. 

### Linear Regression <a class="anchor" id="linreg"></a>

To get an idea of a base MSE that we can achieve with our data set, we perform a linear regression that uses the least squares metric. We use this model to perform a grid search: We look at the MSE that we achieve with our data set taking in to account different number of features (from 100 - 9000) and different number of PCA dimensions. By doing so, we can better understand which method of dimensionality reduction serves our objective of creating a predictive model for the given data set best.

#### Linear Regression, no Regularization


```python
linear_regression = LinearRegression() 

#Linear Regression with all features and 50 PCA Dimensions
eval_linreg = evaluate_model("linreg",linear_regression,900,10,satc_tfidf)
y_star,y_pred = eval_linreg[2:4]

# Plot the predicted vs the actual medv response
fig, ax = plt.subplots(1,1, figsize=(8,6))
ax.scatter(y_pred,y_star , facecolor='None', edgecolor='b')
# add a reference unity line
ax.plot([min(y_pred), max(y_pred)], [min(y_star), max(y_star)], linestyle='--', color='k');
ax.set_xlabel('y_predicted')
ax.set_ylabel('y_star')
```




    Text(0,0.5,'y_star')




![png](output_files/output_44_1.png)



```python
#gridsearch_linreg = pd.DataFrame(columns=["Name","features","pca_dim",'MSE','MAbsE', 'Explained_variance_score','R_squared'])

#for pcadim in range(0,50,10):
 #   if pcadim == 0:
  #      pcadim = None 
   # for featuredim in range (100,9000,500):
    #    gridsearch_linreg.loc[len(gridsearch_linreg)] = evaluate_model("linear_reg",linear_regression,featuredim,pcadim,satc_tfidf)[1]

#gridsearch_linreg.to_csv("gridsearch_linreg.csv")
```


```python
#gridsearch_linreg.head()
gridsearch_linreg= pd.read_csv("https://raw.githubusercontent.com/ranilovi/SATC_rating_prediction/master/gridsearch_linreg.csv").drop(["Unnamed: 0"],axis=1)
```


```python
seaborn.set(context="notebook",style='ticks')

fg = seaborn.FacetGrid(data=gridsearch_linreg, hue='pca_dim', 
                       #hue_order=_genders, 
                       aspect=1.61)
fg.map(pyplot.plot, 'features', 'MSE').add_legend()
```




    <seaborn.axisgrid.FacetGrid at 0x19bea0128d0>




![png](output_files/output_47_1.png)



```python
fg = seaborn.FacetGrid(data=gridsearch_linreg, hue='features',)
fg.map(pyplot.plot, 'pca_dim', 'MSE').add_legend()
```




    <seaborn.axisgrid.FacetGrid at 0x19bebaf1630>




![png](output_files/output_48_1.png)



```python
#20 best mse 
gridsearch_linreg.sort_values("MSE", inplace=True)

gridsearch_linreg.head()
#we get smallest mse 0.054556
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>features</th>
      <th>pca_dim</th>
      <th>MSE</th>
      <th>MAbsE</th>
      <th>Explained_variance_score</th>
      <th>R_squared</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>45</th>
      <td>linear_reg</td>
      <td>4600</td>
      <td>20.0</td>
      <td>0.054556</td>
      <td>0.189874</td>
      <td>-0.022672</td>
      <td>-0.077935</td>
    </tr>
    <tr>
      <th>39</th>
      <td>linear_reg</td>
      <td>1600</td>
      <td>20.0</td>
      <td>0.055822</td>
      <td>0.192558</td>
      <td>-0.017447</td>
      <td>-0.083173</td>
    </tr>
    <tr>
      <th>19</th>
      <td>linear_reg</td>
      <td>600</td>
      <td>10.0</td>
      <td>0.056286</td>
      <td>0.198081</td>
      <td>0.000750</td>
      <td>-0.063915</td>
    </tr>
    <tr>
      <th>59</th>
      <td>linear_reg</td>
      <td>2600</td>
      <td>30.0</td>
      <td>0.056352</td>
      <td>0.187811</td>
      <td>0.008375</td>
      <td>-0.045383</td>
    </tr>
    <tr>
      <th>61</th>
      <td>linear_reg</td>
      <td>3600</td>
      <td>30.0</td>
      <td>0.056699</td>
      <td>0.194365</td>
      <td>-0.023472</td>
      <td>-0.104713</td>
    </tr>
  </tbody>
</table>
</div>



By doing a grid search for all possible combinations of dimension and PCA and ploting them against the MSE, we cannot identify a pattern. Since we prefer a less complex model to a more complex one, if they both possess the same explanatory power, we do another grid search looking at a small number of features and all PCA dimensions.  We get the best results for PCA dimensions <20, regardless of the number of features taken into account. This makes sense, since applying a PCA will modify the variables in a way that we expect to be ignoring most features anyway.

We get an MSE of 0.053225, with an absolute Explained Erorr of around 0.186675 is our baseline score to judge further models. 




```python
gridsearch_linreg_small = pd.DataFrame(columns=["Name","features","pca_dim",'MSE','MAbsE', 'Explained_variance_score','R_squared'])

for featuredim in range (10,500,10):
    for pcadim in range(0,50,10):
        if featuredim < pcadim:
            pcadim = featuredim
        if pcadim == 0:
            pcadim = None 
        gridsearch_linreg_small.loc[len(gridsearch_linreg_small)] = evaluate_model("linear_reg",linear_regression,featuredim,pcadim,satc_tfidf)[1]
```


```python
gridsearch_linreg_small.sort_values("MSE", inplace=True)
gridsearch_linreg_small.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>features</th>
      <th>pca_dim</th>
      <th>MSE</th>
      <th>MAbsE</th>
      <th>Explained_variance_score</th>
      <th>R_squared</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>231</th>
      <td>linear_reg</td>
      <td>470</td>
      <td>10</td>
      <td>0.053819</td>
      <td>0.189354</td>
      <td>-0.032176</td>
      <td>-0.109682</td>
    </tr>
    <tr>
      <th>116</th>
      <td>linear_reg</td>
      <td>240</td>
      <td>10</td>
      <td>0.055721</td>
      <td>0.189530</td>
      <td>-0.008333</td>
      <td>-0.076285</td>
    </tr>
    <tr>
      <th>57</th>
      <td>linear_reg</td>
      <td>120</td>
      <td>20</td>
      <td>0.055905</td>
      <td>0.187267</td>
      <td>-0.041702</td>
      <td>-0.078962</td>
    </tr>
    <tr>
      <th>71</th>
      <td>linear_reg</td>
      <td>150</td>
      <td>10</td>
      <td>0.056932</td>
      <td>0.193023</td>
      <td>-0.031539</td>
      <td>-0.091527</td>
    </tr>
    <tr>
      <th>209</th>
      <td>linear_reg</td>
      <td>420</td>
      <td>40</td>
      <td>0.056973</td>
      <td>0.190777</td>
      <td>-0.089628</td>
      <td>-0.160391</td>
    </tr>
  </tbody>
</table>
</div>




```python
fg3 = seaborn.FacetGrid(data=
gridsearch_linreg_small[gridsearch_linreg_small["features"]%100 == 0], hue='features')
fg3.map(pyplot.scatter, 'pca_dim', 'MSE').add_legend()
```




    <seaborn.axisgrid.FacetGrid at 0x19bebb96240>




![png](output_files/output_53_1.png)


#### Penalized Regression using Lasso

As we are using sparse data, it makes sense to use regularization methods such a Lasso or Ridge regression. We are using the sklearn Lasso linear model with iterative fitting along a regularization path. The best model (with the best regularization parameter) is selected by cross-validation.

The optimization objective for Lasso is:

(1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1

We get a similar MSE (0.053387) as before for 500 features. 


```python
#lasso_model = LassoCV(n_alphas=100, alphas=[.01,.1,1,10,100]) 
lasso_model = LassoCV() 

gridsearch_lasso = pd.DataFrame(columns=["Name","features","pca_dim",'MSE','MAbsE', 'Explained_variance_score','R_squared'])

#pca dim should be in between 20 and 30 
featuredim = 500 

for pcadim in range(0,30):
    if pcadim == 0:
            pcadim = None 
    gridsearch_lasso.loc[len(gridsearch_lasso)] = evaluate_model("lasso_model",lasso_model,featuredim,pcadim,satc_tfidf)[1]

```


```python
gridsearch_lasso.sort_values("MSE", inplace=True)
gridsearch_lasso.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>features</th>
      <th>pca_dim</th>
      <th>MSE</th>
      <th>MAbsE</th>
      <th>Explained_variance_score</th>
      <th>R_squared</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>26</th>
      <td>lasso_model</td>
      <td>500</td>
      <td>26</td>
      <td>0.053431</td>
      <td>0.183750</td>
      <td>-0.002960</td>
      <td>-0.050035</td>
    </tr>
    <tr>
      <th>10</th>
      <td>lasso_model</td>
      <td>500</td>
      <td>10</td>
      <td>0.056352</td>
      <td>0.188959</td>
      <td>-0.004982</td>
      <td>-0.038872</td>
    </tr>
    <tr>
      <th>20</th>
      <td>lasso_model</td>
      <td>500</td>
      <td>20</td>
      <td>0.057118</td>
      <td>0.190622</td>
      <td>-0.007896</td>
      <td>-0.048792</td>
    </tr>
    <tr>
      <th>18</th>
      <td>lasso_model</td>
      <td>500</td>
      <td>18</td>
      <td>0.057181</td>
      <td>0.191799</td>
      <td>-0.007220</td>
      <td>-0.088472</td>
    </tr>
    <tr>
      <th>8</th>
      <td>lasso_model</td>
      <td>500</td>
      <td>8</td>
      <td>0.059729</td>
      <td>0.196377</td>
      <td>-0.010837</td>
      <td>-0.047953</td>
    </tr>
  </tbody>
</table>
</div>



### Decision Tree Regressor <a class="anchor" id="regtrees"></a>

Decision tree learning uses a decision tree (as a predictive model) to go from observations about an item (represented in the branches) to conclusions about the item's target value (represented in the leaves). We can use this method to get an idea of the different feature importances and maybe get to a better result. 

We are comparing two different models and get better results for a tree with reduced depth: If set the minimum samples split size to 10, we get a better MSE then e.g. for split size 2. We will compare different options in the final evaluation. 


```python
# Get the predictors and the response values
X = satc_tfidf[satc_tfidf.columns[4:]].filter(features_words,axis=1).values
y = satc_tfidf['Rating'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3, random_state=0)
```


```python
# Create an sklearn decision tree object using the mse metric for splitting, considering all the features and 
# splitting if there are more than 10 samples at a node.
tree = DecisionTreeRegressor(max_features=None, min_samples_split=10)
tree_est = tree.fit(X_train, y_train)
```

#### All Features, Low Depth
Decision Tree taking into account all features at every splitting step, minimum sample split = 10: 


```python
# use sklearn's export to generate the dot-data string file with all the nodes and their props.
dot_data = export_graphviz(tree_est, out_file='boston_tree.dot',feature_names=satc_tfidf[satc_tfidf.columns[4:]].filter(features_words,axis=1).columns[0:],filled=True, 
                           rounded=True, special_characters=True)

with open('boston_tree.dot') as f:
    dot_graph = f.read()  
# create the source object
I = graphviz.Source(dot_graph, format='png', engine='dot')
# Use ipython Image to shrink the rendered image of the source obj to fit into jupyter nb.
Image(I.render())
```




![png](output_files/output_61_0.png)




```python
feature_importances = pd.Series(data=tree.feature_importances_, index=list(satc_tfidf[satc_tfidf.columns[4:]].filter(features_words,axis=1)))
feature_importances.sort_values(axis=0, ascending=False)[0:15]
```




    soul         0.367642
    cheat        0.224781
    model        0.206799
    game         0.101130
    change       0.057444
    ball         0.042204
    ovary        0.000000
    sam          0.000000
    jeremy       0.000000
    rabbit       0.000000
    sandwich     0.000000
    postit       0.000000
    jim          0.000000
    honeymoon    0.000000
    shrink       0.000000
    dtype: float64



#### 50 Features, High Depth
We create a second model that splits up until "the end": We split as long as there are more then two samples at a node of the tree. We get a much more comprehensive image of the tree. 


```python
tree2 = DecisionTreeRegressor(max_features=20, min_samples_split=2)
tree_est2 = tree2.fit(X_train, y_train)
```


```python
dot_data = export_graphviz(tree_est2, out_file='boston_tree.dot',feature_names=satc_tfidf[satc_tfidf.columns[4:]].filter(features_words,axis=1).columns[0:],filled=True, 
                           rounded=True, special_characters=True)

with open('boston_tree.dot') as f:
    dot_graph = f.read()  
I = graphviz.Source(dot_graph, format='png', engine='dot')
Image(I.render())
```




![png](output_files/output_65_0.png)




```python
feature_importances2 = pd.Series(data=tree2.feature_importances_, index=list(satc_tfidf[satc_tfidf.columns[4:]].filter(features_words,axis=1)))
feature_importances2.sort_values(axis=0, ascending=False)[0:15]
```




    soul      0.282471
    cheat     0.160360
    vogue     0.151730
    game      0.113064
    weight    0.071008
    change    0.066618
    gay       0.037259
    phil      0.029990
    bitsy     0.028824
    money     0.024853
    raw       0.017647
    marry     0.008824
    whore     0.002941
    shoe      0.002206
    ball      0.002206
    dtype: float64




```python
#EVALUATION of the two different trees including plots on 20 pca dimensions 
eval_tree1 = evaluate_model("tree",tree,900,20,satc_tfidf)
eval_tree2 = evaluate_model("tree2",tree2,900,20,satc_tfidf)

y_star1,y_pred1 = eval_tree1[2:4]
y_star2,y_pred2 = eval_tree2[2:4]

# Plot the predicted vs the actual medv response
fig, ax = plt.subplots(1,1, figsize=(8,6))
ax.scatter(y_pred1,y_star1 , facecolor='None', edgecolor='b')

ax.plot([min(y_pred1), max(y_pred1)], [min(y_star1), max(y_star1)], linestyle='--', color='k');
ax.set_xlabel('y_predicted Tree 1')
ax.set_ylabel('y_star')
```




    Text(0,0.5,'y_star')




![png](output_files/output_67_1.png)



```python
# Plot the predicted vs the actual medv response
fig, ax = plt.subplots(1,1, figsize=(8,6))
ax.scatter(y_pred2,y_star2 , facecolor='None', edgecolor='b')

ax.plot([min(y_pred2), max(y_pred2)], [min(y_star2), max(y_star2)], linestyle='--', color='k');
ax.set_xlabel('y_predicted2 Tree2')
ax.set_ylabel('y_star2')
```




    Text(0,0.5,'y_star2')




![png](output_files/output_68_1.png)



```python
print("The first Tree Model results in an MSE of " + str(eval_tree1[1][3]),
     "\nThe second Tree Model results in an MSE of " + str(eval_tree2[1][3]))
```

    The first Tree Model results in an MSE of 0.09166761043173888 
    The second Tree Model results in an MSE of 0.10848214285714279


### Random Forest Regressor <a class="anchor" id="randomforest"></a>

A random forest is a meta estimator that fits a number of classifying decision trees on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. 
Depending on the number of estimators (the number of trees that are averaged out) we get different results for the model. However, we don't get any results better then the MSE of around 0,05 that we have seen before. Plotting the predicted y shows that we still get a fairly bad prediction. 

Doing a grid search shows that don't see different MSE scores for chaning the number of estimators between 20 and 100. 


```python
randomforest = RandomForestRegressor(n_estimators=500)
randomforest_est =  randomforest.fit(X_train, y_train)
```


```python
# Get the predictors and the response values
X = satc_tfidf[satc_tfidf.columns[4:]].filter(features_words,axis=1).values
y = satc_tfidf['Rating'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3, random_state=0)

y_pred = randomforest.predict(X_test)

# Plot the predicted vs the actual medv response
fig, ax = plt.subplots(1,1, figsize=(8,6))
ax.scatter(y_pred, y_test, facecolor='None', edgecolor='b')
# add a reference unity line
ax.plot([min(y_pred), max(y_pred)], [min(y_test), max(y_test)], linestyle='--', color='k');
ax.set_xlabel('y_predicted')
ax.set_ylabel('y_actual')

mse = np.mean((y_pred-y_test)**2)
print("MSE random forest, 200 esitmators = ", mse)
```

    MSE random forest, 200 esitmators =  0.05687050399999854



![png](output_files/output_72_1.png)



```python
#comparing different number of estimators shows we get almost the same MSE for any number of estimators in between 20 and 100 
```


```python
#gridsearch_randomforest= pd.DataFrame(columns=["Name","features","pca_dim",'MSE','MAbsE', 'Explained_variance_score','R_squared'])

#for pcadim in range(0,50,10):
 #   if pcadim == 0:
  #      pcadim = None 
   # for n_estimators in range(1,201,10):
    #    randomforest = RandomForestRegressor(n_estimators=n_estimators)
     #   gridsearch_randomforest.loc[len(gridsearch_randomforest)] = evaluate_model("randomforestst_"+str(n_estimators),randomforest,featuredim,pcadim,satc_tfidf)[1]
```


```python
#gridsearch_randomforest.sort_values("MSE", inplace=True)
#gridsearch_randomforest
```

### XG Boost <a class="anchor" id="xgb"></a>

As we are dealing with a weak regressor, we can use gradient boosting to create an ensemble of weak regression trees. 
XGBoost is a library that uses gradient boosting to achieve this. 

We are using XGBoost for this: XGBoost is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable. 

We get the best results for a tree with depth = 1, this corresponds to a simple linear regression. 


```python
features_df = satc_tfidf[satc_tfidf.columns[4:]].filter(features_words,axis=1)
features = features_df.values
target_df = satc_tfidf['Rating']
target = target_df.values
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.3)
test_df = pd.concat([target_df,features_df],axis=1)[0:60]
train_df = pd.concat([target_df,features_df],axis=1)[60:]
model = xgb.XGBRegressor(max_depth=1)
model.fit(features_train,target_train)
y_pred = model.predict(features_test)
y_test=target_test
# Plot the predicted vs the actual medv response
fig, ax = plt.subplots(1,1, figsize=(8,6))
ax.scatter(y_pred, y_test, facecolor='None', edgecolor='b')
# add a reference unity line
ax.plot([min(y_pred), max(y_pred)], [min(y_test), max(y_test)], linestyle='--', color='k');
ax.set_xlabel('y_predicted')
ax.set_ylabel('y_actual')

mse_ba = np.mean((y_pred-y_test)**2)
print("Test MSE = ", mse_ba)

```

    Test MSE =  0.06593022176052468



![png](output_files/output_77_1.png)



```python
gridsearch_xgb = pd.DataFrame(columns=["Name","features","pca_dim",'MSE','MAbsE', 'Explained_variance_score','R_squared'])

#pca dim should be in between 20 and 30 
featuredim = 500 
pcadim = 20
for pcadim in range(0,50,10):
    if pcadim == 0:
        pcadim = None 
    for depth in range(1,10):
        xgb_model = xgb.XGBRegressor(max_depth=depth)
        name = "xgb" + str(depth)
        gridsearch_xgb.loc[len(gridsearch_xgb)] = evaluate_model(name,xgb_model,featuredim,pcadim,satc_tfidf)[1]
```


```python
gridsearch_xgb.sort_values("MSE", inplace=True)
gridsearch_xgb.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>features</th>
      <th>pca_dim</th>
      <th>MSE</th>
      <th>MAbsE</th>
      <th>Explained_variance_score</th>
      <th>R_squared</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>xgb2</td>
      <td>500</td>
      <td>None</td>
      <td>0.057908</td>
      <td>0.193517</td>
      <td>-0.108268</td>
      <td>-0.164023</td>
    </tr>
    <tr>
      <th>5</th>
      <td>xgb6</td>
      <td>500</td>
      <td>None</td>
      <td>0.061709</td>
      <td>0.198109</td>
      <td>-0.097871</td>
      <td>-0.203339</td>
    </tr>
    <tr>
      <th>18</th>
      <td>xgb1</td>
      <td>500</td>
      <td>20</td>
      <td>0.062844</td>
      <td>0.202364</td>
      <td>-0.064728</td>
      <td>-0.161708</td>
    </tr>
    <tr>
      <th>27</th>
      <td>xgb1</td>
      <td>500</td>
      <td>30</td>
      <td>0.063255</td>
      <td>0.205689</td>
      <td>-0.084683</td>
      <td>-0.222863</td>
    </tr>
    <tr>
      <th>8</th>
      <td>xgb9</td>
      <td>500</td>
      <td>None</td>
      <td>0.065240</td>
      <td>0.209635</td>
      <td>-0.127989</td>
      <td>-0.201448</td>
    </tr>
  </tbody>
</table>
</div>



### Prediction Including the Lines per Character <a class="anchor" id="lpc"></a>

We added four additional columns to the dataframe in order to take into account the lines spoken per character per episode. Plotting the lines spoken per person against their Ratings 


```python
satc_tfidf_LPC.head()
corr_carrie = satc_tfidf_LPC['Rating'].corr(satc_tfidf_LPC['lines_carrie'])
corr_miranda = satc_tfidf_LPC['Rating'].corr(satc_tfidf_LPC['lines_miranda'])
corr_charlotte = satc_tfidf_LPC['Rating'].corr(satc_tfidf_LPC['lines_charlotte'])
corr_samantha = satc_tfidf_LPC['Rating'].corr(satc_tfidf_LPC['lines_samantha'])

print("Correlation between the ratings and lines spoken by Samantha {corr_samantha}\n".format(corr_samantha=corr_samantha))
print("Correlation between the ratings and lines spoken by Miranda {corr_miranda}\n".format(corr_miranda=corr_miranda))
print("Correlation between the ratings and lines spoken by Charlotte {corr_charlotte}\n".format(corr_charlotte=corr_charlotte))
print("Correlation between the ratings and lines spoken by Carrie {corr_carrie}\n".format(corr_carrie=corr_carrie))

```

    Correlation between the ratings and lines spoken by Samantha -0.011113981965701352
    
    Correlation between the ratings and lines spoken by Miranda 0.033266053436784786
    
    Correlation between the ratings and lines spoken by Charlotte -0.022534691937596208
    
    Correlation between the ratings and lines spoken by Carrie 0.11761303375075037
    



```python
satc_tfidf_LPC.plot(x="Rating",y="lines_carrie",kind="scatter",title="Number of Lines spoken by Carrie / Rating")
satc_tfidf_LPC.plot(x="Rating",y="lines_miranda",kind="scatter",title="Number of Lines spoken by Miranda / Rating")
satc_tfidf_LPC.plot(x="Rating",y="lines_charlotte",kind="scatter",title="Number of Lines spoken by Charlotte / Rating")
satc_tfidf_LPC.plot(x="Rating",y="lines_samantha",kind="scatter",title="Number of Lines spoken by Samantha / Rating")

```




    <matplotlib.axes._subplots.AxesSubplot at 0x19bead03f60>




![png](output_files/output_82_1.png)



![png](output_files/output_82_2.png)



![png](output_files/output_82_3.png)



![png](output_files/output_82_4.png)


### Model Comparison <a class="anchor" id="modelcomp"></a>

To conclude, we can say that it is not possible to predict the rating on IMDB based on the sripts. 95% of the ratings fall into a std of 0.28 +/- the mean of all ratings. We are not able to get a Mean Absolute Error of below ~0.19 - all of our prediction models are still predicting pretty much random outcomes. 

One of the reasons for this might be the narrow range in which we find the ratings of all episodes (all ratings lie between 7-8.8). It is also possible, that even though viewers liked episodes differently based on their content, they did not know how to express these differences in the rating. 


```python
model_comparison_df = pd.DataFrame(columns=["Name","features","pca_dim",'MSE','MAbsE', 'Explained_variance_score','R_squared'])
#evaluate_model(model, number_of_features,pca_dim,df):

#get model scores, average of ten iterations 
#lasso = evaluate_model("lasso",lasso_model,50,50,satc_tfidf)[1]

#add to model comparison df 
#model_comparison_df.loc[len(model_comparison_df)] = lasso

### Linear Regression Models OLS
linear_regression = LinearRegression() 
model_comparison_df.loc[len(model_comparison_df)] = evaluate_model("linear_regression",linear_regression,500,20,satc_tfidf)[1]
model_comparison_df.loc[len(model_comparison_df)] = evaluate_model("linear_regression",linear_regression,500,10,satc_tfidf)[1]
model_comparison_df.loc[len(model_comparison_df)] = evaluate_model("linear_regression_LPC",linear_regression,500,20,satc_tfidf_LPC,True)[1]
model_comparison_df.loc[len(model_comparison_df)] = evaluate_model("linear_regression_LPC",linear_regression,500,10,satc_tfidf_LPC,True)[1]

### Lasso: We saw earlier that we get best results for 500 models, 10 PCA dim 
lasso_model = LassoCV()
model_comparison_df.loc[len(model_comparison_df)] = evaluate_model("lasso",lasso_model,500,10,satc_tfidf)[1]
model_comparison_df.loc[len(model_comparison_df)] = evaluate_model("lasso_LPC",lasso_model,500,10,satc_tfidf_LPC,True)[1]

### Decision Tree Regressors: We consider PCA Dimensions 10-30 and min splits 5 and 10 
tree_min10split = DecisionTreeRegressor(max_features=None, min_samples_split=10)
tree_min5split = DecisionTreeRegressor(max_features=None, min_samples_split=5)

model_comparison_df.loc[len(model_comparison_df)] = evaluate_model("tree_min10split",tree_min10split,500,30,satc_tfidf)[1]
model_comparison_df.loc[len(model_comparison_df)] = evaluate_model("tree_min10split",tree_min10split,500,20,satc_tfidf)[1]
model_comparison_df.loc[len(model_comparison_df)] = evaluate_model("tree_min10split",tree_min10split,500,10,satc_tfidf)[1]

model_comparison_df.loc[len(model_comparison_df)] = evaluate_model("tree_min5split",tree_min5split,500,30,satc_tfidf)[1]
model_comparison_df.loc[len(model_comparison_df)] = evaluate_model("tree_min5split",tree_min5split,500,20,satc_tfidf)[1]
model_comparison_df.loc[len(model_comparison_df)] = evaluate_model("tree_min5split",tree_min5split,500,10,satc_tfidf)[1]

model_comparison_df.loc[len(model_comparison_df)] = evaluate_model("tree_min10split_LPC",tree_min10split,900,30,satc_tfidf_LPC,True)[1]
model_comparison_df.loc[len(model_comparison_df)] = evaluate_model("tree_min10split_LPC",tree_min10split,900,20,satc_tfidf_LPC,True)[1]
model_comparison_df.loc[len(model_comparison_df)] = evaluate_model("tree_min10split_LPC",tree_min10split,900,10,satc_tfidf_LPC,True)[1]

### Random Forest 
randomforest_500  = RandomForestRegressor(n_estimators=500)
randomforest_100  = RandomForestRegressor(n_estimators=100)

model_comparison_df.loc[len(model_comparison_df)] = evaluate_model("randomForest_500",randomforest_500,500,20,satc_tfidf)[1]
model_comparison_df.loc[len(model_comparison_df)] = evaluate_model("randomForest_100",randomforest_100,500,20,satc_tfidf)[1]

model_comparison_df.loc[len(model_comparison_df)] = evaluate_model("randomForest_500_LPC",randomforest_500,500,20,satc_tfidf_LPC,True)[1]
model_comparison_df.loc[len(model_comparison_df)] = evaluate_model("randomForest_100_LPC",randomforest_100,500,20,satc_tfidf_LPC,True)[1]

## XGB Boost 
xgb_1 = xgb.XGBRegressor(max_depth=1)
xgb_2 = xgb.XGBRegressor(max_depth=2)
model_comparison_df.loc[len(model_comparison_df)] = evaluate_model("xgb_depth=1",xgb_1,500,None,satc_tfidf)[1]
model_comparison_df.loc[len(model_comparison_df)] = evaluate_model("xgb_depth=2",xgb_2,500,None,satc_tfidf)[1]
model_comparison_df.loc[len(model_comparison_df)] = evaluate_model("xgb_depth=1_LPC",xgb_1,500,None,satc_tfidf_LPC,True)[1]
model_comparison_df.loc[len(model_comparison_df)] = evaluate_model("xgb_deoth=2_LPC",xgb_2,500,None,satc_tfidf_LPC,True)[1]

```


```python
model_comparison_df.sort_values("MSE", inplace=True)
model_comparison_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>features</th>
      <th>pca_dim</th>
      <th>MSE</th>
      <th>MAbsE</th>
      <th>Explained_variance_score</th>
      <th>R_squared</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>lasso_LPC</td>
      <td>500</td>
      <td>10</td>
      <td>0.055177</td>
      <td>0.191726</td>
      <td>-0.001737</td>
      <td>-0.042935</td>
    </tr>
    <tr>
      <th>18</th>
      <td>randomForest_100_LPC</td>
      <td>500</td>
      <td>20</td>
      <td>0.056539</td>
      <td>0.194632</td>
      <td>-0.073289</td>
      <td>-0.210657</td>
    </tr>
    <tr>
      <th>21</th>
      <td>xgb_depth=1_LPC</td>
      <td>500</td>
      <td>None</td>
      <td>0.057827</td>
      <td>0.189875</td>
      <td>0.009964</td>
      <td>-0.062794</td>
    </tr>
    <tr>
      <th>4</th>
      <td>lasso</td>
      <td>500</td>
      <td>10</td>
      <td>0.060384</td>
      <td>0.197222</td>
      <td>-0.005776</td>
      <td>-0.071266</td>
    </tr>
    <tr>
      <th>1</th>
      <td>linear_regression</td>
      <td>500</td>
      <td>10</td>
      <td>0.060944</td>
      <td>0.195784</td>
      <td>-0.009093</td>
      <td>-0.056365</td>
    </tr>
  </tbody>
</table>
</div>



## Episode Lookup <a class="anchor" id="keyword"></a>

Since we aren't very successful at predicting episode ratings, let's just watch some SATC! But which episode to watch?

This simple application recommends episodes based on a keyword query. Using a robust word vectorizer trained on the Google News dataset, it uses a heuristic approach to generate words similar to the lookup term, and rank the most relevant episodes based on the tf-idf score. 

Happy binging!


```python
my_q = 300 # to match dim of GNews word vectors
mcount = 1 # minimal word frequency  

w2v = Word2Vec(size = my_q, min_count = mcount)
w2v.build_vocab(list(satc_text_per_episode["all_text"]))
```


```python
w2v.intersect_word2vec_format("C:\\Users\\dorar_000\\Documents\\GoogleNews-vectors-negative300.bin.gz", binary = True)
#w2v.intersect_word2vec_format("/Users/robertaconrad/documents/07_Ecole_Polytechnique_Studienunterlagen/DataCampCapgemini/session7/GoogleNews-vectors-negative300.bin.gz", binary = True)
```


```python
def recommender(word):
    ## generating similar words and their weights
    words = [word] +list(map(lambda x: x[0], w2v.similar_by_word(word)))
    weights = [1] + list( map(lambda x: x[1], w2v.similar_by_word(word)))
    
    ## raniking the episodes on the weighed average tf-idf
    reco = satc_tfidf.filter(words, axis=1).dot(weights).sort_values(ascending = False)[:10]
    
    ##getting the correct ep indices
    ep_ids = eps.iloc[reco.index]['ID']
    
    for (i, index) in enumerate(list(reco.index)): ##printing episode informtion
        
        print ("Recommendation {:.0f}: Season {:.0f}, Episode {:.0f}".format(i+1,eps.iloc[index]['Season'],eps.iloc[index]['Episode']))
        print (episodes[index].summary())
        print ()
        print ()
    
```


```python
## safe input handling KeyError Exception, and checking input is inside the vocab.
while True:
    try:
        word = input("Enter keyword: ")
        recommender(word)
        break
    except KeyError:
        print("word '%s' not in vocabulary" % word)
        continue
    
```

    Enter keyword: fun
    Recommendation 1: Season 2, Episode 4
    Movie
    =====
    Title: "Sex and the City" They Shoot Single People, Don't They? (1999)
    Genres: Comedy, Drama, Romance.
    Director: John David Coles.
    Writer: Darren Star, Michael Patrick King, Candace Bushnell.
    Cast: Sarah Jessica Parker (Carrie Bradshaw), Kim Cattrall (Samantha Jones), Kristin Davis (Charlotte York), Cynthia Nixon (Miranda Hobbes), Willie Garson (Stanford Blatch).
    Runtime: 30.
    Country: United States.
    Language: English.
    Rating: 7.7 (410 votes).
    Plot: Now none of the girls has a relationship, the quartet hits a salsa bar all night, so Carrie is late for a photo-shoot for Stanford's boyfriend Nevin's magazine; the result is worse then she ever feared. Only Samantha is absent from the power-walking, where Miranda bumps into an ex, hunky ophthalmologist Josh, who just didn't give her orgasms; history repeats itself, and she still fakes coming; once she tells him, his confidence is shaken, so he gets her to coach him- now sex feels like an exam subject. Charlotte enjoys the handy jobs done by and kissing with unemployed actor Tom, too much to see him move to Salt Lake City for a soap part, but is that enough? Samantha was picked up by the salsa club's co-owner William, and invites Carrie to join them for the summer in the East Hamptons, but is stood up. At a party with Stanford, drunk Carrie meets single smoker Jake.
    
    
    Recommendation 2: Season 2, Episode 8
    Movie
    =====
    Title: "Sex and the City" The Man, the Myth, the Viagra (1999)
    Genres: Comedy, Drama, Romance.
    Director: Victoria Hochberg.
    Writer: Darren Star, Michael Patrick King, Candace Bushnell.
    Cast: Sarah Jessica Parker (Carrie Bradshaw), Kim Cattrall (Samantha Jones), Kristin Davis (Charlotte York), Cynthia Nixon (Miranda Hobbes), Chris Noth (Mr. Big).
    Runtime: 30.
    Country: United States.
    Language: English.
    Rating: 7.8 (387 votes).
    Plot: After walking out on her date who turned out to be still married despite his claim to be divorced, Miranda is happy to be picked up by cocky barman Steve, who soon wants more then their one-night stand. Carrie believes her rekindled relationship with Mr. Big is good enough to get him to know her friends a little better. Samantha has an atypical lover, even for a man-eater like her: Donald Trump's single friend Ed is past 70, so will his old-fashioned wooing and lavish gifts outweigh his physical decay?
    
    
    Recommendation 3: Season 6, Episode 7
    Movie
    =====
    Title: "Sex and the City" The Post-It Always Sticks Twice (2003)
    Genres: Comedy, Drama, Romance.
    Director: Alan Taylor.
    Writer: Darren Star, Liz Tuccillo, Candace Bushnell.
    Cast: Sarah Jessica Parker (Carrie Bradshaw), Kim Cattrall (Samantha Jones), Kristin Davis (Charlotte York), Cynthia Nixon (Miranda Hobbes), Jason Lewis (Jerry 'Smith' Jerrod).
    Runtime: 30.
    Country: United States.
    Language: English.
    Rating: 7.8 (383 votes).
    Plot: The quartet's morning meeting hears counterbalancing news: Charlotte is elated to show her engagement ring from Harry while Carrie, infuriatingly pissed off, is determined not to make the day 'the day she got broke up with a post-it', as Berger chose this excessively succinct medium to beat her to the dumping line. Samantha takes the girls to a new club called 'Bed' where Miranda, who just discovered because of being too busy to eat she can fit into her skinny jeans once again, meets a nice man, Peter. Carrie drags everyone out after having lectured Berger's friends about imaginary break-up etiquette and manages to get street kids to sell them pot. Samantha hears Smith on TV putting on the 'I'm still single' act she advised him and finds she doesn't like how it sounds.
    
    
    Recommendation 4: Season 6, Episode 4
    Movie
    =====
    Title: "Sex and the City" Pick-a-Little, Talk-a-Little (2003)
    Genres: Comedy, Drama, Romance.
    Director: David Frankel.
    Writer: Darren Star, Julie Rottenberg, Elisa Zuritsky.
    Cast: Sarah Jessica Parker (Carrie Bradshaw), Kim Cattrall (Samantha Jones), Kristin Davis (Charlotte York), Cynthia Nixon (Miranda Hobbes), Ron Livingston (Jack Berger).
    Runtime: 30.
    Country: United States.
    Language: English.
    Rating: 7.4 (295 votes).
    Plot: Charlotte is shocked and Berger slightly embarrassed when Samantha enthusiastically explains how she enjoys role-plays with her sexy actor Jerry "Smith" Jerrod, including fake rape. Miranda is relieved when he says a man not calling her back just means not interested, as men don't send sexual double messages. When Carrie can't help herself and criticizes a futile detail in Berger's book, even though she loves it, he shuts down; Samantha cuts of Jerry when he confesses being in AA; both couples soon make up. Charlotte makes her first sabbath a grand production- her pride and nitpicking don't make Harry set a wedding date as she desperately wanted but even walk out altogether.
    
    
    Recommendation 5: Season 4, Episode 13
    Movie
    =====
    Title: "Sex and the City" The Good Fight (2002)
    Genres: Comedy, Drama, Romance.
    Director: Charles McDougall.
    Writer: Darren Star, Michael Patrick King, Candace Bushnell.
    Cast: Sarah Jessica Parker (Carrie Bradshaw), Kim Cattrall (Samantha Jones), Kristin Davis (Charlotte York), Cynthia Nixon (Miranda Hobbes), John Corbett (Aidan Shaw).
    Runtime: 30.
    Country: United States.
    Language: English.
    Rating: 7.7 (329 votes).
    Plot: Carrie finds living together with Aidan cramps their space, literally as well as figuratively in secret singular behavior. Samantha is hot enough on Richard to give him a blow job, even in his glass wall-office, but resists any romance. Trey is ready to get on with life without a baby, but Charlotte is offended by any alternative and throws a ladies night which ends quickly as they witness a livid marital row. Miranda has a last hot sex partner, Walker Lewis, before her pregnancy is too far advanced.
    
    
    Recommendation 6: Season 1, Episode 4
    Movie
    =====
    Title: "Sex and the City" Valley of the Twenty-Something Guys (1998)
    Genres: Comedy, Drama, Romance.
    Director: Alison Maclean.
    Writer: Darren Star, Michael Patrick King, Candace Bushnell.
    Cast: Sarah Jessica Parker (Carrie Bradshaw), Kim Cattrall (Samantha Jones), Kristin Davis (Charlotte York), Cynthia Nixon (Miranda Hobbes), Chris Noth (Mr. Big).
    Runtime: 30.
    Country: United States.
    Language: English.
    Rating: 7.5 (483 votes).
    Plot: Carrie and Mr. Big keep bumping into each-other till they date (without that term) at Samantha's hot restaurant PR-opening, but he cancels last minute. The cook, just Jon, is reserved for Samantha's bed, but his hot friend Sam, also twenty-something (those always seem to know the right people) flirts with Carrie; Big turns up saying he was on time but couldn't find her, time to leave town. At a twenties club Sam takes Carrie on his lap and kisses really well, five hours long, and again while she helps him pick a shirt. Samantha eagerly reports on a night of sex with Jon, twenties-men are so hot and consider thirties-women a good deal. Charlotte is embarrassed Bryant explicitly asked for anal sex, Miranda warns her not to give in on respect which is power, Samantha says a hole is just that; Charlotte gets a decent first date. Samantha realizes she'll always feel older with twenties-men, and resolves to give them up. After Big brought his just divorced friend Jack to their drink, Carrie seeks consolation in Sam's bed, in heaven when spooned, only to wake up in his nightmare apartment which comes with a room mate, but no toilet paper, and runs off to her ultimate addiction- the shoe store.
    
    
    Recommendation 7: Season 6, Episode 10
    Movie
    =====
    Title: "Sex and the City" Boy, Interrupted (2003)
    Genres: Comedy, Drama, Romance.
    Director: Timothy Van Patten.
    Writer: Darren Star, Cindy Chupack, Candace Bushnell.
    Cast: Sarah Jessica Parker (Carrie Bradshaw), Kim Cattrall (Samantha Jones), Kristin Davis (Charlotte York), Cynthia Nixon (Miranda Hobbes), Willie Garson (Stanford Blatch).
    Runtime: 30.
    Country: United States.
    Language: English.
    Rating: 7.4 (354 votes).
    Plot: Carrie agrees to a reunion to see how her high-school crush -whom she dumped- is, and finds Jeremy got even more attractive. Back in hot Manhattan Samantha can't get through the waiting list for Soho House club which has a pool with service, but picks up someone's (Annabelle Bronstein's) forgotten pass. When Stanford has paraded his hunk Marcus in front of Charlotte and Anthony, the latter proves with a photo from Honcho magazine that the stud used to work as a gay escort named Paul, Charlotte shows it to Miranda at the Knicks game in Madison Square Gardens where they admire her flame, doctor Robert Leeds, who treats the team but seems interested in a cheer-leader. When Jeremy stays at Carries, he tells her he's in intensive mental therapy in an excellent facility. Just when she invited the other girls and Stanford, who got to see the Honcho photo, Samanatha is found out not to be Annabella -who's British- and thrown out. Stanford his broken up with Marcus and asks Carrie to a gay prom ball- they even make king and queen; Marcus turns up...
    
    
    Recommendation 8: Season 4, Episode 3
    Movie
    =====
    Title: "Sex and the City" Defining Moments (2001)
    Genres: Comedy, Drama, Romance.
    Director: Allen Coulter.
    Writer: Darren Star, Jenny Bicks, Candace Bushnell.
    Cast: Sarah Jessica Parker (Carrie Bradshaw), Kim Cattrall (Samantha Jones), Kristin Davis (Charlotte York), Cynthia Nixon (Miranda Hobbes), Chris Noth (Mr. Big).
    Runtime: 30.
    Country: United States.
    Language: English.
    Rating: 7.0 (327 votes).
    Plot: Miranda is dating Doug, a respected professional, but just can't get comfortable with his loose sense of toilet intimacy. On a 'non-date' with Mr. Big Carrie meets and gets attracted to Jazz musician Ray King, who turns out to own two clubs and is interested in her too. Trey is finally hot enough for Charlotte, but now she finds him too demonstrative in word and deed. Samantha hits on with Maria, a Brazilian painter she meets in Charlotte's gallery, who turns out to be lesbian.
    
    
    Recommendation 9: Season 3, Episode 15
    Movie
    =====
    Title: "Sex and the City" Hot Child in the City (2000)
    Genres: Comedy, Drama, Romance.
    Director: Michael Spiller.
    Writer: Darren Star, Allan Heinberg, Candace Bushnell.
    Cast: Sarah Jessica Parker (Carrie Bradshaw), Kim Cattrall (Samantha Jones), Kristin Davis (Charlotte York), Cynthia Nixon (Miranda Hobbes), Kyle MacLachlan (Trey MacDougal).
    Runtime: 30.
    Country: United States.
    Language: English.
    Rating: 8.0 (390 votes).
    Plot: 34-year old Miranda feels a teenage girl again when her dentist gives her braces. Samantha is jealous of a 13-year old, super-rich client, who seems to have skipped the teen years altogether. Carrie's gorgeous younger boyfriend Wade "Superboy" Adams is a comics-shop-clerk who makes her feel young again, as they do teen- and tween-things together, or perhaps too young: he still lives with his devoted parents... Meanwhile Charlotte resorts to heavy medical therapy for Trey's sexual non-performance, and gets surprising results.
    
    
    Recommendation 10: Season 5, Episode 2
    Movie
    =====
    Title: "Sex and the City" Unoriginal Sin (2002)
    Genres: Comedy, Drama, Romance.
    Director: Charles McDougall.
    Writer: Darren Star, Cindy Chupack, Candace Bushnell.
    Cast: Sarah Jessica Parker (Carrie Bradshaw), Kim Cattrall (Samantha Jones), Kristin Davis (Charlotte York), Cynthia Nixon (Miranda Hobbes), David Eigenberg (Steve Brady).
    Runtime: 30.
    Country: United States.
    Language: English.
    Rating: 7.3 (295 votes).
    Plot: Samantha's announcement she's giving Richard another chance after his unfaithfulness, as nobody's perfect and it was 'only sex', is such a bomb-shell that even Charlotte makes an obscene gesture. Although Miranda is atheist, she can't deny Steve's entire Catholic family having the baby baptized. Just when Carrie felt down having to write about relationships without having one herself, she gets invited by a publisher to select some of her columns as a book. Eternally positive Charlotte drags Carrie along to the Fountain of Faith, but stands up to say faith didn't suffice for her.
    
    



```python

```
