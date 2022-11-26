# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 03:04:14 2019

@author: adraj
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 00:08:42 2019

@author: adraj
"""

import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import pandas_profiling as pp
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split

warnings.simplefilter(action='ignore', category=FutureWarning)



# Point to data
path = os.getcwd() +'\\Participants_Data\\'
train_data = pd.read_excel(path + 'Data_train.xlsx', header = 0, encoding='latin1') #define encoding type to match output from excel
test_data = pd.read_excel(path + 'Data_Test.xlsx', header = 0, encoding='latin1') #define encoding type to match output from excel
#polarity_data = pd.read_csv(path + 'name_sentiment_all_data.csv', header = 0, encoding='latin1')
training_set = pd.concat([train_data, test_data], ignore_index=True,sort=False)


ntrain = len(train_data)
ntest = len(test_data)

#profile = pp.ProfileReport(df2)
#rejected_variables = profile.get_rejected_variables(threshold=0.9)
#profile = train_data.profile_report(title='Pandas Profiling Report')
#profile.to_file(output_file="output_case.html")



import re

def replace_missing_text(string):
    total_list=[]
    miss = 'out of 5 stars'
    for i in range (len(string)):
        if miss not in string[i]:
            total_list.append(string[i])            
    return ( ''.join(total_list) )   

reviews_extra = 'out of 5 stars'
ratings_extra = 'customer reviews'
ratings_extra_singular = 'customer review'


training_set['Reviews'] = training_set.apply(lambda x:x.Reviews.replace(reviews_extra,''),axis=1)
training_set['Ratings'] = training_set.apply(lambda x:x.Ratings.replace(ratings_extra,''),axis=1)
training_set['Ratings'] = training_set.apply(lambda x:x.Ratings.replace(ratings_extra_singular,''),axis=1)



training_set['Year_of_Print'] = training_set['Edition'].str.extract('(\d\d\d\d)', expand=True) 

training_set['Year_of_Print'] = training_set['Year_of_Print'].fillna(training_set['Year_of_Print'].mode()[0])

training_set['Edition_quality'] = training_set.apply(lambda x:x.Edition.split(',')[0],axis=1)

training_set.dtypes
 
del training_set['Edition']

###################################

training_set['com_word_count'] = training_set['Synopsis'].apply(lambda x: len(str(x).split(" ")))

#sentence_count
training_set['com_sent_count'] = training_set['Synopsis'].apply(lambda x: len(str(x).split(".")))

#character_count
training_set['char_count'] = training_set['Synopsis'].str.len() ## this also includes spaces

#average_word_length
def avg_word(sentence):
  words = sentence.split()
  return (sum(len(word) for word in words)/len(words))

training_set['avg_word'] = training_set['Synopsis'].apply(lambda x: avg_word(x))
#training_set[['Commentary','avg_word']].head()

#calculating_stop_words
from nltk.corpus import stopwords
stop = stopwords.words('english')

training_set['stopwords'] = training_set['Synopsis'].apply(lambda x: len([x for x in x.split() if x in stop]))
#training_set[['Commentary','stopwords']].head()

#numeric_in_commentary
training_set['numerics'] = training_set['Synopsis'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
#training_set[['Commentary','numerics']].head()

#commentry_to_lower
training_set['Synopsis'] = training_set['Synopsis'].apply(lambda x: " ".join(x.lower() for x in x.split()))
training_set['Synopsis'].head()

#remove_punctutation
training_set['Synopsis'] = training_set['Synopsis'].str.replace('[^\w\s]','')
training_set['Synopsis'] = training_set['Synopsis'].str.replace('\d+', '')

#remove_stopwords
from nltk.corpus import stopwords
stop = stopwords.words('english')
training_set['Synopsis'] = training_set['Synopsis'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
training_set['Synopsis'].head()

training_set['Commentary_preprocessed'] = training_set['Synopsis']

"""
#frequently_occuring_words
freq = pd.Series(' '.join(training_set['Commentary']).split()).value_counts()[:10]
freq
#rarely_occuring_words
freq = pd.Series(' '.join(training_set['Commentary']).split()).value_counts()[-20:]"""

#ONLY TO BE USED WHEN FULL ENGLISH LANGUAGE PARA IS PROVIDED NOT WITH NAME/ PLACE
from textblob import TextBlob
training_set['Commentary_preprocessed'][:5].apply(lambda x: str(TextBlob(x).correct()))

#stemming
from nltk.stem import PorterStemmer
st = PorterStemmer()
training_set['Commentary_porterstemmer'] = training_set['Commentary_preprocessed'].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))


"""from nltk.stem.snowball import SnowballStemmer
st = SnowballStemmer('english')
training_set['Commentary_snowball'] = training_set['Commentary_preprocessed'].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))
"""
#correct
#from textblob import TextBlob
#TextBlob(text).correct()
#lemmatization
from textblob import Word
training_set['Commentary_preprocessed'] = training_set['Commentary_porterstemmer'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
training_set['Commentary_preprocessed'].head()

import nltk

w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]

training_set['text_lemmatized'] = training_set.Commentary_preprocessed.apply(lemmatize_text)
training_set['text_lemmatized'].head()
training_set['text_lemmatized'] = training_set['text_lemmatized'].apply(', '.join)

training_set['text_lemmatized']

training_set.columns

df= training_set[['text_lemmatized','Title','Author','Reviews','Ratings','Genre','BookCategory',
                  'Year_of_Print','Edition_quality','com_word_count','stopwords','Price',
                  'avg_word','com_sent_count','numerics']]
df.dtypes

from sklearn.feature_extraction.text import TfidfVectorizer


"""
tfidf_vec = TfidfVectorizer(analyzer='word', 
                            stop_words='english',
                            max_features=1000,
                            sublinear_tf= True, #use a logarithmic form for frequency
                            min_df = 5, #minimum numbers of documents a word must be present in to be kept
                            norm= 'l2', #ensure all our feature vectors have a euclidian norm of 1
                            ngram_range= (1,3), #to indicate that we want to consider both unigrams and bigrams.
                            strip_accents='ascii')"""
tfidf_vec = TfidfVectorizer(analyzer='word', 
                            min_df = 5, #minimum numbers of documents a word must be present in to be kept
                            norm= 'l2', #ensure all our feature vectors have a euclidian norm of 1
                            ngram_range= (1,4),
                            max_features=5000,#to indicate that we want to consider both unigrams and bigrams.
                            strip_accents='ascii',
                            sublinear_tf= True,
                            )

tfidf_dense = tfidf_vec.fit_transform(df['text_lemmatized']).todense()
new_cols = tfidf_vec.get_feature_names()
# remove the text column as the word 'text' may exist in the words and you'll get an error
df = df.drop('text_lemmatized',axis=1)
# join the tfidf values to the existing dataframe
df = df.join(pd.DataFrame(tfidf_dense, columns=new_cols))


#test dummy encoding for categorical values
def encode_text_dummy(df, name):
    dummies = pd.get_dummies(df[name])
    for x in dummies.columns:
        dummy_name = "{}-{}".format(name, x)
        df[dummy_name] = dummies[x]
    df.drop(name, axis=1, inplace=True)
    print('Encoding Text Dummy Complete for {}'.format(name)) 
    
encode_text_dummy(df, 'Author')
encode_text_dummy(df, 'Genre')
encode_text_dummy(df, 'BookCategory')
encode_text_dummy(df, 'Year_of_Print')
encode_text_dummy(df, 'Edition_quality')
encode_text_dummy(df, 'Title')



df['Reviews'] = pd.to_numeric(df['Reviews'])
df['Ratings'] = df['Ratings'].str.replace(',','')
df['Ratings'] = pd.to_numeric(df['Ratings'])

#numerics = ['uint8','int16', 'int32', 'int64', 'float16', 'float32', 'float64']
#ny_train_x=df.select_dtypes(include=numerics).fillna(0).values

train_final =df[:ntrain]
test_final=df[ntrain:]

del train_final['Price']
del test_final['Price']

X = train_final.copy()
y = train_data['Price']
y_log = np.log(y)


##############################
df2= training_set[['Reviews','Ratings','Genre','BookCategory',
                  'Year_of_Print','Edition_quality','com_word_count','stopwords']]

df3 = df2[df2['Year_of_Print'] == '1925']


def encode_text_dummy(df, name):
    dummies = pd.get_dummies(df[name])
    for x in dummies.columns:
        dummy_name = "{}-{}".format(name, x)
        df[dummy_name] = dummies[x]
    df.drop(name, axis=1, inplace=True)
    print('Encoding Text Dummy Complete for {}'.format(name)) 
    
encode_text_dummy(df2, 'Genre')
encode_text_dummy(df2, 'BookCategory')
encode_text_dummy(df2, 'Year_of_Print')
encode_text_dummy(df2, 'Edition_quality')

df2['Reviews'] = pd.to_numeric(df2['Reviews'])
df2['Ratings'] = df2['Ratings'].str.replace(',','')
df2['Ratings'] = pd.to_numeric(df2['Ratings'])

df2.columns
df2.dtypes
train_final =df2[:ntrain]
test_final=df2[ntrain:]

del train_final['Price']
del test_final['Price']

X = train_final.copy()
y = train_data['Price']
y_log = np.log(y)

########################



from sklearn.model_selection import KFold, cross_val_score, train_test_split
train, test, y_train, y_test = train_test_split(X, y_log, test_size=0.1, random_state=0)

train_data = lgb.Dataset(train, label=y_train)
test_data = lgb.Dataset(test, label=y_test)

#parameters assigned post grid-search on model
param = {'objective': 'regression',
         'boosting': 'gbdt',  
         'metric': 'root_mean_squared_error',
         'learning_rate': 0.08, 
         'num_iterations': 200,
         'num_leaves': 130,#35
         'max_depth': 30,
         'min_data_in_leaf': 18,#12
         'bagging_fraction': 0.85,
         'bagging_freq': 10,#1
         'feature_fraction': 0.75
         }

lgbm = lgb.train(params=param,
                 verbose_eval=50,
                 train_set=train_data,
                 valid_sets=[test_data])

y_pred_lgbm = lgbm.predict(test)

y_pred_lgbm_final = lgbm.predict(test_final)
y_pred_lgbm_final = np.exp(y_pred_lgbm_final)

pd.DataFrame(y_pred_lgbm_final, columns=['Price']).to_excel('lgbm_06_book_cost.xlsx')   


from sklearn.metrics import  mean_squared_error
print("Root mean square error for test dataset: {}".format(np.round(np.sqrt(mean_squared_error(y_test, y_pred_lgbm)), 4)))
#Root mean square error for test dataset: 0.5531 #4 .758001
#Root mean square error for test dataset: 0.5413 #4 .758848
#for n_it=300 ,Root mean square error for test dataset: 0.5367  0.761522
#for n_it=200 ,Root mean square error for test dataset: 0.5343  0.762919
#hyper changes: 0.767547; Root mean square error for test dataset: 0.531245

###################################################

train, test, y_train, y_test = train_test_split(X, y_log, test_size=0.25, random_state=2)

params = {
    'n_estimators':1000,
    'colsample_bytree': 0.9,
    'objective': 'reg:squarederror',
    'max_depth': 8,
    'min_child_weight': 3,
    'eta': 0.01,
    'subsample': 0.8,
    'seed' : 6,
    'reg_alpha' : 0.7,
    'reg_lambda' : 1
}

dtrain = xgb.DMatrix(train, y_train)
dtest = xgb.DMatrix(test, y_test)

model = xgb.cv(
    params=params,
    dtrain=dtrain,
    num_boost_round=800,
    nfold=15,
    early_stopping_rounds=200
)

# Fit
final_gb = xgb.train(params, dtrain, num_boost_round=len(model))

preds = final_gb.predict(dtest)
#ppp=preds
#for val in ppp:
 #   ppp.
  #  if (ppp[val]):
   #     print("value changing {}".format(ppp[val]))
    #    preds[val]=0

print("Root mean square error for test dataset: {}".format(np.round(np.sqrt(mean_squared_error(y_test, enet_pred_rev)), 4)))

print(np.exp(y_test[0:10]),enet_pred_rev[0:10])


######################
n_folds=5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)
    
    
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3 ))

ENet.fit(train, y_train)

enet_pred =ENet.predict(test.values)

enet_pred_final =ENet.predict(test_final.values)

enet_pred_rev = np.exp(enet_pred)
enet_pred_final = np.exp(enet_pred_final)

pd.DataFrame(enet_pred_final, columns=['Price']).to_excel('enet_04_book_cost.xlsx')   
#Root mean square error for test dataset: 0.5726 #4 .7577
#Root mean square error for test dataset: 509.6798  0.7569
################ 
