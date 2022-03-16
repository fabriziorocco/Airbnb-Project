import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import re
import sys
import itertools
import warnings
import string
warnings.filterwarnings("ignore")

#PATH = 'https://fabriziorocco.it/listings.csv'
#DATA = pd.read_csv(PATH)

def fill_na (data):
    df_prod = data.copy()
    df_prod['description'] = df_prod['description'].fillna('No description')
    df_prod['neighborhood_overview'] = df_prod['neighborhood_overview'].fillna('No neighborhood_overview')
    return df_prod

def drop_cols (data):
    df_prod = fill_na(data)
    df_prod.drop(df_prod.columns[8:25], axis=1, inplace=True)
    df_prod.drop(df_prod.columns[23:56], axis=1, inplace=True)
    return df_prod

def regex_handling (data):
    df_prod = drop_cols(data)
    alphanumeric = lambda x: re.sub('\w*\d\w*', ' ', str(x))
    l_case = lambda x: re.sub('[%s]' % re.escape(string.punctuation), ' ', x.lower())
    df_prod['amenities_upd'] = df_prod['amenities'].apply(lambda x: [re.sub(' ', '_', i) for i in re.findall(r'"([^"]*)"', x)])
    df_prod['name'] = df_prod['name'].map(alphanumeric).map(l_case)
    df_prod['description'] = df_prod['description'].map(alphanumeric).map(l_case)
    df_prod['amenities_upd'] = df_prod['amenities_upd'].map(alphanumeric).map(l_case)
    df_prod['neighborhood_overview'] = df_prod['neighborhood_overview'].map(alphanumeric).map(l_case)
    df_prod['amenities_upd'] = df_prod['amenities_upd'].astype(str).str.replace(r'\[|\]|,', '')
    df_prod['amenities_upd'] = df_prod['amenities_upd'].astype(str).str.replace('_', ' ')
    return df_prod

def clean_currency(data):
    df_prod = regex_handling(data)
    df_prod['price'] = df_prod['price'].astype(str).str.replace('$', '').replace(',', '')
    return df_prod

def generate_count_vectorizer (data):
    df_prod = clean_currency(data)
    data_sub = df_prod[['listing_url','name','description','price', 'amenities_upd']]
    data_sub['vector'] = data_sub[['name', 'description', 'price', 'amenities_upd']].astype(str).apply(lambda x: ','.join(x), axis=1)
    vect = CountVectorizer(stop_words='english', ngram_range=(1, 2)) #Try also (1,1) for unigrams. Reason for 'hair dryer', 'london eye'
    matrix = vect.fit_transform(data_sub['vector'])
    return matrix, vect

def recommend (data, query, matrix, vect):
    user_query = vect.transform([query])
    similarity = cosine_similarity(user_query, matrix)
    top_5 = np.argsort(similarity[0])[-5:]
    best_res = np.argmax(similarity[0])
    return data.loc[[top_5[4], top_5[3], top_5[2], top_5[1], top_5[0]]]

if __name__ == "__main__":
    print ("Executed when invoked directly")
    recommend()

