import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import string
import nltk
from langdetect import detect  # pip install langdetect
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import Counter
from scipy import stats
from dateutil import parser
import streamlit as st
import folium
from folium.plugins import FastMarkerCluster
#from streamlit_yellowbrick import st_yellowbrick

import seaborn as sns
import shapely # pip install shapely
from shapely.geometry import Point
import geopandas as gpd

import warnings
warnings.filterwarnings("ignore")
st.set_option('deprecation.showPyplotGlobalUse', False)
from wordcloud import WordCloud
from nltk.corpus import stopwords
from collections import Counter
from PIL import Image

from sklearn.feature_extraction.text import CountVectorizer
from yellowbrick.text.freqdist import FreqDistVisualizer # pip install yellowbrick
from yellowbrick.style import set_palette

PATH = 'data/processed/module3/final_df.csv'
reviews_en = pd.read_csv(PATH)

# full dataframe with POSITIVE comments
reviews_pos = reviews_en.loc[reviews_en.polarity >= 0.95]

# only corpus of POSITIVE comments
pos_comments = reviews_pos['comments'].tolist()

# full dataframe with NEGATIVE comments
reviews_neg = reviews_en.loc[reviews_en.polarity < 0.0]

# only corpus of NEGATIVE comments
neg_comments = reviews_neg['comments'].tolist()


def plot_wordcloud(wordcloud, language):
    plt.figure(figsize=(12, 10))
    plt.imshow(wordcloud, interpolation = 'bilinear')
    plt.axis("off")
    plt.title(language + ' Comments\n', fontsize=18, fontweight='bold')
    plt.show()


def get_neighbourhood_data(neighbourhood):
    # full dataframe with POSITIVE comments
    reviews_positive = reviews_pos[reviews_pos.neighbourhood_cleansed_x == neighbourhood]

    # only corpus of POSITIVE comments
    positive_comments = reviews_pos['comments'].tolist()

    # full dataframe with NEGATIVE comments
    reviews_negative = reviews_neg[reviews_neg.neighbourhood_cleansed_x == neighbourhood]

    # only corpus of NEGATIVE comments
    negative_comments = reviews_neg['comments'].tolist()

    return reviews_positive, positive_comments, reviews_negative, negative_comments


def get_example_comments(neighbourhood, n_neg_examples, n_pos_examples):
    # read some negative comments
    neg_examples = get_neighbourhood_data(neighbourhood)[3][:n_neg_examples]

    # read some positive comments
    pos_examples = get_neighbourhood_data(neighbourhood)[1][:n_pos_examples]

    return pos_examples, neg_examples

def get_wordcloud_neighbourhood_pos(neighbourhood):
    # postive wordcloud
    reviews_pos = get_neighbourhood_data(neighbourhood)[0]
    wordcloud_pos = WordCloud(max_font_size=200, max_words=200, background_color="palegreen",
                      width= 3000, height = 2000,
                      stopwords = stopwords.words('english')).generate(str(reviews_pos.comments.values))
    plot_wordcloud(wordcloud_pos, '\nPositively Tuned')

def get_wordcloud_neighbourhood_neg(neighbourhood):
    # postive wordcloud
    reviews_neg = get_neighbourhood_data(neighbourhood)[2]
    wordcloud_neg = WordCloud(max_font_size=200, max_words=200, background_color="mistyrose",
                      width= 3000, height = 2000,
                      stopwords = stopwords.words('english')).generate(str(reviews_neg.comments.values))
    plot_wordcloud(wordcloud_neg, '\nNegatively Tuned')


def get_frequency_neg(neighbourhood):
    neg_comments = get_neighbourhood_data(neighbourhood)[3]

    # vectorizing text
    vectorizer = CountVectorizer(stop_words='english')
    docs_neg = vectorizer.fit_transform(neg_comments)
    features = vectorizer.get_feature_names()

    # preparing the plot
    set_palette('pastel')
    plt.figure(figsize=(18, 8))
    plt.title('The Top 30 most frequent words used in NEGATIVE comments\n', fontweight='bold')

    # instantiating and fitting the FreqDistVisualizer, plotting the top 30 most frequent terms
    visualizer_neg = FreqDistVisualizer(features=features, n=30).fit(docs_neg)
    visualizer_neg.poof;


def get_frequency_pos(neighbourhood):
    posi_comments = get_neighbourhood_data(neighbourhood)[1]

    # vectorizing text
    vectorizer = CountVectorizer(stop_words='english')
    docs_pos = vectorizer.fit_transform(posi_comments)
    features = vectorizer.get_feature_names()

    # preparing the plot
    set_palette('pastel')
    plt.figure(figsize=(18, 8))
    plt.title('The Top 30 most frequent words used in POSITIVE comments\n', fontweight='bold')

    # instantiating and fitting the FreqDistVisualizer, plotting the top 30 most frequent terms
    visualizer_pos = FreqDistVisualizer(features=features, n=30).fit(docs_pos)
    visualizer_pos.poof;

def main():
    st.subheader("Part 3: Sentiment analysis of reviews in your neighbourhood")
    option = st.selectbox(
        'Choose your neighbourhood',
        ("Westminster",
         "Camden",
         "Tower Hamlets",
         "Kensington and Chelsea",
         "Islington",
         "Lambeth",
         "Southwark",
         "Hackney",
         "Wandsworth",
         "Hammersmith and Fulham",
         "Brent",
         "Richmond upon Thames",
         "Ealing",
         "Haringey",
         "Lewisham",
         "Newham",
         "Greenwich",
         "Barnet",
         "Hounslow",
         "Waltham Forest",
         "Hillingdon",
         "Croydon",
         "Merton",
         "City of London",
         "Enfield",
         "Kingston upon Thames",
         "Bromley",
         "Redbridge",
         "Harrow",
         "Sutton",
         "Havering",
         "Barking and Dagenham",
         "Bexley"))

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('Positive neighbourhood data')
        st.pyplot(get_wordcloud_neighbourhood_pos(option))
        #st_yellowbrick(get_frequency_pos(option))

    with col2:
        st.markdown('Negative neighbourhood data')
        st.pyplot(get_wordcloud_neighbourhood_neg(option))
        #st_yellowbrick(get_frequency_neg(option))



