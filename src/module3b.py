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
from streamlit_folium import folium_static
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

PATH1 = 'data/processed/module3b/cleansed_listings.csv'
PATH2 = 'data/processed/module3b/london_attractive_spots_df.csv'
cleansed_listings = pd.read_csv(PATH1)
london_attractive_spots_df = pd.read_csv(PATH2)

def map ():
    # Neighborhood distribution according to number (count) of listings -- Map View
    latitude = cleansed_listings['latitude'].tolist()
    longitude = cleansed_listings['longitude'].tolist()

    locations = list(zip(latitude, longitude))

    neighbourhood_map = folium.Map(location=[cleansed_listings["latitude"].mean(), 0],
                                   zoom_start=11,
                                   control_scale=True,
                                   tiles='OpenStreetMap')

    london_attractive_spots_df.apply(lambda row: folium.Marker(location=[row["lat"],
                                                                         row["long"]],
                                                               radius=10,
                                                               popup=row['name']).add_to(neighbourhood_map), axis=1)

    FastMarkerCluster(data=locations).add_to(neighbourhood_map)
    return folium_static(neighbourhood_map)
