import pandas as pd
import numpy as np
import pickle
import streamlit as st
from PIL import Image

pickle_in = open('data/processed/module2/gboost.pkl', 'rb')
model = pickle.load(pickle_in)


def prediction(host_response_time, host_response_rate, host_acceptance_rate,
       host_is_superhost, host_total_listings_count,
       host_identity_verified, room_type, accommodates, bathrooms_text,
       bedrooms, beds, minimum_nights, maximum_nights,
       availability_30, availability_60, availability_90,
       availability_365, number_of_reviews, review_scores_rating,
       review_scores_accuracy, review_scores_cleanliness,
       review_scores_checkin, review_scores_communication,
       review_scores_location, review_scores_value, instant_bookable,
       reviews_per_month, host_since_year, final_neighbourhood,
       HOT_WATER, HEATING, COFFEE_MAKER, FIRE_EXTINGUISHER, IRON,
       DRYER, HAIR_DRYER, STOVE, OVEN, WIFI, COOKING_BASICS,
       LONG_TERM_STAYS_ALLOWED, KITCHEN, CARBON_MONOXIDE_ALARM,
       ESSENTIALS, SMOKE_ALARM, HANGERS, BED_LINENS, SHAMPOO,
       WASHER, REFRIGERATOR, DISHES_AND_SILVERWARE,
       DEDICATED_WORKSPACE, MICROWAVE, TV, DISHWASHER,
       PRIVATE_ENTRANCE, FIRST_AID_KIT):
    prediction = model.predict([[host_response_time, host_response_rate, host_acceptance_rate,
       host_is_superhost, host_total_listings_count,
       host_identity_verified, room_type, accommodates, bathrooms_text,
       bedrooms, beds, minimum_nights, maximum_nights,
       availability_30, availability_60, availability_90,
       availability_365, number_of_reviews, review_scores_rating,
       review_scores_accuracy, review_scores_cleanliness,
       review_scores_checkin, review_scores_communication,
       review_scores_location, review_scores_value, instant_bookable,
       reviews_per_month, host_since_year, final_neighbourhood,
       HOT_WATER, HEATING, COFFEE_MAKER, FIRE_EXTINGUISHER, IRON,
       DRYER, HAIR_DRYER, STOVE, OVEN, WIFI, COOKING_BASICS,
       LONG_TERM_STAYS_ALLOWED, KITCHEN, CARBON_MONOXIDE_ALARM,
       ESSENTIALS, SMOKE_ALARM, HANGERS, BED_LINENS, SHAMPOO,
       WASHER, REFRIGERATOR, DISHES_AND_SILVERWARE,
       DEDICATED_WORKSPACE, MICROWAVE, TV, DISHWASHER,
       PRIVATE_ENTRANCE, FIRST_AID_KIT]])
    print(prediction)
    return prediction

def main():
    st.subheader("Part 2: Set your features, we'll help you finding the most accurate price")

    col1, col2, col3 = st.columns(3)

    with col1:
        host_response_time = st.selectbox('What is the host response time?',(2, 3, 4, 5))
        host_is_superhost = st.selectbox('Is the host a superhost?', (0,1))
        room_type = st.selectbox('Room type: \n 4 = Entire home/apt, \n 3 = Hotel room, \n 2 = Private room, \n 1 = Shared room', (1,2,3,4))
        bedrooms = st.slider('Bedrooms', 1, 4, 1)
        number_of_reviews = st.slider('How many reviews?', 0, 713, 2)
        HOT_WATER = st.selectbox('HOT WATER',(0,1))
        FIRE_EXTINGUISHER = st.selectbox('FIRE EXTINGUISHER',(0,1))
        HAIR_DRYER = st.selectbox('HAIR DRYER', (0, 1))

    with col2:
        host_response_rate = st.slider('What is the host response rate?', 0, 100, 100)
        host_total_listings_count = st.slider('What is the host total number of apartments?', 0, 3750, 1)
        accommodates = st.slider('Number of people accommodates', 1, 16, 2)
        minimum_nights = st.slider('Minimum number of nights?', 1, 365, 2)
        instant_bookable = st.selectbox('Is it instantly bookable?', (0,1))
        HEATING = st.selectbox('HEATING', (0, 1))
        IRON = st.selectbox('IRON', (0, 1))
        STOVE = st.selectbox('STOVE', (0, 1))


    with col3:
        host_acceptance_rate = st.slider('What is the host acceptance rate?', 0, 100, 100)
        host_identity_verified = st.selectbox('Is the identity of the host verified?', (0,1))
        bathrooms_text = st.selectbox('bathrooms', (0, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5))
        maximum_nights = st.slider('Maximum number of nights?', 1, 365, 30)
        final_neighbourhood = st.selectbox('Neighborhood (Check notebook module 2 for further explanations', (1,2,3,4,5,6))
        COFFEE_MAKER = st.selectbox('COFFEE MAKER', (0, 1))
        DRYER = st.selectbox('DRYER', (0, 1))
        OVEN = st.selectbox('OVEN', (0, 1))

    beds = bedrooms
    availability_30 = 0
    availability_60 = 0
    availability_90 = 0
    availability_365 = 0
    review_scores_rating, review_scores_accuracy,  review_scores_cleanliness, review_scores_checkin, review_scores_communication, review_scores_location, review_scores_value = 4.81, 4.81,4.81,4.81,4.81,4.81,4.81
    reviews_per_month = 0.30
    host_since_year = 2016
    WIFI = 1
    COOKING_BASICS = 1
    LONG_TERM_STAYS_ALLOWED = 0
    KITCHEN = 1
    CARBON_MONOXIDE_ALARM = 0
    ESSENTIALS = 1
    SMOKE_ALARM = 0
    HANGERS = 1
    BED_LINENS = 1
    SHAMPOO = 1
    WASHER = 1
    REFRIGERATOR = 1
    DISHES_AND_SILVERWARE = 0
    DEDICATED_WORKSPACE = 0
    MICROWAVE = 0
    TV = 1
    DISHWASHER = 1
    PRIVATE_ENTRANCE = 1
    FIRST_AID_KIT = 0

    if st.button("Predict"):
        result = prediction(host_response_time, host_response_rate, host_acceptance_rate,
       host_is_superhost, host_total_listings_count,
       host_identity_verified, room_type, accommodates, bathrooms_text,
       bedrooms, beds, minimum_nights, maximum_nights,
       availability_30, availability_60, availability_90,
       availability_365, number_of_reviews, review_scores_rating,
       review_scores_accuracy, review_scores_cleanliness,
       review_scores_checkin, review_scores_communication,
       review_scores_location, review_scores_value, instant_bookable,
       reviews_per_month, host_since_year, final_neighbourhood,
       HOT_WATER, HEATING, COFFEE_MAKER, FIRE_EXTINGUISHER, IRON,
       DRYER, HAIR_DRYER, STOVE, OVEN, WIFI, COOKING_BASICS,
       LONG_TERM_STAYS_ALLOWED, KITCHEN, CARBON_MONOXIDE_ALARM,
       ESSENTIALS, SMOKE_ALARM, HANGERS, BED_LINENS, SHAMPOO,
       WASHER, REFRIGERATOR, DISHES_AND_SILVERWARE,
       DEDICATED_WORKSPACE, MICROWAVE, TV, DISHWASHER,
       PRIVATE_ENTRANCE, FIRST_AID_KIT)
        st.success('The price of your house is {}€'.format(result))

