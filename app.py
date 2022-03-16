import streamlit as st
from PIL import Image
from urllib.request import urlopen
import pickle
import src.module1 as main
import src.module2 as module2
import src.module3 as module3
import src.module3b as module3b

PATH = 'https://fabriziorocco.it/listings_sub.pkl'
st.title('The Airbnb AI Assistant')
st.subheader('Group A7')
image = Image.open('data/pict.jpg')
st.image(image)
st.markdown(
    " There are a growing number of apartments on Airbnb, and it can be very difficult for a new landlord to understand what the similar apartments are, how to price their own correctly and how to know what service is appreciated by the guests. "
    "To solve these problems, we have created a platform that landlords can subscribe to so that they can get suggestions and price-range based on similar apartments, and an idea of what services are valued by the customers. Our motivation is based on our interest to use multiple Artificial Intelligence techniques to solve those problems and to create a platform that Airbnb could offer as a paid service."
)


@st.experimental_memo
def load_data(path):
    data = pickle.load(urlopen(path))
    return data

@st.experimental_singleton
def fetch():
    matrix = pickle.load(urlopen("https://fabriziorocco.it/matrix.pkl"))
    vect = pickle.load(urlopen("https://fabriziorocco.it/vectorizor.pkl"))
    # with open(r"https://fabriziorocco.it/listings_sub.pkl", "rb") as input_file:
    #   data_cl = pickle.load(input_file)
    # with open(r"https://fabriziorocco.it/matrix.pkl", "rb") as input_file:
    #   matrix = pickle.load(input_file)
    # with open(r"https://fabriziorocco.it/vectorizor.pkl", "rb") as input_file:
    #   vect = pickle.load(input_file)
    return matrix,vect


def run():
    with st.spinner('Fetching data from the server. Please wait'):
        data = load_data(PATH)
    matrix, vect = fetch()
    st.text('Data downloaded ✅')
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", "66641")
    col2.metric("Columns", "57")
    col3.metric("City", "London, UK")

    query = st.text_area('Look for similar apartments')
    if st.button('Recommend'):
        st.subheader('Top 5 similar apartments')
        st.write(main.recommend(data, query, matrix, vect))

    module2.main()

    module3.main()

    st.subheader("Check how the apartments are located based on POI 📍🎡")
    module3b.map()


if __name__ == "__main__":
    run()
