
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("netflix_titles.csv")
    return df

df = load_data()

# Preprocess
def clean_data(df):
    df['combined'] = df['director'].fillna('') + " " + df['cast'].fillna('') + " " + df['listed_in'].fillna('') + " " + df['description'].fillna('')
    return df

df = clean_data(df)

# Vectorization
cv = CountVectorizer(stop_words='english')
count_matrix = cv.fit_transform(df['combined'])
cosine_sim = cosine_similarity(count_matrix)

# Recommendation Function
def get_recommendations(title, cosine_sim=cosine_sim):
    if title not in df['title'].values:
        return ["Movie not found in dataset."]
    idx = df[df['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices].tolist()

# Streamlit UI
st.title("🎬 Netflix Movie Recommendation System")
st.write("Find similar movies based on your choice!")

movie_list = df['title'].dropna().unique()
selected_movie = st.selectbox("Choose a movie:", movie_list)

if st.button("Get Recommendations"):
    recommendations = get_recommendations(selected_movie)
    st.write("### Recommended Movies:")
    for movie in recommendations:
        st.write(f"- {movie}")
