import streamlit as st
import numpy as np
import pickle
import requests
import pandas as pd

TMDB_API_KEY = "b0ba8a42cf757ef8505822852f799df6"

@st.cache_data(show_spinner=False)
def load_data():
    movies_dict = pickle.load(open("movie_dict.pkl", "rb"))
    similarity_loaded = pickle.load(open("similarity.pkl", "rb"))

    movies = pd.DataFrame(movies_dict).copy()
    movies["title"] = movies["title"].astype(str).str.strip()
    movies["movie_id"] = movies["movie_id"].astype(int)

    similarity = np.array(similarity_loaded, dtype=np.float32)

    # ✅ HARD CHECK: similarity must be NxN matching movies length
    n = len(movies)
    if similarity.ndim != 2 or similarity.shape != (n, n):
        raise ValueError(
            f"Bad similarity.pkl! Expected ({n},{n}), got {similarity.shape}. "
            "Rebuild and re-save similarity.pkl from the same dataframe as movie_dict.pkl."
        )

    return movies, similarity

def fetch_movie_poster(movie_id: int) -> str:
    url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    try:
        r = requests.get(url, params={"api_key": TMDB_API_KEY}, timeout=10)
        data = r.json()
        poster_path = data.get("poster_path")
        if not poster_path:
            return "https://via.placeholder.com/500x750?text=No+Poster"
        return "https://image.tmdb.org/t/p/w500" + poster_path
    except Exception:
        return "https://via.placeholder.com/500x750?text=No+Poster"

def recommend(movie_index: int, movies: pd.DataFrame, similarity: np.ndarray, top_n: int = 5):
    distances = similarity[movie_index]

    ranked = sorted(
        enumerate(distances),
        key=lambda x: float(x[1]),
        reverse=True
    )

    ranked = [x for x in ranked if x[0] != movie_index][:top_n]

    names, posters = [], []
    for idx, _ in ranked:
        rec_movie_id = int(movies.iloc[idx]["movie_id"])
        names.append(movies.iloc[idx]["title"])
        posters.append(fetch_movie_poster(rec_movie_id))

    return names, posters

# -------------------- UI --------------------
st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Movie Recommender")

movies, similarity = load_data()

selected_movie_idx = st.selectbox(
    "What movies do you want to watch?",
    movies.index,
    format_func=lambda i: movies.loc[i, "title"]
)

if st.button("Recommend"):
    names, posters = recommend(selected_movie_idx, movies, similarity, top_n=5)

    cols = st.columns(5)  # ✅ always 5 columns
    for i in range(5):
        with cols[i]:
            st.markdown(
    f"<p style='text-align:center; font-size:16px; font-weight:600;'>{names[i]}</p>",
    unsafe_allow_html=True
)
            st.image(posters[i], use_container_width=True)
