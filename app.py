import os
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px

DATA_PATH = "data/netflix_titles.csv"
MODEL_PATH = "models/ott_rating_model.pkl"
METRICS_PATH = "models/model_metrics.txt"

st.set_page_config(page_title="OTT Content Analytics", page_icon="🎬", layout="wide")

@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        return pd.DataFrame()
    df = pd.read_csv(DATA_PATH)
    return df

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

df = load_data()
model = load_model()

st.title("🎬 OTT Content Analytics & Rating Predictor")
st.caption("Kaggle Netflix Movies and TV Shows dataset + Machine Learning + Streamlit")

if df.empty:
    st.warning("Dataset not found.")
    st.info("Run: python download_dataset.py")
    st.stop()

# Basic cleaning for dashboard
df["rating"] = df["rating"].fillna("Unknown")
df["type"] = df["type"].fillna("Unknown")
df["country"] = df["country"].fillna("Unknown")
df["listed_in"] = df["listed_in"].fillna("Unknown")
df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")

tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔎 Explore Dataset", "🤖 Rating Predictor"])

with tab1:
    st.subheader("Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Titles", len(df))
    c2.metric("Movies", int((df["type"] == "Movie").sum()))
    c3.metric("TV Shows", int((df["type"] == "TV Show").sum()))
    c4.metric("Years Covered", f"{int(df['release_year'].min())}–{int(df['release_year'].max())}")

    left, right = st.columns(2)

    with left:
        type_df = df["type"].value_counts().reset_index()
        type_df.columns = ["Type", "Count"]
        fig = px.pie(type_df, names="Type", values="Count",
                     title="Movies vs TV Shows", hole=0.35)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        rating_df = df["rating"].value_counts().head(10).reset_index()
        rating_df.columns = ["Rating", "Count"]
        fig = px.bar(rating_df, x="Rating", y="Count",
                     title="Top Rating Categories")
        st.plotly_chart(fig, use_container_width=True)

    yearly = df.dropna(subset=["release_year"]).groupby("release_year").size().reset_index(name="Titles")
    fig = px.line(yearly, x="release_year", y="Titles",
                  title="Content Release Trend")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Explore OTT Content")

    f1, f2, f3 = st.columns(3)
    selected_type = f1.multiselect(
        "Content Type", sorted(df["type"].dropna().unique()),
        default=sorted(df["type"].dropna().unique())
    )
    selected_rating = f2.multiselect(
        "Rating", sorted(df["rating"].dropna().unique()),
        default=[]
    )
    min_year = int(df["release_year"].min())
    max_year = int(df["release_year"].max())
    selected_year = f3.slider("Release Year", min_year, max_year, (2000, max_year))

    filtered = df[
        df["type"].isin(selected_type) &
        df["release_year"].between(selected_year[0], selected_year[1], inclusive="both")
    ]

    if selected_rating:
        filtered = filtered[filtered["rating"].isin(selected_rating)]

    st.write(f"Showing **{len(filtered)}** titles.")
    st.dataframe(
        filtered[["show_id", "type", "title", "country", "release_year", "rating", "duration", "listed_in"]]
        .head(100),
        use_container_width=True
    )

with tab3:
    st.subheader("🤖 Predict the OTT Rating Category")

    if model is None:
        st.warning("Trained model not found. Run: python train.py")
        st.stop()

    st.write("Enter content characteristics to predict its rating category.")

    col1, col2 = st.columns(2)

    with col1:
        content_type = st.selectbox("Content Type", ["Movie", "TV Show"])
        release_year = st.number_input(
            "Release Year", min_value=1920, max_value=2026, value=2020, step=1
        )
        duration = st.number_input(
            "Duration / Number of Seasons", min_value=1, max_value=300, value=90, step=1
        )

    with col2:
        common_countries = [
            "United States", "India", "United Kingdom", "Japan",
            "South Korea", "Canada", "France", "Spain", "Germany"
        ]
        country = st.selectbox("Primary Country", common_countries)
        genre = st.selectbox(
            "Primary Genre",
            ["Dramas", "Comedies", "Action & Adventure", "Documentaries",
             "International Movies", "Children & Family Movies",
             "Crime TV Shows", "International TV Shows", "Romantic Movies",
             "Horror Movies", "Anime Series", "Sports Movies"]
        )

    if st.button("Predict Rating", type="primary"):
        input_df = pd.DataFrame([{
            "type": content_type,
            "release_year": release_year,
            "duration_num": duration,
            "country_primary": country,
            "genre_primary": genre
        }])

        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        classes = model.classes_

        result = pd.DataFrame({
            "Rating": classes,
            "Probability": probabilities
        }).sort_values("Probability", ascending=False).head(5)

        st.success(f"Predicted Rating Category: **{prediction}**")
        st.dataframe(result, use_container_width=True)

        fig = px.bar(result, x="Rating", y="Probability",
                     title="Top Prediction Probabilities")
        st.plotly_chart(fig, use_container_width=True)

    if os.path.exists(METRICS_PATH):
        with st.expander("Model Evaluation"):
            st.code(open(METRICS_PATH, encoding="utf-8").read())
