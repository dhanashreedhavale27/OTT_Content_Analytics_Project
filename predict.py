import joblib
import pandas as pd

MODEL_PATH = "models/ott_rating_model.pkl"

model = joblib.load(MODEL_PATH)

def predict_rating(content_type, release_year, duration_num,
                   country_primary, genre_primary):
    data = pd.DataFrame([{
        "type": content_type,
        "release_year": release_year,
        "duration_num": duration_num,
        "country_primary": country_primary,
        "genre_primary": genre_primary
    }])
    return model.predict(data)[0]
