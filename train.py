import os
import re
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = "data/netflix_titles.csv"
MODEL_PATH = "models/ott_rating_model.pkl"
METRICS_PATH = "models/model_metrics.txt"

os.makedirs("models", exist_ok=True)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        "Dataset not found. Run 'python download_dataset.py' first."
    )

df = pd.read_csv(DATA_PATH)

required = ["type", "country", "release_year", "rating", "duration", "listed_in"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

def first_value(value):
    if pd.isna(value):
        return "Unknown"
    return str(value).split(",")[0].strip()

def duration_to_number(value):
    if pd.isna(value):
        return None
    match = re.search(r"\d+", str(value))
    return float(match.group()) if match else None

data = df.copy()
data["country_primary"] = data["country"].apply(first_value)
data["genre_primary"] = data["listed_in"].apply(first_value)
data["duration_num"] = data["duration"].apply(duration_to_number)
data["release_year"] = pd.to_numeric(data["release_year"], errors="coerce")
data["rating"] = data["rating"].fillna("Unknown")

# Keep rating classes with enough examples for a more stable classroom model.
counts = data["rating"].value_counts()
valid_ratings = counts[counts >= 20].index
data = data[data["rating"].isin(valid_ratings)].copy()

features = [
    "type", "release_year", "duration_num",
    "country_primary", "genre_primary"
]
target = "rating"

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

numeric_features = ["release_year", "duration_num"]
categorical_features = ["type", "country_primary", "genre_primary"]

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric_features),
    ("cat", categorical_pipe, categorical_features)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ))
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average="weighted")

joblib.dump(model, MODEL_PATH)

report = classification_report(y_test, predictions, zero_division=0)

metrics = f"""OTT Rating Prediction Model
--------------------------------
Dataset rows used: {len(data)}
Training rows: {len(X_train)}
Testing rows: {len(X_test)}
Model: Random Forest Classifier
Target: Rating category

Accuracy: {accuracy:.4f}
Weighted F1 Score: {f1:.4f}

Classification Report:
{report}
"""

with open(METRICS_PATH, "w", encoding="utf-8") as file:
    file.write(metrics)

print(metrics)
print(f"Model saved to: {MODEL_PATH}")
