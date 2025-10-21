import pandas as pd
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Path
DATA_PATH = "data/data_diagnosa.csv"
MODEL_PATH = "models/diagnosa_model.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"

os.makedirs("models", exist_ok=True)

# Baca data
df = pd.read_csv(DATA_PATH)

# Gabungkan subject + assessment + object + plan
df["combined_text"] = (
    df["subject"].fillna("")
    + " "
    + df["assessment"].fillna("")
    + " "
    + df["object"].fillna("")
    + " "
    + df["plan"].fillna("")
)

X = df["combined_text"]
y = df["diagnosa"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Pipeline training
pipeline = Pipeline(
    [
        (
            "tfidf",
            TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                stop_words=[
                    "dan",
                    "serta",
                    "yang",
                    "pada",
                    "dengan",
                    "untuk",
                    "di",
                    "ke",
                    "dari",
                    "tidak",
                    "ada",
                ],
            ),
        ),
        ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", multi_class="auto")),
    ]
)

print("🧠 Melatih model diagnosa...")
pipeline.fit(X_train, y_train)

# Evaluasi
y_pred = pipeline.predict(X_test)
print("\n✅ Akurasi:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Simpan model dan vectorizer
joblib.dump(pipeline.named_steps["clf"], MODEL_PATH)
joblib.dump(pipeline.named_steps["tfidf"], VECTORIZER_PATH)
print("\n📁 Model & vectorizer tersimpan di folder 'models/'")
