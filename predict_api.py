from flask import Flask, request, jsonify
import joblib
import numpy as np
import requests
import json


app = Flask(__name__)

# Load model dan vectorizer
model = joblib.load("models/diagnosa_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# Konfigurasi Gemini API
API_KEY = "AIzaSyB6BSsIcprpCsUmWpd6yheBT-U_93f8si8"
GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"gemini-2.0-flash:generateContent?key={API_KEY}"
)


def extract_keywords(text):
    """Ekstrak semua kata penting dari teks berdasarkan bobot TF-IDF (tanpa duplikasi)."""
    tfidf_matrix = vectorizer.transform([text])
    feature_array = np.array(vectorizer.get_feature_names_out())
    tfidf_scores = tfidf_matrix.toarray().flatten()

    # Ambil kata yang punya bobot > 0
    nonzero_indices = np.where(tfidf_scores > 0)[0]
    keywords_with_scores = [
        (feature_array[i], tfidf_scores[i]) for i in nonzero_indices
    ]

    # Urutkan berdasarkan skor TF-IDF terbesar
    keywords_sorted = sorted(keywords_with_scores, key=lambda x: x[1], reverse=True)

    # Hapus duplikasi, pertahankan urutan dari skor tertinggi
    seen = set()
    unique_keywords = []
    for word, _ in keywords_sorted:
        if word not in seen:
            seen.add(word)
            unique_keywords.append(word)

    return unique_keywords


@app.route("/parse", methods=["POST"])
def parse_text():
    """Ekstrak Subject, Assessment, Object, Plan, instruction dari percakapan dan info pasien."""
    data = request.get_json()
    conversation = data.get("conversation", "")
    info_pasien = data.get("info", "")

    if not conversation.strip() and not info_pasien.strip():
        return jsonify({"error": "Percakapan dan info pasien kosong"}), 400

    full_text = f"""
    === Informasi Pasien ===
    {info_pasien}

    === Percakapan Dokter-Pasien ===
    {conversation}
    """

    prompt = f"""
Dari teks berikut yang berisi percakapan dokter dan pasien serta info pasien,
ekstrak lima bagian SOPAI:

- Subject (keluhan utama & tambahan pasien)
- Assessment (penilaian dokter)
- Object (hasil pemeriksaan)
- Plan (rencana tindakan)
- instruction (instruksi tambahan dokter)

Teks:
{full_text}

Jawab dalam format JSON:
{{
    "subject": "",
    "assessment": "",
    "object": "",
    "plan": "",
    "instruction": ""
}}
    """

    try:
        response = requests.post(
            GEMINI_URL,
            headers={"Content-Type": "application/json"},
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=40,
        )

        result = response.json()
        raw_output = result["candidates"][0]["content"]["parts"][0]["text"]
        clean_text = raw_output.strip().replace("```json", "").replace("```", "")

        try:
            parsed_json = json.loads(clean_text)
        except json.JSONDecodeError:
            parsed_json = {
                "subject": "",
                "assessment": "",
                "object": "",
                "plan": "",
                "instruction": "",
                "raw_output": raw_output,
                "error": "Gagal parsing JSON dari Gemini",
            }

        return jsonify(parsed_json)

    except Exception as e:
        return jsonify({"error": f"Gagal memanggil Gemini: {e}"}), 500


@app.route("/predict", methods=["POST"])
def predict():
    """Prediksi diagnosa otomatis dengan format JSON yang terstruktur."""
    data = request.get_json()
    subject = data.get("subject", "")
    assessment = data.get("assessment", "")
    object_ = data.get("object", "")
    plan = data.get("plan", "")

    combined_text = f"{subject} {assessment} {object_} {plan}"

    # === Ekstrak kata penting (explainability)
    top_tokens = extract_keywords(combined_text)

    # === Prediksi dengan model
    X = vectorizer.transform([" ".join(top_tokens)])
    probs = model.predict_proba(X)[0]
    classes = model.classes_

    # === Ambil top 3 diagnosis tertinggi
    top_indices = np.argsort(probs)[::-1][:3]

    # === Load ICD mapping
    with open("data/icd10_mapping.json", "r", encoding="utf-8") as f:
        icd_map = json.load(f)

    # === Susun hasil prediksi
    predictions = []
    for i in top_indices:
        name = classes[i]
        predictions.append(
            {
                "icd10": icd_map.get(name, "N/A"),
                "name": name,
                "score": round(float(probs[i]), 2),
            }
        )

    # === Return sesuai format yang diinginkan
    return jsonify(
        {
            "predictions": predictions,
            "explainability": {"top_tokens": top_tokens},  # sudah unik dan relevan
        }
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
