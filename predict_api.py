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


def extract_keywords(text, top_n=10):
    tfidf_matrix = vectorizer.transform([text])
    feature_array = np.array(vectorizer.get_feature_names_out())
    tfidf_sorting = np.argsort(tfidf_matrix.toarray()).flatten()[::-1]
    top_n_keywords = feature_array[tfidf_sorting][:top_n]
    return " ".join(top_n_keywords)


@app.route("/parse", methods=["POST"])
def parse_text():
    """Ekstrak Subject, Assessment, Object, Plan dari percakapan dan info pasien."""
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
ekstrak empat bagian:
- Subject (keluhan utama pasien)
- Assessment (penilaian dokter)
- Object (hasil pemeriksaan)
- Plan (rencana tindakan)

Teks:
{full_text}

Jawab dalam format JSON:
{{
    "subject": "...",
    "assessment": "...",
    "object": "...",
    "plan": "..."
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
                "raw_output": raw_output,
                "error": "Gagal parsing JSON dari Gemini",
            }

        return jsonify(parsed_json)

    except Exception as e:
        return jsonify({"error": f"Gagal memanggil Gemini: {e}"}), 500


@app.route("/predict", methods=["POST"])
def predict():
    """Prediksi diagnosa otomatis."""
    data = request.get_json()
    subject = data.get("subject", "")
    assessment = data.get("assessment", "")
    object_ = data.get("object", "")
    plan = data.get("plan", "")

    combined_text = f"{subject} {assessment} {object_} {plan}"
    keywords_text = extract_keywords(combined_text, top_n=15)

    X = vectorizer.transform([keywords_text])
    probs = model.predict_proba(X)[0]
    classes = model.classes_

    top_indices = np.argsort(probs)[::-1][:3]
    top_diagnoses = [
        {"diagnosa": classes[i], "confidence": float(probs[i])} for i in top_indices
    ]

    return jsonify(
        {
            "keywords": keywords_text.split(),
            "diagnosa_utama": top_diagnoses[0],
            "diagnosa_sekunder_1": top_diagnoses[1],
            "diagnosa_sekunder_2": top_diagnoses[2],
        }
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
