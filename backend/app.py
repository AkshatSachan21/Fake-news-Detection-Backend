import re
import string
import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    data = request.get_json(force=True)
    text = data.get("text", "")

    if not text.strip():
        return jsonify({"error": "Empty text"}), 400

    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    prediction = int(model.predict(vec)[0])
    proba = model.predict_proba(vec)[0]

    label = "REAL" if prediction == 1 else "FAKE"
    confidence = float(proba[prediction]) * 100
    fake_probability = float(proba[0]) * 100

    return jsonify({
        "label": label,
        "confidence": round(confidence, 2),
        "fake_probability": round(fake_probability, 2)
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
