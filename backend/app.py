import re
import string
import pickle
import numpy as np
from flask import Flask, request, jsonify
from newspaper import Article

app = Flask(__name__)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Confidence threshold below which a prediction is marked "Uncertain"
UNCERTAIN_THRESHOLD = 60.0

# Feature names for explainability
FEATURE_NAMES = np.array(vectorizer.get_feature_names_out())

# Average coefficients across calibrated classifiers (for explainability)
try:
    COEFS = np.mean(
        [cc.estimator.coef_[0] for cc in model.calibrated_classifiers_],
        axis=0
    )
except Exception:
    COEFS = None


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


def get_top_words(vec, top_n=5):
    """Return top words pushing toward FAKE and toward REAL for this input."""
    if COEFS is None:
        return {"fake_words": [], "real_words": []}

    vec_array = vec.toarray()[0]
    nonzero_idx = np.nonzero(vec_array)[0]

    if len(nonzero_idx) == 0:
        return {"fake_words": [], "real_words": []}

    contributions = vec_array[nonzero_idx] * COEFS[nonzero_idx]
    words = FEATURE_NAMES[nonzero_idx]

    fake_mask = contributions < 0
    real_mask = contributions > 0

    fake_words = words[fake_mask][np.argsort(contributions[fake_mask])][:top_n]
    real_words = words[real_mask][np.argsort(-contributions[real_mask])][:top_n]

    return {"fake_words": list(fake_words), "real_words": list(real_words)}


def classify_text(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    prediction = int(model.predict(vec)[0])
    proba = model.predict_proba(vec)[0]

    confidence = float(proba[prediction]) * 100
    fake_probability = float(proba[0]) * 100

    if confidence < UNCERTAIN_THRESHOLD:
        label = "UNCERTAIN"
    else:
        label = "REAL" if prediction == 1 else "FAKE"

    top_words = get_top_words(vec)

    return {
        "label": label,
        "confidence": round(confidence, 2),
        "fake_probability": round(fake_probability, 2),
        "top_words": top_words
    }


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

    result = classify_text(text)
    return jsonify(result)


@app.route("/predict_url", methods=["POST", "OPTIONS"])
def predict_url():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    data = request.get_json(force=True)
    url = data.get("url", "")

    if not url.strip():
        return jsonify({"error": "Empty URL"}), 400

    try:
        article = Article(url)
        article.download()
        article.parse()
        extracted_text = (article.title or "") + " " + (article.text or "")
    except Exception as e:
        return jsonify({"error": "Could not extract article from URL: " + str(e)}), 400

    if not extracted_text.strip():
        return jsonify({"error": "No readable article content found at this URL"}), 400

    result = classify_text(extracted_text)
    result["extracted_title"] = article.title
    result["extracted_text_preview"] = extracted_text[:300]
    return jsonify(result)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
