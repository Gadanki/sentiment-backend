from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
from preprocess import preprocess_review

app = Flask(__name__)
CORS(app)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    review = data.get("review", "")

    cleaned = preprocess_review(review)
    print("CLEANED REVIEW:", cleaned)   # <<--- DEBUG PRINT

    vectorized = vectorizer.transform([cleaned])

    proba = model.predict_proba(vectorized)[0]

    positive_score = proba[1]
    negative_score = proba[0]

    # For neutral class – if prediction is not strong enough
    if max(positive_score, negative_score) < 0.60:
        sentiment = "neutral"
        confidence = round(max(positive_score, negative_score) * 100, 2)
    else:
        sentiment = "positive" if positive_score > negative_score else "negative"
        confidence = round(max(positive_score, negative_score) * 100, 2)

    return jsonify({
        "sentiment": sentiment,
        "confidence": confidence,
        "cleaned_text": cleaned
    })


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)
