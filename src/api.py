# src/api.py

from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import pickle
import numpy as np
import os

app = Flask(__name__)

# --- Carregar modelos na inicialização da API ---
MODEL_DIR = './models/'
print("Carregando artefatos...")
embedding_model = SentenceTransformer("zeeshanabbasi2004/finbert-sentiment", cache_folder=MODEL_DIR)

# Carrega os modelos salvos
with open(os.path.join(MODEL_DIR, 'lr_model.pkl'), 'rb') as f:
    classifier_model = pickle.load(f)

with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)

with open(os.path.join(MODEL_DIR, 'label_encoder.pkl'), 'rb') as f:
    label_encoder = pickle.load(f)
print("Artefatos carregados com sucesso.")


@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint para realizar a predição de sentimento."""
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Requisição inválida. Body deve conter a chave "text".'}), 400

    text = data['text']

    # --- Pipeline de Predição ---
    # 1. Gerar embedding
    embedding = embedding_model.encode([text])

    # 2. Normalizar
    scaled_embedding = scaler.transform(embedding)

    # 3. Prever probabilidades
    probabilities = classifier_model.predict_proba(scaled_embedding)[0]

    # 4. Obter classe e confiança
    confidence = np.max(probabilities)
    predicted_class_index = np.argmax(probabilities)
    predicted_sentiment = label_encoder.inverse_transform([predicted_class_index])[0]

    # 5. Formatar probabilidades para a resposta
    class_probabilities = {label_encoder.classes_[i]: prob for i, prob in enumerate(probabilities)}

    # --- Montar Resposta ---
    response = {
        'sentiment': str(predicted_sentiment),
        'confidence': float(confidence),
        'probabilities': class_probabilities
    }

    return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)