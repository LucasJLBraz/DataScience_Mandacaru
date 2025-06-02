# src/train.py

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pickle
import os

# --- Configurações ---
DATA_PATH = './data/raw/data.csv'
MODELS_PATH = './models/'
SEED = 19


def train_and_save_model():
    """
    Carrega os dados, treina os componentes (encoder, scaler, modelo)
    e salva-os em disco.
    """
    # Carrega e Preparara os Dados
    print("Carregando dados...")
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"Erro: Arquivo '{DATA_PATH}' não encontrado.")
        return

    print("Iniciando o treinamento...")

    X = df['Sentence']
    y = df['Sentiment']

    # Label Encoder
    print("Treinando LabelEncoder...")
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y)

    # Embedding Encoder
    print("Carregando SentenceTransformer...")
    encoder = SentenceTransformer("zeeshanabbasi2004/finbert-sentiment", cache_folder=MODELS_PATH)
    X_train_embeddings = encoder.encode(X.tolist(), show_progress_bar=True)

    # Normalizador
    print("Treinando MinMaxScaler...")
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_embeddings)

    # Modelo de Classificação
    model = LogisticRegression(penalty='l2', C=0.01, random_state=SEED)
    print(f"Treinando {model.__class__.__name__}...")
    model.fit(X_train_scaled, y_train_encoded)

    # --- 3. Salvando os Artefatos ---
    print("Salvando artefatos na pasta './models/'...")
    os.makedirs(MODELS_PATH, exist_ok=True)

    with open(os.path.join(MODELS_PATH, 'lr_model.pkl'), 'wb') as f:
        pickle.dump(model, f)

    with open(os.path.join(MODELS_PATH, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    with open(os.path.join(MODELS_PATH, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)

    print("\nTreinamento concluído e modelos salvos com sucesso!")


if __name__ == '__main__':
    train_and_save_model()