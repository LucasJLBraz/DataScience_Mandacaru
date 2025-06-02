# app.py

import streamlit as st
import requests
import pandas as pd
import os
import json

# Configurações da Página
st.set_page_config(
    page_title="Análise de Sentimentos Financeiros",
    page_icon="🤖",
    layout="centered"
)

# URL da nossa API Flask
API_URL = "http://127.0.0.1:5000/predict"
HISTORY_FILE = "./data/processed/query_history.csv"


# Funções Auxiliares
def save_to_history(text, response):
    """Salva a consulta e o resultado em um arquivo CSV."""
    rpst_sentiment = response['sentiment']
    rpst_confidence = response['confidence']
    probabilities_dict = response['probabilities']

    # Formata as probabilidades para JSON
    formatted_probabilities = {
        "positive": float(probabilities_dict.get("positive", 0.0)),
        "negative": float(probabilities_dict.get("negative", 0.0)),
        "neutral": float(probabilities_dict.get("neutral", 0.0))
    }
    probabilities_json_string = json.dumps(formatted_probabilities)

    if not os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            f.write("Texto,Sentimento,Confiança,Probabilidades\n")

    with open(HISTORY_FILE, 'a', encoding='utf-8') as f:
        text_cln = text.replace('"', '""')
        probabilities_cln = probabilities_json_string.replace('"', '""')
        f.write(f'"{text_cln}",{rpst_sentiment},{rpst_confidence},"{probabilities_cln}"\n')


#Inicialização do Session State para o texto do usuário
if 'user_text' not in st.session_state:
    st.session_state.user_text = "The stock market showed strong gains today."
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'analyzed_text' not in st.session_state:
    st.session_state.analyzed_text = ""

# Interface do Usuário
st.title("🤖 Análise de Sentimentos Financeiros")
st.markdown("Use o modelo para classificar o sentimento de um texto (em inglês).")

# Input do usuário e botão de análise na mesma linha
col1, col2 = st.columns([4, 1])  # Coluna do texto maior, coluna do botão menor

with col1:
    st.session_state.user_text = st.text_area(
        "Digite o texto aqui:",
        value=st.session_state.user_text,
        height=100,
        key="text_area_input",
        label_visibility="collapsed"  # Esconde o label se o título já for suficiente
    )

with col2:
    st.write("")  # Espaço
    st.write("")  # Espaço
    analyze_button_clicked = st.button("Analisar", key="analyze_main_button",
                                       help="Analisar o sentimento do texto inserido")

# Exemplos de texto para análise
st.markdown("---")
st.subheader("Ou tente um exemplo:")
example_texts = [
    "Stocks surged today on positive economic news.",
    "The company reported disappointing earnings, sending shares lower.",
    "Market sentiment remains cautious amid global uncertainty.",
    "Analysts are bullish on the tech sector for the next quarter.",
    "Inflation concerns are weighing heavily on investor confidence.",
    "The Federal Reserve's interest rate hike surprised many investors.",
]

# Define o número de colunas para os botões de exemplo
num_example_cols = 3
example_cols = st.columns(num_example_cols)

for i, example in enumerate(example_texts):
    button_text = example[:30] + "..." if len(example) > 30 else example
    col_index = i % num_example_cols
    if example_cols[col_index].button(button_text, key=f"ex_btn_{i}", help=example):  # Tooltip com o texto completo
        st.session_state.user_text = example
        st.session_state.analysis_result = None
        st.rerun()

# Lógica de Análise
if analyze_button_clicked:  # Verifica se o botão de análise foi clicado
    if st.session_state.user_text:
        with st.spinner("Analisando..."):
            try:
                payload = {"text": st.session_state.user_text}
                response = requests.post(API_URL, json=payload, timeout=10)
                response.raise_for_status()

                result = response.json()
                save_to_history(st.session_state.user_text, result)

                st.session_state.analysis_result = result
                st.session_state.analyzed_text = st.session_state.user_text


            except requests.exceptions.ConnectionError:
                st.error(
                    f"Erro de conexão: Não foi possível conectar à API em {API_URL}. Verifique se o servidor Flask está rodando.")
            except requests.exceptions.Timeout:
                st.error("Timeout: A requisição para a API demorou muito para responder.")
            except requests.exceptions.RequestException as e:
                st.error(f"Erro na API: {e}")
            except Exception as e:
                st.error(f"Ocorreu um erro inesperado: {e}")
    else:
        st.warning("Por favor, digite um texto ou selecione um exemplo para analisar.")

# Exibição dos Resultados
if st.session_state.analysis_result:
    result = st.session_state.analysis_result
    analyzed_text = st.session_state.analyzed_text

    st.markdown("---")
    st.subheader(f"Resultado da Análise para: \"{analyzed_text[:70]}...\"")

    sentiment = result['sentiment']
    confidence = result['confidence']
    probabilities = result['probabilities']

    # Define a cor baseada no sentimento
    if sentiment.lower() == 'positive':
        color = "green"
        icon = "👍"
    elif sentiment.lower() == 'negative':
        color = "red"
        icon = "👎"
    else:
        color = "blue"
        icon = "😐"

    st.markdown(f"**{icon} Sentimento:** <span style='color:{color}; font-size: 20px;'>{sentiment.capitalize()}</span>",
                unsafe_allow_html=True)
    st.metric(label="Confiança", value=f"{confidence:.2%}")

    st.write("**Probabilidades por classe:**")

    # Pega as probabilidades e formata para exibição
    prob_display_data = {
        "Positivo": probabilities.get("positive", 0.0),
        "Negativo": probabilities.get("negative", 0.0),
        "Neutro": probabilities.get("neutral", 0.0)
    }

    # Define a ordem desejada para exibição das barras
    display_order = ["Positivo", "Negativo", "Neutro"]

    for class_name in display_order:
        prob_value = prob_display_data.get(class_name, 0.0)  # Pega o valor, default 0.0

        # Layout com nome da classe e porcentagem
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**{class_name}**")
        with col2:
            st.write(f"**{prob_value:.1%}**")

        # Usa a barra de progresso nativa do Streamlit
        st.progress(prob_value)

# Histórico de Consultas
st.markdown("---")
if st.checkbox("Mostrar Histórico de Consultas", key="show_history_checkbox"):
    if os.path.exists(HISTORY_FILE):
        try:
            history_df = pd.read_csv(HISTORY_FILE)
            if not history_df.empty:
                st.dataframe(history_df.tail(10), use_container_width=True)
            else:
                st.info("O histórico de consultas está vazio.")
        except pd.errors.EmptyDataError:
            st.info("O arquivo de histórico está vazio ou mal formatado.")
        except Exception as e:
            st.error(f"Erro ao carregar o histórico: {e}")
    else:
        st.info("Nenhuma consulta foi feita ainda.")

st.markdown("---")
st.caption("Desenvolvido com Streamlit e Flask")
