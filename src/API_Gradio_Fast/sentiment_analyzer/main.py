import gradio as gr
from gradio import themes
import requests
import json
import csv
import pandas as pd
from gradio import Interface, Text, Number, Dropdown, HTML, Label, TabbedInterface  # Import elements directly


def predict_sentiment(text: str):
    data = {"text": text}
    response = requests.post("http://localhost:8000/predict", json=data)
    if response.status_code != 200:
        return "Error: Unable to get prediction from the model."
    # Parse the response
    result = response.json()
    sentiment = result['sentiment']
    confidence = result['confidence']
    probabilities = result['probabilities']

    # Converter os nomes das probabilidades para português
    probabilities = { "Positivo": probabilities["positive"], "Negativo": probabilities["negative"], "Neutro": probabilities["neutral"] }


    print(f"Probabilities: {probabilities}")

    # Convert the sentiment prediction to a simpler format
    if sentiment == "positive":
        sentiment_label = "Positivo"
    elif sentiment == "negative":
        sentiment_label = "Negativo"
    else:
        sentiment_label = "Neutro"


    # Verificar se o arquivo existe
    try:
        with open('query_history.csv', 'r') as file:
            pass
    except FileNotFoundError:
        # Se não existir, criar o arquivo e escrever o cabeçalho
        with open('query_history.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Texto", "Sentimento", "Confiança", "Probabilidades"])
    # Write the input and output to a CSV file
    with open('query_history.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([text, sentiment_label, confidence, probabilities])

    # Color the output based on the sentiment
    if sentiment_label == "Positivo":
        sentiment_label = f"<span style='color:green; font-size:20px'>{sentiment_label}</span>"
    elif sentiment_label == "Negativo":
        sentiment_label = f"<span style='color:red;font-size:20px'>{sentiment_label}</span>"
    else:
        sentiment_label = f"<span style='color:grey;font-size:20px'>{sentiment_label}</span>"
    
    
    return sentiment_label, confidence, probabilities



def load_query_history():
    try:
        df = pd.read_csv('query_history.csv')
        return df.to_html()
    except FileNotFoundError:
        return "No query history found."

iface = TabbedInterface([
    Interface(
    theme=gr.themes.Soft(),
    fn=predict_sentiment,
    inputs=[
        Text(label="Entre com um texto, em inglês, para classificar:")
    ],
    outputs=[
        HTML(label="Sentimento:"),  # Use Dropdown
        Number(label="Confiança:"),
        Label(label="Probabilidades:", num_top_classes=3)
    ],
    examples=[ 
        ["I love mandacaru.dev, it's the best course ever!"],
        ["I'm really happy to show my project to the world!"],
        ["My dogecoin stock droped by 0.123% today."],
        ["My bitcoin stock growed by 0.123% today."],
        ["Etherium did not change today."],
        ["Given the current situation, I'm not sure if I should invest in the stock market."],
        ["Stocks remained stable today, with the S&P 500 showing a minimal change of +0.1%."],
        ["The market exhibited a lack of major movements, with the Dow Jones Industrial Average hovering around the same levels."],
        ["Investors celebrated as the technology sector saw significant gains, pushing the NASDAQ up by 1.5%."],
        ["Despite overall positive sentiment, some blue-chip stocks, like Apple (AAPL), experienced a slight decline of -0.2%."],
        ["Market indices displayed a neutral trend, reflecting the uncertainty in economic conditions."],
        ["Investors faced challenges today as several energy stocks, including ExxonMobil (XOM), recorded notable declines."]
    ],
    title="Classificador de Sentimentos",
    description="Esta é uma interface para o classificador de sentimentos. Digite um texto e clique em 'Submeter' para ver a predição do modelo."
    ),
    Interface(
    theme=gr.themes.Soft(),
    fn=load_query_history,
    inputs=[],
    outputs=[
        HTML(label="Histórico de Consultas:")
    ],
    title="Histórico de Consultas",
    description="Esta é uma interface para visualizar o histórico de consultas do classificador de sentimentos."
    )
    ],
    ["Classificador de Sentimentos", "Histórico de Consultas"]

)

iface.queue()

iface.launch(server_port=7860, share=False, debug=True)
