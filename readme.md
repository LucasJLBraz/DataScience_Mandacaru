
# DemoDay Desafio 2

Este é um projeto do Mandacaru.dev, em que fazemos uma análise e classificação do dataset "Financial Sentiment Analysis" disponível em: https://www.kaggle.com/datasets/sbhatti/financial-sentiment-analysis.
## Abordagem clássica em aprendizado de máquina:

### Seção 1: Pré-processamento dos dados

O pré-processamento dos dados é uma parte fundamental do NLP, para o tratamento dos nossos dados iremos usar:<br>

1 - Remoção de linhas nulas/vazias.<br>
2 - Remoção de alfanuméricos.<br>
3 - Conversão em palavras minúsculas.<br>
4 - Remoção de pontuações e stopwords.<br>
5 - Lemmatização<br>
6 - Divisão de treino/teste<br>
7 - Codificação da variável target<br>
8 - Vetorização dos dados via TF/IDF<br>

### - Seção 2: Treinamento do modelo

Para o treinamento do modelo usaremos 80% do dataset, e 20% para o teste. <br>
Usamos os seguintes classificadores: (SVM, Random Forest, XGBoost, MLP, Naive Bayes).<br> 
Também implementamos o GridSearch para o comparativo do impacto do GridSearch no resultado dos classificadores.

## Abordagem usando o Modelo de Aprendizado Profundo(BERT)

### Seção 1: Importações e Leitura de Dados**
- Realiza importações necessárias, incluindo bibliotecas como NumPy, Pandas, Transformers e outras.
- Lê os dados de um arquivo CSV chamado "data.csv".

### Seção 1.5: Análise Exploratória de Dados**
- Realiza análise exploratória dos dados, gerando gráficos para visualizar a distribuição das classes, o comprimento das frases e uma nuvem de palavras.

### Seção 2: Pré-processamento e Tokenização**
- Remove linhas com valores nulos.
- Atualiza os rótulos usando um codificador de rótulos.
- Divide os dados em conjuntos de treinamento e teste.
  
### Seção 2.5: Aumento de Dados**
- Aumenta os dados para a classe menos dominante substituindo palavras por seus sinônimos.
  
### Seção 2.7: Análise após Aumento de Dados e Tokenização**
- Adiciona uma nova coluna com os nomes originais das classes.
- Gera gráficos para visualizar a distribuição das classes após o aumento de dados e a divisão entre treino e teste.

### Seção 3: Classe do Conjunto de Dados**
- Define uma classe de conjunto de dados para manipular os dados durante o treinamento do modelo.

### Seção 4: Parâmetros do Modelo**
- Define a arquitetura do modelo usando BERT pré-treinado.
- Configura otimizador, critério de perda e agendador de taxa de aprendizado.

### Seção 5: Funções Auxiliares**
- Implementa funções auxiliares para verificar a acurácia e calcular a acurácia média.

### Seção 6: Treinamento com Validação Cruzada**
- Realiza o treinamento do modelo usando validação cruzada e estratificação.
- Armazena perdas de treinamento e validação para cada dobra.

### Seção 6.5: Gráfico de Perda de Treinamento e Acurácia de Validação**
- Gera um gráfico para visualizar as perdas de treinamento e a acurácia de validação ao longo das épocas e dobras.

### Seção 7: Testando o Modelo**
- Avalia o modelo no conjunto de teste final.
- Calcula a precisão de teste e imprime um relatório de classificação e uma matriz de confusão.

### Seção 8: Salvando o Modelo, Tokenizador e Encoder**
- Salva o modelo treinado, o tokenizador BERT e o codificador de rótulos para uso futuro.
