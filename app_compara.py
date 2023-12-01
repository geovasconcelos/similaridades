import streamlit as st
import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Função para pré-processamento do texto
def preprocess_text(text):
    if isinstance(text, str):  # Verifica se é uma string
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
    return text

# Função para calcular similaridade entre o texto inserido e a coluna OBJETO
def calculate_similarity(input_text, df, similaridade):
    input_text = preprocess_text(input_text)

    df['OBJETO'] = df['OBJETO'].apply(preprocess_text)

    # Preencher NaN com uma string vazia
    df['OBJETO'].fillna("", inplace=True)

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(pd.concat([df['OBJETO'], pd.Series([input_text])]))

    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    similar_contracts = []

    for i in range(len(df)):
        if cosine_sim[-1][i] > similaridade:
            similar_contracts.append(df.iloc[i])

    similar_df = pd.DataFrame(similar_contracts)

    return similar_df

# Interface do aplicativo usando Streamlit
def main():
    st.title("Aplicativo de Similaridade de Contratos")

    # Carregando DataFrame
    uploaded_file = st.file_uploader("Carregar arquivo Excel", type=["xlsx"])

    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        similaridade = st.slider("Selecione o limiar de similaridade", min_value=0.0, max_value=1.0, step=0.1)

        # Adicionando campo de input
        input_text = st.text_area("Insira o texto para comparação:", "")

        if st.button("Calcular Similaridade"):
            similar_df = calculate_similarity(input_text, df, similaridade)
            st.write("Contratos Similares:")
            st.write(similar_df)

if __name__ == "__main__":
    main()
