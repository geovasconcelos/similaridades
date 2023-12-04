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
def calculate_similarity(input_text, df, similaridade, colunas_resultado):
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
            similar_contracts.append(df.iloc[i][colunas_resultado])

    similar_df = pd.DataFrame(similar_contracts, columns=colunas_resultado).reset_index(drop=True)

    return similar_df

# Interface do aplicativo usando Streamlit
def main():
    
    image = 'data:image/png;base64,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'
    st.image(image, use_column_width=False)
    
    st.title("Aplicativo de Busca de contratos similares")
    
    st.markdown(
    """
    Aqui você pode buscar contratos ativos em todas as empresas do grupo Eletrobras.
    Você digita o objeto contratual e define o nível de similaridade em sua busca e 
    recebe uma lista dos contratos ativos com objetos similares.
    """)

    # Carregando DataFrame sem criar índice
    uploaded_file = st.file_uploader("Carregar arquivo Excel", type=["xlsx"])

    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file, index_col=None)  # Adicionando index_col=None
        similaridade = st.slider("Selecione o limiar de similaridade", min_value=0.0, max_value=1.0, step=0.1, value=0.7)

        # Adicionando campo de input
        input_text = st.text_area("Insira o texto para comparação:", "")

        # Escolhendo colunas para mostrar no resultado
        colunas_resultado = st.multiselect("Selecione as colunas para mostrar no resultado:", df.columns)

        if st.button("Calcular Similaridade"):
            if not colunas_resultado:
                st.warning("Selecione pelo menos uma coluna para mostrar no resultado.")
            else:
                similar_df = calculate_similarity(input_text, df, similaridade, colunas_resultado)
                st.write("Contratos Similares:")
                st.write(similar_df)

if __name__ == "__main__":
    main()
