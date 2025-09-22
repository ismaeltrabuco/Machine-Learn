import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("✨ Heaven Threshold ✨")
st.write("Modelo cósmico de ascensão baseado na fórmula celestial.")

# Upload do dataset
file = st.file_uploader("Envie seu dataset.csv", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.subheader("Tabela Celestial")
    st.dataframe(df)

    # Gráfico de distribuição
    st.subheader("Distribuição por origem")
    fig, ax = plt.subplots()
    df['origem'].value_counts().plot(kind="bar", ax=ax)
    st.pyplot(fig)

    # Gráfico de quem passa e não passa
    st.subheader("Ascensão (y=1) vs Reprovação (y=-1)")
    fig2, ax2 = plt.subplots()
    df['y'].value_counts().plot(kind="pie", autopct='%1.1f%%', ax=ax2)
    ax2.set_ylabel("")
    st.pyplot(fig2)

    # Comparação por origem
    st.subheader("Taxa de ascensão por origem")
    taxa = df.groupby("origem")['y'].apply(lambda x: (x==1).mean())
    st.bar_chart(taxa)
