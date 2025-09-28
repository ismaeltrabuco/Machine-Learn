# app.py
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')
# ======================
# Configuração da Página
# ======================
st.set_page_config(
    page_title="Roça do Seu João",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)
# ======================
# Geração de Dados Exemplo
# ======================
@st.cache_data(show_spinner=False)
def generate_sample_data(seed=42):
    np.random.seed(seed)
    n_samples = 100
    # Variáveis contínuas
    prod_gourmet = np.random.poisson(850, n_samples)
    visitas_site = np.random.poisson(2500, n_samples)
    ads_google = np.random.exponential(900, n_samples)
    ads_tiktok = np.random.exponential(1200, n_samples)
    chuva = np.random.gamma(4, 50, n_samples)
    hectares = np.random.normal(45, 8, n_samples)
    # Variáveis categóricas (lua)
    fases = ['Nova', 'Crescente', 'Cheia', 'Minguante']
    fase_lua_plantio = np.random.choice(fases, n_samples)
    fase_lua_colheita = np.random.choice(fases, n_samples)
    # Produção em sacas (lua influencia)
    bonus_plantio = np.where(fase_lua_plantio == "Crescente", 300,
                    np.where(fase_lua_plantio == "Cheia", 200, 0))
    bonus_colheita = np.where(fase_lua_colheita == "Cheia", 400,
                     np.where(fase_lua_colheita == "Minguante", -200, 0))
    producao_safra = (
        20 * hectares +
        2 * chuva +
        bonus_plantio +
        bonus_colheita +
        np.random.normal(0, 100, n_samples)
    ).astype(int)
    # Vendas em R$
    vendas = (
        0.5 * visitas_site +
        1.5 * prod_gourmet +
        0.4 * ads_google +
        0.6 * ads_tiktok +
        35 * producao_safra +
        np.random.normal(0, 5000, n_samples)
    ).astype(int)
    # Lucro líquido
    lucro_liquido = (
        vendas -
        (0.3 * ads_google + 0.3 * ads_tiktok + np.random.normal(2000, 500, n_samples))
    ).astype(int)
    data = pd.DataFrame({
        "visitas_site": visitas_site,
        "prod_gourmet_vendas": prod_gourmet,
        "ads_google": ads_google.round(0).astype(int),
        "ads_tiktok": ads_tiktok.round(0).astype(int),
        "chuva_mm": chuva.round(1),
        "hectares": hectares.round(1),
        "fase_lua_plantio": fase_lua_plantio,
        "fase_lua_colheita": fase_lua_colheita,
        "producao_safra_sacas": producao_safra,
        "vendas_safra_reais": vendas,
        "lucro_liquido": lucro_liquido
    })
    return data
# ======================
# Funções de Modelo
# ======================
def preprocess_data(data: pd.DataFrame, target: str):
    data_processed = data.copy()
    features = [c for c in data_processed.columns if c != target]
    # Encodar categóricas
    categorical_cols = data_processed.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        data_processed[col] = le.fit_transform(data_processed[col])
    return data_processed, features
def run_mle_analysis(data, features, target):
    X = data[features]
    y = data[target]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LinearRegression()
    model.fit(X_scaled, y)
    y_pred = model.predict(X_scaled)
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    results = pd.DataFrame({
        'Feature': features,
        'Coeficiente': model.coef_,
        'Magnitude_Absoluta': np.abs(model.coef_)
    }).sort_values('Magnitude_Absoluta', ascending=False)
    return results, r2, mse, scaler, model, y_pred
# ======================
# Gráficos
# ======================
def plot_scatter(x, y, xlabel, ylabel, color="blue"):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, y, alpha=0.7, color=color)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig
def plot_feature_importance(results, target_name="Vendas"):
    fig, ax = plt.subplots(figsize=(10, 6))
    results_sorted = results.sort_values("Magnitude_Absoluta", ascending=True)
    ax.barh(results_sorted["Feature"], results_sorted["Magnitude_Absoluta"], color="cornflowerblue")
    ax.set_xlabel("Importância (coeficientes normalizados)")
    ax.set_title(f"🌾 Importância das Features para {target_name}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig
# ======================
# Cabeçalho Narrativo
# ======================
def show_header():
    st.title("🌾 A Roça do João - Inteligência Aplicada ao Campo")
    st.markdown("""
    A Roça do Seu João ficou famosa por seu **alto rendimento**, mesmo com poucos hectares.
    Fazendeiros, industriais e comerciantes de todos os portes procuram Seu João para consultorias.
    Inspirado nessa tradição, seu bisneto criou o **serviço Roça do João**, um app que, assim como
    uma boa conversa na varanda, analisa os dados da fazenda com **Machine Learning** para revelar:
    - Padrões escondidos 🌱
    - Fatores críticos de produção 🌦️
    - Estratégias de marketing que dão retorno 📈
    👉 Aqui você pode **testar com dados de exemplo** ou carregar os **dados da sua própria fazenda**.
    """)
# ======================
# App Principal
# ======================
def main():
    show_header()
    # Etapa 1: Carregar ou Gerar Dados
    st.header("1️⃣ Dados da Safra")
    option = st.radio("Escolha a fonte de dados:", ["Gerar dados de exemplo", "Upload CSV"])
    if option == "Upload CSV":
        uploaded_file = st.file_uploader("📂 Envie seu arquivo CSV")
        if uploaded_file:
            data = pd.read_csv(uploaded_file)
        else:
            st.stop()
    else:
        data = generate_sample_data()
    st.write("### 📊 Amostra dos dados carregados")
    st.dataframe(data.head())
    # Etapa 2: Visualizações Iniciais
    st.header("2️⃣ Visualizações Iniciais")
    col1, col2 = st.columns(2)
    with col1:
        fig1 = plot_scatter(data["prod_gourmet_vendas"], data["lucro_liquido"],
                            "Produtos Gourmet Vendidos", "Lucro Líquido (R$)", color="steelblue")
        st.pyplot(fig1)
    with col2:
        fig2 = plot_scatter(data["producao_safra_sacas"], data["vendas_safra_reais"],
                            "Produção (sacas)", "Vendas (R$)", color="green")
        st.pyplot(fig2)
    # Etapa 3: Treinar Modelo
    st.header("3️⃣ Treinar Modelo")
    target = st.selectbox("Escolha a variável alvo (target):", ["vendas_safra_reais", "producao_safra_sacas"])
    processed_data, features = preprocess_data(data, target)
    mle_results, r2, mse, scaler, model, y_pred = run_mle_analysis(processed_data, features, target)
    st.subheader(f"📈 Resultados do Modelo Linear para {target}")
    st.metric("R²", f"{r2:.3f}")
    st.metric("MSE", f"{mse:,.0f}")
    st.dataframe(mle_results)
    # Etapa 4: Importância das Features
    st.header("4️⃣ Importância das Features")
    fig_imp = plot_feature_importance(mle_results, target_name=target)
    st.pyplot(fig_imp)
    # Top 3 fatores
    top_factors = mle_results.nlargest(3, "Magnitude_Absoluta")
    st.info(f"🔑 **Top 3 fatores que mais influenciam {target}:** {', '.join(top_factors['Feature'].tolist())}")
    # Etapa 5: Conclusão
    st.header("5️⃣ Conclusão")
    st.success("✨ João agora sabe quais fatores realmente impactam sua produção e vendas. Imagine o que podemos fazer com **os dados da sua fazenda**!")
    st.balloons()
# ======================
# Run
# ======================
if **name** == "**main**":
    main()
