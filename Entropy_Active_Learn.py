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
    page_title="Roça do Seu João - ML do Rancho",
    page_icon="🤠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================
# Geração de Dados Exemplo
# ======================
@st.cache_data(show_spinner=False)
def generate_sample_data(seed=42, bonus_lunar_scale=1.0):
    np.random.seed(seed)
    n_samples = 100
    # Variáveis contínuas
    prod_gourmet = np.random.poisson(850, n_samples)
    visitas_site = np.random.poisson(2500, n_samples)
    ads_google = np.random.exponential(900, n_samples)
    ads_tiktok = np.random.exponential(1200, n_samples)
    chuva = np.random.gamma(4, 50, n_samples)
    hectares = np.random.normal(45, 8, n_samples)
    mao_obra = np.random.uniform(500, 2000, n_samples).astype(int)  # Custo ético de mão de obra
    solo_preservado = np.random.uniform(0, 1, n_samples)  # Fator de sustentabilidade
    # Variáveis categóricas (lua)
    fases = ['Nova', 'Crescente', 'Cheia', 'Minguante']
    fase_lua_plantio = np.random.choice(fases, n_samples)
    fase_lua_colheita = np.random.choice(fases, n_samples)
    # Produção em sacas (lua e sustentabilidade influenciam)
    bonus_plantio = np.where(fase_lua_plantio == "Crescente", 300 * bonus_lunar_scale,
                    np.where(fase_lua_plantio == "Cheia", 200 * bonus_lunar_scale, 0))
    bonus_colheita = np.where(fase_lua_colheita == "Cheia", 400 * bonus_lunar_scale,
                     np.where(fase_lua_colheita == "Minguante", -200 * bonus_lunar_scale, 0))
    producao_safra = (
        15 * hectares +  # Reduzi peso de hectares
        2 * chuva +
        bonus_plantio +
        bonus_colheita +
        10 * solo_preservado * 100 +  # Impacto da preservação
        np.random.normal(0, 100, n_samples)
    ).astype(int)
    # Vendas em R$
    vendas = (
        0.3 * visitas_site +  # Reduzi peso de visitas
        1.2 * prod_gourmet +
        0.3 * ads_google +
        0.4 * ads_tiktok +
        30 * producao_safra +
        5 * mao_obra +  # Impacto de mão de obra justa
        np.random.normal(0, 5000, n_samples)
    ).astype(int)
    # Lucro líquido
    lucro_liquido = (
        vendas -
        (0.2 * ads_google + 0.2 * ads_tiktok + 0.1 * mao_obra + np.random.normal(2000, 500, n_samples))
    ).astype(int)
    data = pd.DataFrame({
        "visitas_site": visitas_site,
        "prod_gourmet_vendas": prod_gourmet,
        "ads_google": ads_google.round(0).astype(int),
        "ads_tiktok": ads_tiktok.round(0).astype(int),
        "chuva_mm": chuva.round(1),
        "hectares": hectares.round(1),
        "mao_obra_etica": mao_obra,
        "solo_preservado": solo_preservado.round(2),
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

# Substituímos o gráfico de importância por uma tabela narrativa
def show_rancho_lessons(results, target_name="Vendas"):
    top_factors = results.nlargest(3, "Magnitude_Absoluta")
    lessons = [
        f"🏜️ **Lições do Rancho**: '{feat}' é chave pra {target_name.lower()} — invista com sabedoria!",
        f"🌾 **Sabedoria Antiga**: '{top_factors['Feature'].iloc[1]}' mostra força na safra!",
        f"🤠 **Resistência**: '{top_factors['Feature'].iloc[2]}' reflete o espírito livre da roça!"
    ]
    return lessons

# ======================
# Cabeçalho Narrativo
# ======================
def show_header():
    st.title("🤠 Roça do Seu João - Sabedoria do Rancho Texano")
    st.markdown("""
    Bem-vindo ao rancho de Seu João, onde o **alto rendimento** nasce da terra e da luta! 🐴  
    Inspirado pela resistência abolicionista e pela tradição roceira, este app carrega o espírito de liberdade e humanidade.  
    Fazendeiros e comerciantes buscam os conselhos de Seu João, e seu bisneto trouxe essa sabedoria pra era digital com **Machine Learning**.  
    Revele os segredos da sua fazenda:  
    - Padrões escondidos na terra 🌱  
    - Fatores que movem a produção 🌦️  
    - Estratégias justas de mercado 📈  
    👉 Teste com dados de exemplo ou envie os **dados da sua roça**.
    """)

# ======================
# App Principal
# ======================
def main():
    show_header()
    # Etapa 1: Carregar ou Gerar Dados
    st.header("1️⃣ Dados da Safra")
    option = st.radio("Escolha a fonte de dados:", ["Gerar dados de exemplo", "Upload CSV"])
    bonus_lunar = st.slider("🌕 Bônus Lunar (ajuste da influência)", 0.5, 1.5, 1.0, 0.1)
    if option == "Upload CSV":
        uploaded_file = st.file_uploader("📂 Envie seu arquivo CSV")
        if uploaded_file:
            data = pd.read_csv(uploaded_file)
        else:
            st.stop()
    else:
        data = generate_sample_data(bonus_lunar_scale=bonus_lunar)
    st.write("### 📊 Amostra dos dados carregados")
    st.dataframe(data.head())
    # Etapa 2: Visualizações Iniciais
    st.header("2️⃣ Visualizações Iniciais")
    col1, col2 = st.columns(2)
    with col1:
        fig1 = plot_scatter(data["prod_gourmet_vendas"], data["lucro_liquido"],
                            "Produtos Gourmet Vendidos", "Lucro Líquido (R$)", color="saddlebrown")
        st.pyplot(fig1)
    with col2:
        fig2 = plot_scatter(data["producao_safra_sacas"], data["vendas_safra_reais"],
                            "Produção (sacas)", "Vendas (R$)", color="forestgreen")
        st.pyplot(fig2)
    # Etapa 3: Treinar Modelo
    st.header("3️⃣ Treinar Modelo")
    target = st.selectbox("Escolha a variável alvo (target):", ["vendas_safra_reais", "producao_safra_sacas", "lucro_liquido"])
    processed_data, features = preprocess_data(data, target)
    mle_results, r2, mse, scaler, model, y_pred = run_mle_analysis(processed_data, features, target)
    st.subheader(f"📈 Resultados do Modelo Linear para `{target}`")
    st.metric("R²", f"{r2:.3f}")
    st.metric("MSE", f"{mse:,.0f}")
    st.dataframe(mle_results)
    # Etapa 4: Lições do Rancho
    st.header("4️⃣ Lições do Rancho")
    lessons = show_rancho_lessons(mle_results, target_name=target)
    for lesson in lessons:
        st.write(lesson)
    # Etapa 5: Conclusão
    st.header("5️⃣ Conclusão")
    st.success("✨ Com os conselhos de Seu João e a força da IA, sua roça pode prosperar com justiça e tradição! 🐴")
    st.balloons()

# ======================
# Run
# ======================
if __name__ == "__main__":
    main()
