# app.py
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
import warnings

warnings.filterwarnings('ignore')

# Configuração visual
st.set_page_config(
    page_title="Roça do João - ML Intelligence",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

# ===================================================
# Funções utilitárias
# ===================================================
@st.cache_data(show_spinner=False)
def generate_sample_data(seed=42):
    np.random.seed(seed)
    n_samples = 120

    visitas_site = np.random.poisson(2500, n_samples)
    prod_gourmet_vendas = np.random.poisson(850, n_samples)
    ads_tiktok = np.random.exponential(1200, n_samples)
    ads_google = np.random.exponential(800, n_samples)
    contratos_export = np.random.poisson(12, n_samples)

    chuva_mm_safra = np.random.gamma(4, 50, n_samples)
    temp_media_safra = np.random.normal(24, 3, n_samples)
    dias_sol_safra = np.random.normal(140, 20, n_samples)

    fase_lua_plantio = np.random.choice(['Nova', 'Crescente', 'Cheia', 'Minguante'], n_samples)
    fase_lua_colheita = np.random.choice(['Nova', 'Crescente', 'Cheia', 'Minguante'], n_samples)

    preco_feijao_saca = np.random.normal(180, 25, n_samples)
    custo_fertilizante = np.random.normal(2500, 300, n_samples)

    hectares_plantados = np.random.normal(45, 8, n_samples)
    funcionarios_safra = np.random.poisson(8, n_samples)

    vendas_base = (
        0.8 * visitas_site +
        1.2 * prod_gourmet_vendas +
        0.3 * ads_tiktok +
        0.4 * ads_google +
        800 * contratos_export +
        2.5 * chuva_mm_safra +
        50 * (temp_media_safra - 20) +
        8 * dias_sol_safra +
        45 * preco_feijao_saca +
        -0.8 * custo_fertilizante +
        450 * hectares_plantados +
        120 * funcionarios_safra +
        np.random.normal(0, 3000, n_samples)
    )
    lua_bonus_plantio = np.where(fase_lua_plantio == 'Crescente', 1500,
                          np.where(fase_lua_plantio == 'Cheia', 1200, 0))
    lua_bonus_colheita = np.where(fase_lua_colheita == 'Cheia', 2000,
                           np.where(fase_lua_colheita == 'Minguante', -800, 0))
    vendas_safra = np.maximum(vendas_base + lua_bonus_plantio + lua_bonus_colheita, 5000)

    producao_safra_sacas = (hectares_plantados * 50 + np.random.normal(0, 200, n_samples)).round(0).astype(int)
    lucro_liquido = vendas_safra - (custo_fertilizante + funcionarios_safra*2500 + np.random.normal(1000,500,n_samples))

    data = pd.DataFrame({
        'visitas_site_ciclo': visitas_site.astype(int),
        'prod_gourmet_vendas': prod_gourmet_vendas.astype(int),
        'ads_tiktok_invest': ads_tiktok.round(0).astype(int),
        'ads_google_invest': ads_google.round(0).astype(int),
        'contratos_export': contratos_export,
        'chuva_mm_safra': chuva_mm_safra.round(1),
        'temp_media_safra': temp_media_safra.round(1),
        'dias_sol_safra': dias_sol_safra.round(0).astype(int),
        'fase_lua_plantio': fase_lua_plantio,
        'fase_lua_colheita': fase_lua_colheita,
        'preco_feijao_saca': preco_feijao_saca.round(2),
        'custo_fertilizante_ha': custo_fertilizante.round(0).astype(int),
        'hectares_plantados': hectares_plantados.round(1),
        'funcionarios_safra': funcionarios_safra,
        'vendas_safra_reais': vendas_safra.round(0).astype(int),
        'producao_safra_sacas': producao_safra_sacas,
        'lucro_liquido': lucro_liquido.round(0).astype(int)
    })
    return data

def preprocess_data(data: pd.DataFrame):
    data_processed = data.copy()
    categorical_cols = data_processed.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        data_processed[col] = le.fit_transform(data_processed[col])
    features = [c for c in data_processed.columns if c not in ['vendas_safra_reais','producao_safra_sacas','lucro_liquido']]
    return data_processed, features

# Modelos básicos
def run_linear(data, features, target):
    X, y = data[features], data[target]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LinearRegression().fit(X_scaled, y)
    return model, scaler, r2_score(y, model.predict(X_scaled))

def run_ridge(data, features, target, scaler):
    X, y = data[features], data[target]
    model = Ridge(alpha=1.0).fit(scaler.transform(X), y)
    return model, r2_score(y, model.predict(scaler.transform(X)))

def run_naive_bayes(data, features, target):
    X, y = data[features], data[target]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    bins = np.linspace(y.min(), y.max(), 4)
    y_classes = np.digitize(y, bins)
    model = GaussianNB().fit(X_scaled, y_classes)
    acc = (model.predict(X_scaled) == y_classes).mean()
    return acc, model

# ===================================================
# UI - Header
# ===================================================
st.title("🌾 Roça do João - Inteligência Artificial no Agronegócio")
st.markdown("""
A Roça do Seu João ficou famosa por seu alto rendimento, mesmo com poucos hectares.  
Seu bisneto inovou criando este serviço que une tradição da roça e inteligência de máquina.  

👉 Aqui, você pode testar diagnósticos e previsões para entender quais fatores mais impactam sua safra.  
""")

# ===================================================
# Etapas (Pipeline com botões)
# ===================================================
if "data" not in st.session_state:
    st.session_state.data = None
if "features" not in st.session_state:
    st.session_state.features = None
if "scaler" not in st.session_state:
    st.session_state.scaler = None
if "models" not in st.session_state:
    st.session_state.models = {}

# Etapa 1 - Dados
if st.button("📂 1. Gerar ou Carregar Dados"):
    opt = st.radio("Escolha:", ["Usar dados da Roça do João","Upload CSV"])
    if opt=="Usar dados da Roça do João":
        st.session_state.data = generate_sample_data()
    else:
        uploaded = st.file_uploader("CSV",type=['csv'])
        if uploaded: st.session_state.data = pd.read_csv(uploaded)
    if st.session_state.data is not None:
        st.success("✅ Dados prontos!")
        st.dataframe(st.session_state.data.head())

# Etapa 2 - Diagnóstico
if st.button("🔎 2. Diagnóstico"):
    if st.session_state.data is None:
        st.error("Carregue dados primeiro!")
    else:
        st.session_state.data, st.session_state.features = preprocess_data(st.session_state.data)
        st.write("📊 Estatísticas da Safra")
        st.dataframe(st.session_state.data.describe())

# Etapa 3 - Treinar Modelos
if st.button("⚙️ 3. Treinar Modelos"):
    if st.session_state.data is None: st.error("Carregue dados primeiro!")
    else:
        data, features = st.session_state.data, st.session_state.features
        lin, scaler, r2_lin = run_linear(data, features, 'vendas_safra_reais')
        ridge, r2_ridge = run_ridge(data, features, 'vendas_safra_reais', scaler)
        acc_nb, nb = run_naive_bayes(data, features, 'producao_safra_sacas')
        st.session_state.scaler = scaler
        st.session_state.models = {"linear":lin,"ridge":ridge,"nb":nb}
        st.success(f"Linear R²={r2_lin:.2f} | Ridge R²={r2_ridge:.2f} | Naive Bayes Acurácia={acc_nb:.2f}")

# Etapa 4 - Resultados
if st.button("📈 4. Visualizar Resultados"):
    if not st.session_state.models:
        st.error("Treine modelos primeiro!")
    else:
        data = st.session_state.data
        st.subheader("Impacto Gourmet no Lucro Líquido")
        fig, ax = plt.subplots()
        ax.scatter(data['prod_gourmet_vendas'], data['lucro_liquido'], alpha=0.6)
        ax.set_xlabel("Produtos Gourmet Vendidos")
        ax.set_ylabel("Lucro Líquido (R$)")
        ax.set_title("Impacto dos Produtos Gourmet no Lucro")
        st.pyplot(fig)

        st.subheader("Produção vs Vendas")
        fig2, ax2 = plt.subplots()
        ax2.scatter(data['producao_safra_sacas'], data['vendas_safra_reais'], c='green', alpha=0.6)
        ax2.set_xlabel("Produção (sacas)")
        ax2.set_ylabel("Vendas (R$)")
        st.pyplot(fig2)

        st.success("🌱 João agora sabe como produção e gourmet impactam seu lucro futuro!")
