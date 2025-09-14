# app.py
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # opcional; não usamos diretamente, mas pode ser útil se você quiser expandir
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
import warnings

warnings.filterwarnings('ignore')

# Configure matplotlib
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

# -----------------------------------
# Configuração da página
# -----------------------------------
st.set_page_config(
    page_title="Roça do João - ML Intelligence",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌾 Análise Inteligente - Roça do João")

# -----------------------------------
# Utilidades
# -----------------------------------
def load_bayesian_libs():
    """Carrega bibliotecas Bayesianas apenas quando necessário"""
    try:
        import pymc as pm
        import arviz as az
        return pm, az, True
    except ImportError as e:
        st.info("ℹ️ Para ativar a análise Bayesiana, instale: `pip install pymc arviz`")
        return None, None, False

@st.cache_data(show_spinner=False)
def generate_sample_data(seed=42):
    """Gera dados realistas da Roça do João - Ciclo de 8 meses por safra"""
    np.random.seed(seed)
    n_samples = 120  # ~10 anos (considerando janelas/séries de 8 meses por ciclo, simplificado)

    # === FEATURES DE MARKETING DIGITAL ===
    visitas_site = np.random.poisson(2500, n_samples)
    prod_gourmet_vendas = np.random.poisson(850, n_samples)
    ads_tiktok = np.random.exponential(1200, n_samples)
    ads_google = np.random.exponential(800, n_samples)

    # === FEATURES DE EXPORTAÇÃO ===
    contratos_export = np.random.poisson(12, n_samples)

    # === FEATURES CLIMÁTICAS ===
    chuva_mm_safra = np.random.gamma(4, 50, n_samples)
    temp_media_safra = np.random.normal(24, 3, n_samples)
    dias_sol_safra = np.random.normal(140, 20, n_samples)

    # === FEATURES LUNARES ===
    fase_lua_plantio = np.random.choice(['Nova', 'Crescente', 'Cheia', 'Minguante'], n_samples)
    fase_lua_colheita = np.random.choice(['Nova', 'Crescente', 'Cheia', 'Minguante'], n_samples)

    # === FEATURES ECONÔMICAS ===
    preco_feijao_saca = np.random.normal(180, 25, n_samples)
    custo_fertilizante = np.random.normal(2500, 300, n_samples)

    # === FEATURES OPERACIONAIS ===
    hectares_plantados = np.random.normal(45, 8, n_samples)
    funcionarios_safra = np.random.poisson(8, n_samples)

    # === TARGET ===
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

    lua_bonus_plantio = np.where(
        fase_lua_plantio == 'Crescente', 1500,
        np.where(fase_lua_plantio == 'Cheia', 1200, 0)
    )
    lua_bonus_colheita = np.where(
        fase_lua_colheita == 'Cheia', 2000,
        np.where(fase_lua_colheita == 'Minguante', -800, 0)
    )

    vendas_safra = np.maximum(vendas_base + lua_bonus_plantio + lua_bonus_colheita, 5000)

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
    })
    return data

def preprocess_data(data: pd.DataFrame):
    """Preprocessa os dados: trata datas, encoda categorias e define target."""
    data_processed = data.copy()

    # Coluna de data (opcional)
    if 'data' in data_processed.columns:
        data_processed['dia_semana'] = pd.to_datetime(data_processed['data']).dt.dayofweek
        data_processed = data_processed.drop('data', axis=1)

    # Encodar categóricas
    categorical_cols = data_processed.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        data_processed[col] = le.fit_transform(data_processed[col])

    # Target
    possible_targets = ['vendas_safra_reais', 'vendas', 'conversoes', 'sales', 'target', 'y']
    target_col = next((t for t in possible_targets if t in data_processed.columns), None)

    if target_col is None:
        numeric_cols = data_processed.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            target_col = numeric_cols[-1]
        else:
            st.error("❌ Nenhuma coluna target válida encontrada!")
            return None, None, None

    features = [c for c in data_processed.columns if c != target_col]
    return data_processed, features, target_col

def run_mle_analysis(data, features, target):
    """Regressão Linear (MLE)"""
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

    return results, r2, mse, scaler, model

def run_map_analysis(data, features, target, scaler):
    """Ridge (MAP)"""
    X = data[features]
    y = data[target]
    X_scaled = scaler.transform(X)

    model = Ridge(alpha=1.0)
    model.fit(X_scaled, y)

    y_pred = model.predict(X_scaled)
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)

    results = pd.DataFrame({
        'Feature': features,
        'Coeficiente': model.coef_,
        'Magnitude_Absoluta': np.abs(model.coef_)
    }).sort_values('Magnitude_Absoluta', ascending=False)

    return results, r2, mse, model

def run_bayesian_analysis(data, features, target, scaler, draws=1000, tune=1000):
    """Regressão Bayesiana com PyMC"""
    pm, az, success = load_bayesian_libs()
    if not success:
        return None

    X = data[features]
    y = data[target]
    X_scaled = scaler.transform(X)

    with pm.Model() as model:
        coefficients = pm.Normal('coefficients', mu=0, sigma=2, shape=len(features))
        intercept = pm.Normal('intercept', mu=y.mean(), sigma=y.std())
        sigma = pm.HalfNormal('sigma', sigma=y.std())

        mu = intercept + pm.math.dot(X_scaled, coefficients)
        _ = pm.Normal('y', mu=mu, sigma=sigma, observed=y)

        try:
            trace = pm.sample(
                draws=draws,
                tune=tune,
                return_inferencedata=True,
                cores=1,
                chains=2,
                target_accept=0.9,
                progressbar=False
            )
        except Exception as e:
            st.error(f"❌ Erro no MCMC: {e}")
            return None

    summary = az.summary(trace, var_names=['coefficients'])
    summary['Feature'] = features
    summary['|Mean|'] = np.abs(summary['mean'])

    results = summary[['Feature', 'mean', 'sd', 'hdi_3%', 'hdi_97%', '|Mean|']].sort_values('|Mean|', ascending=False)
    return results, trace

def compute_entropy(residuals, bins=30):
    """Entropia aproximada dos resíduos (proxy de incerteza)"""
    hist, _ = np.histogram(residuals, bins=bins, density=True)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log(hist)))

def show_entropy_section(y_true, y_pred_mle, y_pred_map, y_pred_bayes=None):
    """Mostra análise de entropia das distribuições de erro"""
    st.header("5️⃣ Análise de Entropia (Incerteza na Fazenda)")
    st.markdown("""
    Entropia mede a **incerteza de uma distribuição**.  
    **Baixa entropia** → previsões mais confiáveis. **Alta entropia** → necessidade de mais dados/ajustes.
    """)

    residuals_mle = y_true - y_pred_mle
    residuals_map = y_true - y_pred_map
    entropy_mle = compute_entropy(residuals_mle)
    entropy_map = compute_entropy(residuals_map)

    entropies = {"MLE (Linear)": entropy_mle, "MAP (Ridge)": entropy_map}

    if y_pred_bayes is not None:
        residuals_bayes = y_true - y_pred_bayes
        entropies["Bayesiano"] = compute_entropy(residuals_bayes)

    entropy_df = pd.DataFrame(entropies.items(), columns=["Modelo", "Entropia"])
    st.dataframe(entropy_df, use_container_width=True)

    best_model = entropy_df.loc[entropy_df['Entropia'].idxmin(), 'Modelo']
    st.success(f"🏆 **Menor incerteza**: {best_model}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(entropies.keys(), entropies.values(), color="green", alpha=0.7)
    ax.set_ylabel("Entropia")
    ax.set_title("🌾 Incerteza dos Modelos - Roça do João")
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

def create_comparison_plot(mle_results, map_results, bayesian_results=None):
    """Gráfico comparativo dos coeficientes"""
    fig, ax = plt.subplots(figsize=(14, 8))

    features = mle_results['Feature'].values
    x_pos = np.arange(len(features))

    if bayesian_results is not None:
        width = 0.25
        ax.bar(x_pos - width, mle_results['Coeficiente'], width, label='MLE', alpha=0.85, color='lightblue')
        ax.bar(x_pos, map_results['Coeficiente'], width, label='MAP (Ridge)', alpha=0.85, color='lightgreen')
        ax.bar(x_pos + width, bayesian_results['mean'], width, label='Bayesiano', alpha=0.85, color='orange')
    else:
        width = 0.35
        ax.bar(x_pos - width/2, mle_results['Coeficiente'], width, label='MLE', alpha=0.85, color='lightblue')
        ax.bar(x_pos + width/2, map_results['Coeficiente'], width, label='MAP (Ridge)', alpha=0.85, color='lightgreen')

    ax.set_xlabel('Fatores da Fazenda')
    ax.set_ylabel('Impacto nas Vendas da Safra (R$)')
    ax.set_title('🌾 Roça do João - Comparação dos Fatores de Impacto')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(features, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    plt.tight_layout()
    return fig

def create_uncertainty_plot(bayesian_results):
    """Gráfico de intervalos de credibilidade (Bayes)"""
    if bayesian_results is None or bayesian_results.empty:
        return None

    fig, ax = plt.subplots(figsize=(12, 10))

    features = bayesian_results['Feature'].values
    means = bayesian_results['mean'].values
    lower = bayesian_results['hdi_3%'].values
    upper = bayesian_results['hdi_97%'].values

    y_pos = np.arange(len(features))

    ax.errorbar(
        means, y_pos, xerr=[means - lower, upper - means],
        fmt='o', capsize=5, capthick=2, markersize=8, color='darkgreen'
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.set_xlabel('Impacto nas Vendas (R$)')
    ax.set_title('🌾 Roça do João - Incerteza dos Fatores (HDI 94%)')
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Sem Impacto')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    return fig

# -----------------------------------
# UI - Cabeçalho
# -----------------------------------
def show_header():
    """Exibe o cabeçalho personalizado da Roça do João"""
    st.markdown("""
    ### 🚀 Transforme dados do seu negócio em decisões inteligentes para maximizar conversões e resultados.

    Este aplicativo, que une ciência de dados, IA e Machine Learn, pode revolucionar o seu negócio, 
    sendo seu consultor desde investimentos em marketing digital aos fatores climáticos e até mesmo 
    tradições como as fases da lua.

    O Exemplo que temos aqui, para ilustrar o nosso funcionamento, é a Roça do seu João.

    João chegou até nós, sabendo que o TikTok é ruim para suas vendas de feijão. 
    Mas seus netos que vão assumir os negócios em breve estão apostando em produtos artesanais, gourmets e processados numa indústria saudável — o que na prática mais consome do que gera lucro.
    É o começo do Pesquisa e Desenvolvimento da Roça.

    Esse mesmo pessoal dos netos do Seu João recomendou que visitasse esse site/app, para entender melhor como o TikTok pode compensar o investimento em P&D dos produtos novos, reforçar sua crença de que a lua é importante na safra e provar matematicamente, com visualizações simples, as **features** que impactam a roça.
    Assim, o próprio app inteligente oferece um feedback aqui.

    **O Desafio do João:** Como otimizar investimentos e operações em cada safra de 8 meses?
    - **Marketing Digital:** TikTok da neta vs Google Ads tradicional  
    - **Exportação:** Contratos internacionais que multiplicam receita  
    - **Clima:** Chuva, sol e temperatura  
    - **Tradição:** Fases da lua influenciam?  
    - **Economia:** Preço do feijão e custo de insumos

    **Insira os dados da sua roça ou explore o dataset de exemplo do Seu João.**
    """)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🌾 Hectares", "45.3", "±8")
    with col2:
        st.metric("📅 Ciclo Safra", "8 meses", "")
    with col3:
        st.metric("💰 Receita Média", "R$ 85k", "por safra")
    with col4:
        st.metric("📈 ROI TikTok", "280%", "vs Google 180%")

    st.markdown("""
    ---
    *"Na roça, cada decisão conta. Agora, João não depende apenas da sorte ou da experiência —  
    ele conta com dados e ciência para maximizar cada safra."*

    Empresas modernas prosperam quando usam dados para:
    - Criar **modelos preditivos**  
    - Medir **ROI de marketing** com precisão  
    - **Otimizar recursos** e reduzir desperdícios  
    - **Reduzir riscos** climáticos, econômicos e de mercado
    """)

# -----------------------------------
# App principal
# -----------------------------------
def main():
    show_header()

    st.markdown("""
    ### 🔑 KPIs como bússolas, dados como chaves escondidas.
    Descubra o **ouro verde** que já existe por trás dos números da fazenda.

    Fazendas modernas precisam de **modelos sob medida**: não apenas relatórios de safra, 
    mas **ferramentas inteligentes** que revelam padrões, otimizam investimentos e reduzem riscos.
    """)
    st.markdown("---")

    # Sidebar
    st.sidebar.header("🎛️ Configurações da Análise")

    data_source = st.sidebar.selectbox(
        "📊 Escolha a fonte dos dados:",
        ["Dados da Roça do João (Exemplo)", "Upload de Arquivo"]
    )

    # Carregar dados
    data = None
    if data_source == "Upload de Arquivo":
        uploaded_file = st.sidebar.file_uploader(
            "📁 Faça upload do seu CSV agrícola",
            type=['csv'],
            help="O arquivo deve conter dados de safra e uma coluna-alvo (ex: vendas/receita)."
        )
        if uploaded_file:
            try:
                data = pd.read_csv(uploaded_file)
                st.sidebar.success("✅ Arquivo carregado!")
            except Exception as e:
                st.sidebar.error(f"❌ Erro ao ler CSV: {e}")
                return
        else:
            st.info("📂 Envie um CSV para analisar seus próprios dados.")
            return
    else:
        data = generate_sample_data()

    # Dataset info
    st.header("📊 Dataset da Fazenda Carregado")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📈 Registros", len(data))
    with col2:
        st.metric("📋 Variáveis", data.shape[1])
    with col3:
        receita_media = data['vendas_safra_reais'].mean() if 'vendas_safra_reais' in data.columns else 0
        st.metric("💰 Receita Média", f"R$ {receita_media:,.0f}")
    with col4:
        missing = int(data.isnull().sum().sum())
        st.metric("❓ Dados Faltantes", missing)

    with st.expander("👀 Visualizar Dados da Fazenda", expanded=False):
        tab1, tab2, tab3 = st.tabs(["📋 Dados Brutos", "📊 Estatísticas", "🌾 Insights Rápidos"])
        with tab1:
            st.dataframe(data, use_container_width=True)
        with tab2:
            st.dataframe(data.describe(include='all'), use_container_width=True)
        with tab3:
            if 'vendas_safra_reais' in data.columns:
                colA, colB = st.columns(2)
                with colA:
                    st.write("**🏆 Melhor Safra:**")
                    melhor_safra = data.loc[data['vendas_safra_reais'].idxmax()]
                    st.write(f"Receita: R$ {melhor_safra['vendas_safra_reais']:,.0f}")
                with colB:
                    st.write("**📉 Pior Safra:**")
                    pior_safra = data.loc[data['vendas_safra_reais'].idxmin()]
                    st.write(f"Receita: R$ {pior_safra['vendas_safra_reais']:,.0f}")

    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Configurações da Análise")

    include_bayesian = st.sidebar.checkbox("🧠 Incluir Análise Bayesiana", value=True)

    if include_bayesian:
        draws = st.sidebar.slider("🎲 Draws:", 500, 2000, 1000, 100)
        tune = st.sidebar.slider("🔧 Tune:", 500, 2000, 1000, 100)
        st.sidebar.info(f"Total de amostras: {draws * 2} (2 chains)")

    st.markdown("---")

    if st.button("🚀 **INICIAR ANÁLISE COMPLETA DA FAZENDA**", type="primary"):
        analysis_container = st.container()
        with analysis_container:
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                # 1) Preprocessamento
                status_text.text("🔄 Preparando dados da fazenda...")
                progress_bar.progress(10)
                processed_data, features, target = preprocess_data(data)
                if processed_data is None:
                    return
                st.success(f"✅ **Dados processados!** Target: `{target}` | Features: `{len(features)}`")
                progress_bar.progress(20)

                # 2) MLE
                status_text.text("🔍 Executando análise MLE (Linear)...")
                progress_bar.progress(35)
                mle_results, mle_r2, mle_mse, scaler, mle_model = run_mle_analysis(processed_data, features, target)

                st.header("1️⃣ Análise Linear (MLE) - Relações Diretas")
                st.markdown("*Mostra o impacto linear de cada fator nas vendas da safra*")
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.dataframe(mle_results, use_container_width=True)
                with col2:
                    st.metric("📊 R²", f"{mle_r2:.3f}")
                    st.metric("📏 MSE", f"{mle_mse:,.0f}")
                    if mle_r2 > 0.8:
                        st.success("🎯 Excelente ajuste!")
                    elif mle_r2 > 0.6:
                        st.warning("⚠️ Ajuste moderado")
                    else:
                        st.error("❌ Ajuste baixo — investigue não-linearidades/mais features")

                progress_bar.progress(50)

                # 3) MAP (Ridge)
                status_text.text("🎯 Executando análise MAP (Ridge)...")
                progress_bar.progress(60)
                map_results, map_r2, map_mse, map_model = run_map_analysis(processed_data, features, target, scaler)

                st.header("2️⃣ Análise Regularizada (MAP) - Controle de Overfitting")
                st.markdown("*Modelo mais conservador, evita superajuste aos dados históricos*")
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.dataframe(map_results, use_container_width=True)
                with col2:
                    st.metric("📊 R²", f"{map_r2:.3f}")
                    st.metric("📏 MSE", f"{map_mse:,.0f}")

                progress_bar.progress(75)

                # 4) Bayes
                bayesian_results = None
                if include_bayesian:
                    status_text.text("🧠 Executando análise Bayesiana...")
                    progress_bar.progress(80)
                    with st.spinner("🔄 MCMC em execução..."):
                        bayesian_output = run_bayesian_analysis(processed_data, features, target, scaler, draws, tune)

                    if bayesian_output:
                        bayesian_results, trace = bayesian_output
                        st.header("3️⃣ Análise Bayesiana - Incerteza Quantificada")
                        st.markdown("*Impacto estimado + intervalos de credibilidade*")
                        st.dataframe(bayesian_results, use_container_width=True)

                        st.subheader("📋 Insights para a Roça do João")
                        st.write("**🎯 Fatores Mais Importantes para as Vendas:**")
                        top_features = bayesian_results.nlargest(3, '|Mean|')
                        for idx, (_, row) in enumerate(top_features.iterrows(), 1):
                            uncertainty_level = "📍 Baixa" if row['sd'] < abs(row['mean'])/3 else "⚠️ Alta"
                            impacto = "positivo 📈" if row['mean'] > 0 else "negativo 📉"
                            st.write(f"**{idx}. {row['Feature']}**: Impacto {impacto} de R$ {row['mean']:,.0f} (±{row['sd']:,.0f}) — Incerteza: {uncertainty_level}")

                        # Alta incerteza
                        high_uncertainty = bayesian_results[bayesian_results['sd'] > bayesian_results['mean'].abs()/2]
                        if len(high_uncertainty) > 0:
                            st.write("**⚠️ Fatores que Precisam de Mais Dados:**")
                            for _, row in high_uncertainty.iterrows():
                                if row['hdi_3%'] < 0 < row['hdi_97%']:
                                    rec = "❓ Efeito indeterminado — coletar mais dados"
                                elif row['hdi_3%'] > 0:
                                    rec = "📈 Provavelmente positivo"
                                else:
                                    rec = "📉 Provavelmente negativo"
                                st.write(f"• **{row['Feature']}**: {rec} (HDI: [R$ {row['hdi_3%']:,.0f}, R$ {row['hdi_97%']:,.0f}])")

                progress_bar.progress(90)

                # 5) Visualizações
                status_text.text("📊 Criando visualizações...")
                st.header("4️⃣ Visualizações Comparativas")
                fig1 = create_comparison_plot(mle_results, map_results, bayesian_results)
                st.pyplot(fig1)

                if bayesian_results is not None and not bayesian_results.empty:
                    fig2 = create_uncertainty_plot(bayesian_results)
                    if fig2:
                        st.pyplot(fig2)

                # 6) Entropia
                status_text.text("📊 Calculando entropia dos modelos...")
                progress_bar.progress(95)
                X_scaled = scaler.transform(processed_data[features])
                y_true = processed_data[target]
                y_pred_mle = mle_model.predict(X_scaled)
                y_pred_map = map_model.predict(X_scaled)
                show_entropy_section(
                    y_true=y_true,
                    y_pred_mle=y_pred_mle,
                    y_pred_map=y_pred_map,
                    y_pred_bayes=None
                )

                # 7) Recomendações
                st.header("6️⃣ Recomendações Estratégicas para João")

                if bayesian_results is not None and not bayesian_results.empty:
                    top_factor = bayesian_results.iloc[0]
                    st.success(f"🌟 **Fator #1 de Impacto:** {top_factor['Feature']}")
                    st.write(f"Cada unidade de melhoria pode gerar ~R$ {top_factor['mean']:,.0f} por safra (em média).")

                    recommendations = []
                    for _, row in bayesian_results.head(5).iterrows():
                        factor = row['Feature'].lower()
                        if 'chuva' in factor or 'temp' in factor or 'sol' in factor or 'clima' in factor:
                            recommendations.append("🌦️ **Clima**: Investir em irrigação/captação de água e monitoramento climático.")
                        if 'tiktok' in factor:
                            recommendations.append("📱 **TikTok**: Escalar criativos e testes A/B; reforçar prova social.")
                        if 'google' in factor:
                            recommendations.append("🔎 **Google Ads**: Focar em palavras-chave de alta intenção.")
                        if 'export' in factor:
                            recommendations.append("🌍 **Exportação**: Priorizar novos contratos e qualidade para mercados externos.")
                        if 'preco' in factor:
                            recommendations.append("💰 **Preço**: Avaliar travas/hedge e timing de venda.")
                        if 'hectares' in factor:
                            recommendations.append("🌾 **Área Plantada**: Estudar expansão gradual com análise de custo/retorno.")

                    if recommendations:
                        st.write("**🎯 Ações Prioritárias:**")
                        for rec in sorted(set(recommendations)):
                            st.write(f"• {rec}")

                # 8) Fechamento narrativo
                st.markdown("---")
                st.success("""
                🔮 **Resumo Narrativo da Análise da Roça do João**

                João descobre que sua roça já tem um **núcleo de fatores fortes e previsíveis**:
                contratos de exportação, clima e marketing digital. Esses elementos sustentam
                a maior parte da receita e oferecem previsões confiáveis.

                Outros pontos — como fases da lua ou custos de insumo — ainda aparecem com
                incerteza. Isso não significa que não importam, mas que João pode **coletar
                mais dados** para confirmar sua influência.

                O resultado é claro: com dados, João deixa de depender apenas da tradição ou
                da intuição. Ele ganha uma **bússola científica** para tomar decisões,
                equilibrando o que já sabe da terra com o que a inteligência artificial revela.
                """)

                # Finalização
                progress_bar.progress(100)
                status_text.text("✅ Análise concluída!")
                st.balloons()
                st.success("**🌾 Análise completa da Roça do João!**")
                st.success("✨ Imagine o que podemos fazer com **os dados da sua fazenda**.")

                # Download
                if bayesian_results is not None and not bayesian_results.empty:
                    csv_data = bayesian_results.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Baixar Análise Bayesiana (CSV)",
                        data=csv_data,
                        file_name=f"analise_roca_joao_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )

            except Exception as e:
                progress_bar.progress(0)
                status_text.text("❌ Erro na análise")
                st.error(f"**Erro durante a análise:** {str(e)}")
                with st.expander("🔍 Detalhes do Erro"):
                    import traceback
                    st.code(traceback.format_exc())

    # Rodapé
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
    <b>🌾 Roça do João - Inteligência Artificial para o Agronegócio</b><br>
    <b>Dica:</b> Para melhores resultados, tenha ao menos ~20 registros de safra<br>
    <b>Algoritmos:</b> Scikit-learn (MLE/MAP) + PyMC (Bayesian MCMC) + Entropia de Resíduos
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------
# Run
# -----------------------------------
if __name__ == "__main__":
    main()
