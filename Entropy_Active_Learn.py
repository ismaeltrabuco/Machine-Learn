import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

# Configuração da página
st.set_page_config(
    page_title="Roça do João - ML Intelligence",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌾 Análise Inteligente - Roça do João")

def show_header():
    """Exibe o cabeçalho personalizado da Roça do João"""
    st.markdown("""
    ### 🚀 Transforme dados da fazenda em decisões inteligentes para maximizar cada safra

    Este aplicativo demonstra como a ciência de dados pode revolucionar o agronegócio, 
    analisando desde investimentos em marketing digital até fatores climáticos e até mesmo 
    tradições rurais como as fases da lua 🌙.

    **O Desafio do João:** Como otimizar investimentos e operações em cada safra de 8 meses?
    - 🎯 **Marketing Digital:** TikTok da neta vs Google Ads tradicional
    - 🌍 **Exportação:** Contratos internacionais que multiplicam receita  
    - 🌦️ **Clima:** Chuva, sol e temperatura - os verdadeiros chefes da fazenda
    - 🌙 **Tradição:** Será que plantar na lua crescente realmente funciona?
    - 💰 **Economia:** Preços do feijão e custos dos insumos

    **Configure seus dados no painel lateral ou explore o dataset de exemplo**
    """)
    
    # Metrics importantes do João
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
    *"Na roça, cada decisão conta. Com dados, João não depende mais só da sorte e da experiência - 
    ele tem ciência para maximizar cada safra."*

    **🚀 Empresas agrícolas modernas precisam de:**
    - Modelos preditivos para planejar safras
    - Análise de ROI em marketing rural  
    - Otimização de recursos baseada em dados
    - Redução de riscos climáticos e econômicos
    ---
    """)

def load_bayesian_libs():
    """Carrega bibliotecas Bayesianas apenas quando necessário"""
    try:
        import pymc as pm
        import arviz as az
        return pm, az, True
    except ImportError as e:
        st.error(f"Erro ao importar bibliotecas Bayesianas: {e}")
        st.info("Execute: pip install pymc arviz")
        return None, None, False

def generate_sample_data():
    """Gera dados realistas da Roça do João - Ciclo de 8 meses por safra"""
    np.random.seed(42)
    n_samples = 120  # 10 anos de dados (120 safras de 8 meses)
    
    # === FEATURES DE MARKETING DIGITAL ===
    visitas_site = np.random.poisson(2500, n_samples)  # Visitas no site da fazenda
    prod_gourmet_vendas = np.random.poisson(850, n_samples)  # Vendas de produtos especiais (feijão orgânico, etc)
    ads_tiktok = np.random.exponential(1200, n_samples)  # Investimento TikTok (neta do João)
    ads_google = np.random.exponential(800, n_samples)   # Ads no Google
    
    # === FEATURES DE EXPORTAÇÃO ===
    contratos_export = np.random.poisson(12, n_samples)  # Número de contratos de exportação
    
    # === FEATURES CLIMÁTICAS (CRUCIAL NO AGRO) ===
    chuva_mm_safra = np.random.gamma(4, 50, n_samples)  # Chuva total na safra (mm)
    temp_media_safra = np.random.normal(24, 3, n_samples)  # Temperatura média (°C)
    dias_sol_safra = np.random.normal(140, 20, n_samples)  # Dias de sol na safra
    
    # === FEATURES LUNARES (TRADIÇÃO RURAL) ===
    fase_lua_plantio = np.random.choice(['Nova', 'Crescente', 'Cheia', 'Minguante'], n_samples)
    fase_lua_colheita = np.random.choice(['Nova', 'Crescente', 'Cheia', 'Minguante'], n_samples)
    
    # === FEATURES ECONÔMICAS ===
    preco_feijao_saca = np.random.normal(180, 25, n_samples)  # Preço da saca (R$)
    custo_fertilizante = np.random.normal(2500, 300, n_samples)  # Custo fertilizante por hectare
    
    # === FEATURES OPERACIONAIS ===
    hectares_plantados = np.random.normal(45, 8, n_samples)  # Área plantada
    funcionarios_safra = np.random.poisson(8, n_samples)  # Número de funcionários
    
    # === VARIÁVEL TARGET: VENDAS POR SAFRA ===
    # Simulando uma relação complexa e realista
    vendas_base = (
        # Marketing digital (impacto moderado)
        0.8 * visitas_site +
        1.2 * prod_gourmet_vendas +
        0.3 * ads_tiktok +
        0.4 * ads_google +
        
        # Exportação (alto impacto)
        800 * contratos_export +
        
        # Clima (MUITO importante no agro)
        2.5 * chuva_mm_safra +
        50 * (temp_media_safra - 20) +  # Temperatura ideal ~20-25°C
        8 * dias_sol_safra +
        
        # Economia
        45 * preco_feijao_saca +
        -0.8 * custo_fertilizante +
        
        # Operacional
        450 * hectares_plantados +
        120 * funcionarios_safra +
        
        # Ruído realista
        np.random.normal(0, 3000, n_samples)
    )
    
    # Ajustes por fase da lua (baseado em crenças populares rurais)
    lua_bonus_plantio = np.where(fase_lua_plantio == 'Crescente', 1500, 
                        np.where(fase_lua_plantio == 'Cheia', 1200, 0))
    lua_bonus_colheita = np.where(fase_lua_colheita == 'Cheia', 2000,
                         np.where(fase_lua_colheita == 'Minguante', -800, 0))
    
    vendas_safra = vendas_base + lua_bonus_plantio + lua_bonus_colheita
    vendas_safra = np.maximum(vendas_safra, 5000)  # Minimum viable safra
    
    # === CRIAR DATAFRAME ===
    data = pd.DataFrame({
        # Marketing Digital
        'visitas_site_ciclo': visitas_site.astype(int),
        'prod_gourmet_vendas': prod_gourmet_vendas.astype(int),
        'ads_tiktok_invest': ads_tiktok.round(0).astype(int),
        'ads_google_invest': ads_google.round(0).astype(int),
        
        # Exportação
        'contratos_export': contratos_export,
        
        # Clima
        'chuva_mm_safra': chuva_mm_safra.round(1),
        'temp_media_safra': temp_media_safra.round(1),
        'dias_sol_safra': dias_sol_safra.round(0).astype(int),
        
        # Lunar
        'fase_lua_plantio': fase_lua_plantio,
        'fase_lua_colheita': fase_lua_colheita,
        
        # Econômico
        'preco_feijao_saca': preco_feijao_saca.round(2),
        'custo_fertilizante_ha': custo_fertilizante.round(0).astype(int),
        
        # Operacional
        'hectares_plantados': hectares_plantados.round(1),
        'funcionarios_safra': funcionarios_safra,
        
        # TARGET
        'vendas_safra_reais': vendas_safra.round(0).astype(int)
    })
    
    return data

def preprocess_data(data):
    """Preprocessa os dados"""
    data_processed = data.copy()
    
    # Tratar coluna de data se existir
    if 'data' in data_processed.columns:
        data_processed['dia_semana'] = pd.to_datetime(data_processed['data']).dt.dayofweek
        data_processed = data_processed.drop('data', axis=1)

    # Codificar variáveis categóricas
    categorical_cols = data_processed.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        data_processed[col] = le.fit_transform(data_processed[col])

    # Identificar target - agora procura por 'vendas_safra_reais' primeiro
    possible_targets = ['vendas_safra_reais', 'vendas', 'conversoes', 'sales', 'target', 'y']
    target_col = None
    
    for target in possible_targets:
        if target in data_processed.columns:
            target_col = target
            break
    
    if target_col is None:
        # Use a última coluna numérica como target
        numeric_cols = data_processed.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            target_col = numeric_cols[-1]
        else:
            st.error("❌ Nenhuma coluna target válida encontrada!")
            return None, None, None
    
    features = [col for col in data_processed.columns if col != target_col]
    
    return data_processed, features, target_col

def run_mle_analysis(data, features, target):
    """Executa análise MLE"""
    X = data[features]
    y = data[target]

    # Normalizar os dados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_scaled, y)

    # Criar dataframe de resultados
    results = pd.DataFrame({
        'Feature': features,
        'Coeficiente': model.coef_,
        'Magnitude_Absoluta': np.abs(model.coef_)
    }).sort_values('Magnitude_Absoluta', ascending=False)

    # Métricas
    y_pred = model.predict(X_scaled)
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)

    return results, r2, mse, scaler, model

def run_map_analysis(data, features, target, scaler):
    """Executa análise MAP (Ridge)"""
    X = data[features]
    y = data[target]
    
    X_scaled = scaler.transform(X)

    model = Ridge(alpha=1.0)
    model.fit(X_scaled, y)

    results = pd.DataFrame({
        'Feature': features,
        'Coeficiente': model.coef_,
        'Magnitude_Absoluta': np.abs(model.coef_)
    }).sort_values('Magnitude_Absoluta', ascending=False)

    # Métricas
    y_pred = model.predict(X_scaled)
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)

    return results, r2, mse, model

def run_bayesian_analysis(data, features, target, scaler, draws=1000, tune=1000):
    """Executa análise Bayesiana"""
    pm, az, success = load_bayesian_libs()
    if not success:
        return None

    X = data[features]
    y = data[target]
    X_scaled = scaler.transform(X)

    with pm.Model() as model:
        # Priors mais informativos
        coefficients = pm.Normal('coefficients', mu=0, sigma=2, shape=len(features))
        intercept = pm.Normal('intercept', mu=y.mean(), sigma=y.std())
        sigma = pm.HalfNormal('sigma', sigma=y.std())

        # Likelihood
        mu = intercept + pm.math.dot(X_scaled, coefficients)
        likelihood = pm.Normal('y', mu=mu, sigma=sigma, observed=y)

        # Sampling com configurações seguras
        try:
            trace = pm.sample(
                draws=draws, 
                tune=tune, 
                return_inferencedata=True,
                cores=1,  # Single core para evitar problemas
                chains=2,  # Menos chains
                target_accept=0.9,
                progressbar=False  # Desabilitar progress do PyMC
            )
        except Exception as e:
            st.error(f"❌ Erro durante MCMC sampling: {e}")
            return None

    # Resumo das distribuições posteriores
    summary = az.summary(trace, var_names=['coefficients'])
    summary['Feature'] = features
    summary['|Mean|'] = np.abs(summary['mean'])

    results = summary[['Feature', 'mean', 'sd', 'hdi_3%', 'hdi_97%', '|Mean|']].sort_values('|Mean|', ascending=False)

    return results, trace

def compute_entropy(residuals, bins=30):
    """Calcula entropia aproximada de uma distribuição a partir dos resíduos"""
    hist, bin_edges = np.histogram(residuals, bins=bins, density=True)
    hist = hist[hist > 0]  # remover valores nulos
    entropy = -np.sum(hist * np.log(hist))
    return entropy

def show_entropy_section(y_true, y_pred_mle, y_pred_map, y_pred_bayes=None):
    """Mostra análise de entropia das distribuições de erro"""
    st.header("5️⃣ Análise de Entropia (Incerteza na Fazenda)")
    st.markdown("""
    Entropia mede a **incerteza de uma distribuição** - crucial para o agronegócio!  
    
    **🌾 Para a Roça do João:**
    - **Baixa entropia** → Previsões confiáveis → João pode planejar com segurança
    - **Alta entropia** → Muita incerteza → João precisa de mais dados ou fatores externos
    
    Fórmula da entropia diferencial:
    $$
    H(p) = -\int p(z) \\ln p(z) \\, dz
    $$
    """)

    residuals_mle = y_true - y_pred_mle
    residuals_map = y_true - y_pred_map
    entropy_mle = compute_entropy(residuals_mle)
    entropy_map = compute_entropy(residuals_map)

    entropies = {
        "MLE (Linear)": entropy_mle,
        "MAP (Ridge)": entropy_map
    }

    if y_pred_bayes is not None:
        residuals_bayes = y_true - y_pred_bayes
        entropy_bayes = compute_entropy(residuals_bayes)
        entropies["Bayesiano"] = entropy_bayes

    # Mostrar resultados
    st.subheader("📊 Entropia dos Modelos de Predição")
    entropy_df = pd.DataFrame(entropies.items(), columns=["Modelo", "Entropia"])
    st.dataframe(entropy_df)

    # Interpretação para o João
    best_model = entropy_df.loc[entropy_df['Entropia'].idxmin(), 'Modelo']
    st.success(f"🏆 **Melhor modelo para João:** {best_model} (menor incerteza)")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(entropies.keys(), entropies.values(), color="green", alpha=0.7)
    ax.set_ylabel("Entropia (Incerteza)")
    ax.set_title("🌾 Incerteza dos Modelos - Roça do João")
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

def create_comparison_plot(mle_results, map_results, bayesian_results=None):
    """Cria gráfico de comparação específico para agronegócio"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    features = mle_results['Feature'].values
    x_pos = np.arange(len(features))
    
    if bayesian_results is not None:
        width = 0.25
        ax.bar(x_pos - width, mle_results['Coeficiente'], width, label='MLE', alpha=0.8, color='lightblue')
        ax.bar(x_pos, map_results['Coeficiente'], width, label='MAP (Ridge)', alpha=0.8, color='lightgreen')
        ax.bar(x_pos + width, bayesian_results['mean'], width, label='Bayesian', alpha=0.8, color='orange')
    else:
        width = 0.35
        ax.bar(x_pos - width/2, mle_results['Coeficiente'], width, label='MLE', alpha=0.8, color='lightblue')
        ax.bar(x_pos + width/2, map_results['Coeficiente'], width, label='MAP (Ridge)', alpha=0.8, color='lightgreen')

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
    """Cria gráfico de incerteza Bayesiana para agronegócio"""
    if bayesian_results is None:
        return None
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    features = bayesian_results['Feature'].values
    means = bayesian_results['mean'].values
    lower = bayesian_results['hdi_3%'].values
    upper = bayesian_results['hdi_97%'].values
    
    y_pos = np.arange(len(features))

    # Plot error bars
    ax.errorbar(means, y_pos, xerr=[means - lower, upper - means], 
                fmt='o', capsize=5, capthick=2, markersize=8, color='darkgreen')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.set_xlabel('Impacto nas Vendas (R$)')
    ax.set_title('🌾 Roça do João - Incerteza dos Fatores (Intervalo de Credibilidade 94%)')
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Sem Impacto')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    return fig

def main():
    # Mostrar cabeçalho personalizado
    show_header()
    
    st.markdown("""
    ### 🔑 KPIs como bússolas, dados como chaves escondidas.
    Descubra o **ouro verde** que já existe por trás dos números da fazenda.

    Fazendas modernas precisam de **modelos sob medida**:  
    não apenas relatórios de safra, mas **ferramentas inteligentes** que revelam padrões climáticos, 
    otimizam investimentos e reduzem riscos na agricultura.  
    """)
    
    st.markdown("---")

    # Sidebar
    st.sidebar.header("🎛️ Configurações da Análise")
    
    # Opção de dados
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
            help="O arquivo deve conter dados de safras com uma coluna target (vendas, receita, etc.)"
        )
        if uploaded_file:
            try:
                data = pd.read_csv(uploaded_file)
                st.sidebar.success("✅ Arquivo carregado!")
            except Exception as e:
                st.sidebar.error(f"❌ Erro: {e}")
                return
        else:
            st.info("📂 Faça upload de um arquivo CSV para analisar seus próprios dados agrícolas")
            return
    else:
        data = generate_sample_data()

    # Mostrar informações dos dados
    st.header("📊 Dataset da Fazenda Carregado")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📈 Safras Analisadas", len(data))
    with col2:
        st.metric("📋 Variáveis", data.shape[1])
    with col3:
        receita_media = data['vendas_safra_reais'].mean() if 'vendas_safra_reais' in data.columns else 0
        st.metric("💰 Receita Média", f"R$ {receita_media:,.0f}")
    with col4:
        missing = data.isnull().sum().sum()
        st.metric("❓ Dados Faltantes", missing)

    # Preview dos dados
    with st.expander("👀 Visualizar Dados da Fazenda", expanded=False):
        tab1, tab2, tab3 = st.tabs(["📋 Dados Brutos", "📊 Estatísticas", "🌾 Insights Rápidos"])
        
        with tab1:
            st.dataframe(data, use_container_width=True)
        
        with tab2:
            st.dataframe(data.describe(), use_container_width=True)
            
        with tab3:
            if 'vendas_safra_reais' in data.columns:
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**🏆 Melhor Safra:**")
                    melhor_safra = data.loc[data['vendas_safra_reais'].idxmax()]
                    st.write(f"Receita: R$ {melhor_safra['vendas_safra_reais']:,.0f}")
                    
                with col2:
                    st.write("**📉 Pior Safra:**")
                    pior_safra = data.loc[data['vendas_safra_reais'].idxmin()]
                    st.write(f"Receita: R$ {pior_safra['vendas_safra_reais']:,.0f}")

    # Configurações da análise
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Configurações da Análise")
    
    include_bayesian = st.sidebar.checkbox("🧠 Incluir Análise Bayesiana", value=True)
    
    if include_bayesian:
        st.sidebar.markdown("**Parâmetros MCMC:**")
        draws = st.sidebar.slider("🎲 Draws:", 500, 2000, 1000, 100)
        tune = st.sidebar.slider("🔧 Tune:", 500, 2000, 1000, 100)
        st.sidebar.info(f"Total de amostras: {draws * 2} (2 chains)")

    # BOTÃO PRINCIPAL
    st.markdown("---")
    
    if st.button("🚀 **INICIAR ANÁLISE COMPLETA DA FAZENDA**", type="primary"):
        
        # Container para toda a análise
        analysis_container = st.container()
        
        with analysis_container:
            
            # Barra de progresso
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # 1. Preprocessamento
                status_text.text("🔄 Preparando dados da fazenda...")
                progress_bar.progress(10)
                
                processed_data, features, target = preprocess_data(data)
                if processed_data is None:
                    return
                
                st.success(f"✅ **Dados processados!** Target: `{target}` | Features: `{len(features)}`")
                progress_bar.progress(20)
                
                # 2. MLE Analysis
                status_text.text("🔍 Executando análise MLE (Linear)...")
                progress_bar.progress(35)
                
                mle_results, mle_r2, mle_mse, scaler, mle_model = run_mle_analysis(processed_data, features, target)
                
                st.header("1️⃣ Análise Linear (MLE) - Relações Diretas")
                st.markdown("*Mostra o impacto linear de cada fator nas vendas da safra*")
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.dataframe(mle_results, use_container_width=True)
                with col2:
                    st.metric("📊 R² Score", f"{mle_r2:.3f}")
                    st.metric("📏 MSE", f"{mle_mse:,.0f}")
                    
                    # Interpretação para João
                    if mle_r2 > 0.8:
                        st.success("🎯 Excelente ajuste!")
                    elif mle_r2 > 0.6:
                        st.warning("⚠️ Ajuste moderado")
                    else:
                        st.error("❌ Ajuste ruim")
                
                progress_bar.progress(50)
                
                # 3. MAP Analysis
                status_text.text("🎯 Executando análise MAP (Ridge)...")
                progress_bar.progress(60)
                
                map_results, map_r2, map_mse, map_model = run_map_analysis(processed_data, features, target, scaler)
                
                st.header("2️⃣ Análise Regularizada (MAP) - Controle de Overfitting")
                st.markdown("*Modelo mais conservador, evita superajuste aos dados históricos*")
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.dataframe(map_results, use_container_width=True)
                with col2:
                    st.metric("📊 R² Score", f"{map_r2:.3f}")
                    st.metric("📏 MSE", f"{map_mse:,.0f}")
                
                progress_bar.progress(75)
                
                # 4. Bayesian Analysis (se habilitado)
                bayesian_results = None
                if include_bayesian:
                    status_text.text("🧠 Executando análise Bayesiana (pode demorar)...")
                    progress_bar.progress(80)
                    
                    with st.spinner("🔄 Executando MCMC Sampling..."):
                        bayesian_output = run_bayesian_analysis(processed_data, features, target, scaler, draws, tune)
                    
                    if bayesian_output:
                        bayesian_results, trace = bayesian_output
                        
                        st.header("3️⃣ Análise Bayesiana - Incerteza Quantificada")
                        st.markdown("*Mostra não apenas o impacto, mas também a confiança em cada estimativa*")
                        
                        st.dataframe(bayesian_results, use_container_width=True)
                        
                        # Relatório Bayesiano específico para João
                        st.subheader("📋 Insights para a Roça do João")
                        
                        # Features mais importantes
                        st.write("**🎯 Fatores Mais Importantes para as Vendas:**")
                        top_features = bayesian_results.nlargest(3, '|Mean|')
                        for idx, (_, row) in enumerate(top_features.iterrows(), 1):
                            uncertainty_level = "📍 Baixa" if row['sd'] < row['mean']/3 else "⚠️ Alta"
                            impacto = "positivo 📈" if row['mean'] > 0 else "negativo 📉"
                            st.write(f"**{idx}. {row['Feature']}**: Impacto {impacto} de R$ {row['mean']:,.0f} (±{row['sd']:,.0f}) - Incerteza: {uncertainty_level}")
                        
                        # Features com alta incerteza
                        high_uncertainty = bayesian_results[bayesian_results['sd'] > bayesian_results['mean'].abs()/2]
                        if len(high_uncertainty) > 0:
                            st.write("**⚠️ Fatores que Precisam de Mais Dados:**")
                            for _, row in high_uncertainty.iterrows():
                                if row['hdi_3%'] < 0 < row['hdi_97%']:
                                    rec = "❓ Efeito indeterminado - coletar mais dados"
                                elif row['hdi_3%'] > 0:
                                    rec = "📈 Provavelmente positivo"
                                else:
                                    rec = "📉 Provavelmente negativo"
                                st.write(f"• **{row['Feature']}**: {rec} (HDI: [R$ {row['hdi_3%']:,.0f}, R$ {row['hdi_97%']:,.0f}])")
                
                progress_bar.progress(90)
                
                # 5. Visualizações
                status_text.text("📊 Criando visualizações...")
                
                st.header("4️⃣ Visualizações Comparativas")
                
                # Gráfico de comparação
                fig1 = create_comparison_plot(mle_results, map_results, bayesian_results)
                st.pyplot(fig1)
                
                # Gráfico de incerteza (se Bayesian disponível)
                if bayesian_results is not None:
                    fig2 = create_uncertainty_plot(bayesian_results)
                    if fig2:
                        st.pyplot(fig2)

                # 6. Entropia
                status_text.text("📊 Calculando entropia dos modelos...")
                progress_bar.progress(95)

                # Predições dos modelos para análise de entropia
                X_scaled = scaler.transform(processed_data[features])
                y_true = processed_data[target]
                y_pred_mle = mle_model.predict(X_scaled)
                y_pred_map = map_model.predict(X_scaled)

                # Análise de entropia
                show_entropy_section(
                    y_true=y_true,
                    y_pred_mle=y_pred_mle,
                    y_pred_map=y_pred_map,
                    y_pred_bayes=None
                )
                
                # 7. Recomendações para João
                st.header("6️⃣ Recomendações Estratégicas para João")
                
                # Análise dos resultados mais importantes
                if bayesian_results is not None:
                    top_factor = bayesian_results.iloc[0]
                    st.success(f"🌟 **Fator #1 de Impacto:** {top_factor['Feature']}")
                    st.write(f"Cada unidade de melhoria pode gerar R$ {top_factor['mean']:,.0f} a mais por safra")
                    
                    # Recomendações específicas baseadas nos fatores
                    recommendations = []
                    for _, row in bayesian_results.head(3).iterrows():
                        factor = row['Feature']
                        impact = row['mean']
                        
                        if 'clima' in factor.lower() or 'chuva' in factor.lower():
                            recommendations.append(f"🌦️ **{factor}**: Investir em irrigação ou sistema de captação de água")
                        elif 'tiktok' in factor.lower():
                            recommendations.append(f"📱 **{factor}**: Expandir estratégia de marketing digital da neta")
                        elif 'export' in factor.lower():
                            recommendations.append(f"🌍 **{factor}**: Focar em conseguir mais contratos de exportação")
                        elif 'preco' in factor.lower():
                            recommendations.append(f"💰 **{factor}**: Monitorar preços e timing de venda")
                        elif 'hectares' in factor.lower():
                            recommendations.append(f"🌾 **{factor}**: Considerar expansão da área plantada")
                    
                    st.write("**🎯 Ações Prioritárias:**")
                    for rec in recommendations:
                        st.write(f"• {rec}")
                
                # Finalizar
                progress_bar.progress(100)
                status_text.text("✅ Análise concluída!")
                
                # Mensagem de sucesso
                st.balloons()
                st.success("**🌾 Análise completa da Roça do João! Todos os modelos foram executados com sucesso.**")
                st.success("✨ Imagine o que podemos fazer com **os dados da sua fazenda**.")
                
                # Mensagem de contato específica para agronegócio
                st.markdown("""
                💡 **Do plantio à colheita, transformamos dados em decisões inteligentes.**  
                📈 **Da tradição à inovação, otimizamos cada safra com ciência de dados.**  

                👉 **Entre em contato e vamos descobrir juntos os segredos da sua terra**  
                📧 [contato@plexonatural.com](mailto:contato@plexonatural.com)
                🌾 **Especialistas em IA para Agronegócio**
                """)
                
                # Download
                if bayesian_results is not None:
                    csv_data = bayesian_results.to_csv(index=False)
                    st.download_button(
                        label="📥 Baixar Análise Completa da Fazenda",
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

    # Rodapé informativo para agronegócio
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
    <b>🌾 Roça do João - Inteligência Artificial para o Agronegócio</b><br>
    <b>Dica:</b> Para melhores resultados, certifique-se de ter dados de pelo menos 20 safras<br>
    <b>Algoritmos:</b> Scikit-learn (MLE/MAP) + PyMC (Bayesian MCMC) + Análise de Entropia
    </div>
    """, unsafe_allow_html=True)

# IMPORTANTE: Só executar main() se for o arquivo principal
if __name__ == "__main__":
    main()
