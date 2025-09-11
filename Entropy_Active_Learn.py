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
    page_title="Entropy Business Intelligence",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔬 Análise Bayesiana Inteligente")
st.markdown("""
### Explore decisões de investimento com ciência de dados
Este aplicativo mostra um modelo de análise Bayesiana, mas lembre-se: cada negócio é único e requer soluções personalizadas.  
Aqui vamos comparar o impacto e a incerteza de investir em **TikTok Ads** e **Snapchat Ads**, considerando até fatores inusitados, como as fases da lua 🌙.  
O objetivo é ajudá-lo a entender como decisões podem variar de acordo com diferentes cenários — insights que podem não ser óbvios para quem não conhece a área.
""")


st.title("A Alquimia Aplicada ao Seu Negócio")
st.markdown("""
### KPIs como bússolas, dados como chaves escondidas.
Descubra o **ouro** que já existe por trás dos seus números.

Empresas modernas precisam de **modelos sob medida**:  
não apenas relatórios, mas **ferramentas inteligentes** que revelam padrões, reduzem incertezas e criam novas oportunidades.  
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
    """Gera dados de exemplo mais realistas"""
    np.random.seed(42)
    n_samples = 100
    
    # Features de entrada
    visitantes_tiktok = np.random.poisson(13000, n_samples)
    visitantes_snapchat = np.random.poisson(9000, n_samples)
    ads_tiktok = np.random.exponential(2800, n_samples)
    ads_snapchat = np.random.exponential(2000, n_samples)
    feedback_feeling = np.random.normal(7.5, 1.0, n_samples)
    fase_lunar = np.random.choice(['Nova', 'Crescente', 'Cheia', 'Minguante'], n_samples)
    dia_semana = np.random.randint(0, 7, n_samples)
    
    # Target com relação realista
    vendas = (
        0.01 * visitantes_tiktok +
        0.015 * visitantes_snapchat +
        0.1 * ads_tiktok +
        0.12 * ads_snapchat +
        50 * feedback_feeling +
        np.random.normal(0, 50, n_samples)
    )
    vendas = np.maximum(vendas, 0)  # Garantir valores positivos
    
    data = pd.DataFrame({
        'visitantes_tiktok': visitantes_tiktok,
        'visitantes_snapchat': visitantes_snapchat,
        'ads_tiktok': ads_tiktok,
        'ads_snapchat': ads_snapchat,
        'feedback_feeling': feedback_feeling,
        'fase_lunar': fase_lunar,
        'dia_semana': dia_semana,
        'vendas': vendas
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

    # Identificar target
    possible_targets = ['vendas', 'conversoes', 'sales', 'target', 'y']
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
    st.header("5️⃣ Análise de Entropia (Incerteza)")
    st.markdown("""
    Entropia mede a **incerteza de uma distribuição**.  
    Distribuições mais espalhadas (maior variância) → maior entropia.  
    Distribuições concentradas → menor entropia.  
    
    Fórmula da entropia diferencial:
    $$
    H(p) = -\int p(z) \\ln p(z) \\, dz
    $$
    
    Para uma Gaussiana multivariada:
    $$
    H(\\mathcal{N}(\\mu, \\Sigma)) = \\tfrac{d}{2} \\ln(2 \\pi e |\\Sigma|)
    $$
    """)

    residuals_mle = y_true - y_pred_mle
    residuals_map = y_true - y_pred_map
    entropy_mle = compute_entropy(residuals_mle)
    entropy_map = compute_entropy(residuals_map)

    entropies = {
        "MLE": entropy_mle,
        "MAP": entropy_map
    }

    if y_pred_bayes is not None:
        residuals_bayes = y_true - y_pred_bayes
        entropy_bayes = compute_entropy(residuals_bayes)
        entropies["Bayesiano"] = entropy_bayes

    # Mostrar resultados
    st.subheader("📊 Entropia das Distribuições de Resíduos")
    entropy_df = pd.DataFrame(entropies.items(), columns=["Modelo", "Entropia"])
    st.dataframe(entropy_df)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(entropies.keys(), entropies.values(), color="skyblue", alpha=0.8)
    ax.set_ylabel("Entropia (Incerteza)")
    ax.set_title("Comparação de Entropia entre Modelos")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

def create_comparison_plot(mle_results, map_results, bayesian_results=None):
    """Cria gráfico de comparação"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    features = mle_results['Feature'].values
    x_pos = np.arange(len(features))
    
    if bayesian_results is not None:
        width = 0.25
        ax.bar(x_pos - width, mle_results['Coeficiente'], width, label='MLE', alpha=0.8, color='skyblue')
        ax.bar(x_pos, map_results['Coeficiente'], width, label='MAP (Ridge)', alpha=0.8, color='lightgreen')
        ax.bar(x_pos + width, bayesian_results['mean'], width, label='Bayesian', alpha=0.8, color='salmon')
    else:
        width = 0.35
        ax.bar(x_pos - width/2, mle_results['Coeficiente'], width, label='MLE', alpha=0.8, color='skyblue')
        ax.bar(x_pos + width/2, map_results['Coeficiente'], width, label='MAP (Ridge)', alpha=0.8, color='lightgreen')

    ax.set_xlabel('Features')
    ax.set_ylabel('Valor do Coeficiente')
    ax.set_title('Comparação dos Métodos de Estimação')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(features, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_uncertainty_plot(bayesian_results):
    """Cria gráfico de incerteza Bayesiana"""
    if bayesian_results is None:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    features = bayesian_results['Feature'].values
    means = bayesian_results['mean'].values
    lower = bayesian_results['hdi_3%'].values
    upper = bayesian_results['hdi_97%'].values
    
    y_pos = np.arange(len(features))

    # Plot error bars
    ax.errorbar(means, y_pos, xerr=[means - lower, upper - means], 
                fmt='o', capsize=5, capthick=2, markersize=8, color='darkblue')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.set_xlabel('Valor do Coeficiente')
    ax.set_title('Distribuições Posteriores com Intervalos de Credibilidade (94% HDI)')
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Zero')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    return fig

def main():
    # Título e descrição
    st.title("🔬 Análise Bayesiana Completa")
    st.markdown("""
    ### Esse aplicativo modelo é um exemplo, cada negócio carece de soluções modeladas para cada fim específico, normalmente, melhoria nos numeros de conversão, custos e produção.
    
    Aqui vamos comparar a importância e incertezas associadas aos investimentos em Tik Tok e Snapchat Ads, e os efeitos das fases da lua (pense que seu negócio pode ter características muito distintas de outros, que pessoas que não são da área podem não entender, mesmo sua empresa.)
    """)
    
    st.markdown("---")

    # Sidebar
    st.sidebar.header("🎛️ Configurações")
    
    # Opção de dados
    data_source = st.sidebar.selectbox(
        "📊 Escolha a fonte dos dados:",
        ["Dados de Exemplo", "Upload de Arquivo"]
    )
    
    # Carregar dados
    data = None
    if data_source == "Upload de Arquivo":
        uploaded_file = st.sidebar.file_uploader(
            "📁 Faça upload do seu CSV", 
            type=['csv'],
            help="O arquivo deve conter uma coluna target (vendas, conversoes, etc.)"
        )
        if uploaded_file:
            try:
                data = pd.read_csv(uploaded_file)
                st.sidebar.success("✅ Arquivo carregado!")
            except Exception as e:
                st.sidebar.error(f"❌ Erro: {e}")
                return
        else:
            st.info("📂 Faça upload de um arquivo CSV para começar")
            return
    else:
        data = generate_sample_data()

    # Mostrar informações dos dados
    st.header("📊 Dados Carregados")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📈 Amostras", len(data))
    with col2:
        st.metric("📋 Colunas", data.shape[1])
    with col3:
        st.metric("💾 Memória", f"{data.memory_usage(deep=True).sum()/1024:.0f} KB")
    with col4:
        missing = data.isnull().sum().sum()
        st.metric("❓ Valores Faltantes", missing)

    # Preview dos dados
    with st.expander("👀 Visualizar Dados", expanded=False):
        tab1, tab2 = st.tabs(["📋 Dados", "📊 Estatísticas"])
        
        with tab1:
            st.dataframe(data)
        
        with tab2:
            st.dataframe(data.describe())

    # Configurações da análise
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Configurações da Análise")
    
    include_bayesian = st.sidebar.checkbox("🧠 Incluir Análise Bayesiana", value=True)
    
    if include_bayesian:
        st.sidebar.markdown("**Parâmetros MCMC:**")
        draws = st.sidebar.slider("🎲 Draws:", 500, 2000, 1000, 100)
        tune = st.sidebar.slider("🔧 Tune:", 500, 2000, 1000, 100)
        st.sidebar.info(f"Total de amostras: {draws * 2} (2 chains)")

    # BOTÃO PRINCIPAL - SÓ AQUI QUE EXECUTA ALGO!
    st.markdown("---")
    
    if st.button("🚀 **INICIAR ANÁLISE COMPLETA**", type="primary"):
        
        # Container para toda a análise
        analysis_container = st.container()
        
        with analysis_container:
            
            # Barra de progresso
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # 1. Preprocessamento
                status_text.text("🔄 Preparando dados...")
                progress_bar.progress(10)
                
                processed_data, features, target = preprocess_data(data)
                if processed_data is None:
                    return
                
                st.success(f"✅ **Dados processados!** Target: `{target}` | Features: `{len(features)}`")
                progress_bar.progress(20)
                
                # 2. MLE Analysis
                status_text.text("🔍 Executando análise MLE...")
                progress_bar.progress(35)
                
                mle_results, mle_r2, mle_mse, scaler, mle_model = run_mle_analysis(processed_data, features, target)
                
                st.header("1️⃣ Análise MLE (Maximum Likelihood)")
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.dataframe(mle_results)
                with col2:
                    st.metric("R² Score", f"{mle_r2:.3f}")
                    st.metric("MSE", f"{mle_mse:.1f}")
                
                progress_bar.progress(50)
                
                # 3. MAP Analysis
                status_text.text("🎯 Executando análise MAP...")
                progress_bar.progress(60)
                
                map_results, map_r2, map_mse, map_model = run_map_analysis(processed_data, features, target, scaler)
                
                st.header("2️⃣ Análise MAP (Ridge Regression)")
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.dataframe(map_results)
                with col2:
                    st.metric("R² Score", f"{map_r2:.3f}")
                    st.metric("MSE", f"{map_mse:.1f}")
                
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
                        
                        st.header("3️⃣ Análise Bayesiana Completa")
                        st.dataframe(bayesian_results)
                        
                        # Relatório Bayesiano
                        st.subheader("📋 Insights Bayesianos")
                        
                        # Features mais importantes
                        st.write("**🎯 Features Mais Relevantes:**")
                        top_features = bayesian_results.nlargest(3, '|Mean|')
                        for _, row in top_features.iterrows():
                            uncertainty_level = "📍 Baixa" if row['sd'] < 1.0 else "⚠️ Alta"
                            st.write(f"• **{row['Feature']}**: {row['mean']:.3f} (±{row['sd']:.3f}) - Incerteza: {uncertainty_level}")
                        
                        # Features com alta incerteza
                        high_uncertainty = bayesian_results[bayesian_results['sd'] > 1.0]
                        if len(high_uncertainty) > 0:
                            st.write("**⚠️ Features que Precisam de Mais Dados:**")
                            for _, row in high_uncertainty.iterrows():
                                if row['hdi_3%'] < 0 < row['hdi_97%']:
                                    rec = "❓ Efeito indeterminado"
                                elif row['hdi_3%'] > 0:
                                    rec = "📈 Provavelmente positivo"
                                else:
                                    rec = "📉 Provavelmente negativo"
                                st.write(f"• **{row['Feature']}**: {rec} (HDI: [{row['hdi_3%']:.2f}, {row['hdi_97%']:.2f}])")
                
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

                # Por enquanto, não temos predições diretas do Bayesiano
                show_entropy_section(
                    y_true=y_true,
                    y_pred_mle=y_pred_mle,
                    y_pred_map=y_pred_map,
                    y_pred_bayes=None
                )
                
                # Finalizar
                progress_bar.progress(100)
                status_text.text("✅ Análise concluída!")
                
                # Mensagem de sucesso
                st.balloons()
                st.success("**Análise completa! Todos os modelos foram executados com sucesso.**")
                st.success("✨ Imagine o que podemos fazer com **os dados da sua empresa**.")
st.markdown("""
💡 De KPI em KPI, desenhamos o **modelo que você precisa**.  
📈 Do campo ao mercado digital, transformamos incerteza em clareza.  

👉 Entre em contato e vamos descobrir juntos as **chaves escondidas do seu negócio** contato@plexonatural.com.
""")
except Exception as e:
    st.error(f"Ocorreu um erro na apresentação dos resultados: {e}")

                
                # Download
                if bayesian_results is not None:
                    csv_data = bayesian_results.to_csv(index=False)
                    st.download_button(
                        label="📥 Baixar Resultados Bayesianos",
                        data=csv_data,
                        file_name=f"analise_bayesiana_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )
                
            except Exception as e:
                progress_bar.progress(0)
                status_text.text("❌ Erro na análise")
                st.error(f"**Erro durante a análise:** {str(e)}")
                
                with st.expander("🔍 Detalhes do Erro"):
                    import traceback
                    st.code(traceback.format_exc())

    # Rodapé informativo
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
    💡 <b>Dica:</b> Para melhores resultados, certifique-se de que seus dados tenham pelo menos 50 amostras<br>
    🔬 <b>Algoritmos:</b> Scikit-learn (MLE/MAP) + PyMC (Bayesian MCMC)
    </div>
    """, unsafe_allow_html=True)

# IMPORTANTE: Só executar main() se for o arquivo principal
if __name__ == "__main__":
    main()
