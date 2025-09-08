import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pymc as pm # Changed import from pymc3 to pymc
import arviz as az

# Configure matplotlib to use a font that supports a wider range of characters
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Segoe UI Emoji', 'Noto Color Emoji']


class BayesianSalesAnalyser:
    def __init__(self):
        self.data = None
        self.features = None
        self.target = None
        self.model = None
        self.trace = None
        self.summary = None

    def load_data(self, data):
        """Carrega e prepara os dados para análise"""
        self.data = data.copy()

        # Pré-processamento básico
        if 'data' in self.data.columns:
            self.data['dia_semana'] = pd.to_datetime(self.data['data']).dt.dayofweek
            self.data = self.data.drop('data', axis=1)

        # Codificar variáveis categóricas
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            self.data[col] = le.fit_transform(self.data[col])

        # Separar features e target
        self.target = 'vendas'
        self.features = [col for col in self.data.columns if col != self.target]

        return self.data

    def analyze_dataset(self):
        """Executa a análise completa do dataset"""
        print("INICIANDO ANÁLISE BAYESIANA COMPLETA") # Removed emoji
        print("=" * 50)

        # 1. Modelo Simples (MLE)
        print("\n1. MODELO DE MÍNIMOS QUADRADOS (MLE)") # Removed emoji
        X = self.data[self.features]
        y = self.data[self.target]

        # Normalizar os dados
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model_mle = LinearRegression()
        model_mle.fit(X_scaled, y)

        mle_coefs = pd.DataFrame({
            'Feature': self.features,
            'Coeficiente_MLE': model_mle.coef_,
            'Magnitude_Absoluta': np.abs(model_mle.coef_)
        }).sort_values('Magnitude_Absoluta', ascending=False)

        print("Coeficientes do modelo MLE (ordenados por importância):")
        print(mle_coefs.to_string(index=False))

        # 2. Modelo Regularizado (MAP - Ridge)
        print("\n2. MODELO REGULARIZADO (MAP - RIDGE)") # Removed emoji
        model_map = Ridge(alpha=1.0)
        model_map.fit(X_scaled, y)

        map_coefs = pd.DataFrame({
            'Feature': self.features,
            'Coeficiente_MAP': model_map.coef_,
            'Magnitude_Absoluta': np.abs(model_map.coef_)
        }).sort_values('Magnitude_Absoluta', ascending=False)

        print("Coeficientes do modelo MAP (Ridge Regression):")
        print(map_coefs.to_string(index=False))

        # 3. Inferência Bayesiana Completa
        print("\n3. INFERÊNCIA BAYESIANA COMPLETA") # Removed emoji
        print("Calculando distribuições posteriores...")

        with pm.Model() as self.model:
            # Priors para os coeficientes
            coefficients = pm.Normal('coefficients', mu=0, sigma=1, shape=len(self.features))

            # Prior para o intercepto
            intercept = pm.Normal('intercept', mu=y.mean(), sigma=1)

            # Prior para o erro
            sigma = pm.HalfNormal('sigma', sigma=1)

            # Likelihood
            mu = intercept + pm.math.dot(X_scaled, coefficients)
            likelihood = pm.Normal('y', mu=mu, sigma=sigma, observed=y)

            # Amostragem da posterior
            self.trace = pm.sample(1000, tune=1000, return_inferencedata=True)

        # Resumo das distribuições posteriores
        self.summary = az.summary(self.trace, var_names=['coefficients'])
        self.summary['Feature'] = self.features
        self.summary['|Mean|'] = np.abs(self.summary['mean'])

        print("\nDistribuições posteriores dos coeficientes:")
        bayesian_results = self.summary[['Feature', 'mean', 'sd', 'hdi_3%', 'hdi_97%', '|Mean|']].sort_values('|Mean|', ascending=False)
        print(bayesian_results.to_string(index=False))

        return mle_coefs, map_coefs, bayesian_results

    def generate_report(self):
        """Gera relatório com insights e recomendações"""
        print("\n4. RELATÓRIO FINAL E RECOMENDAÇÕES") # Removed emoji
        print("=" * 50)

        # Features mais importantes
        top_features = self.summary.nlargest(3, '|Mean|')
        print("\nFEATURES MAIS RELEVANTES (alto impacto, baixa incerteza):") # Removed emoji
        for _, row in top_features.iterrows():
            if row['sd'] < 0.5:  # Baixa incerteza
                print(f"  - {row['Feature']}: {row['mean']:.3f} (±{row['sd']:.3f})")

        # Features com alta incerteza
        print("\nFEATURES COM ALTA INCERTEZA (precisam de 'zoom'):") # Removed emoji
        high_uncertainty = self.summary[self.summary['sd'] > 1.0]

        if len(high_uncertainty) > 0:
            for _, row in high_uncertainty.iterrows():
                print(f"  - {row['Feature']}: {row['mean']:.3f} (±{row['sd']:.3f})")
                print(f"    Intervalo de 94% de confiança: [{row['hdi_3%']:.2f}, {row['hdi_97%']:.2f}]")

                # Recomendações based on the confidence interval
                if row['hdi_3%'] < 0 and row['hdi_97%'] > 0:
                    print("    RECOMENDAÇÃO: Efeito indeterminado. Necessita mais dados para definir se é positivo ou negativo.") # Removed emoji
                elif row['hdi_3%'] > 0:
                    print("    RECOMENDAÇÃO: Efeito provavelmente positivo. Vale investir em mais experimentos.") # Removed emoji
                else:
                    print("    RECOMENDAÇÃO: Efeito provavelmente negativo. Considerar reduzir investimento.") # Removed emoji
        else:
            print("  Nenhuma feature com incerteza crítica identificada.")

        # Previsão para novos cenários
        print("\nPREVISÃO PARA NOVOS CENÁRIOS:") # Removed emoji
        print("   O modelo está pronto para previsões bayesianas com intervalos de confiança.")
        print("   Use o método predict() com novos dados para obter previsões probabilísticas.")

        return self.summary

    def predict(self, new_data):
        """Faz previsões para novos dados com intervalos de credibilidade"""
        # Pré-processamento dos novos dados
        new_data_processed = new_data.copy()
        categorical_cols = new_data_processed.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            # Ajuste para lidar com categorias não vistas
            unique_vals = np.unique(np.concatenate([self.data[col].values, new_data_processed[col].values]))
            le.fit(unique_vals)
            new_data_processed[col] = le.transform(new_data_processed[col])

        # Garantir mesma ordem de colunas
        new_data_processed = new_data_processed[self.features]

        # Fazer previsões
        with self.model:
            # Use pm.Data for the new data
            new_features_data = pm.Data('new_features_data', new_data_processed.values)
            mu = self.model.intercept + pm.math.dot(new_features_data, self.model.coefficients)
            predictions = pm.sample_posterior_predictive(self.trace, var_names=['y'], random_seed=123) # Added random_seed for reproducibility

        return predictions

# Exemplo de uso
if __name__ == "__main__":
    # Dados de exemplo
    data = {
        'data': pd.date_range(start='2024-01-01', periods=10, freq='D'),
        'visitantes_tiktok': [12500, 13200, 11800, 14500, 15600, 14200, 13800, 16200, 14900, 15500],
        'visitantes_snapchat': [8900, 9200, 8500, 9700, 10200, 9500, 9100, 10500, 9800, 10100],
        'ads_tiktok': [2500, 2800, 2300, 3000, 3200, 2900, 2700, 3400, 3100, 3300],
        'ads_snapchat': [1800, 1900, 1700, 2100, 2200, 2000, 1850, 2300, 2150, 2250],
        'vendas': [420, 450, 390, 480, 520, 470, 440, 550, 500, 530],
        'fase_lunar': ['Nova', 'Crescente', 'Crescente', 'Cheia', 'Cheia', 'Minguante', 'Minguante', 'Nova', 'Crescente', 'Crescente'],
        'feedback_feeling': [7.2, 7.5, 6.8, 8.1, 8.4, 7.9, 7.6, 8.6, 8.2, 8.3]
    }

    df = pd.DataFrame(data)

    # Criar e executar o analisador
    analyzer = BayesianSalesAnalyser()
    analyzer.load_data(df)
    mle, map, bayesian = analyzer.analyze_dataset()
    report = analyzer.generate_report()
    # @title 📊 CÉLULA DE VISUALIZAÇÃO DOS RESULTADOS

def visualize_results(analyzer, mle_coefs, map_coefs, bayesian_results):
    """Visualiza os resultados da análise de forma gráfica"""

    # Configurações de estilo
    plt.style.use('default')
    sns.set_palette("husl")

    # Crear figura con subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('ANÁLISE COMPLETA DOS RESULTADOS BAYESIANOS', fontsize=16, fontweight='bold') # Removed the problematic character

    # 1. Comparación entre MLE, MAP y Bayesian Mean
    ax1 = axes[0, 0]
    features = bayesian_results['Feature']
    x_pos = np.arange(len(features))

    width = 0.25
    ax1.bar(x_pos - width, mle_coefs['Coeficiente_MLE'], width, label='MLE', alpha=0.8)
    ax1.bar(x_pos, map_coefs['Coeficiente_MAP'], width, label='MAP (Ridge)', alpha=0.8)
    ax1.bar(x_pos + width, bayesian_results['mean'], width, label='Bayesian Mean', alpha=0.8)

    ax1.set_xlabel('Features')
    ax1.set_ylabel('Valor do Coeficiente')
    ax1.set_title('Comparação dos Métodos: MLE vs MAP vs Bayesian')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(features, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Distribuições Posteriores con HDI
    ax2 = axes[0, 1]
    colors = plt.cm.viridis(np.linspace(0, 1, len(features)))

    for i, (_, row) in enumerate(bayesian_results.iterrows()):
        ax2.errorbar(row['mean'], i,
                    xerr=[[row['mean'] - row['hdi_3%']], [row['hdi_97%'] - row['mean']]],
                    fmt='o', capsize=5, capthick=2,
                    label=row['Feature'] if i < 5 else "",  # Mostrar apenas 5 labels
                    color=colors[i], markersize=8)

    ax2.set_yticks(range(len(features)))
    ax2.set_yticklabels(features)
    ax2.set_xlabel('Valor do Coeficiente com Intervalo de 94% HDI')
    ax2.set_title('Distribuições Posteriores com Intervalos de Credibilidade')
    ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='lower right')

    # 3. Importância das Features (Magnitude Absoluta)
    ax3 = axes[1, 0]
    importance_df = bayesian_results.copy()
    importance_df['importance'] = np.abs(importance_df['mean'])
    importance_df = importance_df.sort_values('importance', ascending=True)

    ax3.barh(importance_df['Feature'], importance_df['importance'],
             color=plt.cm.plasma(importance_df['importance']/importance_df['importance'].max()))

    ax3.set_xlabel('Importância (Magnitude Absoluta)')
    ax3.set_title('Importância Relativa das Features')
    ax3.grid(True, alpha=0.3)

    # 4. Incerteza vs Magnitude
    ax4 = axes[1, 1]
    scatter = ax4.scatter(bayesian_results['mean'], bayesian_results['sd'],
                         s=np.abs(bayesian_results['mean'])*100,  # Tamanho proporcional à magnitude
                         c=np.abs(bayesian_results['mean']),
                         cmap='viridis', alpha=0.7)

    # Adicionar labels para os pontos
    for i, (_, row) in enumerate(bayesian_results.iterrows()):
        ax4.annotate(row['Feature'], (row['mean'], row['sd']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)

    ax4.set_xlabel('Valor Médio do Coeficiente')
    ax4.set_ylabel('Incerteza (Desvio Padrão)')
    ax4.set_title('Relação entre Magnitude e Incerteza dos Coeficientes')
    ax4.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    ax4.grid(True, alpha=0.3)

    # Adicionar colorbar
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Magnitude Absoluta')

    plt.tight_layout()
    plt.show()

    # 5. Visualização das Distribuições Posteriores individuais
    print("\n" + "="*60)
    print("DISTRIBUIÇÕES POSTERIORES DETALHADAS") # Removed emoji
    print("="*60)

    # Selecionar as 4 features mais importantes para visualização detalhada
    top_4_features = bayesian_results.nlargest(4, '|Mean|')

    fig2, axes2 = plt.subplots(2, 2, figsize=(15, 10))
    axes2 = axes2.flatten()

    for i, (multi_index, row) in enumerate(top_4_features.iterrows()):
        # The index comes as 'coefficients[0]', 'coefficients[1]', ...
        if "coefficients" in multi_index:
            # extract the integer index
            feature_index = int(multi_index.split("[")[1].split("]")[0])
        else:
            feature_index = 0  # fallback for intercept or sigma

        # Extrair as amostras da posterior para esta feature
        feature_samples = analyzer.trace.posterior['coefficients'][:, :, feature_index].values.flatten() # Changed idx to feature_index

        # Plotar histograma e KDE
        sns.histplot(feature_samples, kde=True, ax=axes2[i], stat='density')
        axes2[i].axvline(x=row['mean'], color='red', linestyle='-', label=f'Média: {row["mean"]:.3f}')
        axes2[i].axvline(x=row['hdi_3%'], color='orange', linestyle='--', label='HDI 3%')
        axes2[i].axvline(x=row['hdi_97%'], color='orange', linestyle='--', label='HDI 97%')
        axes2[i].axvline(x=0, color='black', linestyle=':', alpha=0.5)

        axes2[i].set_title(f'Distribuição Posterior: {row["Feature"]}')
        axes2[i].set_xlabel('Valor do Coeficiente')
        axes2[i].set_ylabel('Densidade')
        axes2[i].legend()
        axes2[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 6. Matriz de Correlação das Features
    print("\n" + "="*60)
    print("MATRIZ DE CORRELAÇÃO DAS FEATURES") # Removed emoji
    print("="*60)

    fig3, ax5 = plt.subplots(figsize=(10, 8))
    correlation_matrix = analyzer.data.corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm',
                center=0, square=True, ax=ax5)
    ax5.set_title('Matriz de Correlação entre Features')
    plt.show()

    # 7. Visualização adicional: Traceplot para diagnóstico do MCMC
    print("\n" + "="*60)
    print("DIAGNÓSTICO DO MCMC - TRACEPLOT") # Removed emoji
    print("="*60)

    # Plotar o traceplot para verificar a convergência
    az.plot_trace(analyzer.trace, var_names=['coefficients'])
    plt.tight_layout()
    plt.show()

    # 8. Sumário estatístico completo
    print("\n" + "="*60)
    print("SUMÁRIO ESTATÍSTICO COMPLETO") # Removed emoji
    print("="*60)
    print(az.summary(analyzer.trace, var_names=['coefficients', 'intercept', 'sigma']))


# Exemplo de uso
if __name__ == "__main__":
    # Dados de exemplo
    data = {
        'data': pd.date_range(start='2024-01-01', periods=10, freq='D'),
        'visitantes_tiktok': [12500, 13200, 11800, 14500, 15600, 14200, 13800, 16200, 14900, 15500],
        'visitantes_snapchat': [8900, 9200, 8500, 9700, 10200, 9500, 9100, 10500, 9800, 10100],
        'ads_tiktok': [2500, 2800, 2300, 3000, 3200, 2900, 2700, 3400, 3100, 3300],
        'ads_snapchat': [1800, 1900, 1700, 2100, 2200, 2000, 1850, 2300, 2150, 2250],
        'vendas': [420, 450, 390, 480, 520, 470, 440, 550, 500, 530],
        'fase_lunar': ['Nova', 'Crescente', 'Crescente', 'Cheia', 'Cheia', 'Minguante', 'Minguante', 'Nova', 'Crescente', 'Crescente'],
        'feedback_feeling': [7.2, 7.5, 6.8, 8.1, 8.4, 7.9, 7.6, 8.6, 8.2, 8.3]
    }

    df = pd.DataFrame(data)

    # Criar e executar o analisador
    analyzer = BayesianSalesAnalyser()
    analyzer.load_data(df)
    mle, map, bayesian = analyzer.analyze_dataset()
    report = analyzer.generate_report()
# Executar as visualizações após a análise
print("\n" + "="*60)
print("GERANDO VISUALIZAÇÕES DOS RESULTADOS") # Removed emoji
print("="*60)

visualize_results(analyzer, mle, map, bayesian) # Changed bayesian_results to bayesian

# Visualização adicional: Traceplot para diagnóstico do MCMC
print("\n" + "="*60)
print("DIAGNÓSTICO DO MCMC - TRACEPLOT") # Removed emoji
print("="*60)

# Plotar o traceplot para verificar a convergência
az.plot_trace(analyzer.trace, var_names=['coefficients'])
plt.tight_layout()
plt.show()

# Sumário estatístico completo
print("\n" + "="*60)
print("SUMÁRIO ESTATÍSTICO COMPLETO") # Removed emoji
print("="*60)
print(az.summary(analyzer.trace, var_names=['coefficients', 'intercept', 'sigma']))
