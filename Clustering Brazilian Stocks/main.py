from sklearn.cluster import KMeans
from sklearn.decomposition import PCA             # Análise de Componentes Principais (ou Principal Component Analysis)
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler  # Padronizador e Redutor da Escala de Dados

import numpy
import pandas
import plotly.express as plotexp


K = 10
#
# Número de Clusters


def bar_chart(group, key):
    x, y = 'Stock', 'D.Yield'

    labels, title = {'Stock': 'Ação', 'D.Yield': 'Dividend Yield Percentual'}, 'Cluster ' + str(key)

    range_y, template = [0, 30], 'seaborn'

    bar_plot = plotexp.bar(group[key], x=x, y=y, labels=labels, range_y=range_y, title=title, template=template)
    bar_plot.show()
#
# Função de plotagem dos gráficos de barras que serão gerados a partir dos K clusters obtidos ao término do procedimento
# de clusterização.


def clusters(dataset):
    dataset = {key: dataset[dataset['Cluster'] == key - 1] for key in range(1, K + 1)}

    for key in dataset.keys():
        dataset[key] = drop(dropped_columns(), dataset[key])

        dataset[key] = dataset[key].sort_values(by='Stock', ascending=True)

        dataset[key]['D.Yield'] *= 100

    return dataset


def drop(columns, data):
    return data.drop(columns, axis=1)


def dropped_columns():
    return ['index', 'Price', 'P/L', 'P/VP', 'EV/EBIT', 'ROIC', 'ROE', 'Liq. L2M', 'Pat. Liq.', 'Div. Brut./Pat. Liq.',
            'CRL L5A']
#
# Colunas que serão descartadas após a conclusão do procedimento de clusterização.


def print_data(data):
    return print(f'\n{data}'
                 f'\n')


def processing_steps():
    return [('scaler', StandardScaler()), ('PCA', PCA(n_components=2, random_state=SEED))]
#
# Etapas que serão processadas no fluxo de operação do pipeline instanciado na linha 97.


def unused_columns():
    return ['PSR', 'P/Ativos', 'P/Cap. Giro', 'P/EBIT', 'P/ACL', 'EV/EBITDA', 'Marg. EBIT', 'Marg. Liq.', 'Liq. Corr.']
#
# Colunas não utilizadas como critérios de similaridade para a clusterização.


if __name__ == '__main__':
    stocks = pandas.read_csv('Brazilian Stocks.csv', delimiter=';')
    #
    # `Brazilian Stocks.csv` contém dados obtidos em 13 de setembro de 2023, pós-fechamento da Bolsa de Valores.

    print_data(stocks)

    stocks = drop(unused_columns(), stocks)
    #
    # Seleção de 12 dentre as 21 colunas existentes no conjunto de dados `Brazilian Stocks.csv`.
    #
    # . Colunas Selecionadas
    #       Stock, Price, P/L, P/VP, D.Yield, EV/EBIT, ROIC, ROE, Liq. L2M, Pat. Liq., Div. Brut./Pat. Liq. e CRL L5A

    print_data(stocks)

    stocks = stocks[stocks['D.Yield'] > 0.05]
    stocks = stocks.reset_index()
    #
    # Seleção dos dados referentes às ações de empresas que nos últimos 12 meses apresentam um histórico de
    # distribuição de dividendos superior a 5%.

    print_data(stocks)

    SEED = 1225
    numpy.random.seed(SEED)

    pipeline = Pipeline(processing_steps())
    #
    # Execução do pipeline de dados a partir da aplicação da padronização e redução da escala dos dados, seguida
    # da redução de dimensionalidade (11D --> 2D) realizada pelo método de decomposição PCA.

    data_pairs = pandas.DataFrame(columns=['x', 'y'], data=pipeline.fit_transform(drop(['index', 'Stock'], stocks)))
    #
    # Obtenção do conjunto de pares de dados (ou de pares ordenados) que correspondem ao resultado da redução de
    # dimensionalidade aplicada sobre os dados contidos em `stocks`.
    #
    # Os dados das 11 colunas numéricas de `stocks` encontram-se condensados ou reduzidos na forma de duas colunas
    # apenas, de modo que cada par (x,y) em cada uma das linhas de `data_pairs` representa uma Ação contida em `stocks`.

    print_data(data_pairs)

    model = KMeans(n_clusters=K, n_init='auto', random_state=SEED, verbose=False)

    model.fit(data_pairs)
    #
    # Treinamento do modelo de clusterização aplicado sobre o conjunto de pares de dados.

    data_pairs['Cluster'], stocks['Cluster'] = model.predict(data_pairs), model.predict(data_pairs)
    #
    # Atribuição dos números de identificação de cada cluster ao conjunto de dados reduzidos e ao conjunto de
    # dados original.

    data_pairs['Stock'], data_pairs['D.Yield'] = stocks['Stock'], stocks['D.Yield']

    chart = plotexp.scatter(data_pairs, x='x', y='y', color='Cluster', hover_data=['Stock', 'D.Yield'])

    chart.show()
    #
    # Apresentação da distribuição dos K clusters resultantes da aplicação do método K-Means.

    data_pairs, stocks = data_pairs.sort_values(by='Cluster'), stocks.sort_values(by='Cluster')
    #
    # Ordenação dos dados contidos em `data_pairs` e `stocks`, tomando como referência de ordenação ascendente a coluna
    # contendo o número de identificação de cada cluster.

    print_data(data_pairs)
    print_data(stocks)

    cluster = clusters(stocks)
    #
    # Obtenção dos subconjuntos de dados provenientes do conjunto princiapl `stocks`, de modo que cada subconjunto está
    # identificado a partir de um número atribuído ao cluster que representa este subconjunto.

    for index in cluster.keys():
        bar_chart(cluster, index)
    #
    # Apresentação de cada um dos cinco clusters gerados pelo K-Means a partir da utilização de um gráfico de barras.
    #
    # Neste gráfico, as ações de cada cluster são apresentdas tomando como referêcnia o valor percentual do Dividend
    # Yield anual associado a cada um dos ativos.

    print(f'\nMétricas de Avaliação (ou Validação Interna dos Resultados)'
          f'\n'
          f'\nNúmero de Clusters = {K:5.2f}'
          f'\n'
          f'\n           Inertia = {model.inertia_:5.2f}'
          f'\n'
          f'\n  Silhouette Score = {silhouette_score(drop("Stock", data_pairs), sorted(model.labels_)):5.2f}'
          f'\n')
    #
    # A Inércia (ou Inertia) mede a compactação ou a junção dos clusters, sendo útil para avaliar a qualidade do
    # procedimento de clusterização.
    #
    # Quanto menor for o valor (não negativo) da inércia, melhor será a qualidade dos agrupamentos.
    #
    # A Silhouette Score (ou Pontuação de Silhueta) mede a qualidade da coesão intra-cluster e da separação
    # inter-cluster.
    #
    # Esta métrica pode assumir valores reais entre -1 e 1, sendo que valores mais próximos de 1 indicam que os clusters
    # estão bem separados e têm uma boa coesão interna.
