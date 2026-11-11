import math
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn import preprocessing
from sklearn.cluster import KMeans
from pickle import dump

dados = pd.read_csv('HousingData.csv')
dados = dados.fillna(dados.median())
dados['CHAS'] = dados['CHAS'].astype(int)

dados_cat = dados['CHAS']
dados_num = dados.drop(columns=['CHAS'])

normalizador = preprocessing.MinMaxScaler()
modelo_normalizador = normalizador.fit(dados_num)
dump(modelo_normalizador, open('normalizador_housing.model', 'wb'))
dados_num_normalizados = pd.DataFrame(modelo_normalizador.transform(dados_num), columns=dados_num.columns)

dados_cat_normalizado = pd.get_dummies(dados_cat, prefix='CHAS', dtype=int)
dump(list(dados_cat_normalizado.columns), open('colunas_categoricas.model', 'wb'))

dados_normalizados = dados_num_normalizados.join(dados_cat_normalizado)

distorcoes = []
K = range(1, 21)
for i in K:
    cluster_model = KMeans(n_clusters=i, random_state=42).fit(dados_normalizados)
    distorcoes.append(
        sum(
            np.min(
                cdist(dados_normalizados, cluster_model.cluster_centers_, 'euclidean'),
                axis=1) / dados_normalizados.shape[0]
        )
    )

x0 = K[0]
y0 = distorcoes[0]
xn = K[-1]
yn = distorcoes[-1]
distancias = []
for i in range(len(distorcoes)):
    x = K[i]
    y = distorcoes[i]
    numerador = abs((yn - y0) * x - (xn - x0) * y + xn * y0 - yn * x0)
    denominador = math.sqrt((yn - y0) ** 2 + (xn - x0) ** 2)
    distancias.append(numerador / denominador)

n_clusters_otimo = K[distancias.index(max(distancias))]
print('Numero otimo de grupos:', n_clusters_otimo)

housing_cluster = KMeans(n_clusters=n_clusters_otimo, random_state=42).fit(dados_normalizados)
dump(housing_cluster, open('housing_cluster.model', 'wb'))
