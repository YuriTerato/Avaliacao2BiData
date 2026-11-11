import pandas as pd
from pickle import load

modelo_normalizador = load(open('normalizador_housing.model', 'rb'))
colunas_categoricas = load(open('colunas_categoricas.model', 'rb'))
housing_cluster = load(open('housing_cluster.model', 'rb'))

nova_residencia = pd.DataFrame({
    'CRIM': [0.25], 'ZN': [12.5], 'INDUS': [7.0], 'CHAS': [0], 'NOX': [0.52],
    'RM': [6.3], 'AGE': [65.0], 'DIS': [4.0], 'RAD': [4], 'TAX': [300],
    'PTRATIO': [17.5], 'B': [390.0], 'LSTAT': [10.0], 'MEDV': [22.0]
})

nova_num = nova_residencia[list(modelo_normalizador.feature_names_in_)]
nova_num_normalizada = pd.DataFrame(modelo_normalizador.transform(nova_num), columns=nova_num.columns)

nova_cat_normalizada = pd.get_dummies(nova_residencia['CHAS'], prefix='CHAS', dtype=int)
nova_cat_normalizada = nova_cat_normalizada.reindex(columns=colunas_categoricas, fill_value=0)

nova_normalizada = nova_num_normalizada.join(nova_cat_normalizada)

grupo = housing_cluster.predict(nova_normalizada)
print('A residencia pertence ao grupo:', grupo[0])

centroide = pd.DataFrame(housing_cluster.cluster_centers_[grupo], columns=nova_normalizada.columns)

centroide_num = pd.DataFrame(modelo_normalizador.inverse_transform(centroide[nova_num.columns]), columns=nova_num.columns)

categoria_mais_forte = centroide[colunas_categoricas].idxmax(axis=1)
centroide_cat_01 = pd.get_dummies(categoria_mais_forte, dtype=int).reindex(columns=colunas_categoricas, fill_value=0)
centroide_cat_natural = pd.from_dummies(centroide_cat_01, sep='_')

centroide_natural = centroide_num.join(centroide_cat_natural)
print('Centroide do grupo (desnormalizado):')
print(centroide_natural.round(2).T)
