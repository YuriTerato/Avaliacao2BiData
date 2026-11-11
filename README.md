# KMeans com Boston Housing

Agrupa residências da base Boston Housing com KMeans.

* `treino_housing.py`: normaliza a base, encontra o número de grupos pelo método do cotovelo, treina e salva o modelo.
* `inferencia_housing.py`: recebe uma nova residência, descobre o grupo e mostra o centroide desnormalizado.

## Como executar

```bash
pip install -r requirements.txt
python treino_housing.py
python inferencia_housing.py
```
