# Aprendizado de Máquina (UFF)

## Trabalho Prático

> Stephen Makonin, [*“HUE: The Hourly Usage of Energy Dataset for Buildings in British Columbia,”*](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/N3HGRN) Data in Brief, vol. 23, no. 103744, pp. 1-4 (2019).

## Passos

Os passos a seguir foram testados no Ubuntu 20.04.

1. Baixar o conjunto de dados [HUE](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/N3HGRN);
2. Mover os arquivos do HUE para a pasta `~/uff-ml/pratica/dataset/hue`;
3. Instalar os *softwares* necessários, através do terminal:

```bash
pip install seaborn sklearn  # graficos e aprendizado de maquina
apt install -y python3-tk    # graficos
```

4. Executar o pré-processamento:

```bash
cd pratica
python preprocessing.py
python object2numeral.py
```
5. Executar o aprendizado:

```bash
python train_and_test.py
python validation_analysis.py
```
