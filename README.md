# Aprendizado de Máquina (UFF)

## Aula 1

Dia 16/04/2020

- [weka](https://waikato.github.io/weka-wiki/downloading_weka/)

- [scikit-learn](https://scikit-learn.org/stable/getting_started.html)

  - [X] [Decision Trees](https://scikit-learn.org/stable/modules/tree.html) feito em [tree_example.py](aula1/1tree_example.py)

## Aula 2

Dia 30/04/2020

```bash
$ pip install numpy scipy pandas seaborn matplotlib sklearn
$ apt install python3-tk
```

Familiarizar-se com os pacotes Python:

- [X] **Numpy** processamento de vetores e matrizes [scipy.py](aula2/2scipy.py)
- [X] **Scipy** cálculo de estatísticas [scipy.py](aula2/2scipy.py)
- [X] **Pandas** processamento de séries e dataframes [pandas.py](aula2/1pandas.py)
  - Muito útil para processar os dados para serem manipulados pelos algoritmos de aprendizado de máquina
- [X] **Matplotlib** para desenho de gráficos unidimensionais e bidimensionais [seaborn.py](aula2/3seaborn.py)
- [X] **Seaborn** gráficos mais elaborados que o Matplotlib [seaborn.py](aula2/3seaborn.py)

Utilizando os pacotes mencionados:

- [X] Utilize o Pandas (Dataframe) para importar o conjunto de dados Iris [scipy.py](aula2/2scipy.py)
- [ ] Utilize o Scipy para calcular todas as estatísticas vistas na aula de hoje [scipy.py](aula2/2scipy.py)
- [X] Construa boxplots para cada um dos 4 atributos do conjunto de dados Iris [seaborn.py](aula2/3seaborn.py)
  - [X] Pesquise a diferença entre os boxplots e os violinplots, e plote também esses gráficos [seaborn.py](aula2/3seaborn.py)
- [X] Construa uma matriz de correlação entre os atributos do conjunto de dados [seaborn.py](aula2/3seaborn.py)
- [ ] Fazer análises do conjunto de dados para tentar entender qual a dificuldade de construir classificadores considerando a distribuição dos dados nas classes
