from sklearn.datasets import load_iris
import pandas as pd

iris_dataset = load_iris()
iris_data = iris_dataset['data']                    # objetos
iris_features = iris_dataset['feature_names']       # atributos
iris_targets = iris_dataset['target']               # atributos / classes
iris_target_names = iris_dataset['target_names']    # classes

data = pd.DataFrame(iris_data, columns=iris_features)
data['target'] = iris_targets
data['target'] = data['target'].apply(lambda x: iris_target_names[x])
correlation = data.corr()

print(f'>>> Summary:\n{data}\n')
print(f'>>> Information:\n{data.describe()}\n')
print(f'>>> Correlation:\n{correlation}\n')
print(f'>>> Data Description:')
data.info()
