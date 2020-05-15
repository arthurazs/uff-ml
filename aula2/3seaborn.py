import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd

iris_dataset = load_iris()
iris_data = iris_dataset['data']                    # objetos
iris_features = iris_dataset['feature_names']       # atributos
iris_features = [name[:-5] for name in iris_features]
iris_targets = iris_dataset['target']               # atributos / classes
iris_target_names = iris_dataset['target_names']    # classes

data = pd.DataFrame(iris_data, columns=iris_features)
data['target'] = iris_targets
data['target'] = data['target'].apply(lambda x: iris_target_names[x])

# <-- univariados -->
# <-- localizacao ou tendencia central -->

print('boxplot')
plt.figure()
plt.subplot(221)
sns.boxplot(x='target', y='petal length', data=data)
plt.subplot(222)
sns.boxplot(x='target', y='sepal length', data=data)
plt.subplot(223)
sns.boxplot(x='target', y='petal width', data=data)
plt.subplot(224)
sns.boxplot(x='target', y='sepal width', data=data)

print('violin plot')
plt.figure()
plt.subplot(221)
sns.violinplot(x='target', y='petal length', data=data)
plt.subplot(222)
sns.violinplot(x='target', y='sepal length', data=data)
plt.subplot(223)
sns.violinplot(x='target', y='petal width', data=data)
plt.subplot(224)
sns.violinplot(x='target', y='sepal width', data=data)


# <-- univariados -->
# <-- dispersao ou espelhamento -->

print('pairplot / scatterplot')
sns.pairplot(data, hue='target')


# <-- multivariados -->

print('heatmap')
plt.figure()
sns.heatmap(data.corr(), annot=True)

plt.show()
