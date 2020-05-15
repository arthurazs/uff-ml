import matplotlib.pyplot as plot
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.tree import export_text

iris = load_iris()
clf = DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)

text_clf = export_text(clf, feature_names=iris['feature_names'])
print(text_clf)

plot_tree(clf)
plot.show()
