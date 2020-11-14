from sklearn.tree import export_graphviz
from sklearn import tree
import graphviz
import numpy as np

X = [[0], [0.1],[0.2],[0.3],[0.4],[0.45],[0.5],[0.6],[0.7],[0.8],[0.9],[0.95],[1.0]]
y = [4,2.4,1.5,1.0,1.2,1.5,1.8,2.6,3.0,4.0,4.5,5.0,6.0]
clf = tree.DecisionTreeRegressor()
clf = clf.fit(X, y)


dot_data = export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
graph.render('tree')


X = np.array([[1,1,1,0],[1,1,1,1],[2,1,1,0],[3,2,1,0],[3,3,2,0],[3,3,2,1],[2,3,2,1],[1,2,1,0],[1,3,2,0],[3,2,2,0],[1,2,2,1],[2,2,1,1],[2,1,2,0],[3,2,1,1]])
y = np.array([0,0,1,1,1,0,1,0,1,1,1,1,1,0])
clf = tree.DecisionTreeClassifier(criterion="entropy")
clf = clf.fit(X, y)

dot_data = export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
graph.render('tree_id3')

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)

dot_data = export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
graph.render('tree_cart')