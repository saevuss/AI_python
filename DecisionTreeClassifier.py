#binary tree flowchart where eacgh node splits a group of observations according to some feature variable
import pydotplus
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import collections

#providing dataset
x=[[165,19],[175,32],[136,35],[174,65],[141,28],[176,15],[131,32],[166,6],[128, 32],[179,10],[136,34],[186,2],[126,25],[176,28],[112,38],[169,9],[171,36],[116, 25],[196,25]]
y = ['Man','Woman','Woman','Man','Woman','Man','Woman','Man','Woman','Man','Woman', 'Man','Woman','Woman','Woman','Man','Woman','Woman','Man'] #target
data_feature_names = ['height','length of hair']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.40, random_state=5) #60% training and 40% testing

#fit the model
clf = tree.DecisionTreeClassifier() #creating the classifier
clf = clf.fit(x_train, y_train) #training the model to create the tree wich separates man from woman
prediction = clf.predict([[133, 37]]) #asks the model to predict the class of a person with height of 133cm and hair-lenght of 37cm
print(prediction)

dot_data = tree.export_graphviz(clf, feature_names=data_feature_names, out_file = None, filled=True, rounded=True) #dot format is the language of graphs
graph = pydotplus.graph_from_dot_data(dot_data)
colors = ('orange', 'yellow') 
edges = collections.defaultdict(list)
for edge in graph.get_edge_list():
    edges[edge.get_source()].append(int((edge.get_destination())))

for edge in edges:
    edges[edge].sort()
    for i in range(min(2, len(edges[edge]))):
        dest = graph.get_node(str(edges[edge][i]))[0]
        dest.set_fillcolor(colors[i])
graph.write_png('tree.png')

