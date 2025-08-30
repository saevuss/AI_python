#bettr than single decision ree because while retaining the predictive
#powers it can reduce over-fitting by averaging the results

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import numpy as np

cancer = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
#fitting the model
forest = RandomForestClassifier(n_estimators=50, random_state=0)
forest.fit(x_train, y_train)

print('Accuracy on the training subset:(:.3f)', format(forest.score(x_train, y_train)))
print('Accuracy on the test subset:(:.3f)', format(forest.score(x_test, y_test)))

#like decision tree random forest has the feature_importance module which will provide a better view of feature weight than decision tree
n_features = cancer.data.shape[1]
plt.barh(range(n_features), forest.feature_importances_, align='center')
plt.yticks(np.arange(n_features), cancer.feature_names)
plt.xlabel('Feature importance')
plt.ylabel('Feature')
plt.show()
