from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn import datasets
#Iris Dataset
iris = datasets.load_iris()
X = iris.data

#KMeans
km = KMeans(n_clusters=3)
y_means = km.fit_predict(X)
labels = km.labels_

# Visualizing the Clusters
plt.scatter(X[y_means == 0, 0], X[y_means == 0, 1], s = 100, c = 'red', label = 'Group 1')
plt.scatter(X[y_means == 1, 0], X[y_means == 1, 1], s = 100, c = 'blue', label = 'Group 2')
plt.scatter(X[y_means == 2, 0], X[y_means == 2, 1], s = 100, c = 'green', label = 'Group 3')
''' This is the normal way to do it. Regular ole 2D Scatter Plotting.'''

# Visualizing the Clusters, with EdgeColor on: 
plt.scatter(X[y_means == 0, 0], X[y_means == 0, 1], s = 100, c = 'red', edgecolor = 'blue', label = 'Group 1')
plt.scatter(X[y_means == 1, 0], X[y_means == 1, 1], s = 100, c = 'blue', edgecolor = 'slate', label = 'Group 2')
plt.scatter(X[y_means == 2, 0], X[y_means == 2, 1], s = 100, c = 'green', edge color = 'black', label = 'Group 3')
''' The edge color makes the 2D graph much more visually appealing.'''

#Plotting 3D - The sweet new way I found to represent data. (K-Means is applied to the 3D space)
fig = plt.figure(1, figsize=(7,7))
ax = Axes3D(fig, rect=[0, 0, 0.95, 1])
ax.scatter(X[:, 3], X[:, 0], X[:, 2],
          c=labels.astype(np.float), edgecolor="k", s=50)
ax.set_xlabel("Petal width")
ax.set_ylabel("Sepal length")
ax.set_zlabel("Petal length")
plt.title("K Means", fontsize=14)
plt.show()
