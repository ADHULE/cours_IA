# FileExample_11_9.py
# Implementsclusteringusingscikit-learn.
# Importthenecessarylibraries:
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from scipy.spatial.distance import cdist
import numpy as np
# Createthedataset:
dataset =make_blobs(n_samples=200,centers=4,n_features=2,\
cluster_std =1.6,random_state=50)
points =dataset[0]
# Plotthedataset:
plt.scatter(dataset[0][:, 0],dataset[0][:,1])
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()
# Findtheoptimalnumberofclusterswiththeinertia
# anddistortionfunctions:
inertia =[]
distortion =[]
# Scriptcontinuesnextpage.
# Scriptcontinuedfrompreviouspage.
for i in range(1,10):
    k_means =KMeans(n_clusters=i,n_init= 'auto')
    k_means.fit(points)
    inertia.append(k_means.inertia_)
    distortion.append(sum(np.min(cdist(points,\
    k_means.cluster_centers_, 'euclidean'), \
    axis =1))/points.shape[0])
# Plottheinertiavalues:
plt.plot([1,2,3,4,5,6,7,8,9], inertia)
plt.ylabel('Inertia')
plt.xlabel('Number of clusters')
plt.show()
# Plotthedistortionvalues:
plt.plot([1,2,3,4,5,6,7,8,9], distortion)
plt.ylabel('Distortion')
plt.xlabel('Number of clusters')
plt.show()
# RuntheKMeansmethodforn=4:
kmeans =KMeans(n_clusters=4,n_init= 'auto')
kmeans.fit(dataset[0])
# Theclusterpointsaregivenby:
clusters =kmeans.cluster_centers_
print("The clusterpointsare:\n", clusters)
# Thedatapointsandthecentroidsareplottedwith:
color0 = ['r','g', 'y', 'c', 'k'] #'m','r',
y_kn =kmeans.fit_predict(points)
index =len(clusters)
for i in range(index):
     plt.scatter(points[y_kn ==i,0],points[y_kn==i,1],\
     s =50,color=color0[i])
for i in range(index):
     plt.scatter(clusters[i][0], clusters[i][1],
     marker = '*', color= 'black', s=120)
plt.show()