#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 14:23:12 2019

@author: zhengleo
"""

import matplotlib.pyplot as plt
import numpy
import pandas

import sklearn.cluster as cluster

Spiral = pandas.read_csv('Spiral.csv')
print("4a. the scatterplot belowing:")
nObs = Spiral.shape[0]
plt.scatter(Spiral[['x']], Spiral[['y']], c = Spiral[['id']].values.tolist())
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()

trainData = Spiral[['x','y']]

kmeans = cluster.KMeans(n_clusters=4, random_state=0).fit(trainData)

print("Cluster Centroids = \n", kmeans.cluster_centers_)

Spiral['KMeanCluster'] = kmeans.labels_

for i in range(2):
    print("Cluster Label = ", i)
    print(Spiral.loc[Spiral['KMeanCluster'] == i])
    
print("4b.the scatterplot belowing:")
plt.scatter(Spiral[['x']], Spiral[['y']], c = Spiral[['KMeanCluster']].values.tolist())
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()

import math
import sklearn.neighbors

# Three nearest neighbors
kNNSpec = sklearn.neighbors.NearestNeighbors(n_neighbors = 8, algorithm = 'brute', metric = 'euclidean')
nbrs = kNNSpec.fit(trainData)
d3, i3 = nbrs.kneighbors(trainData)

# Retrieve the distances among the observations
distObject = sklearn.neighbors.DistanceMetric.get_metric('euclidean')
distances = distObject.pairwise(trainData)

# Create the Adjacency and the Degree matrices
Adjacency = numpy.zeros((nObs, nObs))
Degree = numpy.zeros((nObs, nObs))

for i in range(nObs):
    for j in i3[i]:
        if (i <= j):
            Adjacency[i,j] = math.exp(- distances[i][j])
            Adjacency[j,i] = Adjacency[i,j]

for i in range(nObs):
    sum = 0
    for j in range(nObs):
        sum += Adjacency[i,j]
    Degree[i,i] = sum
        
Lmatrix = Degree - Adjacency

from numpy import linalg as LA
evals, evecs = LA.eigh(Lmatrix)

# Series plot of the smallest ten eigenvalues to determine the number of clusters
print("4d.the scatterplot belowing:")
plt.scatter(numpy.arange(0,9,1), evals[0:9,])
plt.xlabel('Sequence')
plt.ylabel('Eigenvalue')
plt.show()

# Inspect the values of the selected eigenvectors 
print("4e.the scatterplot belowing:")
Z = evecs[:,[0,1]]

plt.scatter(Z[[0]], Z[[1]])
plt.xlabel('Z[0]')
plt.ylabel('Z[1]')
plt.show()

kmeans_spectral = cluster.KMeans(n_clusters=2, random_state=0).fit(Z)

Spiral['SpectralCluster'] = kmeans_spectral.labels_

plt.scatter(Spiral[['x']], Spiral[['y']], c = Spiral[['SpectralCluster']].values.tolist())
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()