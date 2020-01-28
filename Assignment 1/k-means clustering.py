#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 11:51:25 2019

@author: zhengleo
"""

import numpy as np
import sklearn.cluster as cluster
import sklearn.metrics as metrics
import pandas as pd


dataset = pd.read_csv('cars.csv')
ncar = dataset.shape[0]

Horsepower_list = dataset['Horsepower'].values.tolist()
Weight_list = dataset['Weight'].values.tolist()
traindata = pd.DataFrame({'Horsepower':Horsepower_list,'Weight':Weight_list})
traindata = np.reshape(np.asarray(traindata),(ncar,2))


Nclusters = np.zeros(15)
Elbow = np.zeros(15)
Silhouette = np.zeros(15)
TotalWCSS = np.zeros(15)
Inertia = np.zeros(15)


for c in range(15):
   Kclusters = c + 1
   Nclusters[c] = Kclusters

   kmeans = cluster.KMeans(n_clusters=Kclusters, random_state=60616).fit(traindata)

   # The Inertia value is the within cluster sum of squares deviation from the centroid
   Inertia[c] = kmeans.inertia_
   
   if (1 < Kclusters):
       Silhouette[c] = metrics.silhouette_score(traindata, kmeans.labels_)
   else:
       Silhouette[c] = np.NaN

   WCSS = np.zeros(Kclusters)
   nC = np.zeros(Kclusters)

   for i in range(428):
      k = kmeans.labels_[i]
      nC[k] += 1
      diff = (traindata[i][0]-kmeans.cluster_centers_[k][0])**2+(traindata[i][1]-kmeans.cluster_centers_[k][1])**2
      WCSS[k] += diff

   Elbow[c] = 0
   for k in range(Kclusters):
      Elbow[c] += WCSS[k] / nC[k]
      TotalWCSS[c] += WCSS[k]

   print("Cluster Assignment:", kmeans.labels_)
   for k in range(Kclusters):
      print("Cluster ", k)
      print("Centroid = ", kmeans.cluster_centers_[k])
      print("Size = ", nC[k])
      print("Within Sum of Squares = ", WCSS[k])
      print(" ")
      
print("N Clusters\t Inertia\t Total WCSS\t Elbow Value\t Silhouette Value:\n")
for c in range(15):
   print('3a.{:.0f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'
         .format(Nclusters[c], Inertia[c], TotalWCSS[c], Elbow[c], Silhouette[c]))
   
   
acceleration = np.zeros(15)
slop = np.zeros(15)
for i in range(0,14):
    slop[i] = (Elbow[i]-Elbow[i+1])/-1.0
slop =slop.tolist()
for i in range(0,13):
    acceleration[i] = slop[i+1]-slop[i]
    
    
    


import matplotlib.pyplot as plt

plt.plot(Nclusters, Elbow, linewidth = 2, marker = 'o')
plt.grid(True)
plt.xlabel("Number of Clusters")
plt.ylabel("Elbow Value")
plt.xticks(np.arange(1, 16, step = 1))
plt.show()

plt.plot(Nclusters, Silhouette, linewidth = 2, marker = 'o')
plt.grid(True)
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Value")
plt.xticks(np.arange(1, 16, step = 1))
plt.show() 

print("3b. According to the graph Elbow value,so I suggest the number of clusters is 13.")