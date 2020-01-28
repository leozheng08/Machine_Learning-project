#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 14:30:33 2019

@author: zhengleo
"""

import matplotlib.pyplot as plt
import numpy
import pandas
import sklearn.cluster as cluster
import sklearn.decomposition as decomposition
import sklearn.metrics as metrics
import sklearn.linear_model as linear_model

ChicagoDiabetes = pandas.read_csv('ChicagoDiabetes.csv',
                               delimiter=',')
crudeRate = ['Crude Rate 2000','Crude Rate 2001','Crude Rate 2002','Crude Rate 2003','Crude Rate 2004','Crude Rate 2005','Crude Rate 2006','Crude Rate 2007','Crude Rate 2008','Crude Rate 2009','Crude Rate 2010','Crude Rate 2011']

X = ChicagoDiabetes[crudeRate]

nObs = X.shape[0]
nVar = X.shape[1]

#pandas.plotting.scatter_matrix(X, figsize=(20,20), c = 'red',
                              # diagonal='hist', hist_kwds={'color':['burlywood']})

# Calculate the Correlations among the variables
XCorrelation = X.corr(method = 'pearson', min_periods = 1)

print('Empirical Correlation: \n', XCorrelation)

# Extract the Principal Components
_thisPCA = decomposition.PCA(n_components = nVar)
_thisPCA.fit(X)
EVR = _thisPCA.explained_variance_ratio_
cumsum_variance_ratio = numpy.cumsum(_thisPCA.explained_variance_ratio_)

print('Explained Variance: \n', _thisPCA.explained_variance_)
print('Explained Variance Ratio: \n', _thisPCA.explained_variance_ratio_)
print('Cumulative Explained Variance Ratio: \n', cumsum_variance_ratio)
print('Principal Components: \n', _thisPCA.components_)

print("1.b Plot the Explained Variances below:")
plt.plot(_thisPCA.explained_variance_, marker = 'o')
plt.xlabel('Index')
plt.ylabel('Explained Variance')
plt.xticks(numpy.arange(0,nVar))
plt.axhline((1/nVar), color = 'r', linestyle = '--')
plt.grid(True)
plt.show()

cumsum_variance_ratio = numpy.cumsum(_thisPCA.explained_variance_ratio_)
plt.plot(cumsum_variance_ratio, marker = 'o')
plt.xlabel('Index')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.xticks(numpy.arange(0,nVar))
plt.grid(True)
plt.show()
print(f"1.d the the cumulative explained variance ratio is {cumsum_variance_ratio[0]},{cumsum_variance_ratio[1]}")

first2PC = _thisPCA.components_[:, [0,1]]
print('Principal COmponent: \n', first2PC)
print("1.a the max number of principle component is 2 beacause the first two principle component account for 96.69% of the total variance")
PC1 = numpy.zeros(12)
PC2 = numpy.zeros(12)
for i in range(12):
    PC1[i] = first2PC[i][0]
    PC2[i] = first2PC[i][1]
first2PC = pandas.DataFrame({'PC1':PC1,'PC2':PC2},index = ['Crude Rate 2000','Crude Rate 2001','Crude Rate 2002','Crude Rate 2003','Crude Rate 2004','Crude Rate 2005','Crude Rate 2006','Crude Rate 2007','Crude Rate 2008','Crude Rate 2009','Crude Rate 2010','Crude Rate 2011'])

print("1.c the first principle component is dominated by Crude Rate 2001,Crude Rate 2005,Crude Rate 2006,Crude Rate 2010  ")
print("the second principle component is dominated by Crude Rate 2001,Crude Rate 2002,Crude Rate 2005,Crude Rate 2009,Crude Rate 2001,Crude Rate 2005,Crude Rate 2006,Crude Rate 2011 ")

_thisPCA = decomposition.PCA(n_components = 2)
X_transformed = pandas.DataFrame(_thisPCA.fit_transform(X))

# Find clusters from the transformed data
maxNClusters = 10

nClusters = numpy.zeros(maxNClusters-1)
Elbow = numpy.zeros(maxNClusters-1)
Silhouette = numpy.zeros(maxNClusters-1)
TotalWCSS = numpy.zeros(maxNClusters-1)
Inertia = numpy.zeros(maxNClusters-1)

for c in range(maxNClusters-1):
   KClusters = c + 2
   nClusters[c] = KClusters

   kmeans = cluster.KMeans(n_clusters=KClusters, random_state=20181010).fit(X_transformed)

   # The Inertia value is the within cluster sum of squares deviation from the centroid
   Inertia[c] = kmeans.inertia_
   
   if (KClusters > 1):
       Silhouette[c] = metrics.silhouette_score(X_transformed, kmeans.labels_)
   else:
       Silhouette[c] = float('nan')

   WCSS = numpy.zeros(KClusters)
   nC = numpy.zeros(KClusters)

   for i in range(nObs):
      k = kmeans.labels_[i]
      nC[k] += 1
      diff = X_transformed.iloc[i,] - kmeans.cluster_centers_[k]
      WCSS[k] += diff.dot(diff)

   Elbow[c] = 0
   for k in range(KClusters):
      Elbow[c] += (WCSS[k] / nC[k])
      TotalWCSS[c] += WCSS[k]

   print("The", KClusters, "Cluster Solution Done")

print("N Clusters\t Inertia\t Total WCSS\t Elbow Value\t Silhouette Value:")
for c in range(maxNClusters-1):
   print('{:.0f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'
         .format(nClusters[c], Inertia[c], TotalWCSS[c], Elbow[c], Silhouette[c]))

print("1.e Plot the Elbow and the Silhouette charts below")
# Draw the Elbow and the Silhouette charts  
plt.plot(nClusters, Elbow, linewidth = 2, marker = 'o')
plt.grid(True)
plt.xlabel("Number of Clusters")
plt.ylabel("Elbow Value")
plt.xticks(numpy.arange(2, maxNClusters+1, 1))
plt.show()

plt.plot(nClusters, Silhouette, linewidth = 2, marker = 'o')
plt.grid(True)
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Value")
plt.xticks(numpy.arange(2, maxNClusters+1, 1))
plt.show()

acceleration = numpy.zeros(9)
slop = numpy.zeros(9)
for i in range(0,8):
    slop[i+1] = (Elbow[i]-Elbow[i+1])/-1.0
slop =slop.tolist()
for i in range(0,7):
    acceleration[i+1] = slop[i+2]-slop[i+1]   
         
chart = pandas.DataFrame({'Nclusters':nClusters,'Elbow_value':Elbow,'Silhouette_value':Silhouette,'slop':slop,'acceleration':acceleration})
print(f"1.f the Elbow and Silhouette chart is \n{chart},according to the chart  you can find when the kcluters is 4 the acceleration is biggest ")

kmeans = cluster.KMeans(n_clusters=4, random_state=20181010).fit(X_transformed)
X_transformed['Cluster ID'] = kmeans.labels_

Community = ChicagoDiabetes[['Community']]
Community['Cluster ID'] = kmeans.labels_
Ncommunity = Community.shape[0]
Cluster_0 = []
Cluster_1 = []
Cluster_2 = []
Cluster_3 = []
for i in range(Ncommunity):
    if Community.iloc[i,1] == 0:
        Cluster_0.append(Community.iloc[i,0])
    if Community.iloc[i,1] == 1:
        Cluster_1.append(Community.iloc[i,0])
    if Community.iloc[i,1] == 2:
        Cluster_2.append(Community.iloc[i,0])
    if Community.iloc[i,1] == 3:
        Cluster_3.append(Community.iloc[i,0])
index = ['Cluster_0','Cluster_1','Cluster_2','Cluster_3']
Cluster = numpy.asarray([Cluster_0,Cluster_1,Cluster_2,Cluster_3])  
CommunityEachCluster = pandas.DataFrame({'Community':Cluster},index = index)
print(f"1.g the names of the communities in each cluster is {CommunityEachCluster}")



# Draw the first two PC using cluster label as the marker color 
carray = ['red', 'orange', 'green', 'black']
plt.figure(figsize=(10,10))
for i in range(4):
    subData = X_transformed[X_transformed['Cluster ID'] == i]
    plt.scatter(x = subData[0],
                y = subData[1], c = carray[i], label = i, s = 25)
plt.grid(True)
plt.axis(aspect = 'equal')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.axis(aspect = 'equal')
plt.legend(title = 'Cluster ID', fontsize = 12, markerscale = 2)
plt.show()

dataCHR = ChicagoDiabetes[crudeRate]
Hosptialization = ['Hospitalizations 2000','Hospitalizations 2001','Hospitalizations 2002','Hospitalizations 2003','Hospitalizations 2004','Hospitalizations 2005','Hospitalizations 2006','Hospitalizations 2007','Hospitalizations 2008','Hospitalizations 2009','Hospitalizations 2010','Hospitalizations 2011']
dataHD = ChicagoDiabetes[Hosptialization]

traindataC = numpy.reshape(numpy.asarray(dataCHR),(46,12))
traindataH = numpy.reshape(numpy.asarray(dataHD),(46,12))
traindataTP = numpy.zeros((46,12))
for i in range(46):
    for j in range(12):
        traindataTP[i][j] = traindataH[i][j]/traindataC[i][j]*10000
dataTP = pandas.DataFrame(traindataTP,columns = ['TP 2000','TP 2001','TP 2002','TP 2003','TP 2004','TP 2005','TP 2006','TP 2007','TP 2008','TP 2009','TP 2010','TP 2011'] )

AnnualTP = dataTP.sum().values.tolist()
AnnualHD = dataHD.sum().values.tolist()
AnnualCD = numpy.zeros(12)
for i in range(12):
    AnnualCD[i] = AnnualHD[i]/AnnualTP[i]*10000
year = numpy.asarray([2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011])
ChicagoAnnualCHR = pandas.DataFrame({' Chicago’s annual crude hospitalization rates':AnnualCD},index = year)

dataHD['Cluster ID'] = kmeans.labels_
dataHD = dataHD.groupby(['Cluster ID']).sum()
dataTP['Cluster ID'] = kmeans.labels_
dataTP = dataTP.groupby(['Cluster ID']).sum()
newTableHD = dataHD.values.tolist()
newTableTP = dataTP.values.tolist()
newTableCHR= numpy.zeros((4,12))
for i in range(4):
    for j in range(12):
        newTableCHR[i][j] = (newTableHD[i][j]/newTableTP[i][j])*10000
dataCHR = pandas.DataFrame(newTableCHR,columns =crudeRate )
print(f"1.i Plot the crude hospitalization rates in each cluster against the years below:")

plt.figure(figsize=(6,6))

plt.plot(year, AnnualCD, marker = 'o',label = 'reference curve',
         color = 'red', linestyle = ':')
plt.plot(year, newTableCHR[0], marker = 'o',label = 'cluster 0 ',
         color = 'C0',linestyle = 'solid', linewidth = 1, markersize = 6)
plt.plot(year, newTableCHR[1], marker = 'o',label = 'cluster 1',
         color = 'C1', linestyle = 'solid', linewidth = 1, markersize = 6)
plt.plot(year, newTableCHR[2], marker = 'o',label = 'cluster 2',
         color = 'C2',linestyle = 'solid', linewidth = 1, markersize = 6)
plt.plot(year, newTableCHR[3], marker = 'o', label = 'cluster 3',
         color = 'C3', linestyle = 'solid', linewidth = 1, markersize = 6)
plt.title('the crude hospitalization rates')
plt.grid(True)
plt.xlabel("year")
plt.ylabel("Crude Hospitaliztion rate")
#plt.axis(aspect = 'equal')
plt.legend()
#plt.savefig("11-d_ds")
#print("figure saved")



















