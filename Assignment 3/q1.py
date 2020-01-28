#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 13:14:02 2019

@author: zhengleo
"""

import pandas 
import numpy
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import sklearn.metrics as metrics

data = pandas.read_csv('ChicagoDiabetes.csv')
data = data.dropna()
crudeRate = ['Crude Rate 2000','Crude Rate 2001','Crude Rate 2002','Crude Rate 2003','Crude Rate 2004','Crude Rate 2005','Crude Rate 2006','Crude Rate 2007','Crude Rate 2008','Crude Rate 2009','Crude Rate 2010','Crude Rate 2011']
hospitalization = ['Hospitalizations 2000','Hospitalizations 2001','Hospitalizations 2002','Hospitalizations 2003','Hospitalizations 2004','Hospitalizations 2005','Hospitalizations 2006','Hospitalizations 2007','Hospitalizations 2008','Hospitalizations 2009','Hospitalizations 2010','Hospitalizations 2011']
trainDataC = data[crudeRate]
trainDataH = data[hospitalization]

traindataC = numpy.reshape(numpy.asarray(trainDataC),(46,12))
traindataH = numpy.reshape(numpy.asarray(trainDataH),(46,12))
traindataTP = numpy.zeros((46,12))
for i in range(46):
    for j in range(12):
        traindataTP[i][j] = traindataH[i][j]/traindataC[i][j]*10000
trainDataTP = pandas.DataFrame(traindataTP,columns = ['TP 2000','TP 2001','TP 2002','TP 2003','TP 2004','TP 2005','TP 2006','TP 2007','TP 2008','TP 2009','TP 2010','TP 2011'] )

Nclusters = numpy.zeros(9)
Elbow = numpy.zeros(9)
Silhouette = numpy.zeros(9)
TotalWcss = numpy.zeros(9)
NumberClusters = []
k_label=[]



for c in range(0,9):
    kcluster = c+2
    Nclusters[c] = kcluster
    kmeans = cluster.KMeans(n_clusters=kcluster, random_state=20190405).fit(traindataC)
    k_label.append(kmeans.labels_)
    
    if (1 < kcluster):
       Silhouette[c] = metrics.silhouette_score(traindataC, kmeans.labels_)
    else:
       Silhouette[c] = numpy.NaN
       
    WCSS = numpy.zeros(kcluster)
    nC = numpy.zeros(kcluster)
   
    elbow = 0
    for i in range(kcluster):
        count = 0
        sum = 0
        CommunityEachCluster = []
        for j in range(0,46):
            if (kmeans.labels_[j] == i):
                count= count+1
                for k in range(0,12):
                    sum += (traindataC[j][k]-kmeans.cluster_centers_[i][k])**2
                
               
        WCSS[i] = sum
        nC[i] = count
        elbow+=WCSS[i]/nC[i]
       
    Elbow[c] = elbow
    NumberClusters.append(nC)

Cluster_label = pandas.DataFrame({'Cluster_label':k_label[0]}) 
table = trainDataH.join(Cluster_label)
tableTP = trainDataTP.join(Cluster_label)
new_table_0 = table.groupby(['Cluster_label']).sum()
new_tableTP_0 = tableTP.groupby(['Cluster_label']).sum()
newTableHD = new_table_0.values.tolist()
newTableTP = new_tableTP_0.values.tolist()
newTableCR_0 = numpy.zeros((2,12))
for i in range(2):
    for j in range(12):
        newTableCR_0[i][j] = (newTableHD[i][j]/newTableTP[i][j])*10000
new_tableCR_0 = pandas.DataFrame(newTableCR_0,columns =crudeRate )
        


Cluster_label = pandas.DataFrame({'Cluster_label':k_label[1]}) 
table = trainDataH.join(Cluster_label)
tableTP = trainDataTP.join(Cluster_label)
new_table_1 = table.groupby(['Cluster_label']).sum()
new_tableTP_1 = tableTP.groupby(['Cluster_label']).sum()
newTableHD = new_table_1.values.tolist()
newTableTP = new_tableTP_1.values.tolist()
newTableCR_1 = numpy.zeros((3,12))
for i in range(3):
    for j in range(12):
        newTableCR_1[i][j] = (newTableHD[i][j]/newTableTP[i][j])*10000
new_tableCR_1 = pandas.DataFrame(newTableCR_1,columns =crudeRate )


Cluster_label = pandas.DataFrame({'Cluster_label':k_label[2]}) 
table = trainDataH.join(Cluster_label)
tableTP = trainDataTP.join(Cluster_label)
new_table_2 = table.groupby(['Cluster_label']).sum()
new_tableTP_2 = tableTP.groupby(['Cluster_label']).sum()
newTableHD = new_table_2.values.tolist()
newTableTP = new_tableTP_2.values.tolist()
newTableCR_2 = numpy.zeros((4,12))
for i in range(4):
    for j in range(12):
        newTableCR_2[i][j] = (newTableHD[i][j]/newTableTP[i][j])*10000
new_tableCR_2 = pandas.DataFrame(newTableCR_2,columns =crudeRate )


Cluster_label = pandas.DataFrame({'Cluster_label':k_label[3]}) 
table = trainDataH.join(Cluster_label)
tableTP = trainDataTP.join(Cluster_label)
new_table_3 = table.groupby(['Cluster_label']).sum()
new_tableTP_3 = tableTP.groupby(['Cluster_label']).sum()
newTableHD = new_table_3.values.tolist()
newTableTP = new_tableTP_3.values.tolist()
newTableCR_3 = numpy.zeros((5,12))
for i in range(5):
    for j in range(12):
        newTableCR_3[i][j] = (newTableHD[i][j]/newTableTP[i][j])*10000
new_tableCR_3 = pandas.DataFrame(newTableCR_3,columns =crudeRate )

Cluster_label = pandas.DataFrame({'Cluster_label':k_label[4]}) 
table = trainDataH.join(Cluster_label)
tableTP = trainDataTP.join(Cluster_label)
new_table_4 = table.groupby(['Cluster_label']).sum()
new_tableTP_4 = tableTP.groupby(['Cluster_label']).sum()
newTableHD = new_table_4.values.tolist()
newTableTP = new_tableTP_4.values.tolist()
newTableCR_4 = numpy.zeros((6,12))
for i in range(6):
    for j in range(12):
        newTableCR_4[i][j] = (newTableHD[i][j]/newTableTP[i][j])*10000
new_tableCR_4 = pandas.DataFrame(newTableCR_4,columns =crudeRate )

Cluster_label = pandas.DataFrame({'Cluster_label':k_label[5]}) 
table = trainDataH.join(Cluster_label)
tableTP = trainDataTP.join(Cluster_label)
new_table_5 = table.groupby(['Cluster_label']).sum()
new_tableTP_5 = tableTP.groupby(['Cluster_label']).sum()
newTableHD = new_table_5.values.tolist()
newTableTP = new_tableTP_5.values.tolist()
newTableCR_5 = numpy.zeros((7,12))
for i in range(7):
    for j in range(12):
        newTableCR_5[i][j] = (newTableHD[i][j]/newTableTP[i][j])*10000
new_tableCR_5 = pandas.DataFrame(newTableCR_5,columns =crudeRate )


Cluster_label = pandas.DataFrame({'Cluster_label':k_label[6]}) 
table = trainDataH.join(Cluster_label)
tableTP = trainDataTP.join(Cluster_label)
new_table_6 = table.groupby(['Cluster_label']).sum()
new_tableTP_6 = tableTP.groupby(['Cluster_label']).sum()
newTableHD = new_table_6.values.tolist()
newTableTP = new_tableTP_6.values.tolist()
newTableCR_6 = numpy.zeros((8,12))
for i in range(8):
    for j in range(12):
        newTableCR_6[i][j] = (newTableHD[i][j]/newTableTP[i][j])*10000
new_tableCR_6 = pandas.DataFrame(newTableCR_6,columns =crudeRate )


Cluster_label = pandas.DataFrame({'Cluster_label':k_label[7]}) 
table = trainDataH.join(Cluster_label)
tableC = trainDataC.join(Cluster_label)
tableTP = trainDataTP.join(Cluster_label)
new_table_7 = table.groupby(['Cluster_label']).sum()
new_tableC_7 = tableC.groupby(['Cluster_label']).sum()
new_tableTP_7 = tableTP.groupby(['Cluster_label']).sum()
newTableHD = new_table_7.values.tolist()
newTableTP = new_tableTP_7.values.tolist()
newTableCR_7 = numpy.zeros((9,12))
for i in range(9):
    for j in range(12):
        newTableCR_7[i][j] = (newTableHD[i][j]/newTableTP[i][j])*10000
new_tableCR_7 = pandas.DataFrame(newTableCR_7,columns =crudeRate )

Cluster_label = pandas.DataFrame({'Cluster_label':k_label[8]}) 
table = trainDataH.join(Cluster_label)
tableTP = trainDataTP.join(Cluster_label)
new_table_8 = table.groupby(['Cluster_label']).sum()
new_tableTP_8 = tableTP.groupby(['Cluster_label']).sum()
newTableHD = new_table_8.values.tolist()
newTableTP = new_tableTP_8.values.tolist()
newTableCR_8 = numpy.zeros((10,12))
for i in range(10):
    for j in range(12):
        newTableCR_8[i][j] = (newTableHD[i][j]/newTableTP[i][j])*10000
new_tableCR_8 = pandas.DataFrame(newTableCR_8,columns =crudeRate )


AnnualTP = trainDataTP.sum().values.tolist()
AnnualHD = trainDataH.sum().values.tolist()
AnnualCD = numpy.zeros(12)
for i in range(12):
    AnnualCD[i] = AnnualHD[i]/AnnualTP[i]*10000
year = numpy.asarray([2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011])
   
plt.figure(figsize=(6,6))
plt.title('KCluster = 2')
plt.plot(year, AnnualCD, marker = 'o',label = 'aunnal',
         color = 'red', linestyle = 'solid', linewidth = 1, markersize = 6)
plt.plot(year, newTableCR_0[0], marker = 'o',
         color = 'C0', linestyle = 'solid', linewidth = 1, markersize = 6)
plt.plot(year, newTableCR_0[1], marker = 'o',
         color = 'C1', linestyle = 'solid', linewidth = 1, markersize = 6)
plt.xlabel("year")
plt.ylabel("Crude Hospitaliztion rate")
#plt.xticks(numpy.arange(1, 12, step = 1))
plt.show()    

plt.figure(figsize=(6,6))
plt.title('KCluster = 3')
plt.plot(year, AnnualCD, marker = 'o',
         color = 'red', linestyle = 'solid', linewidth = 1, markersize = 6)
plt.plot(year, newTableCR_1[0], marker = 'o',
         color = 'C0', linestyle = 'solid', linewidth = 1, markersize = 6)
plt.plot(year, newTableCR_1[1], marker = 'o',
         color = 'C1', linestyle = 'solid', linewidth = 1, markersize = 6)
plt.plot(year, newTableCR_1[2], marker = 'o',
         color = 'C2', linestyle = 'solid', linewidth = 1, markersize = 6)
plt.xlabel("year")
plt.ylabel("Crude Hospitaliztion rate")
plt.show()     
    
 
plt.figure(figsize=(6,6))
plt.title('KCluster = 4')
plt.plot(year, AnnualCD, marker = 'o',
         color = 'red', linestyle = 'solid', linewidth = 1, markersize = 6)
plt.plot(year, newTableCR_2[0], marker = 'o',
         color = 'C0', linestyle = 'solid', linewidth = 1, markersize = 6)
plt.plot(year, newTableCR_2[1], marker = 'o',
         color = 'C1', linestyle = 'solid', linewidth = 1, markersize = 6)
plt.plot(year, newTableCR_2[2], marker = 'o',
         color = 'C2', linestyle = 'solid', linewidth = 1, markersize = 6)
plt.plot(year, newTableCR_2[3], marker = 'o',
         color = 'C3', linestyle = 'solid', linewidth = 1, markersize = 6)
plt.xlabel("year")
plt.ylabel("Crude Hospitaliztion rate")
plt.show()

plt.figure(figsize=(6,6))
plt.title('KCluster = 5')
plt.plot(year, AnnualCD, marker = 'o',
         color = 'red', linestyle = 'solid', linewidth = 1, markersize = 6)
plt.plot(year, newTableCR_3[0], marker = 'o',
         color = 'C0', linestyle = 'solid', linewidth = 1, markersize = 6)
plt.plot(year, newTableCR_3[1], marker = 'o',
         color = 'C1', linestyle = 'solid', linewidth = 1, markersize = 6)
plt.plot(year, newTableCR_3[2], marker = 'o',
         color = 'C2', linestyle = 'solid', linewidth = 1, markersize = 6)
plt.plot(year, newTableCR_3[3], marker = 'o',
         color = 'C3', linestyle = 'solid', linewidth = 1, markersize = 6)
plt.plot(year, newTableCR_3[4], marker = 'o',
         color = 'C4', linestyle = 'solid', linewidth = 1, markersize = 6)
plt.xlabel("year")
plt.ylabel("Crude Hospitaliztion rate")
plt.show()

plt.figure(figsize=(6,6))
plt.title('KCluster = 6')
plt.plot(year, AnnualCD, marker = 'o',
         color = 'red', linestyle = 'solid', linewidth = 1, markersize = 6)
plt.plot(year, newTableCR_4[0], marker = 'o',
         color = 'C0', linestyle = 'solid', linewidth = 1, markersize = 6)
plt.plot(year, newTableCR_4[1], marker = 'o',
         color = 'C1', linestyle = 'solid', linewidth = 1, markersize = 6)
plt.plot(year, newTableCR_4[2], marker = 'o',
         color = 'C2', linestyle = 'solid', linewidth = 1, markersize = 6)
plt.plot(year, newTableCR_4[3], marker = 'o',
         color = 'C3', linestyle = 'solid', linewidth = 1, markersize = 6)
plt.plot(year, newTableCR_4[4], marker = 'o',
         color = 'C4', linestyle = 'solid', linewidth = 1, markersize = 6)
plt.plot(year, newTableCR_4[5], marker = 'o',
         color = 'C5', linestyle = 'solid', linewidth = 1, markersize = 6)
plt.xlabel("year")
plt.ylabel("Crude Hospitaliztion rate")
plt.show()

plt.figure(figsize=(6,6))
plt.title('KCluster = 7')
plt.plot(year, AnnualCD, marker = 'o',
         color = 'red', linestyle = 'solid', linewidth = 1, markersize = 6)
plt.plot(year, newTableCR_5[0], marker = 'o',
         color = 'C0', linestyle = 'solid', linewidth = 1, markersize = 6)
plt.plot(year, newTableCR_5[1], marker = 'o',
         color = 'C1', linestyle = 'solid', linewidth = 1, markersize = 6)
plt.plot(year, newTableCR_5[2], marker = 'o',
         color = 'C2', linestyle = 'solid', linewidth = 1, markersize = 6)
plt.plot(year, newTableCR_5[3], marker = 'o',
         color = 'C3', linestyle = 'solid', linewidth = 1, markersize = 6)
plt.plot(year, newTableCR_5[4], marker = 'o',
         color = 'C4', linestyle = 'solid', linewidth = 1, markersize = 6)
plt.plot(year, newTableCR_5[5], marker = 'o',
         color = 'C5', linestyle = 'solid', linewidth = 1, markersize = 6)
plt.plot(year, newTableCR_5[6], marker = 'o',
         color = 'C6', linestyle = 'solid', linewidth = 1, markersize = 6)
plt.xlabel("year")
plt.ylabel("Crude Hospitaliztion rate")
plt.show()

plt.figure(figsize=(6,6))
plt.title('KCluster = 8')
plt.plot(year, AnnualCD, marker = 'o',
         color = 'red', linestyle = 'solid', linewidth = 1, markersize = 6)
plt.plot(year, newTableCR_6[0], marker = 'o',
         color = 'C0', linestyle = 'solid', linewidth = 1, markersize = 6)
plt.plot(year, newTableCR_6[1], marker = 'o',
         color = 'C1', linestyle = 'solid', linewidth = 1, markersize = 6)
plt.plot(year, newTableCR_6[2], marker = 'o',
         color = 'C2', linestyle = 'solid', linewidth = 1, markersize = 6)
plt.plot(year, newTableCR_6[3], marker = 'o',
         color = 'C3', linestyle = 'solid', linewidth = 1, markersize = 6)
plt.plot(year, newTableCR_6[4], marker = 'o',
         color = 'C4', linestyle = 'solid', linewidth = 1, markersize = 6)
plt.plot(year, newTableCR_6[5], marker = 'o',
         color = 'C5', linestyle = 'solid', linewidth = 1, markersize = 6)
plt.plot(year, newTableCR_6[6], marker = 'o',
         color = 'C6', linestyle = 'solid', linewidth = 1, markersize = 6)
plt.plot(year, newTableCR_6[7], marker = 'o',
         color = 'C7', linestyle = 'solid', linewidth = 1, markersize = 6)
plt.xlabel("year")
plt.ylabel("Crude Hospitaliztion rate")
plt.show()

plt.figure(figsize=(6,6))
plt.title('KCluster = 9')
plt.plot(year, AnnualCD, marker = 'o',
         color = 'red', linestyle = 'solid', linewidth = 1, markersize = 6)
plt.plot(year, newTableCR_7[0], marker = 'o',
         color = 'C0', linestyle = 'solid', linewidth = 1, markersize = 6)
plt.plot(year, newTableCR_7[1], marker = 'o',
         color = 'C1', linestyle = 'solid', linewidth = 1, markersize = 6)
plt.plot(year, newTableCR_7[2], marker = 'o',
         color = 'C2', linestyle = 'solid', linewidth = 1, markersize = 6)
plt.plot(year, newTableCR_7[3], marker = 'o',
         color = 'C3', linestyle = 'solid', linewidth = 1, markersize = 6)
plt.plot(year, newTableCR_7[4], marker = 'o',
         color = 'C4', linestyle = 'solid', linewidth = 1, markersize = 6)
plt.plot(year, newTableCR_7[5], marker = 'o',
         color = 'C5', linestyle = 'solid', linewidth = 1, markersize = 6)
plt.plot(year, newTableCR_7[6], marker = 'o',
         color = 'C6', linestyle = 'solid', linewidth = 1, markersize = 6)
plt.plot(year, newTableCR_7[7], marker = 'o',
         color = 'C7', linestyle = 'solid', linewidth = 1, markersize = 6)
plt.plot(year, newTableCR_7[8], marker = 'o',
         color = 'C8', linestyle = 'solid', linewidth = 1, markersize = 6)
plt.xlabel("year")
plt.ylabel("Crude Hospitaliztion rate")
plt.show()

plt.figure(figsize=(6,6))
plt.title('KCluster = 10')
plt.plot(year, AnnualCD, marker = 'o',
         color = 'red', linestyle = 'solid', linewidth = 1, markersize = 6)
plt.plot(year, newTableCR_8[0], marker = 'o',
         color = 'C0', linestyle = 'solid', linewidth = 1, markersize = 6)
plt.plot(year, newTableCR_8[1], marker = 'o',
         color = 'C1', linestyle = 'solid', linewidth = 1, markersize = 6)
plt.plot(year, newTableCR_8[2], marker = 'o',
         color = 'C2', linestyle = 'solid', linewidth = 1, markersize = 6)
plt.plot(year, newTableCR_8[3], marker = 'o',
         color = 'C3', linestyle = 'solid', linewidth = 1, markersize = 6)
plt.plot(year, newTableCR_8[4], marker = 'o',
         color = 'C4', linestyle = 'solid', linewidth = 1, markersize = 6)
plt.plot(year, newTableCR_8[5], marker = 'o',
         color = 'C5', linestyle = 'solid', linewidth = 1, markersize = 6)
plt.plot(year, newTableCR_8[6], marker = 'o',
         color = 'C6', linestyle = 'solid', linewidth = 1, markersize = 6)
plt.plot(year, newTableCR_8[7], marker = 'o',
         color = 'C7', linestyle = 'solid', linewidth = 1, markersize = 6)
plt.plot(year, newTableCR_8[8], marker = 'o',
         color = 'C8', linestyle = 'solid', linewidth = 1, markersize = 6)
plt.plot(year, newTableCR_8[9], marker = 'o',
         color = 'C9', linestyle = 'solid', linewidth = 1, markersize = 6)
plt.xlabel("year")
plt.ylabel("Crude Hospitaliztion rate")
plt.show()   


plt.plot(Nclusters, Elbow, linewidth = 2, marker = 'o', markersize = 6)
plt.grid(True)
plt.xlabel("Number of Clusters")
plt.ylabel("Elbow Value")
plt.xticks(numpy.arange(2, 11, step = 1))
plt.show()

plt.plot(Nclusters, Silhouette, linewidth = 2, marker = 'o',markersize = 6)
plt.grid(True)
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Value")
plt.xticks(numpy.arange(2, 11, step = 1))
plt.show() 

acceleration = numpy.zeros(9)
slop = numpy.zeros(9)
for i in range(0,8):
    slop[i+1] = (Elbow[i]-Elbow[i+1])/-1.0
slop =slop.tolist()
for i in range(0,7):
    acceleration[i+2] = slop[i+1]-slop[i]   
         
chart = pandas.DataFrame({'Nclusters':Nclusters,'Elbow_value':Elbow,'Silhouette_value':Silhouette,'slop':slop,'acceleration':acceleration})
print(f"11.a the Elbow and Silhouette chart is \n{chart},according to the chart  you can find when the kcluters is 6 the acceleration is biggest ")

kmeans_2 = cluster.KMeans(n_clusters=6, random_state=20190405).fit(traindataC)

trainData_Community = data[['Community']]
trainData_Community['K_label'] = kmeans_2.labels_
Ncommunity = trainData_Community.shape[0]
Cluster_0 = []
Cluster_1 = []
Cluster_2 = []
Cluster_3 = []
Cluster_4 = []
Cluster_5 = []
for i in range(Ncommunity):
    if trainData_Community.iloc[i,1] == 0:
        Cluster_0.append(trainData_Community.iloc[i,0])
    if trainData_Community.iloc[i,1] == 1:
        Cluster_1.append(trainData_Community.iloc[i,0])
    if trainData_Community.iloc[i,1] == 2:
        Cluster_2.append(trainData_Community.iloc[i,0])
    if trainData_Community.iloc[i,1] == 3:
        Cluster_3.append(trainData_Community.iloc[i,0])
    if trainData_Community.iloc[i,1] == 4:
        Cluster_4.append(trainData_Community.iloc[i,0])
    if trainData_Community.iloc[i,1] == 5:
        Cluster_5.append(trainData_Community.iloc[i,0])
index = ['Cluster_0','Cluster_1','Cluster_2','Cluster_3','Cluster_4','Cluster_5']
Cluster = numpy.asarray([Cluster_0,Cluster_1,Cluster_2,Cluster_3,Cluster_4,Cluster_5])  
CommunityEachCluster = pandas.DataFrame({'Community':Cluster},index = index)