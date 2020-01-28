#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 14:45:32 2019

@author: zhengleo
"""

import numpy
import pandas
import random
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import sklearn.model_selection as model_selection
import statsmodels.api as stats
import matplotlib.patches as mpatches


data = pandas.read_csv('FiveRing.csv')
y = data['ring'].astype('category')
X = stats.add_constant(data[['x','y']],prepend = True)
logit = stats.MNLogit(y,X)
print("Name of Target Variable:", logit.endog_names)
print("Name(s) of Predictors:", logit.exog_names)
thisFit = logit.fit(method='newton', full_output = True, maxiter = 1000, tol = 1e-8)
thisParameter = thisFit.params.round(4)
print("1.a list the parameter estimates in a table:")
print(round(thisParameter,4))
y_predProb = thisFit.predict(X)
y_predProb_list = y_predProb.values.tolist()
y_list = []
Y_list = []
for i in range(len(y_predProb_list)):
    max = 0
    for i in y_predProb_list[i]:
        if i >= max:
            max = i
    y_list.append(max)
    
for j in range(len(y_predProb_list)):
    for i in range(len(y_predProb_list[j])):
        if y_predProb_list[j][i] == y_list[j]:
            Y_list.append(i)
data['Predict_ring'] = Y_list
MR_list = data[['Predict_ring','ring']].values.tolist()
MissclassificationRate = 0
for i in MR_list:
    if i[0]!=i[1]:
        MissclassificationRate+=1
MissclassificationRate = MissclassificationRate/20010
print(f"1.b the Misclassification rate is {MissclassificationRate}")
RASE_L = 0.0
for i in range(20010):
    ring = MR_list[i][1]
    predProb = y_predProb_list[i]
    for j in range(5):
        if j!=ring:
            RASE_L+=(0-predProb[j])**2
        else:
            RASE_L+=(1-predProb[j])**2
RASE_L = numpy.sqrt(RASE_L/(2*20010))
print(f"1.c the Root Average Squared error is {RASE_L} ")
PredicRing = data[['Predict_ring']].values.tolist()
color =[]
for i in PredicRing:
    if i[0] == 0:
        color.append('Orange')
    if i[0]==1:
        color.append('Green')
    if i[0]==2:
        color.append('Blue')
    if i[0] ==3:
        color.append('Black')
    if i[0] ==4:
        color.append('Red')
        
orange = mpatches.Patch(color = 'Orange', label = '0')
green = mpatches.Patch(color = 'Green', label = '1')
blue = mpatches.Patch(color = 'Blue', label = '2')
black = mpatches.Patch(color = 'Black', label = '3')
red = mpatches.Patch(color = 'Red', label = '4')
print(f"1.d the picture below:")
plt.figure(figsize=(16,9))
plt.legend(handles = [orange,green,blue,black,red])        
plt.scatter(data[['x']], data[['y']], c = color)
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()


def sample_wr (inData):
    n = len(inData)
    outData = numpy.empty((n,1))
    for i in range(n):
        j = int(random.random() * n)
        outData[i] = inData[j]
    return outData

def bootstrap_MNLogit (x_train, y_train, nB):
   x_index = x_train.index
   nT = len(y_train)
   outProb = numpy.zeros((nT,5))
   #outThreshold = numpy.zeros((nB, 1))
   #classTree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=2, random_state=60616)

   # Initialize internal state of the random number generator.
   random.seed(20190430)

   for iB in range(nB):
      bootIndex = sample_wr(x_index)
      x_train_boot = x_train.loc[bootIndex[:,0]]
      y_train_boot = y_train.loc[bootIndex[:,0]]
      #outThreshold[iB] = len(y_train_boot[y_train_boot['BAD'] == 1]) / len(y_train_boot)
      y_train_boot = y_train_boot['ring'].astype('category')
  
      logit = stats.MNLogit(y_train_boot,x_train_boot)
      thisFit = logit.fit(method='newton', full_output = True, maxiter = 1000, tol = 1e-8)
      #treeFit = classTree.fit(x_train_boot, y_train_boot['BAD'])
      outProb = outProb + thisFit.predict(x_train)
   outProb = outProb / nB
   #print('Mean Threshold: {:.7f}' .format(outThreshold.mean()))
   #print('  SD Threshold: {:.7f}' .format(outThreshold.std()))
   return outProb




def RASE_MissClassification(y_predProb,data):
     y_predProb_list = y_predProb.values.tolist()
     y_list = []
     Y_list = []
     for i in range(len(y_predProb_list)):
         max = 0
         for i in y_predProb_list[i]:
             if i >= max:
                 max = i
         y_list.append(max)
    
     for j in range(len(y_predProb_list)):
        for i in range(len(y_predProb_list[j])):
            if y_predProb_list[j][i] == y_list[j]:
                Y_list.append(i)
     data['Predict_ring'] = Y_list
     MR_list = data[['Predict_ring','ring']].values.tolist()
     MissclassificationRate = 0
     for i in MR_list:
         if i[0]!=i[1]:
             MissclassificationRate+=1
     MissclassificationRate = MissclassificationRate/20010

     RASE_L = 0.0
     for i in range(20010):
         ring = MR_list[i][1]
         predProb = y_predProb_list[i]
         for j in range(5):
             if j!=ring:
                 RASE_L+=(0-predProb[j])**2
             else:
                 RASE_L+=(1-predProb[j])**2
     RASE_L = numpy.sqrt(RASE_L/(2*20010))
    
     return RASE_L, MissclassificationRate

y_train = data[['ring']]
x_train = X


nB_list = [10,20,30,40,50,60,70,80,90,100]
RASE_MissClass = []
for i in nB_list:
    data1 = pandas.read_csv('FiveRing.csv')
    MNLogitPredProb = bootstrap_MNLogit (x_train, y_train, i)
    RASE, MisClassRate = RASE_MissClassification(MNLogitPredProb,data1)
    restore = []
    restore.append(RASE)
    restore.append(MisClassRate)
    RASE_MissClass.append(restore)
    print('      Number of Bootstraps: ', i)
    print('    Misclassification Rate: {:.7f}' .format(MisClassRate))
    print('Root Average Squared Error: {:.7f}' .format(RASE))
    
    PredicRing = data1[['Predict_ring']].values.tolist()
    color =[]
    for i in PredicRing:
        if i[0] == 0:
            color.append('Orange')
        if i[0]==1:
            color.append('Green')
        if i[0]==2:
            color.append('Blue')
        if i[0] ==3:
            color.append('Black')
        if i[0] ==4:
            color.append('Red')
               
    plt.figure(figsize=(16,9))
    orange = mpatches.Patch(color = 'Orange', label = '0')
    green = mpatches.Patch(color = 'Green', label = '1')
    blue = mpatches.Patch(color = 'Blue', label = '2')
    black = mpatches.Patch(color = 'Black', label = '3')
    red = mpatches.Patch(color = 'Red', label = '4')
    print("1.f redraw the picture")
    plt.legend(handles = [orange,green,blue,black,red]) 
    #plt.title(f'nb = {i}')
    plt.scatter(data1[['x']], data1[['y']], c = color)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.show()

nb_list = [0,10,20,30,40,50,60,70,80,90,100]
misClasRate_list = [MissclassificationRate]
RASE_list = [RASE_L]
for i in RASE_MissClass:
    misClasRate_list.append(i[1])
    RASE_list.append(i[0])
ModelMetric = pandas.DataFrame({'Number of Bootestraps':nb_list,'Root Average Squared Error':RASE_list,'Misclassification Rate':misClasRate_list})
print(f"1.e List the Misclassification Rate and the Root Average Squared Error of the bootstrap results:{ModelMetric }")

















            
    
   