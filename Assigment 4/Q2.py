#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 15:35:41 2019

@author: zhengleo
"""

import graphviz
import matplotlib.pyplot as plt
import numpy
import pandas
import random
import sklearn.metrics as metrics
import sklearn.tree as tree
import math
import matplotlib.patches as mpatches

data = pandas.read_csv('FiveRing.csv')
x_train = data[['x','y']]
y_train = data[['ring']]
classTree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=2, random_state=20190415)
treeFit = classTree.fit(x_train, y_train['ring'])
treePredProb = classTree.predict_proba(x_train)

def RASE_MissClassification(y_predProb,data):
     y_predProb_list = y_predProb
     y_list = []
     Y_list = []
     for j in range(len(y_predProb_list)):
         y_list.append(numpy.amax(y_predProb_list[j]))
         for i in range(len(y_predProb_list[j])):
            if y_predProb_list[j][i] == y_list[j]:
                Y_list.append(i)
     data['Predict_ring'] = Y_list
     MR_list = data[['Predict_ring','ring']].values.tolist()
     MissclassificationRate = 0
     RASE_L = 0.0
     for i in range(len(MR_list)):
         if MR_list[i][0]!=MR_list[i][1]:
             MissclassificationRate+=1
         ring = MR_list[i][1]
         predProb = y_predProb_list[i]
         for j in range(5):
             if j!=ring:
                 RASE_L+=(0-predProb[j])**2
             else:
                 RASE_L+=(1-predProb[j])**2
     MissclassificationRate = MissclassificationRate/20010
     RASE_L = numpy.sqrt(RASE_L/(2*20010))
     return RASE_L, MissclassificationRate
       
     #RASE_L = 0.0
     #for i in range(20010):
         #ring = MR_list[i][1]
         #predProb = y_predProb_list[i]
         #for j in range(5):
             #if j!=ring:
                 #RASE_L+=(0-predProb[j])**2
             #else:
                 #RASE_L+=(1-predProb[j])**2
     #RASE_L = numpy.sqrt(RASE_L/(2*20010))
    
     #return RASE_L, MissclassificationRate
 
RASE, MisClassRate = RASE_MissClassification(treePredProb,data)
print(f"2.a the Misclassification Rate is {MisClassRate}")
print(f"2.b the Root Average Squared Error is {RASE}")

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
        
        
plt.figure(figsize=(16,9))
orange = mpatches.Patch(color = 'Orange', label = '0')
green = mpatches.Patch(color = 'Green', label = '1')
blue = mpatches.Patch(color = 'Blue', label = '2')
black = mpatches.Patch(color = 'Black', label = '3')
red = mpatches.Patch(color = 'Red', label = '4')
plt.legend(handles = [orange,green,blue,black,red]) 
plt.scatter(data[['x']], data[['y']], c = color)
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()

w_train = []
for i in range(20010):
    w_train.append(1.0)
w_train = numpy.array(w_train)



def iteration(iteration,data_1):
    accuracy_list = []
    treePredProb_list = []
    for iter in range(iteration) :
        classTree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=2, random_state=20190415)
        treeFit = classTree.fit(x_train, y_train['ring'], w_train)
        treePredProb = classTree.predict_proba(x_train)
        accuracy = classTree.score(x_train, y_train, w_train)
        #print(f'{iter} Accuracy = ', accuracy)
        accuracy_list.append(accuracy)
        treePredProb_list.append(treePredProb)
        y_train_list = y_train['ring'].values.tolist()
        y_train_arr = numpy.array(y_train)
        eventError = numpy.empty((20010,5))
        eventErrorAverage = numpy.empty((20010,1))
        predClass = numpy.empty((20010,1))
    
        for i in range(20010):
            
            max_v = numpy.amax(treePredProb[i])
            for j in range(5):
                if treePredProb[i][j] == max_v:
                    predClass[i] = j
                if y_train_arr[i]==j :
                    for k in range(5):
                        if k != j:
                            eventError[i][k] = 0 - treePredProb[i,k]
                        else:
                            eventError[i][k] = 1 - treePredProb[i,k]

            for k in range(len(eventError[i])):
                eventErrorAverage[i] += numpy.abs(eventError[i][k])
            eventErrorAverage[i] = eventErrorAverage[i]*(1/5.0)

        for e in range(20010):         
            if (predClass[e][0] != y_train_arr[e][0]):
                w_train[e] = 1.0 + eventErrorAverage[e]
            else:
                w_train[e] = eventErrorAverage[e]
                
        RASE_iter, MisClassRate_iter = RASE_MissClassification(treePredProb,data_1)
        if MisClassRate_iter == 0:
            break
    print('      Number of iteration: ', iteration)
    totalAccuracy = 0
    for i in range(iteration):
        totalAccuracy = totalAccuracy+accuracy_list[i]
    #for i in range(iteration):
        #accuracy_list[i]  =  accuracy_list[i]/totalAccuracy
    finalTreePredProb = treePredProb_list[0]*accuracy_list[0]
    for i in range(1,iteration):
        finalTreePredProb+=treePredProb_list[i]*accuracy_list[i]
    finalTreePredProb =finalTreePredProb/totalAccuracy
    RASE_iter_final, MisClassRate_iter_final = RASE_MissClassification(finalTreePredProb,data_1)
    #print('    Misclassification Rate: {:.7f}' .format(MisClassRate_iter_final))
    #print('Root Average Squared Error: {:.7f}' .format(RASE_iter_final))
    for i in range(20010):
        max_v = numpy.amax(finalTreePredProb[i])
        for j in range(5):
            if treePredProb[i][j] == max_v:
                predClass[i] = j
    return RASE_iter_final, MisClassRate_iter_final,predClass

def draw(data1):
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
    plt.legend(handles = [orange,green,blue,black,red]) 
    plt.scatter(data1[['x']], data1[['y']], c = color)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.show()

iteration_list = [0,100,200,300,400,500,600,700,800,900,1000]
RASE_list = [RASE]
MisClassRate_list = [MisClassRate]
data_1 = data = pandas.read_csv('FiveRing.csv')



RASE_iter_1, MisClassRate_iter_1,predclass_1= iteration(iteration_list[1],data_1)
RASE_list.append(RASE_iter_1)
MisClassRate_list.append(MisClassRate_iter_1)
draw(data_1)

RASE_iter_2, MisClassRate_iter_2,predclass_list_2= iteration(iteration_list[2],data_1)
RASE_list.append(RASE_iter_2)
MisClassRate_list.append(MisClassRate_iter_2)
draw(data_1)

RASE_iter_3, MisClassRate_iter_3,predclass_list_3 = iteration(iteration_list[3],data_1)
RASE_list.append(RASE_iter_3)
MisClassRate_list.append(MisClassRate_iter_3)
draw(data_1)

RASE_iter_4, MisClassRate_iter_4,predclass_list_4 = iteration(iteration_list[4],data_1)
RASE_list.append(RASE_iter_4)
MisClassRate_list.append(MisClassRate_iter_4)
draw(data_1)

RASE_iter_5, MisClassRate_iter_5,predclass_list_5 = iteration(iteration_list[5],data_1)
RASE_list.append(RASE_iter_5)
MisClassRate_list.append(MisClassRate_iter_5)
draw(data_1)

RASE_iter_6, MisClassRate_iter_6,predclass_list_6 = iteration(iteration_list[6],data_1)
RASE_list.append(RASE_iter_6)
MisClassRate_list.append(MisClassRate_iter_6)
draw(data_1)

RASE_iter_7, MisClassRate_iter_7,predclass_list_7 = iteration(iteration_list[7],data_1)
RASE_list.append(RASE_iter_7)
MisClassRate_list.append(MisClassRate_iter_7)
draw(data_1)


RASE_iter_8, MisClassRate_iter_8,predclass_list_8 = iteration(iteration_list[8],data_1)
RASE_list.append(RASE_iter_8)
MisClassRate_list.append(MisClassRate_iter_8)
draw(data_1)

RASE_iter_9, MisClassRate_iter_9,predclass_list_9 = iteration(iteration_list[9],data_1)
RASE_list.append(RASE_iter_9)
MisClassRate_list.append(MisClassRate_iter_9)
draw(data_1)

RASE_iter_10, MisClassRate_iter_10,predclass_list_10 = iteration(iteration_list[10],data_1)
RASE_list.append(RASE_iter_10)
MisClassRate_list.append(MisClassRate_iter_10)
draw(data_1)



modelMatric = pandas.DataFrame({'the number of iterations':iteration_list,'the Misclassification Rate':MisClassRate_list,'the Root Average Squared Error':RASE_list})
    
            
            
    
    
    
    
    
    
