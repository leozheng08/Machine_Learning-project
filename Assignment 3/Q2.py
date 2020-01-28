#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 16:23:26 2019

@author: zhengleo
"""
import matplotlib.pyplot as plt
import numpy
import pandas
import sklearn.neural_network as nn
import sklearn.metrics as metrics


X1 = numpy.asarray([0,0,1,1])
X2 = numpy.asarray([0,1,0,1])
Y = numpy.asarray([0,1,1,1])
Y_AND = numpy.asarray([0,0,0,1])
X = pandas.DataFrame({'X1':X1,'X2':X2, 'Y':Y})


plt.scatter(X[['X1']], X[['X2']], c = X[['Y']].values)
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()

nnObj = nn.MLPClassifier(hidden_layer_sizes = 1*1,
                            activation = 'relu',solver = 'lbfgs', verbose = True,
                            max_iter = 1000, random_state = 20190403)
x = X[['X1', 'X2']]
y = X[['Y']]
fit_nn = nnObj.fit(x, y) 
weight = nnObj.coefs_
pred_nn = nnObj.predict(x)

print('Output Activation Function:', nnObj.out_activation_)
print(' Mean Accuracy:', nnObj.score(x, y))

X['Y'] = pred_nn.copy()
plt.scatter(X[['X1']], X[['X2']], c = X[['Y']].values)
plt.xlabel('X1')
plt.ylabel('X2')
#plt.title("%d Hidden Layers, %d Hidden Neurons" % (1, 1))
#plt.legend(fontsize = 12, markerscale = 3)
plt.grid(True)
plt.show()





X_AND = pandas.DataFrame({'X1':X1,'X2':X2, 'Y':Y_AND})
plt.scatter(X_AND[['X1']], X_AND[['X2']], c = X_AND[['Y']].values)
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()

nnObj = nn.MLPClassifier(hidden_layer_sizes = 1*1,
                            activation = 'relu',solver = 'lbfgs', verbose = True,
                            max_iter = 1000, random_state = 20190403)
x = X_AND[['X1', 'X2']]
y = X_AND[['Y']]
fit_nn = nnObj.fit(x, y)
biasAND = nnObj.intercepts_  
weightAND = nnObj.coefs_
pred_nn = nnObj.predict(x)

print('Output Activation Function:', nnObj.out_activation_)
print(' Mean Accuracy:', nnObj.score(x, y))

X_AND['Y'] = pred_nn.copy()
plt.scatter(X_AND[['X1']], X_AND[['X2']], c = X_AND[['Y']].values)
plt.xlabel('X1')
plt.ylabel('X2')
#plt.title("%d Hidden Layers, %d Hidden Neurons" % (1, 1))
#plt.legend(fontsize = 12, markerscale = 3)
plt.grid(True)
plt.show()


X1 = numpy.asarray([0,0,1,1])
X2 = numpy.asarray([0,1,0,1])
Y_XAND = numpy.asarray([1,0,0,1])
X_XAND = pandas.DataFrame({'X1':X1,'X2':X2,'Y':Y_XAND})
plt.scatter(X_XAND[['X1']], X_XAND[['X2']], c = X_XAND[['Y']].values)
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()

nnObj = nn.MLPClassifier(hidden_layer_sizes = 2*1,
                            activation = 'relu',solver = 'lbfgs', verbose = True,
                            max_iter = 1000, random_state = 20190403)
x = X_XAND[['X1', 'X2']]
y = X_XAND[['Y']]
fit_nn = nnObj.fit(x, y)
biasXAND = nnObj.intercepts_ 
weightXAND = nnObj.coefs_
pred_nn = nnObj.predict(x)

print('Output Activation Function:', nnObj.out_activation_)
print(' Mean Accuracy:', nnObj.score(x, y))

X_AND['Y'] = pred_nn.copy()
plt.scatter(X_AND[['X1']], X_AND[['X2']], c = X_AND[['Y']].values)
plt.xlabel('X1')
plt.ylabel('X2')
#plt.title("%d Hidden Layers, %d Hidden Neurons" % (1, 1))
#plt.legend(fontsize = 12, markerscale = 3)
plt.grid(True)
plt.show()

print("2.a below is the proof function of X1 OR X2")
def PerceptronOR(x1,x2,p):
    y = []
    for i in range(len(x1)):
        if x1[i]*p[0]+x2[i]*p[1]>=p[2]:
            y.append(1)
        else:
            y.append(0)
    return y
X1 = numpy.asarray([0,0,1,1])
X2 = numpy.asarray([0,1,0,1])
X = numpy.asarray([X1,X2])
POR = numpy.asarray([1,1,1])
Y = PerceptronOR(X1,X2,POR)

print("2.b below is the proof function of X1 and X2")
def PerceptronAND(x1,x2,p):
    y = []
    for i in range(len(x1)):
        if x1[i]*p[0]+x2[i]*p[1]>=p[2]:
            y.append(1)
        else:
            y.append(0)
    return y

PAND = numpy.asarray([1,1,2])
Y = PerceptronAND(X1,X2,PAND)

print("2.c below is the proof function of X1 Xand X2")
def PerceptronXAND(x1,x2,p):
    h1=[]
    h2 =[]
    y= []
    for i in range(4):
        if x1[i]*p[0]+x2[i]*p[1]>=p[2]:
            h1.append(1)
        else:
            h1.append(0)
        if x1[i]*p[3]+x2[i]*p[4]>=p[5]:
            h2.append(1)
        else:
            h2.append(0)
    for i in range(4):
        if h1[i]*p[6]+h2[i]*p[7]>=p[8]:
            y.append(1)
        else:
            y.append(0)
    return h2
        

PXAND = numpy.asarray([-1,1,0,1,-1,0,1,1,2])
H2 = PerceptronAND(X1,X2,PXAND)













