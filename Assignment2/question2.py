#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 15:00:23 2019

@author: zhengleo
"""
import math
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import statsmodels.api as stats

def TargetPercentByNominal (
   targetVar,       # target variable
   predictor,rownames,colnames):      # nominal predictor

   countTable = pd.crosstab(index = predictor, columns = targetVar, rownames = rownames, colnames = colnames,margins = True, dropna = True)
   x = countTable.drop('All', 1)
   percentTable = countTable.div(x.sum(1), axis='index')*100

   print("Frequency Table: \n")
   print(countTable)
   print( )
   print("Percent Table: \n")
   print(percentTable)

   return countTable,percentTable


data = pd.read_csv('Purchase_Likelihood.csv')
data_intercept = data.groupby('A').size()



print(f"2.a) the number of parameters in this model is {2}")

print(f"2.b) the marginal counts of the categories of the target variable A are separately category 0: {data_intercept.iloc[0]},category 1: {data_intercept.iloc[1]}, category 2 : {data_intercept.iloc[2]}")

data_c = data[['A']]
crossTable = pd.crosstab(0, columns = data_c['A'], margins = True, dropna = True)
p_list = []
for i in range(0,3): 
    p_list.append(crossTable.iloc[0,i]/(crossTable.iloc[0,0]+crossTable.iloc[0,1]+crossTable.iloc[0,2]))

print(f"2.c) the maximum likelihood estimates of the predicted probabilities are separately {p_list[0]}, {p_list[1]}, {p_list[2]}")
    
betas_0 = math.log(p_list[0]/p_list[0])
betas_1 = math.log(p_list[1]/p_list[0])
betas_2 = math.log(p_list[2]/p_list[0])
print(f"2.e) beta 0 is {betas_0}, beta 1 is {betas_1}, beta 2 is {betas_2}")
    
max_likelihood = 0

for i in range(0,3):
    max_likelihood = max_likelihood+crossTable.iloc[0,i]*(math.log(p_list[i]))
print(f"2.d) Model Log-Likelihood Value: {max_likelihood}")




contingency_row = data.iloc[:,0:3]
row_group_size = contingency_row['group_size']
row_homeowner = contingency_row['homeowner']
row_married_couple = contingency_row['married_couple']
contingency_column = data.iloc[:,3]
print(f"2.f) the contingency table is belowing:")
countTable2,contingencyTable = TargetPercentByNominal(contingency_column,[row_group_size,row_homeowner,row_married_couple],['group_size','homeowner','married_couple'],['A'])
print(contingencyTable)





y = data['A'].astype('category')
y_category = y.cat.categories

X= data[['group_size','homeowner','married_couple']].astype('category')
X = pd.get_dummies(X)
X = stats.add_constant(X, prepend=True)

logit = stats.MNLogit(y, X)
print("Name of Target Variable:", logit.endog_names)
print("Name(s) of Predictors:", logit.exog_names)

thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
thisParameter = thisFit.params

print("Model Parameter Estimates:\n", thisFit.params)
print("Model Log-Likelihood Value:\n", logit.loglike(thisParameter.values))

max_pr_1_0 =math.exp(thisParameter.loc['const',0]+thisParameter.loc['group_size_2',0]+thisParameter.loc['homeowner_1',0]+thisParameter.loc['married_couple_1',0])
print(f"2.i) the values of group_size, homeowner, and married_couple such that the odd Prob(A=1)/Prob(A = 0) will attain its maximum")
print(f"is group_size=1,homeowner=1,married_couple=1,the maximum odd Prob(A = 1)/Prob(A = 0) value is {max_pr_1_0}")

odd_ratio =math.exp(thisParameter.loc['group_size_3',1]-thisParameter.loc['group_size_1',1])
print(f"2.j)the odds ratio for group_size = 3 versus group_size = 1, and A = 2 versus A = 0 is {odd_ratio}")

odd_ratio1 =math.exp((thisParameter.loc['group_size_1',1]-thisParameter.loc['group_size_1',0])-(thisParameter.loc['group_size_3',1]-thisParameter.loc['group_size_3',0]))
print(f"2.k)the odds ratio for group_size = 1 versus group_size = 3, and A = 2 versus A = 1 is {odd_ratio1}")