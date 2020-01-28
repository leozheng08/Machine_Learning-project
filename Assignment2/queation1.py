#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 14:26:48 2019

@author: zhengleo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('CustomerSurveyData.csv')
data = data[['CreditCard','JobCategory','CarOwnership']]
data.fillna('missing',inplace = True)
data.isna().sum()
#data = data.dropna()

X_inputs = data[['CreditCard','JobCategory']]
Y_targets = data[['CarOwnership']]

cat_X = X_inputs.astype('category')
X_dummy = pd.get_dummies(cat_X)
X_dummy_name = X_dummy.columns.values.tolist()

from sklearn import tree
classTree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=60616)
claimHistory_CART = classTree.fit(X_dummy, Y_targets)

import graphviz
dot_data = tree.export_graphviz(claimHistory_CART,
                                out_file=None,
                                impurity = True, filled = True,
                                feature_names = X_dummy_name,
                                class_names = ['Lease', 'None','Own'])

graph = graphviz.Source(dot_data)

graph

graph.render('claimHistory_CART')
graph.view('claimHistory_CART')


Credit_inputs = X_inputs['CreditCard']
Job_inputs = X_inputs['JobCategory']

crossTable = pd.crosstab([Credit_inputs,Job_inputs],Y_targets['CarOwnership'],rownames=['CreditCard','JobCategory'], colnames=['CarOwnership'], margins = True, dropna = True)   
print(crossTable)

Entropy_root_node = -((799/5000)*np.log2(799/5000)+(497/5000)*np.log2(497/5000)+(3704/5000)*np.log2(3704/5000))

print(f'1.a) the Entropy for the root node is {Entropy_root_node}')


     
     
     
cat_CreditCard = X_inputs[['CreditCard']].astype('category')
CreditCard_inputs = pd.get_dummies(cat_CreditCard)
inData = CreditCard_inputs.join(Y_targets)


def EntropyNominalSplit(inData,split,length):
    dataTable = inData
    
    def EntropyCalculate(table):
        crossTable = table
        nRows = crossTable.shape[0]
        nColumns = crossTable.shape[1]
        tableEntropy = 0
        for iRow in range(nRows-1):
            rowEntropy = 0
            for iColumn in range(nColumns):
                proportion = crossTable.iloc[iRow,iColumn] / crossTable.iloc[iRow,(nColumns-1)]
                if (proportion > 0):
                    rowEntropy -= proportion * np.log2(proportion)
            print('Row = ', iRow, 'Entropy =', rowEntropy)
            tableEntropy += rowEntropy *  crossTable.iloc[iRow,(nColumns-1)]
        tableEntropy = tableEntropy /  crossTable.iloc[(nRows-1),(nColumns-1)]
        print('Split Entropy = ', tableEntropy)
        print('..............................')
        return tableEntropy
            
    if length==5:
        Entropy_list=[]
        left_branch = []
        right_branch = []
        
        print("split propotion is 1:4 ")
        start =0
        right_string =''
        for i in range(start,length):
            print(f"split index is {list(dataTable.columns)[i]}")
            left_branch.append(list(dataTable.columns)[i])
            for x in range(len(dataTable.columns)):
                if(dataTable.columns[x]!=dataTable.columns[i]):
                    right_string = right_string+' '+dataTable.columns[x]
            right_branch.append(right_string)
            dataTable['LE_Split'] = (dataTable.iloc[:,i] <= split)
            crossTable = pd.crosstab(index = dataTable['LE_Split'], columns = dataTable.iloc[:,5], margins = True, dropna = True)   
            print(crossTable)
            tableEntropy = EntropyCalculate(crossTable)
            Entropy_list.append(tableEntropy)
        print("split propotion is 2:3 ")
        length = length-1
        right_string =''
        for i in range(start,length):
            start=start+1
            for j in range(start,5):
                print(f"split index is {list(dataTable.columns)[i]},{list(dataTable.columns)[j]}")
                left_branch.append(list(dataTable.columns)[i]+' '+list(dataTable.columns)[j])
                for x in range(len(dataTable.columns)):
                    if dataTable.columns[x]!=dataTable.columns[i] and dataTable.columns[x]!=dataTable.columns[j]:
                        right_string = right_string+' '+dataTable.columns[x]
                right_branch.append(right_string)
                dataTable['LE_Split'] = ((dataTable.iloc[:,i]+dataTable.iloc[:,j]) <= split)
                crossTable = pd.crosstab(index = dataTable['LE_Split'], columns = dataTable.iloc[:,5], margins = True, dropna = True)   
                print(crossTable)
                tableEntropy = EntropyCalculate(crossTable)
                Entropy_list.append(tableEntropy)
        return Entropy_list,left_branch,right_branch
        
                
    if length==7:
        Entropy_list=[]
        left_branch = []
        right_branch = []
        print("split propotion is 1:6 ")
        start = 0
        right_string =''
        for i in range(start,length):
            print(f"split index is {list(dataTable.columns)[i]}")
            left_branch.append(list(dataTable.columns)[i])
            for x in range(len(dataTable.columns)):
                if(dataTable.columns[x]!=dataTable.columns[i]):
                    right_string = right_string+' '+dataTable.columns[x]
            right_branch.append(right_string)
            dataTable['LE_Split'] = (dataTable.iloc[:,i] <= split)
            crossTable = pd.crosstab(index = dataTable['LE_Split'], columns = dataTable.iloc[:,7], margins = True, dropna = True)   
            print(crossTable)
            tableEntropy = EntropyCalculate(crossTable)
            Entropy_list.append(tableEntropy)
            
        print("split propotion is 2:5 ")
        length = length-1
        start = 0
        right_string =''
        for i in range(start,length):
            start=start+1
            for j in range(start,7):
                print(f"split index is {list(dataTable.columns)[i]},{list(dataTable.columns)[j]}")
                left_branch.append(list(dataTable.columns)[i]+' '+list(dataTable.columns)[j])
                for x in range(len(dataTable.columns)):
                    if dataTable.columns[x]!=dataTable.columns[i] and dataTable.columns[x]!=dataTable.columns[j]:
                        right_string = right_string+' '+dataTable.columns[x]
                right_branch.append(right_string)
                dataTable['LE_Split'] = ((dataTable.iloc[:,i]+dataTable.iloc[:,j]) <= split)
                crossTable = pd.crosstab(index = dataTable['LE_Split'], columns = dataTable.iloc[:,7], margins = True, dropna = True)   
                print(crossTable)
                tableEntropy = EntropyCalculate(crossTable)
                Entropy_list.append(tableEntropy)
                
        print("split propotion is 3:4 ")
        length = length-1
        start1 = 0
        right_string =''
        for i in range(start1,length):
            start1 = start1+1
            start = start1
            for j in range(start,length+1):
                start = start+1
                for k in range(start,length+2):
                    print(f"split index is {list(dataTable.columns)[i]},{list(dataTable.columns)[j]},{list(dataTable.columns)[k]}")
                    left_branch.append(list(dataTable.columns)[i]+' '+list(dataTable.columns)[j]+' '+list(dataTable.columns)[k])
                    for x in range(len(dataTable.columns)):
                        if dataTable.columns[x]!=dataTable.columns[i] and dataTable.columns[x]!=dataTable.columns[j] and dataTable.columns[x]!=dataTable.columns[k]:
                            right_string = right_string+' '+dataTable.columns[x]
                    right_branch.append(right_string)
                    dataTable['LE_Split'] = ((dataTable.iloc[:,i]+dataTable.iloc[:,j]+dataTable.iloc[:,k]) <= split)
                    crossTable = pd.crosstab(index = dataTable['LE_Split'], columns = dataTable.iloc[:,7], margins = True, dropna = True)   
                    print(crossTable)
                    tableEntropy = EntropyCalculate(crossTable)
                    Entropy_list.append(tableEntropy)
        return Entropy_list,left_branch,right_branch
                
                
        


total_binary_splits = 2**(5-1)-1
print(f"1.b) the numeber of possible binary-splits that you can generate from the CreditCard predictor is {total_binary_splits}")

print("1.c)  the Entropy metric is belowing:")    
Entropy_list,left_branch,right_branch = EntropyNominalSplit(inData,0.5,5)
branch = []
branch_string = ''
for i in range(len(left_branch)):
    branch_string = '{'+left_branch[i]+'}'+'{'+right_branch[i]+'}'
    branch.append(branch_string)
table = pd.DataFrame({'branch':branch,'table_Entropy':Entropy_list})
print(table)
print("1.d) the optimal split is [CreditCard_Others,CreditCard_Visa] and [CreditCard_American Express,CreditCard_Discover,CreditCard_MasterCard]")

cat_JobCategory = X_inputs[['JobCategory']].astype('category')
JobCategory_inputs = pd.get_dummies(cat_JobCategory)
inData = JobCategory_inputs.join(Y_targets)

total_binary_splits = 2**(7-1)-1
print(f"1.e) the numeber of possible binary-splits that you can generate from the JobCategory predictor is {total_binary_splits}")

 
print("1.f)  the Entropy metric is belowing:")
Entropy_list_6,left_branch_6,right_branch_6 = EntropyNominalSplit(inData,0.5,7)
branch_6 = []
branch_string_6 = ''
for i in range(len(left_branch_6)):
    branch_string_6 = '{'+left_branch_6[i]+'}'+'{'+right_branch_6[i]+'}'
    branch_6.append(branch_string_6)
table_6 = pd.DataFrame({'branch':branch_6,'table_Entropy':Entropy_list_6})
print(table_6)




import copy
Entropy_list2 = copy.deepcopy(Entropy_list_6)
min_Entropy = 0
for i in range(len(Entropy_list2)-1):
    if(Entropy_list2[i]<=Entropy_list2[i+1]):
        min_Entropy = Entropy_list2[i]
        Entropy_list2[i+1] = min_Entropy
    else:
        min_Entropy = Entropy_list2[i+1]
        Entropy_list2[i+1] = min_Entropy
print(f" the min-value of Entropy is {min_Entropy}")
print("1.g) the optimal split is [JobCategory_Agriculture,JobCategory_sales] and [JobCategory_professional,JobCategory_Service,JobCategory_Crafts,JobCategory_Labor,JobCategory_missing]")

 






