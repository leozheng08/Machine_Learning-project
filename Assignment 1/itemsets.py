#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 16:37:00 2019

@author: zhengleo
"""

datalist = ['A','B','C','D','E','F','G']
length = len(datalist)

totalItemset = []

def recurrsion(array):
    for i in range(length):
        for j in range(i,length):
            data = []
            for k in range(i,j+1):
                data.append(datalist[k])
            totalItemset.append(data)    
    return totalItemset
    
print(recurrsion(datalist))

print(f"the number of possible itemsets is {len(totalItemset)}")

def traversing(numOfItemset):
    num = numOfItemset
    itemset = []
    for i in totalItemset:
        if(len(i)==num):
            itemset.append(i)
    print(f"all the possible {num}-itemsets is:{itemset} ,the number is {len(itemset)}")
 
for i in range(1,len(datalist)+1):
    traversing(i)
        
    