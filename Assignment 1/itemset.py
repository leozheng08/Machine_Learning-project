#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 12:24:19 2019

@author: zhengleo
"""

from itertools import combinations

itemList = ['A','B','C','D','E','F','G']
itemlist = ['b','c','d','e','f','g','h']

list
def traversing(itemList):
    length = len(itemList)
    totalNumItemsets = 2**length-1
    print(f"1a.the number of possible itemsets is {totalNumItemsets}")
    for i in range(1,length+1):
        ItemSet = list(combinations(itemList,i))
        print(f"1{itemlist[i-1].lower()}.All the possible {i}-itemsets is {ItemSet}")
        
traversing(itemList)
        