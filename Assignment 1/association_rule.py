import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Groceries.csv')

listset = dataset.groupby(['Customer'])['Item'].apply(list).values.tolist()
print(f"2a.the number of customer is {len(listset)}")



from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()
te_ary = te.fit(listset).transform(listset)
ItemIndicator = pd.DataFrame(te_ary, columns=te.columns_)
print(f"2b.the number of unique items in the market basket across all customers is {ItemIndicator.shape[1]}")

Items_count_list=[]
for i in listset:
    Item_count = len(i)
    Items_count_list.append(Item_count)
    
Customers_index_list=[]
for i in range(len(listset)):
    Customers_index_list.append(i+1)
    

P3,median,P1 = np.percentile(Items_count_list,[75,50,25])
iqr = P3-P1
h = 2*iqr*len(Items_count_list)**(-1/3.0)
max_value = max(Items_count_list)
min_value = min(Items_count_list)
range = max_value-min_value




NitemPcustoemr_dic = {'Customers':Customers_index_list,'Items':Items_count_list}
NitemPcustoemr_set = pd.DataFrame(data = NitemPcustoemr_dic)
NitemPcustoemr_set.hist(column='Items',bins=int(range/h)+1)
plt.title("Histogram of unique items in each customer's market basket")
plt.xlabel("customers")
plt.ylabel("items")
print(f"2c.the median and the 25th percentile and 75th percentile in this histogram are seperately {median},{P1},{P3}")



#NitemPcustomer_series = dataset.groupby(['Customer'])['Item'].count()
#freqTable_series = pd.value_counts(NitemPcustomer_series)
#freqTable_set = pd.DataFrame({'Item':freqTable_series.index,'Frequency':freqTable_series.data})
#freqTable_set = freqTable_set.sort_values(by =['Frequency'])
#print(freqTable_set)
#print(f"the number of itemset which appeared at least 75 customers is 14 and the highest k is 14")
#NitemPcustomer_series.describe()

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

#ItemIndicator.replace(False,np.NAN)
#Item_rate = ItemIndicator.count()
#Item_min_value = Item_rate.min()
#Item_value_sum = Item_rate.sum()
#Item_min_support = float(Item_min_value/Item_value_sum)


ItemIndicator_new = pd.DataFrame(te_ary, columns=te.columns_)
Item_min_support = 75/len(listset)
# Find the frequent itemsets
frequent_itemsets = apriori(ItemIndicator_new, min_support = Item_min_support, use_colnames = True)
length = []
for i in frequent_itemsets['itemsets']:
    length.append(len(i))
k_highest = max(length)

print(f"2d. the number of itemsets which appeared at least 75 customers is {frequent_itemsets.shape[0]}, the highest k is {k_highest}")

# Discover the association rules
assoc_rules1 = association_rules(frequent_itemsets, metric = "confidence", min_threshold = 0.01)
print("2e.the number of association rules is 1228,the association rules found in assoc_rules1")

print("2f. the graph belowing:")
import matplotlib.pyplot as plt
plt.figure(figsize=(6,4))
plt.scatter(assoc_rules1['confidence'], assoc_rules1['support'], s = assoc_rules1['lift'])
plt.grid(True)
plt.xlabel("Confidence")
plt.ylabel("Support")
plt.show()

# Find the frequent itemsets
frequent_itemsets = apriori(ItemIndicator_new, min_support = Item_min_support, use_colnames = True)



# Discover the association rules
assoc_rules = association_rules(frequent_itemsets, metric = "confidence", min_threshold = 0.6)
print("2g the rules are found in assoc_rules")
import matplotlib.pyplot as plt
plt.figure(figsize=(6,4))
plt.scatter(assoc_rules['confidence'], assoc_rules['support'], s = assoc_rules['lift'])
plt.grid(True)
plt.xlabel("Confidence")
plt.ylabel("Support")
plt.show()
