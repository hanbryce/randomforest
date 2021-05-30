#!/usr/bin/env python
# coding: utf-8

# In[103]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from collections import Counter

def readtoarray(directory):
    read = pd.read_csv(directory)
    array = read.values
    if len(array[0])==1: return array.T[0]
    return array

def str_to_int(arr):
    arr = np.array(arr)
    arr[arr == 'unacc'] = 0
    arr[arr == 'small'] = 1
    arr[arr == 'low'] = 1
    arr[arr == '1'] = 1
    arr[arr == 'acc'] = 1
    arr[arr == 'med'] = 2
    arr[arr == 'good'] = 2
    arr[arr == '2'] = 2
    arr[arr == 'big'] = 3
    arr[arr == 'high'] = 3
    arr[arr == '3'] = 3
    arr[arr == 'vgood'] = 3
    arr[arr == 'vhigh'] = 4
    arr[arr == '4'] = 4
    arr[arr == '5more'] = 5
    arr[arr == 'more'] = 5
    arr = arr.astype(int)
    return arr


# In[104]:


class MulticlassLR:
    
    def __init__(self, rate, n):
        self.rate = rate
        self.n = n
        self.clflist = []
    
    def fit(self, X, y):
        for i in range(max(y) + 1):
            LR = LogisticRegression(self.rate, self.n)
            LR.fit(X, np.where(y == i, 1, 0))
            self.clflist.append(LR)
    
    def predict(self, X):
        temp = []
        for clf in self.clflist: temp.append(clf.net_input(X))
        tempnp = np.array(temp)
        predlist = np.array([])
        for i in tempnp.T: predlist = np.append(predlist, np.argmax(i))
        return predlist

    def test(self, X, y):
        result = self.predict(X)
        accuracy = 0.
        for i in range(len(y)):
            if result[i] == y[i]: 
                accuracy += 1./(len(y) + 1)
        return accuracy
 
    
class LogisticRegression:
    
    def __init__(self, rate, n):
        self.rate = rate
        self.n = n
        
    def fit(self, X, y):
        self.weight = np.random.RandomState().normal(loc = 0.0, scale = 0.01, size = 1 + X.shape[1])
        for i in range(self.n):
            output = 1. / (1. + np.exp(-np.clip(self.net_input(X), -250, 250)))
            errors = (y - output)
            self.weight[1:] += self.rate * X.T.dot(errors)
            self.weight[0] += self.rate * errors.sum()
        return self
    
    def net_input(self, X):
        return np.dot(X,self.weight[1:]) + self.weight[0]
        
    def predict(self, X):
        return np.where(self.net_input(X) >= 0., 1, 0)


# In[105]:


class RandomForest:
    
    def __init__(self, n_sample, n_tree, n_feature, min_value, measure, maxdepth):
        self.n_sample = n_sample
        self.n_tree = n_tree
        self.n_feature = n_feature
        self.min_value = min_value
        self.measure = measure
        self.maxdepth = maxdepth
        
    def fit(self, X, y):
        self.treelist = []
        D = np.concatenate((X, np.expand_dims(y.T, axis = 1)), axis = 1)
        for i in range(self.n_tree):
            batch_list = np.random.RandomState().choice(list(range(len(D))), self.n_sample, replace = True)
            batch = []
            for i in batch_list: 
                batch.append([D[i]])
            DT = DecisionTree(self.n_feature, self.min_value, self.measure, self.maxdepth)
            DT.fit(batch)
            self.treelist.append(DT)
        return self
    
    def predict(self, X):
        predlist = np.array([])
        for i in X: 
            valuelist = []
            for t in self.treelist: valuelist.append(t.tree.find_value(i))
            predlist = np.append(predlist, max(valuelist,key=valuelist.count))     
        return predlist
    
    def test(self, X, y):
        result = self.predict(X)
        accuracy = 0.
        for i in range(len(y)):
            if result[i] == y[i]: 
                accuracy += 1./(len(y) + 1)
        return accuracy
            
class DecisionTree:
    
    def __init__(self, n_feature, min_value, measure, maxdepth):
        self.n_feature = n_feature
        self.measure = measure
        self.maxdepth = maxdepth
        self.min_value = min_value
        
    def fit(self, D):
        D = np.array(D)
        D = np.squeeze(D, axis = 1)
        self.featurelist = np.random.RandomState().choice(list(range(len(D[0])-1)), self.n_feature, replace = False)
        self.tree = Node()
        self.build_tree(self.tree, D, 0)
        return self
        
    def build_tree(self, node, D, depth):
        D = np.array(D)
        if depth >= self.maxdepth or self.info(D) <= self.min_value :
            if len(D) == 0: node.result = 0
            else:
                elements, counts = np.unique(D[:,-1],return_counts = True)
                node.result = elements[np.argmax(counts)]
        else:
            node.left, node.right = Node(), Node()
            node.feature, node.value = self.find_split(D)
            L, R = self.split(D, node.feature, node.value)
            self.build_tree(node.left , L, depth+1)
            self.build_tree(node.right , R, depth+1)
    
    def find_split(self, D):
        D = np.array(D)
        best_feature, best_value = None, None
        min_info = 100
        for feature in (self.featurelist):
            for i in D[:,feature]:
                left, right = self.split(D, feature, i)
                total_info = (self.info(left)*len(left)/len(D)) + (self.info(right)*len(right)/len(D))
                if total_info < min_info: min_info, best_feature, best_value = total_info, feature, i        
        return best_feature, best_value
    
    def split(self, D, feature, value):
        left,right = [],[]
        for i in D:
            if i[feature] < value: left.append(i)
            else: right.append(i)
        return left, right
    
    def predict(self, X):
        predlist = np.array([])
        for i in X: predlist = np.append(predlist, self.tree.find_value(i))
        return predlist     
    
    def info(self, D):
        D = np.array(D)
        if len(D) == 0: return 0
        elements, counts = np.unique(D[:,-1],return_counts = True)
        if self.measure == 'entropy':
            entropy = 0
            for i in counts: entropy += -(i/(len(D)))*np.log2(i/(len(D)))
            return entropy
        if self.measure == 'gini':
            gini = 1
            for i in counts: gini -= (i/(len(D))**2)
            return gini
        if self.measure == 'error':
            return (1-(max(counts)/len(D)))
    
class Node:
    
    def __init__(self, feature = None, value = None):
        self.left = None
        self.right = None
        self.feature = feature
        self.value = value
        self.result = None
        
    def find_value(self, i):
        if self.result!=None:
            return self.result
        else: 
            if i[self.feature] < self.value: return self.left.find_value(i)
            else: return self.right.find_value(i)


# In[340]:




iris_trainf = readtoarray('dataset_files/iris_X_train.csv')
iris_trainl = readtoarray('dataset_files/iris_y_train.csv')

iris_testf = readtoarray('dataset_files/iris_X_test.csv')
iris_testl = readtoarray('dataset_files/iris_y_test.csv')

car_trainf = readtoarray('dataset_files/car_X_train.csv')
car_trainf = str_to_int(car_trainf)
car_trainl = readtoarray('dataset_files/car_y_train.csv')
car_trainl = str_to_int(car_trainl)

car_testf = readtoarray('dataset_files/car_X_test.csv')
car_testf = str_to_int(car_testf)
car_testf = car_testf.astype(float)
car_testl = readtoarray('dataset_files/car_y_test.csv')
car_testl = str_to_int(car_testl)









"""
    RF = RandomForest(200, 30, 5, 0.2, 'entropy', 10)
    RF.fit(car_trainf,car_trainl)
    print("%.1f" % (RF.test(car_testf,car_testl) * 100) + '%')

    ILR = MulticlassLR(0.0004,800)
    ILR.fit(car_trainf,car_trainl)
    print("%.1f" % (ILR.test(car_testf,car_testl)*100)+'%')    
"""


# In[ ]:




