# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 00:14:36 2020

@author: SUCHARITA
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn import tree

data= pd.read_csv("F:\\ExcelR\\Assignment\\Decision Tree\\Company_Data.csv")
data.describe() # need to scale the data
data.info() #float64(1), int64(7), object(3)
data.isnull().sum()
categorical = [var for var in data.columns if data[var].dtype =='O']
print('There are {} categorical variables.\n'.format(len(categorical)))
print('The categoricall variables are:\n\n',categorical)
string_values =  ['ShelveLoc', 'Urban', 'US']
data[categorical].isnull().sum()
data.head()
plt.hist(data['Sales']) # max sales are in range of 6-10 k; three sales group are there (0-6)k, (>6-10)k, (>0)k
# as we have to decide the influencing factors for good sales, we have to classify sales

stype = []  # define a new value
for value in data["Sales"]: 
    if value <6: 
        stype.append("bad") 
    else: 
        stype.append("good") 
       
data["stype"] = stype    # add stype to the  dataframe of "data" 
plt.hist(data['stype'])

# label encode all categorical variables

string_values =  ['ShelveLoc', 'Urban', 'US','stype']
n = preprocessing.LabelEncoder()
for i in string_values:
    data[i]= n.fit_transform(data[i])

data.columns
data.drop(["Sales"], axis =1, inplace = True)
data.columns
c= data.columns
data.shape # 400, 11
# train test split
train,test = train_test_split(data, test_size =0.2)
x_train = train.iloc[:,0:10]
y_train = train.iloc[:,10:11]
x_test = test.iloc[:,0:10]
y_test = test.iloc[:,10:11]
y_train.stype.value_counts() # 1:211 and 0: 109
y_test.stype.value_counts() # 1:59 and 0: 21


# model building with criteria = gini
modelg= DecisionTreeClassifier(criterion= "gini", max_depth=6, random_state=0)
plt.figure(figsize= (12,8))
tree.plot_tree(modelg.fit(x_train, y_train), filled = True)
plt.show()
modelg.fit(x_train, y_train)
modelg.score
modelg.feature_importances_
#[0.11637034, 0.11495919, 0.08838064, 0.02381657, 0.33837768, 0.23739232, 0.02876953, 0.03886865, 0., 0.01306509]
#['CompPrice', 'Income', 'Advertising', 'Population', 'Price','ShelveLoc', 'Age', 'Education', 'Urban', 'US']
# we can eliminate features which have less importance such as: population,age,urban, US 
#lets create another set of train and test data
x_train1 = x_train.drop(['Population', 'Age', 'Urban', 'US'], axis= 1)
x_train1.columns
x_test1 = x_test.drop(['Population', 'Age', 'Urban', 'US'], axis= 1)
x_test1.columns
# again build model
modelg.fit(x_train1, y_train)
modelg.score
modelg.feature_importances_

train_predg = modelg.predict(x_train1)
test_predg = modelg.predict(x_test1)

# accuracy check for model with gini value
train_accuracy = accuracy_score(y_train,train_predg)
train_accuracy # 93.4%
train_confusion = confusion_matrix(y_train,train_predg)
train_confusion
#[[102,   7],
#       [ 14, 197]]
print(classification_report(y_train, train_predg)) # accuracy = 93%

test_accuracy = accuracy_score(y_test, test_predg)
test_accuracy # 83.75 % 
test_confusion = confusion_matrix(y_test, test_predg)
test_confusion
#[[14,  7],
 #      [ 6, 53]]
print(classification_report(y_test, test_predg)) # accuracy=80%

#model building with criteria = entropy
modele= DecisionTreeClassifier(criterion='entropy', max_depth=6, random_state=0)
plt.figure(figsize=(12,8))
tree.plot_tree(modele.fit(x_train1,y_train), filled = True)
plt.show()
modele.fit(x_train, y_train)
modele.feature_importances_
#[0.10204959, 0.05518117, 0.09380208, 0. , 0.32066513, 0.32507312, 0.06854118, 0., 0., 0.03468773]
#['CompPrice', 'Income', 'Advertising', 'Population', 'Price','ShelveLoc', 'Age', 'Education', 'Urban', 'US']
# we can eliminate features which have less importance such as: population,education, age,urban, US 

#lets create another set of train and test data
x_train2 = x_train.drop(['Population', 'Age', 'Education','Urban', 'US'], axis= 1)
x_train2.columns
x_test2 = x_test.drop(['Population', 'Age','Education', 'Urban', 'US'], axis= 1)
x_test2.columns
# again build model


train_prede= modele.predict(x_train2)
test_prede = modele.predict(x_test2)

# accuracy checkfor model with entropy value
train_accuracy_e = accuracy_score(y_train, train_prede)
train_accuracy_e # 89%
train_confusion_e = confusion_matrix(y_train, train_prede)
train_confusion_e 
#[[ 89,  20],
#      [ 15, 196]]
print(classification_report(y_train, train_prede)) # accuracy = 89%

test_accuracy_e = accuracy_score(y_test, test_prede)
test_accuracy_e # 80%
test_confusion_e = confusion_matrix(y_test, test_prede)
test_confusion_e
#[[12,  9],
#       [ 7, 52]]
print(classification_report(y_test, test_prede)) # accuracy = 80%

