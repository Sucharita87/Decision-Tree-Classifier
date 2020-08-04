import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree
fraud= pd.read_csv("F:\\ExcelR\\Assignment\\Decision Tree\\Fraud_check.csv")
fraud.info() # 3 object, 3 int
fraud.describe()
fraud.shape
fraud.isnull().sum()

categorical = [var for var in fraud.columns if fraud[var].dtype=='O']
print('There are {} categorical variables\n'.format(len(categorical)))
print('The categorical variables are :\n\n', categorical)
string_values = ['Undergrad', 'Marital.Status', 'Urban']
fraud[categorical].isnull().sum() # shows no null values
fraud.rename(columns= {"Taxable.Income":"y"}, inplace = True)
plt.hist(fraud['y'])  


# add income column which will serve as the classifier
fraud["income"]="<=30000"
fraud.loc[fraud["y"]>=30000,"income"]="Good"
fraud.loc[fraud["y"]<=30000,"income"]="Risky"
plt.hist(fraud['income'])
fraud.income.value_counts() # good: 476, risky:124
fraud.income.value_counts().plot(kind="pie") # imbalanced data

# label encoding of all string values(also include "income")
string_new= ['Undergrad', 'Marital.Status', 'Urban','income']
number= preprocessing.LabelEncoder()
for i in string_new:
   fraud[i] = number.fit_transform(fraud[i])

# now that classification is done as "good" and "Risky", we will drop the taxable income(y) column
fraud.drop(["y"], axis =1, inplace = True)
fraud.describe()
train,test = train_test_split(fraud, test_size = 0.2)
x_train = train.iloc[:,[0,1,2,3,4]]
y_train = train.iloc[:,5:6]
x_test = test.iloc[:,[0,1,2,3,4]]
y_test = test.iloc[:,5:6]
y_test.income.value_counts() # 0:99   1:21


# model building 
model = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
plt.figure(figsize=(12,8))
tree.plot_tree(model.fit(x_train, y_train),filled=True)
plt.show()
model.fit(x_train, y_train)
train_pred =model.predict(x_train)
test_pred = model.predict(x_test)

#model_pred is predicted values of test data
from sklearn.metrics import accuracy_score
train_accuracy = accuracy_score(y_train,train_pred)
train_accuracy # 79.8%
train_confusion = confusion_matrix(y_train,train_pred)
train_confusion
#array([[383,   0],
#       [  97, 0]]

##Prediction on test data

test_accuracy = accuracy_score(y_test,test_pred)
test_accuracy # 77.5%
confusion_test = confusion_matrix(y_test,test_pred)
confusion_test
#array([[93,  0],
#       [27,  0]]

print(classification_report(y_test, test_pred)) # accuracy = 78%

#almost similar train and test result


# gini index
# model building 
model2 = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=0)
plt.figure(figsize=(12,8))
tree.plot_tree(model2.fit(x_train, y_train),filled=True)
plt.show()
model2.fit(x_train, y_train)
train_pred2 =model2.predict(x_train)
test_pred2 = model2.predict(x_test)

#model_pred is predicted values of test data
from sklearn.metrics import accuracy_score
train_accuracy2 = accuracy_score(y_train,train_pred2)
train_accuracy2 # 80.41%
train_confusion2 = confusion_matrix(y_train,train_pred2)
train_confusion2
#array([[382,   1],
#       [  93, 4]]

##Prediction on test data

test_accuracy2 = accuracy_score(y_test,test_pred2)
test_accuracy2 # 77.5%
confusion_test2 = confusion_matrix(y_test,test_pred2)
confusion_test2
#array([[93,  0],
#       [27,  0]]

print(classification_report(y_test, test_pred2)) # accuracy = 78%

#almost similar train and test result





