library(party)
library(caret)
library(C50)
library(datasets)
data(iris)
summary(iris)
table(iris$Species)
str(iris)
barplot(iris$Sepal.Length) 
barplot(iris$Sepal.Width)
barplot(iris$Petal.Length)
barplot(iris$Petal.Width)
sum(is.na(iris))

# Data partion for model building and testing
inTraininglocal <- createDataPartition(iris,p=.75,list=F)
train <- iris[-inTraininglocal,]
View(train)
test <- iris[inTraininglocal,]
View(test)
table(train$Species)
table(test$Species)

#model building
model <- C5.0(train$Species~.,data = train,trails = 40)
summary(model)
plot(model)
View(model)
pred <- predict.C5.0(model,test[,-5]) # remove the existing "Species" col
table(pred)
a <- table(test$Species,pred)
a
sum(diag(a)/sum(a)) # accuracy= 1

###Bagging####
acc<-c()
for(i in 1:100)
{
  print(i)
  inTraininglocal<-createDataPartition(iris,p=.75,list=F)
  train1<-iris[-inTraininglocal,]
  test1<-iris[inTraininglocal,]
  
  fittree<-C5.0(train1$Species~.,data=train1)
  pred<-predict.C5.0(fittree,test1[,-5])
  a<-table(test1$Species,pred)
  
  acc<-c(acc,sum(diag(a))/sum(a))
  
}

acc
summary(acc)
#Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#1       1       1       1       1       1 

# creating another model using ctree

iris_ctree <- ctree(train$Species~., data=train)
train_predict <- predict(iris_ctree,train,type="response")
table(train_predict,train$Species)
#train_predict setosa versicolor virginica
#setosa         45          0         0
#versicolor      0         49         5
#virginica       0          1        45

match <- table(train$Species,train_predict)
match
sum(diag(match)/sum(match)) # 95.87%
mean(train_predict != train$Species) * 100 # finding out how many mismatch is there between predicted and original values
# 4.13% mismatch i.e 95.87% accuracy for train data

# testing model accuracy on test data
test_predict<-predict(iris_ctree, test, type ="response")
table(test_predict, test$Species)
#test_predict setosa versicolor virginica
#setosa          5          0         0
#versicolor      0          0         0
#virginica       0          0         0

match1 <- table(test$Species,test_predict)
match1
sum(diag(match1)/sum(match1)) # 100%
mean(test_predict != test$Species) * 100 # no mismatch thus model is 100% accurate for test data
