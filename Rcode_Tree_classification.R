#################################
#### Tree based methods
### Part II: RandomForest for classifications
# 1) Single Trees
# 2) Random Forest
# 3) Comments

##################################################
### Part II: RandomForeset for classifications ###
library(randomForest)
library(tree)
library(pROC)

### Random Forest for classification is similar to that
# of regression trees except for the criterion used for growing a tree is different.

# Algorithm: 

# For b=1 to B
#   I) Take a bootstrap sample of size n
#   II) Build a tree using the bootstrap sample recursively until the n_min is met
#         i) Randomly select m variables
#         ii) For each variable, find the best split point such that
#                    the misclassification errors is minimized by majority vote
#         iii) Find the best variable and split point, split the node into two
#         iv) The end node will output the majority vote either 0 or 1
# The final aggregated tree will report the prop. of 1's among B trees

### Remarks on the splitting criterion. 
# Suppose  (n_1, p_1) and (n_2, p_2) are the number of obs'n and sample prop of "1"'s
# on the left and right split points

#### Three criterions are commonly used
#      i) Mis classification errors for majority vote (RF uses)
#      ii) deviance: -2loglik                                 (Better)
#      iii) Gini index p_1 (1-p_1) + p_2 ( 1-p_2)             (Better)
#           Gini index: misclassification errors for a random procedure that
#           assign hat y="1" with prob. p_1

### Framingham data for classification trees for simplicity

rm(list=ls()) # Remove all the existing variables

dir=c("E:/Data Mining- STAT 571") # school
setwd(dir)
data = read.table("Framingham.dat", sep=",", header=T, as.is=T)
data1=na.omit(data)    # no missing value
names(data1)[1]="HD"
data1$HD=as.factor(data1$HD)
data1$SEX=as.factor(data1$SEX)
table(as.factor(data1$CIG))  # The levels are very unbalanced. We turn this to a categorical variable
data1$CIG=as.factor(data1$CIG)
str(data1)



### 1) A single tree using package tree
# Note this package is different from RandomForest

# a) Use SBP alone to see the effect of two different criteria: deviance and gini 

fit.dev=tree(HD~SBP, data1, split="deviance")  
#split = c("deviance", "gini"), "deviance" as default. 
plot(fit.dev)
text(fit.dev, pretty=TRUE)  # plot the labels
fit.dev$frame 
summary(fit.dev)

fit.gini=tree(HD~SBP, data1, split="gini")  #split = c("deviance", "gini") as an option
plot(fit.gini)
text(fit.gini, pretty=TRUE)  # plot the labels
fit.gini$frame 
summary(fit.gini)

### Notice the two trees built above are different. 




# b) Use all predictors
fit.tree=tree(HD~., data1)
plot(fit.tree)
text(fit.tree, pretty=TRUE)  # plot the labels
fit.tree$frame  
# yprob.1=sample prop of 1's in the end node
# yval is majority vote class
# we can set our own rules by thresholding yprob

# Use ROC/AUC ect. to measure the performance
prob.1=predict(fit.tree, data1)[, 2] # Prob(y=1|x)
roc(data1$HD, prob.1, plot=T)  


# c) A single tree with categorical predictors

#To convince you that trees take categorical var. Grow
# a tree with only CIG+SBP (Obviously from b), CIG didn't get in)

fit.tree=tree(HD~CIG+SBP, data1)
plot(fit.tree)
text(fit.tree, pretty=TRUE)
fit.tree$frame 
(unique(as.factor(round(predict(fit.tree)[, 2], 5)))) # To check it agrees with fit.tree$frame


### 2) Random Forest

# 1)  First run only a few trees
set.seed(1)
fit.rf=randomForest(HD~., data1, mtry=4, ntree=2)   
# By default mtry=sqrt(p). In this case is 2 or 3
names(fit.rf)
fit.rf$mtry
fit.rf$votes[1:50, ] # output the prob of 0 and 1 using oob's 
fit.rf$predicted[1:50] # same as above
fit.rf$err.rate # 
# Mis-classification rate of OOB, misclassification errors for "0" and "1"
plot(fit.rf)


# 2) RandomForest, more general

fit.rf=randomForest(HD~., data1, mtry=4, ntree=500) 
#fit.rf=randomForest(HD~., data.fram, cutoff=c(.5, .5), ntree=500) 
plot(fit.rf)
fit.rf$confusion    # confusion mitrax
fit.rf$votes  # output the prob of 0 and 1 using oob's 

fit.rf.pred=predict(fit.rf, type="prob")  # output the prob of "0" and "1"

fit.rf.pred[1:10,  ]   # same as fit.rf$votes
fit.rf$votes[1:10, ]
fit.rf$predicted[1:10] # lables

## using training and testing data
n=nrow(data1)
n1=(2/3)*n
train.index=sample(n, n1,replace=FALSE)
length(train.index)
data.train=data1[train.index, ]
data.test=data1[-train.index, ]


fit.rf.train=randomForest(HD~., data.train)
plot(fit.rf.train)
predict.rf=predict(fit.rf.train, newdata=data.test)
predict.rf=predict(fit.rf.train, newdata=data.test, type="prob")
predict.rf[1:10, ]

### 3) Comments about RF classifications

# Pro: pretty accurate; fast; automatic output for more categories in response. 
# Con: loose interpretation, may discriminate against categorical predictors.
#      Extremely cautious about
#      . Importance plot
#      . The validity of using a cut off point other than 1/2.

#### We will see how other competitors work

