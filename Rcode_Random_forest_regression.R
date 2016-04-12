#################################
#### Tree based methods  ########
# Text book
# 5.2: Bootstrap
# 8.1: Single trees
# 8.2: Ensemble methods
#    . Bagging
#    . Random Forest



### Part I: regression trees

# 1) Single Tree
# 2) Bagging
# 3) Random Forest (Bagging is a special case)

# 4) Appendix: Bootstrap Technique applied to trees: a separate lecture prepared

### Part II: RandomForest for classifications ( a separate file)

library(tree)              # regression/classification trees
library(randomForest)      # to see how a package is evolved to be better: rfNews()


#####  Part I: Regression trees

### For the purpose of demonstration we use the 
### Baseball data from the book

library(ISLR)
help(Hitters)
data.comp=na.omit(Hitters)  
dim(data.comp)            # We are keeping 263 players
data.comp$LogSalary=log(data.comp$Salary)  # add LogSalary
names(data.comp)
data1=data.comp[,-19]        # Take Salary out


### 1) A single tree ###

# The idea is to partition the space into boxes. 
#   Binary split: Take the best predictor 
#   Recursive: Repeat the search in the next half space
#   The prediction:  sample mean in each box.
#                    (one could use different method: reg for example)

# A single tree is not good 
# . not stable 
# . overfitting


# We use it here to illustrate the idea of binary, top-down, recursive trees.

### Model 1: Use all the obs'n but only include two variables CHits and CAtBat

fit1.single=tree(LogSalary~CAtBat+CHits, data1) # The order plays no role
fit1.single  
names(fit1.single)
fit1.single$frame 

# var   n         dev     yval splits.cutleft splits.cutright
# 1  CAtBat 263 207.1537331 5.927222          <1452           >1452
# 2   CHits 103  36.2195305 5.092883           <182            >182
# 4   CHits  56  18.3590026 4.771243          <49.5           >49.5
# 8  <leaf>   5   9.3315033 5.660001                               
# 9   CHits  51   4.6908471 4.684110           <132            >132
# 18 <leaf>  34   1.6206148 4.533326                               
# 19 <leaf>  17   0.7511846 4.985678                               
# 5  <leaf>  47   5.1645460 5.476113                               
# 3   CHits 160  53.0765908 6.464327           <669            >669
# 6  <leaf>  50   9.5004560 6.213632                               
# 7  <leaf> 110  39.0053795 6.578279   


plot(fit1.single)
text(fit1.single, pretty=0)      # pretty=0 only affect the categorical var's. The names will be shown.
# It has 6 terminal nodes. That means we partition CAtBat and Chits into six boxes. 
# The predicted values are the sample means in each box.

fit1.single.result=summary(fit1.single)
names(fit1.single.result)

fit1.single.result$dev # dev=RSS. 
fit1.single.result$size  # number of nodes
fit1.single.result$df  # n- number of nodes

# RSS=65.37. It should be same as the sum of RSS at each terminal node
# 9.332+1.621+0.751+0.751+5.165+9.500+39.010=66.13 (should be the same)

yhat=predict(fit1.single, data1)
RSS.tree=sum((data1$LogSalary-yhat)^2 )   # another way to get RSS
RSS.tree
# [1] 65.37368

plot(data1$LogSalary, yhat, pch=16, col="blue",
     xlab="LogSal",
     ylab="Yhat")

# As we already knew there are only 6 predicted values being used.

# How does the above tree perform comparing with our old friend lm in terms of the in sample errors?
fit.lm=summary(lm(LogSalary~CAtBat+CHits, data1))
SSR.lm=(263-2)*(fit.lm$sigma)^2
SSR.lm ## Oppps much worse than even a single tree. 
# [1] 127.2509   RSR.tree=65


### An alternative tree (no need to use it. like its output)
library(rpart)
fit.single.rp=rpart(LogSalary~CAtBat+CHits, data1)
fit.single.rp
plot(fit.single.rp)
text(fit.single.rp, pretty = TRUE)
################################

# Model 2:  use all the predictors
fit1.single.full=tree(LogSalary~., data1)

# We can control how large a tree we want to build. 
# control=tree.control( minsize = 6, mindev = 0.05))      
# the default tree.control(nobs, mincut = 5, minsize = 10, mindev = 0.01)
# mincut: min number of obs's to be included in a child
# minsize: number of end nodes
# mindev: the dev needs to be reduced by mindev fold within the branch


# Here is the tree:
plot(fit1.single.full)  
text(fit1.single.full, pretty=0)
fit1.single.full$frame


fit1.single.full.s=summary(fit1.single.full)
names(fit1.single.full.s)
names(fit1.single.full)

# 7 variables are used
# "CAtBat" "CHits"  "AtBat"  "CRuns"  "Hits"   "Walks"  "CRBI" 
# There are 9 terminal nodes

fit1.single.full.s$dev  # RSS=43.03
fit1.single.full.s$used # Var's included 

RSS.lm=(263-8)*((summary(lm(LogSalary~CAtBat+CHits+AtBat+CRuns+Hits+Walks+CRBI, data1)))$sigma)^2
RSS.lm   # Still pretty big 101.5803




### 2)  Bagging: general method

# 1) Take B many bootstrap samples
# 2) One tree for each B sample
# 3) The final predictor = Average of all B many trees
# 4) It is a special case for Random Forest

# Pro: Reduce the var while maintain similar bias. 
# Con: The trees are correlated (on higher level). 




#################################
### 3) Random Forest  #########

# 1) Take B many bootstrap samples

# 2) Build a deep random tree for each Bootstrap sample by
#   . Split only m (mtry) randomly chosen predictors at each split

# 3) Bag all the random trees by taking ave

# 4) Use Out of Bag testing errors to tune mtry!!!!! 


# Pro: Decorrelate the trees - reduce var more
# Con: Tuning parameter m is introduced 
#      . m too small - miss important var's
#      . m too large - more cor between trees


### Remark: 
### 1) nodesize:5 for reg's and 1 for classifications
### 2) when mtry=p=19, randomForest gives us bagging estimates.



library(randomForest)

# i) For a fixed mtry

fit.rf=randomForest(LogSalary~., data1, mtry=10, ntree=100)
str(fit.rf)

#############################################################
### default settings: 
### mtry=p/3, (sqrt(p) in classification tree)
### Bootstrap size B=ntree=500 
### when mtry=p=19, randomForest gives us bagging estimates.
### nodesize:5 for reg's and 1 for classifications. 

# Based on the 100 random trees, we get hat y= ave (all 100 trees).
# How does perform? 

yhat=predict(fit.rf, data1)   # predicted values we use 

plot(data1$LogSalary, yhat, pch=16,  # add a 45 degree line:looks very good!
     main="Y vs. Predicted Y", col="blue")
abline(0, 1, lwd=5, col="red")

mse.train=mean((data1$LogSalary-yhat)^2)
mse.train      # Training mse approx. .03!!!
mse.oob=mean((data1$LogSalary-fit.rf$predicted)^2)  #OOB testing error 
mse.oob        # about .18  

# ii) Compare a single tree, a bagging wiht 100 trees and a RF with mtry=7, 100 trees

fit.single=randomForest(LogSalary~., data1, mtry=19, ntree=1)
fit.bagging=randomForest(LogSalary~., data1, mtry=19, ntree=100)
fit.rf=randomForest(LogSalary~., data1, mtry=7, ntree=100)

par(mfrow=c(3,1))
ylim.0=c(3.5, 7.5)
plot(data1$LogSalary, predict(fit.single), pch=16,  # add a 45 degree line:looks very good!
     ylim=ylim.0,
     main="A single tree", col="blue")
abline(0, 1, lwd=5, col="red")

plot(data1$LogSalary, predict(fit.bagging), pch=16,  # add a 45 degree line:looks very good!
     ylim=ylim.0,
     main="A bagging tree, 100 trees", col="blue")
abline(0, 1, lwd=5, col="red")

plot(data1$LogSalary, predict(fit.rf), pch=16,  # add a 45 degree line:looks very good!
     ylim=ylim.0,
     main="A RF tree, mtry=7, 100 trees", col="blue")
abline(0, 1, lwd=5, col="red")
par(mfrow=c(1,1))
### RF is the best but it should have perfomed much better than a bagging estimate.



# iii) Zoom in to a RF estimate
fit.rf=randomForest(LogSalary~., data1, mtry=10, ntree=100)
str(fit.rf)

# a) Where is cross validation testing error? It is cleverly replaced by OOB mse

plot(fit.rf$mse, xlab="number of trees", col="blue",
     ylab="ave mse up to i many trees using OOB predicted",
     pch=16) # We only need about 100 trees for this 
# We get the above plot by 
plot(fit.rf, type="p", pch=16,col="blue" )

fit.rf$oob.times  # how many times each obs'n belong to OOB. We expect to see 1/e=1/3 (.37)

#   fit.rf$mse is OOB testing errors = mse of only using the OOB fitted values up to tree number
#   fit.rf$predicted is obtained only using the OOB obs'n

fit.rf$predicted   # predicted values based on all OOB values
predict(fit.rf, data1)  # predicted values based on the RF. 

plot(fit.rf$predicted,predict(fit.rf, data1), pch=16, col="blue",
     main="RF prediction vs. OOB prediction",
     xlab="Fitted using OOB only",
     ylab="Fitted using RF")
abline(0, 1, col="red", lwd=5)   # They differ but not by too much.

# b) Ready to tune mtry and B=number of the trees in the bag

# ntree effect: given mtry and ntree, we see the effect of ntree first
fit.rf=randomForest(LogSalary~., data1, mtry=10, ntree=500)
plot(fit.rf, col="red", pch=16, type="p", main="default plot")
# We may need 250 trees to settle the OOB testing errors


# The effect of mtry: the number of random split at each leaf

# Now we fix ntree=250, We only want to compare the OOB mse[250] to see the mtry effects.

# Here we loop mtry from 1 to 19 and returen the testing OOB errors

rf.error.p=1:19 
for (p in 1:19)
{
  fit.rf=randomForest(LogSalary~., data1, mtry=p, ntree=250)
  rf.error.p[p]=fit.rf$mse[100]
}
rf.error.p   
plot(1:19, rf.error.p, pch=16,
     xlab="mtry",
     ylab="mse of mtry")  


#Run above loop a few time, it is not very unstable. 
#The recommended mtry for reg trees are mtry=p/3=19/3 about 6 or 7. Are you convinced with p/3?

###### We should treat mtry to be a tuning parameter!!!!!!!!  ###########
#########################################################################

# iv) We could also get testing errors w/o using OOB idea

n=nrow(data1)
#set.seed(1)
train.index=sample(n,n*3/4) # we use about 3/4 of the subjects as the training data.
train.index
data.train=data1[train.index,]
data.test=data1[-train.index, ]

fit.rf.train=randomForest(LogSalary~., data.train,  mtry=6, ntree=500) 
## this will output the oob errors

## To get testing error 
fit.rf.testing=randomForest(LogSalary~., data.train, xtest=data.test[, -20], 
                          ytest=data.test[,20], mtry=6, ntree=500)
fit.rf.testing$mse
## this will output the testing errors
plot(fit.rf.testing$mse)

plot(fit.rf.train) # when xtest and ytest are given, the output will be for testing data, like y, mse, etc
# The testing errors seem to agree with what we found using OOB errors.

### Let's put testing errors and OOB errors together
plot(1:500, fit.rf.testing$mse, col="red", pch=16,
     xlab="number of trees",
     ylab="mse",
     main="mse's of RF: blue=oob erros, red=testing errors")
points(1:500, fit.rf.train$mse, col="blue", pch=16)
# OOB erros seem to do ok to estimate the testing errors!!!
########


# v) The final fit
fit.rf.final=randomForest(LogSalary~., data1, mtry=6, ntree=250)
plot(fit.rf.final)

person=data1[1, ]   # Let's predict rownames(data1)[1]: "-Alan Ashby"
fit.person=predict(fit.rf.final, person)
fit.person # the fitted salary in log scale is 6.196343 
# -Alan Ashby 
# 6.170148 

# Alan Ashby's true log sal 
data1$LogSalary[1]
# [1] 6.163315

#### Not bad at all.....

####### End of randomForest in regressions.




###############################################
### Appendix:  Bootstrap 

#  Method 
#  . Randomly take n obs'n from the n data point with replacement
#  . Treat the Bootstrap sample as another sample from the population
#  . Standard errors for any estimator
#  . When n is large the Bootstrap estimates converge to the original estimate
################################################


#### Trees built by resampling the data through BOOTSTRAP method
#### Take a bootstrap sample and exam the tree built by the sample

# i) About 2/3 of the original subjects are chosen in each Bootstrap sample
# ii) The trees are different
# ii) mse changes a lot



### Effects for different Bootstrap samples


RSS=0  # initial values
n.unique=0
n=nrow(data1); K=1
for (i in 1:K)
{
  index1=sample(n, n, replace=TRUE)   
  Sample1=data1[index1, ]               # Take a bootstrap sample
  fit1.boot=tree(LogSalary~., Sample1)  # Get a tree fit
  RSS[i]=summary(fit1.boot)$dev  # output RSS for each bootstrap tree
  plot(fit1.boot, 
       main="Trees with a Bootstrap sample") 
  text(fit1.boot, pretty=0)
  n.unique[i]=length(unique(index1))
  Sys.sleep(2)                        # Pause for 2 seconds before running for next round
}

hist(RSS, breaks=30, 
     col="blue",
     main="RSS from different Bootstrap trees")


hist(n.unique, breaks=30,
     col="red", 
     main="number of unique subjects included in each Bootstrap sample")

hist(n-n.unique, breaks=30,
     col="green", 
     main="number of OOB subjects not included in each Bootstrap sample")

