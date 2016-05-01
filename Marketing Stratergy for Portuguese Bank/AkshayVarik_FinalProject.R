# Advanced Statistics for Management (STAT 471/571/701)
# Spring 2016
# Final Project
# Prof: Dr. Linda Zhao
# Name: Akshay Varik
# Penn ID: 73531118


# Task: 
# The data is related with direct marketing campaigns of a Portuguese banking 
# institution. The marketing campaigns were based on phone calls. Often, more than one 
# contact to the same client was required, in order to access if the product (bank term 
# deposit) would be ('yes') or not ('no') subscribed. The classification goal is to predict
# if the client will subscribe (yes/no) a term deposit.


# Loading the file and set the working directory
rm(list=ls()) # Remove all the existing variables
dir=c("E:/Data Mining- STAT 571")   # my laptop
setwd(dir)


# Include all the libraries
library(plyr)
library(dplyr)
library(leaps)
library(glmnet)
library(pROC)
library(MASS)
library(car)
library(data.table)
library(rockchalk)
library(caret)
library(e1071)
library(ROCR)
library(calibrate)
library(gridExtra)
library(boot)
library(rpart)
library(tree)
library(randomForest)
library(rattle)
library(rpart.plot)

# Read the data file
data=read.csv("Data_FinalProject.csv",header=T)


# Preliminary familiarity with the data and cleaning the data
dim(data)
names(data)[21]="TD" # Rename the response varible as TD (Term deposit)
levels(data$TD)  # Get the levels in the response variable

a=length(which(data$TD == "yes")) # Checking out the proportion of the response
yes.percentage=a/dim(data)[1]*100


data$TD=as.factor(data$TD)
summary(data)
str(data)

data1=data[,-11] # Removed the duration variable (as per the suggestion on the site to
                 # obtain realistic predictive model)
                 # Since after the call the outcome of it would be known

data1$Sl_no=1:nrow(data1) # Created a column of serial number

sum(is.na(data)) # Check for missing values

# To subset data randomly i.e. extract 5000 rows
# data2=data1[sample(1:nrow(data1), 5000, replace=FALSE),]

# But I am creating a new file of 55% no and 45% yes in 5000 random subset
set.seed(1)
data_TD_no=subset(data1, TD=="no") # obtain all TD=no rows
data_TD_yes=subset(data1, TD=="yes") # obtain all TD=yes rows
data2_TD_no=data_TD_no[sample(1:nrow(data_TD_no), 2750, replace=FALSE),] # 3500 random values of TD=no
data2_TD_yes=data_TD_yes[sample(1:nrow(data_TD_yes), 2250, replace=FALSE),] # 1500 random values of TD=yes
data2= rbind(data2_TD_no, data2_TD_yes) # concatenated the two dataframes to get 1 dataframe
                                        # This data frame will kind of constitue our entire dataset.

a=length(which(data2$TD == "yes")) # Cross checking out the proportion of the response
yes.percentage=a/dim(data2)[1]*100

data2=data2[sample(nrow(data2)),] # Randomly shuffled the rows of the dataframe 

data3=data1[!(data1$Sl_no %in% data2$Sl_no),] # Data1-Data2 (All the rows we did not pick)


# Data: Original Dataset as I downloaded
# Data1: Here I have added the serial no column and removed the Duration column from Data
# Data2: I will work on this. Contains 55% no and 45% yes TD response. Randomly obtained from Data1
# Data3: All the rows of Data2 not included in Data1
write.csv(data1,file="E:/Data Mining- STAT 571/Data1.csv") # save the data files
write.csv(data2,file="E:/Data Mining- STAT 571/Data2.csv")
write.csv(data3,file="E:/Data Mining- STAT 571/Data3.csv")



data.cleaned=read.csv("Data2.csv",header=T) # read Data2 that I will be using
data.cleaned$TD=as.factor(data.cleaned$TD) # generate levels for categorical variable
data.cleaned$TD = ifelse(data.cleaned$TD=="yes", 1, 0) # add new comun of 1 for TD=yes and for TD=no
a=length(which(data.cleaned$TD == 1)) # Cross checking out the proportion of the response
yes.percentage=a/dim(data.cleaned)[1]*100

summary(data.cleaned)
str(data.cleaned)

data.cleaned=data.cleaned[-c(1,22)] # Dropped the unnecessary serial number columns

cor(data.cleaned[,unlist(lapply(data.cleaned, is.numeric))]) # correlation between
                                                # numeric varaibles in the dataset

# Preliminary Visualization
require(ggplot2)
# pairs(data.cleaned[1:20], pch = 21)
# df.m = melt(data.cleaned, id.var = "TD")
# p <- ggplot(data = df.m, aes(x=variable, y=value)) 
# p <- p + geom_boxplot(aes(fill=TD))
# p <- p + facet_wrap( ~ variable, scales="free", ncol=4)
# p <- p + xlab("x-axis") + ylab("y-axis") + ggtitle("Box-plots")
# p <- p + guides(fill=guide_legend(title="TD"))
# p





# Now in this I divide the dataset into training data-80% and testing data-20%
data.cleaned=na.omit(data.cleaned)
set.seed(1)  # set a random seed so that we will be able to reproduce the random sample
index.train=sample(dim(data.cleaned)[1], 4000) # Take a random sample of n=4000 from 1 to N=5000
data.cleaned.train=data.cleaned[index.train,] # Set the 1000 randomly chosen subjects as a training data
data.cleaned.test=data.cleaned[-index.train,] # The remaining subjects will be reserved for testing purposes.
dim(data.cleaned.train)
dim(data.cleaned.test)





# Model 1: Performing Logistic Regression with all variables.
fit1=glm(TD~., data.cleaned.train, family=binomial(logit))  
summary(fit1)
chi.sq= 5512.4-4149.5     # get the Chi-square stat
pchisq(chi.sq, 1, lower.tail=FALSE)  # p-value: from the likelihood Ratio test
anova(fit1, test="Chisq") # to test if the model is useful: null hypothesis is all (but the intercept) coeff's are 0
confint.default(fit1) # obtain the confidence level of the coefficient of 
                      # the variables in this model.

# The chi-square distribution
par(mfrow=c(2,1))
hist(rchisq(4000, 2), freq=FALSE, breaks=20) 
hist(rchisq(4000, 20), freq=FALSE, breaks=20) 
# When DF is getting larger, Chi-Squared dis is approx. normal 

#prediction on training data
fit1.pred.train=rep("0", 4000)   # prediction step 1
fit1.pred.train[fit1$fitted > 0.9]="1"  # prediction step 2 to get a classifier
fit1.pred.train=as.factor(fit1.pred.train)
cm.train=table(fit1.pred.train, data.cleaned.train$TD)
#Training error
fit1.mce.train=mean(fit1.pred.train != data.cleaned.train$TD)

#prediction on test data
fit1.predict=predict(fit1, data.cleaned.test, type="response", interval="confidence", se.fit=T)
fit1.pred.test=rep("0", 1000)   # prediction step 1
fit1.pred.test[fit1.predict$fit > 0.9]="1"  # prediction step 2 to get a classifier
fit1.pred.test=as.factor(fit1.pred.test)
data.frame(data.cleaned.test$TD, fit1.pred.test) # put observed y and predicted y's together
cm=table(fit1.pred.test, data.cleaned.test$TD)
confusionMatrix(data=fit1.pred.test, data.cleaned.test$TD)
#Testing error
fit1.mce.test=mean(fit1.pred.test != data.cleaned.test$TD)

sensitivity=cm[2,2]/sum(data.cleaned.test$TD =="1")  
specificity=cm[1,1]/ sum(data.cleaned.test$TD == "0")
false.positive=cm[2,1]/sum(data.cleaned.test$TD == "0") 

#ROC Curve
fit1.roc=roc(data.cleaned.train$TD, fit1$fitted, plot=T, col="blue")
names(fit1.roc)
auc(fit1.roc)

##### False Positive vs. Sensitivity curve is called ROC 
plot(1-fit1.roc$specificities, fit1.roc$sensitivities, col="red", pch=16,
     xlab="False Positive", 
     ylab="Sensitivity")

#### Given a False positive rate, locate the prob threshold
plot(1-fit1.roc$specificities, fit1.roc$thresholds, col="green", pch=16,  
     xlab="False Positive",
     ylab="Threshold on prob")

# Tried to plot classifier boundary, but due to high dimension its hard!. Visualization 
# would be hard. 






#Model 2:Done regsubset generation to obtain 8 variables and logistic model fit
#Exhaustive search
fit2.exh=regsubsets(data.cleaned.train$TD~.,data.cleaned.train, nvmax=8, method="exhaustive", really.big=T)
fit2.e=summary(fit2.exh)
fit2.e$bic

par(mfrow=c(2,1))     # Compare different criterions: as expected rsq ^ when p is larger
plot(fit2.e$rsq, xlab="Number of predictors", ylab="rsq", col="red", type="p", pch=16)
plot(fit2.e$rss, xlab="Number of predictors", ylab="rss", col="blue", type="p", pch=16)

coef(fit2.exh,8) 

par(mfrow=c(3,1))
plot(fit2.e$cp, xlab="Number of predictors", 
     ylab="cp", col="red", type="p", pch=16)
plot(fit2.e$bic, xlab="Number of predictors", 
     ylab="bic", col="blue", type="p", pch=16)
plot(fit2.e$adjr2, xlab="Number of predictors", 
     ylab="adjr2", col="green", type="p", pch=16)

Reg.var=rownames(as.matrix(coef(fit2.exh,8))) # variables chosen

fit2.1=glm(TD~month+poutcome+emp.var.rate+cons.price.idx      #Building a logistic regression model
         +loan, data.cleaned.train, family=binomial(logit))
summary(fit2.1)

anova(fit1,fit2.1) # Compare Model 1 and Model 2.1


# Forward selection
fit2.for=regsubsets(data.cleaned.train$TD~.,data.cleaned.train, nvmax=8, method="forward", really.big=T)
fit2.f=summary(fit2.for)

fit2.f$cp

coef(fit2.for,8)

Reg.var=rownames(as.matrix(coef(fit2.for,8)))

fit2.2=glm(TD~month+poutcome+emp.var.rate+cons.price.idx      #Building a logistic regression model
           +loan, data.cleaned.train, family=binomial(logit))
summary(fit2.2)


# Backward Selection
fit2.bac=regsubsets(data.cleaned.train$TD~.,data.cleaned.train, nvmax=8, method="backward", really.big=T)
fit2.b=summary(fit2.bac)

fit2.b$rsq

coef(fit2.bac,8)

Reg.var=rownames(as.matrix(coef(fit2.bac,8)))

fit2.3=glm(TD~month+poutcome+emp.var.rate+cons.price.idx      #Building a logistic regression model
           +loan, data.cleaned.train, family=binomial(logit))
summary(fit2.3)

fit2=fit2.3 

par(mfrow=c(2,1))
plot(fit2,1)     
plot(fit2,2)

chi.sq= 5512.4-4226.9     # get the Chi-square stat
pchisq(chi.sq, 1, lower.tail=FALSE)  # p-value: from the likelihood Ratio test
anova(fit2, test="Chisq") # to test if the model is useful: null hypothesis is all (but the intercept) coeff's are 0


#prediction on training data
fit2.pred.train=rep("0", 4000)   # prediction step 1
fit2.pred.train[fit1$fitted > 0.9]="1"  # prediction step 2 to get a classifier
fit2.pred.train=as.factor(fit2.pred.train)
cm.train=table(fit2.pred.train, data.cleaned.train$TD)
#Training error
fit2.mce.train=mean(fit2.pred.train != data.cleaned.train$TD)

#prediction on test data
fit2.predict=predict(fit2, data.cleaned.test, type="response", interval="confidence", se.fit=T)
fit2.pred.test=rep("0", 1000)   # prediction step 1
fit2.pred.test[fit2.predict$fit > 0.9]="1"  # prediction step 2 to get a classifier
fit2.pred.test=as.factor(fit2.pred.test)
data.frame(data.cleaned.test$TD, fit2.pred.test) # put observed y and predicted y's together
cm=table(fit2.pred.test, data.cleaned.test$TD)
confusionMatrix(data=fit2.pred.test, data.cleaned.test$TD)
#Testing error
fit2.mce.test=mean(fit2.pred.test != data.cleaned.test$TD)

sensitivity=cm[2,2]/sum(data.cleaned.test$TD =="1")  
specificity=cm[1,1]/ sum(data.cleaned.test$TD == "0")
false.positive=cm[2,1]/sum(data.cleaned.test$TD == "0") 

#ROC Curve
fit2.roc=roc(data.cleaned.train$TD, fit2$fitted, plot=T, col="blue")
names(fit2.roc)
auc(fit2.roc)

##### False Positive vs. Sensitivity curve is called ROC 
plot(1-fit2.roc$specificities, fit2.roc$sensitivities, col="red", pch=16,
     xlab="False Positive", 
     ylab="Sensitivity")

#### Given a False positive rate, locate the prob threshold
plot(1-fit2.roc$specificities, fit2.roc$thresholds, col="green", pch=16,  
     xlab="False Positive",
     ylab="Threshold on prob")

# Tried to plot classifier boundary, but due to high dimension its hard!. Visualization 
# would be hard. 








# Model3: Using Regularization Techniques 
X.fl=model.matrix(~., data.cleaned.train)   # put data.frame into a matrix
colnames(X.fl)
Y=X.fl[, 53]  # extract y
X.fl=X.fl[, -c(53)]
fit3.lambda=cv.glmnet(X.fl, Y, alpha=1,nfolds=10)

names(fit3.lambda)
plot(fit3.lambda)
plot(fit3.lambda$lambda)
meancverror=fit3.lambda$cvm               # the mean cv error
plot(fit3.lambda$lambda, fit3.lambda$cvm, xlab="lambda", ylab="mean cv errors")
fit3.lambda$lambda.min        # min lambda changes a lot as a function of nfolds!
nonzeros=fit3.lambda$nzero
plot(fit3.lambda$lambda, fit3.lambda$nzero, xlab="lambda", ylab="number of non-zeros")

#output beta's from lambda.1se (this way we use smaller set of variables.)
coef.1se=coef(fit3.lambda, s="lambda.1se")  
coef.1se=coef.1se[which(coef.1se !=0),] 
pvariables=rownames(as.matrix(coef.1se))

# Fit the model
#glm.input=as.formula(paste("TD", "~", paste(pvariables[-1], collapse = "+"))) # formula
fit3=glm(TD~job+default+contact+month+campaign+poutcome+emp.var.rate
         +nr.employed, data=data.cleaned.train)
summary(fit3)

anova(fit1,fit3)
anova(fit2,fit3)

chi.sq= 991.81-702.35     # get the Chi-square stat
pchisq(chi.sq, 1, lower.tail=FALSE)  # p-value: from the likelihood Ratio test
anova(fit3, test="Chisq") # to test if the model is useful: null hypothesis is all (but the intercept) coeff's are 0

#prediction on training data
fit3.pred.train=rep("0", 4000)   # prediction step 1
fit3.pred.train[fit1$fitted > 0.9]="1"  # prediction step 2 to get a classifier
fit3.pred.train=as.factor(fit3.pred.train)
cm.train=table(fit3.pred.train, data.cleaned.train$TD)
#Training error
fit3.mce.train=mean(fit3.pred.train != data.cleaned.train$TD)

#prediction on test data
fit3.predict=predict(fit3, data.cleaned.test, type="response", interval="confidence", se.fit=T)
fit3.pred.test=rep("0", 1000)   # prediction step 1
fit3.pred.test[fit3.predict$fit > 0.9]="1"  # prediction step 2 to get a classifier
fit3.pred.test=as.factor(fit3.pred.test)
data.frame(data.cleaned.test$TD, fit3.pred.test) # put observed y and predicted y's together
cm=table(fit3.pred.test, data.cleaned.test$TD)
confusionMatrix(data=fit3.pred.test, data.cleaned.test$TD)
#Testing error
fit3.mce.test=mean(fit3.pred.test != data.cleaned.test$TD)

sensitivity=cm[2,2]/sum(data.cleaned.test$TD =="1")  
specificity=cm[1,1]/ sum(data.cleaned.test$TD == "0")
false.positive=cm[2,1]/sum(data.cleaned.test$TD == "0") 

#ROC Curve
fit3.roc=roc(data.cleaned.train$TD, fit3$fitted, plot=T, col="blue")
names(fit3.roc)
auc(fit3.roc)

##### False Positive vs. Sensitivity curve is called ROC 
plot(1-fit3.roc$specificities, fit3.roc$sensitivities, col="red", pch=16,
     xlab="False Positive", 
     ylab="Sensitivity")

#### Given a False positive rate, locate the prob threshold
plot(1-fit3.roc$specificities, fit3.roc$thresholds, col="green", pch=16,  
     xlab="False Positive",
     ylab="Threshold on prob")

# Tried to plot classifier boundary, but due to high dimension its hard!. Visualization 
# would be hard. 









# Model 4: Cross Validation on training set of Model 1
ctrl <- trainControl(method = "repeatedcv", number = 10, savePredictions = TRUE)

fit4 <- train(TD ~.,data=data.cleaned.train, method="glm", family="binomial",
                  trControl = ctrl, tuneLength = 5) # Fit the model

summary(fit4)

pred = predict(fit4, data.cleaned.test) # predict on the testing data set

# try=rep("0", 1000)   # prediction step 1
# try[pred > 0.9]="1"  # prediction step 2 to get a classifier
# try=as.factor(try)
# data.frame(data.cleaned.test$TD, try) # put observed y and predicted y's together
# cm=table(try, data.cleaned.test$TD)
# confusionMatrix(data=try, data.cleaned.test$TD)
# try.mce=mean(try != data.cleaned.test$TD)

abc=prediction(pred,data.cleaned.test$TD)
AUC = as.numeric(performance(abc, "auc")@y.values)
ACC= performance(abc, "acc")
mean(ACC@y.values[[1]])
plot(performance(abc, 'tpr', 'fpr'))
plot(ACC)

Sensitivity= performance(abc, "sens")
mean(Sensitivity@y.values[[1]])
Specificity= performance(abc, "spec")
mean(Specificity@y.values[[1]])










# Model 5: Cross Validation on training set of Model 2
ctrl <- trainControl(method = "repeatedcv", number = 10, savePredictions = TRUE)

fit5 <- train(TD ~month+poutcome+emp.var.rate+cons.price.idx+loan,
              data=data.cleaned.train, method="glm", family="binomial",
              trControl = ctrl, tuneLength = 5) # Fit the model

summary(fit5)

pred = predict(fit5, data.cleaned.test) # predict on the testing data set

abc=prediction(pred,data.cleaned.test$TD)
AUC = as.numeric(performance(abc, "auc")@y.values)
ACC= performance(abc, "acc")
max(ACC@y.values[[1]])
plot(performance(abc, 'tpr', 'fpr'))
plot(ACC)

Sensitivity= performance(abc, "sens")
mean(Sensitivity@y.values[[1]])
Specificity= performance(abc, "spec")
mean(Specificity@y.values[[1]])









# Model 6: Cross Validation on training set of Model 3
ctrl <- trainControl(method = "repeatedcv", number = 10, savePredictions = TRUE)

fit6 <- train(TD ~job+default+contact+month+campaign+poutcome+emp.var.rate
                  +nr.employed,  data=data.cleaned.train, method="glm", family="binomial",
                  trControl = ctrl, tuneLength = 5) # Fit the model

summary(fit6)

pred = predict(fit6, data.cleaned.test) # predict on the testing data set

abc=prediction(pred,data.cleaned.test$TD)
AUC = as.numeric(performance(abc, "auc")@y.values)
ACC= performance(abc, "acc")
max(ACC@y.values[[1]])
plot(performance(abc, 'tpr', 'fpr'))
plot(ACC)

Sensitivity= performance(abc, "sens")
mean(Sensitivity@y.values[[1]])
Specificity= performance(abc, "spec")
mean(Specificity@y.values[[1]])







# Model 7: LDA on all variables
fit7 <- lda(data.cleaned.train$TD ~., data=data.cleaned.train) # fit the lda model
plda <- predict(object = fit7,newdata = data.cleaned.test) #predict on test data
summary(fit7)

plda.class.1=predict(fit7, data.cleaned.test)$class # gives the class of the test data
plda.class.train.1=predict(fit7, data.cleaned.train)$class # gives the class of the train data

# create a histogram of the discriminant function values
ldahist(data = plda$x[,1], g=data.cleaned.test$TD)

# create a scatterplot of the discriminant function values
plot(plda$x[,1], type="n", ylab=c("LDA Axis 1"))
text(plda$x[,1], row.names(data.cleaned.test),  col=c(as.numeric(data.cleaned.test$TD)+10))

# Compute the misclasification error of the model
ct <- table(data.cleaned.test$TD, plda$class)
(ct[1,1]+ct[2,2])/sum(ct) 









# Model 8: LDA on variables otained from regsubsets 
fit8 <- lda(data.cleaned.train$TD ~ month+poutcome+emp.var.rate+cons.price.idx
            +loan, data=data.cleaned.train) # fit the lda model
plda <- predict(object = fit8,newdata = data.cleaned.test) #predict on test data
summary(fit8)

plda.class.2=predict(fit8,data.cleaned.test)$class # gives the class of the test data
plda.class.train.2=predict(fit8,data.cleaned.train)$class # gives the class of the train data

# create a histogram of the discriminant function values
ldahist(data = plda$x[,1], g=data.cleaned.test$TD)

# create a scatterplot of the discriminant function values
plot(plda$x[,1], type="n", ylab=c("LDA Axis 1"))
text(plda$x[,1], row.names(data.cleaned.test),  col=c(as.numeric(data.cleaned.test$TD)+10))

# Compute the misclasification error of the model
ct <- table(data.cleaned.test$TD, plda$class)
(ct[1,1]+ct[2,2])/sum(ct) 








# Model 9: LDA on variables obtained from LASSO
fit9 <- lda(data.cleaned.train$TD ~job+default+contact+month+campaign+poutcome+emp.var.rate
            +nr.employed, data=data.cleaned.train) # fit the lda model
plda <- predict(object = fit9,newdata = data.cleaned.test) #predict on test data
summary(fit7)

plda.class.3=predict(fit9, data.cleaned.test)$class # gives the class of the test data
plda.class.train.3=predict(fit9, data.cleaned.train)$class # gives the class of the train data

# create a histogram of the discriminant function values
ldahist(data = plda$x[,1], g=data.cleaned.test$TD)

# create a scatterplot of the discriminant function values
plot(plda$x[,1], type="n", ylab=c("LDA Axis 1"))
text(plda$x[,1], row.names(data.cleaned.test),  col=c(as.numeric(data.cleaned.test$TD)+10))

# Compute the misclasification error of the model
ct <- table(data.cleaned.test$TD, plda$class)
(ct[1,1]+ct[2,2])/sum(ct) 









# Model 10: Random Forest on all variables
fit10.1= tree(TD~., data=data.cleaned.train)
plot(fit10.1)
text(fit10.1, pretty=0)
fit10.1$frame
fit10.1.result=summary(fit10.1)
fit10.1.result$dev
#xyz=summary(glm(TD~nr.employed+euribor3m+month,data.cleaned.train, family=binomial(logit)))
#names(xyz)
#RSS.LogReg=(4000-4)*((xyz)$deviance)^2

fit.tree=rpart(TD~., data.cleaned.train)
fancyRpartPlot(fit.tree)   # The plot shows the split together with more information
fit.tree$frame

# Split on gini
fit10.1.gini=tree(TD~., data.cleaned.train, split="gini")  
plot(fit10.1.gini)
text(fit10.1.gini, pretty=TRUE)  # plot the labels
fit10.1.gini$frame 
summary(fit10.1.gini)$dev


#Bootstrap
RSS=0  # initial values
n.unique=0
n=nrow(data.cleaned.train)
for (i in 1:100)
{
  index1=sample(n, n, replace=TRUE)   
  Sample1=data.cleaned.train[index1, ]  # Take a bootstrap sample
  fit1.boot=tree(TD~., Sample1)  # Get a tree fit
  plot(fit1.boot, 
       title="Trees with a Bootstrap sample")  
  text(fit1.boot, pretty=0)
  RSS[i]=summary(fit1.boot)$dev  # output RSS for each bootstrap tree
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


#Random Forest
rf.error.p=1:19 
for (p in 1:19)
{
  fit.rf=randomForest(TD~., data.cleaned.train, mtry=p, ntree=100)
  rf.error.p[p]=fit.rf$mse[100]
}
rf.error.p   
plot(1:19, rf.error.p, pch=16,
     xlab="mtry",
     ylab="mse of mtry")  

# For a fixed mtry= 4
fit10.2=randomForest(TD~., data.cleaned.train, mtry=4, ntree=100)
str(fit10.2)
plot(fit10.2)
summary(fit10.2)

plot(fit10.2$mse, xlab="number of trees",
     ylab="ave mse of the 100 trees",
     pch=16) 

# oob times for each obs'n
fit10.2$oob.times  # Out of bags for each observation.
hist(fit10.2$oob.times)

trainingerror=mean((fit10.2$y-fit10.2$predicted)^2) #  this will output the oob errors

pred1=predict(fit10.2, data.cleaned.test) # make predictions on the test data
pred1.train=predict(fit10.2, data.cleaned.train) # predictions on trained data

try1=rep("0", 1000)   
try1[pred1 > 0.9]="1"  
try1=as.factor(try1)
data.frame(data.cleaned.test$TD, try1) # put observed y and predicted y's together
cm=table(try1, data.cleaned.test$TD)
confusionMatrix(data=try1, data.cleaned.test$TD)
try1.mce=mean(try1 != data.cleaned.test$TD)

try1.train=rep("0", 4000)   
try1.train[pred1.train > 0.9]="1"  
try1.train=as.factor(try1.train)

abc=prediction(pred1,data.cleaned.test$TD)
AUC = as.numeric(performance(abc, "auc")@y.values)
ACC= performance(abc, "acc")
mean(ACC@y.values[[1]])
plot(performance(abc, 'tpr', 'fpr'))
plot(ACC)

Sensitivity= performance(abc, "sens")
mean(Sensitivity@y.values[[1]])
Specificity= performance(abc, "spec")
mean(Specificity@y.values[[1]])











# Model 11: Random Forest on variables obtained from Regsubsets
fit11.1= tree(TD~month+poutcome+emp.var.rate+cons.price.idx
              +loan, data=data.cleaned.train)
plot(fit11.1)
text(fit11.1, pretty=0)
fit11.1$frame
fit11.1.result=summary(fit11.1)
fit11.1.result$dev

fit.tree=rpart(TD~month+poutcome+emp.var.rate+cons.price.idx
               +loan, data.cleaned.train)
fancyRpartPlot(fit.tree)   # The plot shows the split together with more information
fit.tree$frame

# Split on gini
fit11.1.gini=tree(TD~month+poutcome+emp.var.rate+cons.price.idx
                  +loan, data.cleaned.train, split="gini")  
plot(fit11.1.gini)
text(fit11.1.gini, pretty=TRUE)  # plot the labels
fit11.1.gini$frame 
summary(fit11.1.gini)$dev


#Bootstrap
RSS=0  # initial values
n.unique=0
n=nrow(data.cleaned.train)
for (i in 1:100)
{
  index1=sample(n, n, replace=TRUE)   
  Sample1=data.cleaned.train[index1, ]  # Take a bootstrap sample
  fit1.boot=tree(TD~month+poutcome+emp.var.rate+cons.price.idx
                 +loan, Sample1)  # Get a tree fit
  plot(fit1.boot, 
       title="Trees with a Bootstrap sample")  
  text(fit1.boot, pretty=0)
  RSS[i]=summary(fit1.boot)$dev  # output RSS for each bootstrap tree
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


#Random Forest
rf.error.p=1:4 
for (p in 1:4)
{
  fit.rf=randomForest(TD~month+poutcome+emp.var.rate+cons.price.idx
                      +loan, data.cleaned.train, mtry=p, ntree=500)
  rf.error.p[p]=fit.rf$mse[500]
}
rf.error.p   
plot(1:4, rf.error.p, pch=16,
     xlab="mtry",
     ylab="mse of mtry")  

# For a fixed mtry= 2
fit11=randomForest(TD~month+poutcome+emp.var.rate+cons.price.idx
                   +loan, data.cleaned.train, mtry=2, ntree=500)
str(fit11)
plot(fit11)
summary(fit11)

plot(fit11$mse, xlab="number of trees",
     ylab="ave mse of the 500 trees",
     pch=16) 

# oob times for each obs'n
fit11$oob.times  # Out of bags for each observation.
hist(fit11$oob.times)

trainingerror=mean((fit11$y-fit11$predicted)^2) #  this will output the oob errors

pred2=predict(fit11, data.cleaned.test) # make predictions on the test data
pred2.train=predict(fit11, data.cleaned.train) # make predictions on the train data

try2=rep("0", 1000)   
try2[pred2 > 0.9]="1"  
try2=as.factor(try2)
data.frame(data.cleaned.test$TD, try2) # put observed y and predicted y's together
cm=table(try2, data.cleaned.test$TD)
confusionMatrix(data=try2, data.cleaned.test$TD)
try2.mce=mean(try2 != data.cleaned.test$TD)

try2.train=rep("0", 4000)   
try2.train[pred2.train > 0.9]="1"  
try2.train=as.factor(try2.train)

abc=prediction(pred2,data.cleaned.test$TD)
AUC = as.numeric(performance(abc, "auc")@y.values)
ACC= performance(abc, "acc")
mean(ACC@y.values[[1]])
max(ACC@y.values[[1]])
plot(performance(abc, 'tpr', 'fpr'))
plot(ACC)

Sensitivity= performance(abc, "sens")
mean(Sensitivity@y.values[[1]])
Specificity= performance(abc, "spec")
mean(Specificity@y.values[[1]])










# Model 12: Random Forest on variables obtained from LASSO
fit12.1= tree(TD~job+default+contact+month+campaign+poutcome+emp.var.rate
              +nr.employed, data=data.cleaned.train)
plot(fit12.1)
text(fit12.1, pretty=0)
fit12.1$frame
fit12.1.result=summary(fit12.1)
fit12.1.result$dev

fit.tree=rpart(TD~job+default+contact+month+campaign+poutcome+emp.var.rate
               +nr.employed, data.cleaned.train)
fancyRpartPlot(fit.tree)   # The plot shows the split together with more information
fit.tree$frame

# Split on gini
fit12.1.gini=tree(TD~job+default+contact+month+campaign+poutcome+emp.var.rate
                  +nr.employed, data.cleaned.train, split="gini")  
plot(fit12.1.gini)
text(fit12.1.gini, pretty=TRUE)  # plot the labels
fit12.1.gini$frame 
summary(fit12.1.gini)$dev


#Bootstrap
RSS=0  # initial values
n.unique=0
n=nrow(data.cleaned.train)
for (i in 1:100)
{
  index1=sample(n, n, replace=TRUE)   
  Sample1=data.cleaned.train[index1, ]  # Take a bootstrap sample
  fit1.boot=tree(TD~job+default+contact+month+campaign+poutcome+emp.var.rate
                 +nr.employed, Sample1)  # Get a tree fit
  plot(fit1.boot, 
       title="Trees with a Bootstrap sample")  
  text(fit1.boot, pretty=0)
  RSS[i]=summary(fit1.boot)$dev  # output RSS for each bootstrap tree
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


#Random Forest
rf.error.p=1:7 
for (p in 1:7)
{
  fit.rf=randomForest(TD~job+default+contact+month+campaign+poutcome+emp.var.rate
                      +nr.employed, data.cleaned.train, mtry=p, ntree=500)
  rf.error.p[p]=fit.rf$mse[500]
}
rf.error.p   
plot(1:7, rf.error.p, pch=16,
     xlab="mtry",
     ylab="mse of mtry")  

# For a fixed mtry= 3
fit12=randomForest(TD~job+default+contact+month+campaign+poutcome+emp.var.rate
                     +nr.employed, data.cleaned.train, mtry=3, ntree=500)
str(fit12)
plot(fit12)
summary(fit12)

plot(fit12$mse, xlab="number of trees",
     ylab="ave mse of the 500 trees",
     pch=16) 

# oob times for each obs'n
fit12$oob.times  # Out of bags for each observation.
hist(fit12$oob.times)

trainingerror=mean((fit12$y-fit12$predicted)^2) #  this will output the oob errors

pred3=predict(fit12, data.cleaned.test) # make predictions on the test data
pred3.train=predict(fit12, data.cleaned.train) # make predictions on the train data

try3=rep("0", 1000)   
try3[pred3 > 0.9]="1"  
try3=as.factor(try3)
data.frame(data.cleaned.test$TD, try3) # put observed y and predicted y's together
cm=table(try3, data.cleaned.test$TD)
confusionMatrix(data=try3, data.cleaned.test$TD)
try3.mce=mean(try3 != data.cleaned.test$TD)

try3.train=rep("0", 4000)   
try3.train[pred3.train > 0.9]="1"  
try3.train=as.factor(try3.train)

abc=prediction(pred3,data.cleaned.test$TD)
AUC = as.numeric(performance(abc, "auc")@y.values)
ACC= performance(abc, "acc")
max(ACC@y.values[[1]])
plot(performance(abc, 'tpr', 'fpr'))
plot(ACC)

Sensitivity= performance(abc, "sens")
mean(Sensitivity@y.values[[1]])
Specificity= performance(abc, "spec")
mean(Specificity@y.values[[1]])









# Ensemble of all the above models
#Model1
df1.1=data.frame(fit1.pred.train)
colnames(df1.1)="TD.Label.Predicted1"
#Model2
df2.1=data.frame(fit2.pred.train)
colnames(df2.1)="TD.Label.Predicted2"
#Model3
df3.1=data.frame(fit3.pred.train)
colnames(df3.1)="TD.Label.Predicted3"
#Model7
df7.1=data.frame(plda.class.train.1)
colnames(df7.1)="TD.Label.Predicted7"
#Model8
df8.1=data.frame(plda.class.train.2)
colnames(df8.1)="TD.Label.Predicted8"
#Model9
df9.1=data.frame(plda.class.train.3)
colnames(df9.1)="TD.Label.Predicted9"
#Model10
df10.1=data.frame(try1.train)
colnames(df10.1)="TD.Label.Predicted10"
#Model11
df11.1=data.frame(try2.train)
colnames(df11.1)="TD.Label.Predicted11"
#Model12
df12.1=data.frame(try3.train)
colnames(df12.1)="TD.Label.Predicted12"

# combine all the dataframes
df.predictions=cbind(df1.1, df2.1, df3.1, df7.1, df8.1, df9.1, df10.1, df11.1, df12.1)
str(df.predictions)
# convert the dataframe to numeric type
indx <- sapply(df.predictions, is.factor)
df.predictions[indx] <- lapply(df.predictions[indx], function(x) as.numeric(as.character(x)))
#get mean value of predictions
df.predictions.mean=data.frame(Mean.Prediction=rowMeans(df.predictions))

# Final Classifier By Equal Weights
try.final1=rep("0", 4000)   
try.final1[df.predictions.mean > 0.5]="1"  
try.final1=as.factor(try.final1)
data.frame(data.cleaned.train$TD, try.final1) # put observed y and predicted y's together
cm=table(try.final1, data.cleaned.train$TD)
confusionMatrix(data=try.final1, data.cleaned.train$TD)
try.final1.mce=mean(try.final1 != data.cleaned.train$TD)


# Linear Regression 
df.main=data.frame(data.cleaned.train$TD, df.predictions)

fitfinal1=lm(data.cleaned.train.TD~., data=df.main)
summary(fitfinal1)

fitfinal2=update(fitfinal1, .~. -TD.Label.Predicted2 -TD.Label.Predicted3) # Removed coorelated variables
summary(fitfinal2)







# FINAL MODEL!!!!!!!!!!!!!!!!!!
# Random Forest

#Divide the dataset into training and testing
set.seed(1)  # set a random seed so that we will be able to reproduce the random sample
index.train1=sample(dim(df.main)[1], 3500) # Take a random sample of n=3500 from 1 to N=4000
df.main.train=df.main[index.train1,] # Set the 500 randomly chosen subjects as a training data
df.main.test=df.main[-index.train1,]

# Fit a random forest tree
fit.tree123=randomForest(data.cleaned.train.TD~., df.main.train, mtry=3, ntree=500)
fit.tree1234=rpart(data.cleaned.train.TD~., df.main.train)
fancyRpartPlot(fit.tree1234)   # The plot shows the split together with more information
fit.tree1234$frame
summary(fit.tree123)
plot(fit.tree123)

finaltree.pred=predict(fit.tree123, df.main.test)
try5=rep("0", 500)   
try5[finaltree.pred > 0.5]="1"  
try5=as.factor(try5)
data.frame(df.main.test$data.cleaned.train.TD, try5) # put observed y and predicted y's together
cm=table(try5, df.main.test$data.cleaned.train.TD)
confusionMatrix(data=try5, df.main.test$data.cleaned.train.TD)
try5.mce=mean(try5 != df.main.test$data.cleaned.train.TD)
abc1=prediction(finaltree.pred, df.main.test$data.cleaned.train.TD)
AUC1 = as.numeric(performance(abc1, "auc")@y.values)
ACC1= performance(abc1, "acc")
max(ACC1@y.values[[1]])
plot(performance(abc1, 'tpr', 'fpr'))
plot(ACC1)

Sensitivity= performance(abc1, "sens")
mean(Sensitivity@y.values[[1]])
Specificity= performance(abc1, "spec")
mean(Specificity@y.values[[1]])
