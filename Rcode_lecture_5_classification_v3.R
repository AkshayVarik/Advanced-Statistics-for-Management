
### Classification through logistic regression
## This lecture consists of materials that are either scattered 
## in the text book or not even covered in the book. You 
## will need to rely on my lecture.

## Rearrange the lecture and add linear boundary!!!!!!!!!

#  1) Classification rules through p(y=1|x)
#  2) Misclassification errors
#       a) Sensitivity
#       b) Specificity (False Positive)
#       c) Misclassification errors
#       d) ROC curves and AUC
#  3) Bayes rules
#     - Unequal costs
#     - Weighted Mis-Classification Errors
#  4) Training and Testing errors



###################################################
##  Need package pROC()
# install.packages("pROC")
# list all the packages installed and their versions
as.data.frame(installed.packages())[, c(1, 3)]
library(pROC)   #http://cran.r-project.org/web/packages/pROC/pROC.pdf
###################################################

# Case Study: Framingham Heart Study. The goal is given a set
# of conditions, we need to classify the person to be HD=1 or HD=0.
# To be more specific:

#  1) Given a person with the following features
#         AGE    SEX  SBP DBP CHOL FRW CIG
#         45  FEMALE  100  80  180 110   5
#     We want to classify her to be HD=1 or 0.
#  2) Data: 1406 health professionals. Conditions gathered at the beginning 
#          of the study (early 50th). Both the original subjects and their next generations
#          have been included in the study. HD=0=No Heart Disease
  

######### Set up working directory###########

rm(list=ls()) # Remove all the existing variables
dir=c("E:/Data Mining- STAT 571")   # my laptop
setwd(dir)

### Read Framingham data/ Clean it little bit.

data = read.table("Framingham.dat", sep=",", header=T, as.is=T)
# Renames, setting the variables with correct natures...
names(data)[1]="HD"
data$HD=as.factor(data$HD)
data$SEX=as.factor(data$SEX)
data.new=data[1407,] # The female whose HD will be predicted.
data=data[-1407,]  # take out the last row 
#############




######### Classifications
# Use Logistic reg model to estimate the prob's 
# Classify subjects into "1" or "0" by setting a threshold on prob's of y=1.
# Start with using SBP alone.

fit1=glm(HD~SBP, data, family=binomial(logit))  # Logistic fit
summary(fit1)

# Classifier 1: hat Y=1 if p(y=1|x) >2/3

fit1.pred.67=rep("0", 1406)   # prediction step 1
fit1.pred.67[fit1$fitted > 2/3]="1"  # prediction step 2 to get a classifier
fit1.pred.67=as.factor(fit1.pred.67)

# Or through classification boundary:
# exp(-3.66+.0159SBP)/(1+exp(-3.66+.0159SBP)) >2/3 is same as
# -3.66+.0159SBP > log(2) which is equivalent to 
# hat y=1 if SBP > (log(2)+3.66)/.0159=273.78.  This is called the classification boundary. 

# Or log(odds ratio)=-3.66+.0159SBP
#  On the left: log((2/3)/(1/3))=-3.66+.0159SBP
#                         log(2)=-3.66+.0159SBP


## Let's plot the classifier boundary

par(mfrow=c(1,1))
plot(jitter(as.numeric(data$HD), factor=1) ~ data$SBP, 
     pch=as.numeric(data$HD)+2, col=data$HD, 
     ylab="Obs'n", xlab="SBP")
abline(v=273.78, lwd=5, col="blue")
title("All the subjects on the right side
      will be classified as HD=1")
legend("topleft", legend=c("0", "1"),
       lty=c(1,1), lwd=c(2,2), col=unique(data$HD))

#############


##### Misclassification error:

# a) Sensitivity: Prop(hat Y = 1| Y=1). (Not an error)
#    True Positive Rate= P( Classified as positive | Positive)

# b) Specificity: Prop(hat Y = 0| Y=0)
#    Specificity = P( Classified as Negative | Negative)

#    False Positive=1-Specificity=P( hat Y=1 | Y=0)
#    False Positive = P( Classified as Positive| Negative)

# c) Misclassification error= Mean(miss-classification)

# We can get all three quantities through confusion matrix or directly find the 
# mis-classification errors


data.frame(data$HD, fit1.pred.67, fit1$fitt)[sample(1406, 10),] 
# put observed y and predicted y's together, randomly take 10 subjects
# we see there are many mislabels.

# confusion matrix: a 2 by 2 table 
cm.67=table(fit1.pred.67, data$HD) # confusion matrix: 
cm.67

# fit1.pred    0    1
#         0 1090  310
#         1    5    1

sensitivity=cm.67[2,2]/sum(data$HD == "1")  # 1/311
sensitivity

### Specificity = P( Classified as Negative | Negative)
### False Positive = P( Classified as Positive| Negative)

specificity=cm.67[1,1]/ sum(data$HD == "0")
false.positive=cm.67[2,1]/sum(data$HD == "0")  # 5/1095
false.positive

### Mis-classification error (MCE): 
error.training=(cm.67[1,2]+cm.67[2,1])/length(fit1.pred.67) # training error
error.training

# An alternative formula to get misclassification error :
sum(fit1.pred.67 != data$HD)/length(fit1.pred.67)  #or
mce.67=mean(fit1.pred.67 != data$HD)
mce.67   #0.224


### If we change the threshold, we get a different classifier
### Example 2. hat y= 1 if P(y=1|SBP) > .5
#   1) To get the boundary: we set
#   log(odds ratio)=-3.66+.0159SBP=log{(1/2)/(1/2)}=log(1)=0
#     hat y =1 if SBP > 3.66/.0159=230.18

fit1.pred.5=rep("0", nrow(data) )
fit1.pred.5[data$SBP > 230.18] ="1"  # Notice a different way of doing so.

cm.5=table(fit1.pred.5, data$HD)
cm.5

# fit1.pred.5    0    1

#           0 1084  302
#           1   11    9

sensitivity=cm.5[2,2]/sum(data$HD == "1")
sensitivity
false.positive=cm.5[2,1]/sum(data$HD == "0")
false.positive

mce.5=mean(fit1.pred.5 != data$HD)
mce.5   #.223

### Putting two classifiers together
### Put both classifiers together
par(mfrow=c(1,1))
plot(jitter(as.numeric(data$HD), factor=1) ~ data$SBP, 
     pch=as.numeric(data$HD)+2, col=data$HD, 
     ylab="Obs'n", xlab="SBP")
legend("topleft", legend=c("0", "1"),
       lty=c(1,1), lwd=c(2,2), col=unique(data$HD))

abline(v=273.78, lwd=5, col="blue")
abline(v=230.4, lwd=5, col="red")
title("Two classifiers based on SBP: red:prob>.5,
      blue: prob>2/3")



####### ROC curve and AUC:
#### For each model or process, 
#### given a threshold, or a classifier, 
#### there will be a pair of sensitivity and specificity.
#### We will graph all the pairs of False Positive as x 
#### and True Positive as y to have a curve: ROC

library(pROC)
fit1=glm(HD~SBP, data, family=binomial(logit))
fit1.roc=roc(data$HD, fit1$fitted, plot=T, col="blue")  
# Notice the ROC curve here is Sensitivity vs. Specificity
# Most of the ROC is drawn using False Positive rate as x.

names(fit1.roc)
auc(fit1.roc)    # area under the curve, the larger the better.


##### False Positive vs. Sensitivity curve is called ROC 
plot(1-fit1.roc$specificities, fit1.roc$sensitivities, col="red", pch=16,
     xlab="False Positive", 
     ylab="Sensitivity")

### We can get more from fit1.roc.
#### Given a False positive rate, locate the prob threshold
plot(1-fit1.roc$specificities, fit1.roc$thresholds, col="green", pch=16,  
     xlab="False Positive",
     ylab="Threshold on prob")

### Which rule to use???? Depends on your criterion.

###############################################################
##### If we use different model we will have different set of 
# classifiers. For example if we take more features: AGE and SBP

fit2=glm(HD~SBP+AGE, family= binomial, data)
summary(fit2)

# logit=-6.554+0.0144SBP+.0589AGE

# rule 1: theshodling the prob at .67
#   -6.554+0.0144SBP+.0589AGE = log(2)
#    SBP=-.0589/.0144 AGE + (log(2)+6.554)/.0144
#    SBP=-4.09 AGE+503, this is called linear boundary
#    Y hat = 1 if SBP > -4.09 AGE+503

# rule 2: thresholding the prob at .5
#   -6.554+0.0144SBP+.0589AGE = log(1)
#    SBP=-4.09 AGE+455


#Let's put two linear boundaries together.

plot(jitter(data$AGE, factor=1.5), data$SBP, col=data$HD, 
     pch=as.numeric(data$HD)+2,
     xlab="AGE", ylab="SBP")
legend("topleft", legend=c("HD=1", "HD=0"),
       lty=c(1,1), lwd=c(2,2), col=c("red", "black"))
abline(a=455, b=-4.09, lwd=5, col="red")
abline(a=503, b=-4.09, lwd=5, col="blue")
title("Linear Boundary. red: threshod at .5, blue: at 2/3")


# We can get sensitivity, false positive rate and MCE for each
# rules.

### Here is the ROC curve based on fit2

fit2.roc=roc(data$HD, fit2$fitted, plot=TRUE)
auc(fit2.roc)   #.654
names(fit2.roc)  
data.frame(fit2.roc$thresh, fit2.roc$sen, fit2.roc$spe)[1: 10, ]
# take a look at a few rows of fit2.roc


### Compare fit1 and fit2 by overlaying the two ROC's

plot(1-fit1.roc$specificities, fit1.roc$sensitivities, col="red", pch=16,
     xlab="False Positive", 
     ylab="Sensitivity")
points(1-fit2.roc$specificities, fit2.roc$sensitivities, col="blue", pch=16)
title("Blue line is for fit2, and red for fit1")

auc(fit1.roc)  # .637
auc(fit2.roc)  # .654

## Question: Which model should we use????


###############################################
####### Bayes rule with unequal losses ########

# Let a_{1,0}=L(Y=1, hat Y=0), the loss (cost) of making a "1" to a "0"
# Let a_{0,1}=L(Y=0, hat Y=1), the loss of making a "0" to a "1"

# Then 
# mean(L(Y, hat Y=1))=P(Y=1) * L(Y=1, hat Y=1) + P(Y=0) * L(Y=0, hat Y=1)
#                    =a_{0,1} * P(Y=0)
# Similarly, 
# mean(L(Y, hat Y=0))=a_{1,0} * P(Y=1)

# To minimize the two mean losses, we choose 
# hat y=1 if mean(L(Y, hat Y=1)) < mean(L(Y, hat Y=0)).

# Plugging in: 
#               a_{0,1} * P(Y=0) < a_{1,0} * P(Y=1)

# which is equivalent to
# hat y=1 if    P(Y=1|x)/P(Y=0|x) > a_{0,1}/a_{1,0} 
# OR 
#               P(Y=1|x) > (a_{0,1}/a_{1,0})/(1+(a_{0,1}/a_{1,0}) )

# The above rule is called Bayes rule.

### An example: Suppose (a_{0,1}/a_{1,0})=1/5=.2, then 
#The threshold over the prob(Y=1|x) > .2/(1+.2)=.17 or

# logit > log(.17/.83)=-1.59 gives us the Bayes rule!!!

#If we use 
fit2=glm(HD~SBP+AGE, data, family=binomial)
summary(fit2)  # logit=-6.554+0.0144SBP+.0589AGE

# The Bayes linear boundary is -6.554+0.0144SBP+.0589AGE = -1.59
#   0.0144SBP+.0589AGE=-1.59+6.554
#   SBP=-4.09AGE + 344.7 

#Let's draw the linear boundary of the Bayes rule when a10/a01=5 

plot(jitter(data$AGE, factor=1.5), data$SBP, col=data$HD, 
     pch=as.numeric(data$HD)+2,
     xlab="AGE", ylab="SBP")
legend("topleft", legend=c("HD=1", "HD=0"),
       lty=c(1,1), lwd=c(2,2), col=c("red", "black"))
abline(a=344.7, b=-4.09, lwd=5, col="red")
title("Linear Boundary of the Bayes Rule, when a10/a01=5")


### Finally we get the weighted mis-classification error
# MCE=(a10 sum(y != hat y|y=1) + a01 sum(y != hat y|y=0))/n

# Get the classes first
fit2.pred.bayes=rep("0", 1406)
fit2.pred.bayes[fit2$fitted > .17]="1" 

MCE.bayes=(sum(5*(fit2.pred.bayes[data$HD == "1"] != "1")) 
           + sum(fit2.pred.bayes[data$HD == "0"] != "0"))/length(data$HD)
MCE.bayes # .7
  
# On the other hand if we were to use 1/2 as the prob threshold then the
# weighted loss would be
fit2.pred.5=rep("0", 1406)
fit2.pred.5[fit2$fitted > .5]="1" 
MCE.5=(sum(5*(fit2.pred.5[data$HD == "1"] != "1")) + sum(fit2.pred.5[data$HD == "0"] != "0"))/length(data$HD)
MCE.5  #1.07

## Alternatively

fit2.pred.5=factor(ifelse(fit2$fitted > .5, "1", "0"))
MCE.5=(sum(5*(fit2.pred.5[data$HD == "1"] != "1")) + sum(fit2.pred.5[data$HD == "0"] != "0"))/length(data$HD)
MCE.5


###### Training and Testing errors

# In order to evaluate the performance of
# each procedure, we need to estimate errors using unseen data. 

# Split the data to two subsamples.
# Training Data: fit a model
# Testing Data: estimate the performance

###### Will choose the model, for example, the one with the largest AUC 
#########################################################################


# Split the data:
data1=na.omit(data)
N=length(data1$HD)
#set.seed(10)  # set a random seed so that we will be able to reproduce the random sample
index.train=sample(N, 1000) # Take a random sample of n=1000 from 1 to N=1393
data.train=data1[index.train,] # Set the 1000 randomly chosen subjects as a training data
data.test=data1[-index.train,] # The remaining subjects will be reserved for testing purposes.
dim(data.train)
dim(data.test)

# Let us compare two models
fit1.train=glm(HD~SBP, data=data.train, family=binomial)
summary(fit1.train)

fit5.train=glm(HD~SBP+SEX+AGE+CHOL+CIG, data=data.train, family=binomial)
summary(fit5.train)

# Get the fitted prob's using the testing data

fit1.fitted.test=predict(fit1.train, data.test, type="response") # fit1 prob
fit5.fitted.test=predict(fit5.train, data.test, type="response") # fit5 prob

data.frame(fit1.fitted.test, fit5.fitted.test)[1:20, ]  # look at the first 20 rows. Notice
# row names is the subject number chosen in the testing data
# estimated prob for each row differ by using fit1 and fit5

# Compare the performances with the testing data

fit1.test.roc=roc(data.test$HD,fit1.fitted.test, plot=T )
fit5.test.roc=roc(data.test$HD,fit5.fitted.test, plot=T )

# Overlaying the two ROC curves using testing data:

plot(1-fit5.test.roc$specificities, fit5.test.roc$sensitivities, col="red", pch=16,
     xlab=paste("AUC(fit5.test)=",round(auc(fit5.test.roc),2),"  AUC(fit1.test)=",round(auc(fit1.test.roc),2) ), 
     ylab="Sensitivities")   # For some reason it needs a return here?

points(1-fit1.test.roc$specificities, fit1.test.roc$sensitivities, col="blue", pch=16)
legend("topleft", legend=c("fit5.test w five variables", "fit1.test w one variable"),
       lty=c(1,1), lwd=c(2,2), col=c("red", "blue"))
title("Comparison of two models using testing data: one with x1, the other with x1-x5")

###### Conclusion: we will use fit5 for its larger testing auc curve....

## Note: If you comment out set.seed and repeat running the above training/testing
#session, what have you observed from the two ROC curves and the two AUC values?
# Are you surprised?


