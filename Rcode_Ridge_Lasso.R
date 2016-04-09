###### LASSO and Ridge Regression

#1) Ridge Regression 
#2) LASSO (Least Absolute Shrinkage and Selection Operator)
#3) Elastic Net: Combination of Ridge and LASSO
    # Penalty function (1-α)/2||β||_2^2+α||β||_1. 
    # α=0 gives us Ridge, α= 1 gives us lasso
#4) Final fit through regsubsets/lm


### Case study: What is a set of important factors relate to violent crime rates?
# The data set is regarding the crimes and various other useful information about the population and police
# enforcement in an sample of communities from almost all the states. 

### The data set aggregate socio-economic information, law enforcement data from 1990 and 
  # the crime data in 1995. There are 147 variables, among which 18 variables are various crimes.
  # The definition of each variable is self-explanatory by names. The data version here is her clean data
  # We are using violentcrimes.perpop: violent crimes per 100K people in 1995

library(leaps)
library(glmnet)  # glmnet{glmnet} is used for Lasso/Ridge and Elastic net in general



### Read the data and a quick exploration ##########
####################################################
rm(list=ls()) # Remove all the existing variables

dir=c("E:/Data Mining- STAT 571")   # my laptop
setwd(dir)

data=read.csv("CrimeData.csv", header=T, na.string=c("", "?"))
names(data)
dim(data)      #2215 communities, 147 variables. The first variable is the identity. 
str(data)
####

### A quick exploration of the data
#######################################


sum(is.na(data)) # On average there are 20 missing value for each community
data1=data[,c(2,6:103,121,122,123, 130:147)]  # take a subset by leaving the variables about 
    # police departments out, because of large amount of missing values. 
    # They are column 104:120, and 124:129; 
    # Col 130-147 are various crimes. Any concerns of doing so????
names(data1)

#### Crime variables
summary(data1[, 103:120])  # crime information
pairs(data1[, 103:120]) # how different crimes are correlated. The pairwise scatter plots are hard to see. I am attempting to do
 # some simple missing value imputations. One way we could have done is to regress violentcrimes.perpop vs. other crimes and use
 # predicted values to fill in missing values.


data2=data1[-c(103:118, 120)] # take other crimes out. 
names(data2)
str(data2)
data.fl=data2[data2$state=="FL",-1] # take state out
data.ca=data2[data2$state=="CA",-1]

#write.csv(data.fl, "CrimeData_FL", row.names = FALSE)


data3=na.omit(data2) # no missing values
################### Data preparations #################


#### Three data sets
 #data1: all variables without information about police departments due to too many missing values
 #data2: data1 and exclude all crime statistics but violent crimes
 #data3: data2 with no missing values
 #data.fl: a subset of data2 for FL
 #data.ca: a subset of data2 for CA



### Ordinary lm first
### Try lm() first. What are the difference among the two models fitted below????

# Model 1
fit.lm=lm(violentcrimes.perpop~., data=data2)
summary(fit.lm)    
## warning: 222 observations deleted due to missing values!!!!!!!
## Notice that we  couldn't produce the coefficients for
## two variables "ownoccup.qrange" and "rent.qrange", why not?????


# Model 2: we concentrate on FL alone.

data.fl=data2[data2$state=="FL",-1]   # subset for FL and take state out of the subset
dim(data.fl)     #n=90, p=102. The first col is Y Here it is in the situation that number of predictors is larger than number of the obs'n!!!
sum(is.na(data.fl))
names(data.fl)
str(data.fl)
fit.fl.lm=lm(violentcrimes.perpop~., data=data.fl)
summary(fit.fl.lm)  

# oops: no degrees of freedom left for errors! p is larger than n!!!

##### What about we try forward or exhaustive search by setting nvmax=20 say.
# Why we can't do backward selection?

# Take "ownoccup.qrange", "rent.qrange" out of the data due to collinearity.

data.fl1=data.fl[, -which( names(data.fl) %in% c("ownoccup.qrange", "rent.qrange"))]

fit.fl.ms=regsubsets(violentcrimes.perpop~., nvmax=10, method="exhau", really.big=TRUE, data.fl1)
# NO, it almost crushed my lovely laptop!
# Have to buy a better one because of this job.

# Tried forward selection and it worked though.

####### Give up the idea of using lm(). Let's try to cut a large number of "not useful" predictors
# first, then come back with lm()...





########################################
#### Ridge/LASSO Regression/Elasticnet. ###########
########################################

# Given a model, we minimize the residual sum of squares,
# but with a penalty term. Lambda is a tuning parameter.

# min RSS + \lambda (1-α)/2||β||_2^2+α||β||_1

#Let us concentrate on FL and CA for the moment
library(glmnet)     # install package glmnet
help(glmnet )       #(1-α)/2||β||_2^2+α||β||_1. α=0 gives us Ridge, α= 1 gives us lasso
                    # X, the design matrix is standardized by default. The output is transformed
                    # back to the original scales.
# glmnet 
# min 1/2 (RSS/n) + \lambda (1-α)/2||β||_2^2+α||β||_1

#### First for FL #######
########################

#1) Ridge Regression

#### Ridge reg: alpha=0

### We need to extract the X variables first
data.fl=read.csv("CrimeData_FL")
dim(data.fl)         # n=90, p=102 

X.fl=model.matrix(violentcrimes.perpop~., data=data.fl)[, -1]
     # get X variables as a matrix. it will also code the categorical 
     # variables correctly!. The first col of model.matrix is violentcrimes.perpop

dim(X.fl)  
colnames(X.fl)

# OR
X.fl=(model.matrix(~., data=data.fl[, -102]))[, -1]  # take the 1's out. 
typeof(X.fl)
dim(X.fl)            # 90 by 101

Y=data.fl[,102]     # extract the response var



### Ready to get Ridge Reg or LASSO

# glmnet with a specified lambda
fit.fl.lambda=glmnet(X.fl, Y, alpha=0, lambda=100 )    

# (1-α)/2||β||_2^2+α||β||_1. α=0 gives us Ridge, α= 1 gives us lasso
names(fit.fl.lambda)
fit.fl.lambda$beta  # The coefficients are functions of lambda. It only outputs the coefs of features.
                    # Notice none of the coef's are 0 though a lot of them are very close to 0
fit.fl.lambda$df    # number of non-zero coeff's
fit.fl.lambda$a0    # est. of beta_0 
coef(fit.fl.lambda) # beta hat of predictors in the original scales. 


# If we don't specify lambda, then 100 lambda values (the lambda path) will be given.

fit.fl.lambda=glmnet(X.fl, Y, alpha=0 ) # In this case the output will consists of 100 output, one for each lambda 
names(fit.fl.lambda)
str(fit.fl.lambda)
plot(fit.fl.lambda)  # It shows that each hat beta is shrinking towards 0 as
      # l2 norm of beta is smaller, which is equvalent to lambda is getting larger


## As we expected that beta is a function of lambda.
# Also the lambda values have a huge variability, log is applied here
# Take a look at coef's of household.size as a function of lambda

par(mfrow=c(1,1))
plot(log(fit.fl.lambda$lambda), fit.fl.lambda$beta[2,], 
     xlab="log(lambda)", ylab="beta for household.size", pch=16,col="blue",
     main="coef of household.size as a func of lambda")
#notice the variability of coef's
hist(fit.fl.lambda$beta[2,], breaks=30,
     xlab="coefficients of household.size ")
par(mfrow=c(1,1))


######### WHICH lambda to use?????? We use k fold cross validation

#### Let's explore the effect of lambda. We may use cv.glmnet() to find the best lambda
#cv.glmnet() outputs the nfolds of errors for each lambda
help(cv.glmnet)   


fit.fl.cv=cv.glmnet(X.fl, Y, alpha=0, nfolds=10)  
# by default nfolds=10, the smallest value can be 3 and largest can be n
# type.measure = "deviance" by default which is mse in our case
# type.measure="class" will be classification error
# type.measure can also be "auc", "mse"


# the cv.glmnet chooses a set of lambda's

names(fit.fl.cv)
plot(fit.fl.cv$lambda)      # There are 100 lambda values used
fit.fl.cv$cvm               # the mean cv errors for 100 lambda's

#plot(log(fit.fl.cv$lambda), fit.fl.cv$cvm, xlab="log(lambda)", ylab="mean cv errors",
#     pch=16, col="red")

plot(fit.fl.cv$lambda, fit.fl.cv$cvm, xlab="lambda", ylab="mean cv errors",
     pch=16, col="red")

# The minimizer of lambda is the smallest lambda value specified by cv.glm
min(fit.fl.cv$lambda) # [1] 7448.86

# We choose our own lambda

lambda.manu=seq(0, 8000, by=160)
fit.fl.cv=cv.glmnet(X.fl, Y, alpha=0, nfolds=10 , lambda = lambda.manu)  
plot(fit.fl.cv$lambda, fit.fl.cv$cvm, xlab="lambda", ylab="mean cv errors",
     pch=16, col="red")

# We could output the minimizer of lambda. But normally we may not use this lambda. More to come later.

fit.fl.cv$lambda.min   # min lambda can change a lot as a function of nfolds!



# Here let's plot a bunch of min lambdas as a function of different partitions of training/testing data

lambda.min=numeric(50)
mse=numeric(50)
for (i in 1:50) {
  fit.fl.cv=cv.glmnet(X.fl, Y, alpha=0, nfolds=5, lambda=lambda.manu ) 
  lambda.min[i]=fit.fl.cv$lambda.min
  mse[i]=fit.fl.cv$cvm
}
par(mfrow=c(1,1))
hist(lambda.min)  # lambda.min varies around 3000. 

## For cv.glmnet also outputs number of none-zero coefficients 

fit.fl.cv$nzero   # number of non-zero coefficients. we notice that all the coef's are non-zeros although
                  # some of them is very small in magnitude.

plot(fit.fl.cv)   # This plot gives us a three dimension view of lambda, mse and number of non-zero coef's which
                  # is very useful for us to choose a final lambda value.



#2) Lasso

### We have seen that Ridge Reg doesn't shrink coefficients all the way to zero. It doesn't really
# do model selection. But LASSO does!!!!!

####### For LASSO: alpha=1
### glmnet() is used again. alpha=1 here. The usage is the same as above...

# Step 0: prepare the input X matrix and the response Y
X.fl=model.matrix(~., data.fl)   # put data.frame into a matrix
colnames(X.fl)
Y=X.fl[, 103]  # extract y

X.fl=X.fl[, -c(1, 103)] # extract design matrix by kicking out the 1's and Y's

#fit.fl.lasso=glmnet(X.fl, Y, alpha=1)
#plot(fit.fl.lasso)  # from this plot we see the effect of L1 norm over
           # the size of each hat beta


# Step 1: choose best lambda
fit.fl.cv=cv.glmnet(X.fl, Y, alpha=1, nfolds=10 )   # alpha=1
plot(fit.fl.cv)
#### Quick review
plot(fit.fl.cv$lambda)      # There are 100 lambda values used
fit.fl.cv$cvm               # the mean cv error
plot(fit.fl.cv$lambda, fit.fl.cv$cvm, xlab="lambda", ylab="mean cv errors")
fit.fl.cv$lambda.min        # min lambda changes a lot as a function of nfolds!
fit.fl.cv$nzero
plot(fit.fl.cv$lambda, fit.fl.cv$nzero, xlab="lambda", ylab="number of non-zeros")
#### Quick review


# use the default plot to explore possible values of lambda.min

plot(fit.fl.cv)
    #the top margin: number of nonzero beta's. 
    #the first v line is lambda.min and the second v line is lambda.1se
    #we may choose any lambda between lambd.min and lambd.1st


# Step 2: Output variables for the lambda chosen

#output beta's from lambda.min, as an example
coef.min=coef(fit.fl.cv, s="lambda.min")  #s=c("lambda.1se","lambda.min")
coef.min=coef.min[which(coef.min !=0),]   # get the non=zero coefficients
coef.min    
rownames(as.matrix(coef.min))     # the set of predictors chosen.


#output beta's from lambda.1se (this way you are using smaller set of variables.)
coef.1se=coef(fit.fl.cv, s="lambda.1se")  
coef.1se=coef.1se[which(coef.1se !=0),] 
coef.1se
rownames(as.matrix(coef.1se))

# we may specify the s value ourselves. We may want to use a number between lambda.min and
# lambda.1se, say we take s=exp(4.6). Not sure we can use cv.glmnet to get all the final variables as
# above when we choose our own lambda


fit.fl.4.6=glmnet(X.fl, Y, alpha=1, lambda = exp(4.6))
coef.4.6=coef(fit.fl.4.6)
coef.4.6
var.names=coef.4.6[which(coef.4.6 !=0), ]
var.names
var.4.6=rownames(as.matrix(var.names))
var.4.6

# [1] "(Intercept)"           "pct.pop.underpov"      "male.pct.divorce"      "pct.kids2parents"     
# [5] "pct.youngkids2parents" "num.kids.nvrmarried"   "pct.kids.nvrmarried"   "pct.people.dense.hh"  
# [9] "med.yr.house.built"    "pct.house.nophone"    

# Step 3: fit the final model. Summary of the findings....

# get the lasso output
fit.fl.4.6=glmnet(X.fl, Y, alpha=1, lambda = exp(4.6))
var.names=coef.4.6[which(coef.4.6 !=0), ]
var.names


# get the lm output
lm.input=as.formula(paste("violentcrimes.perpop", "~", paste(var.4.6[-1], collapse = "+")))
lm.input

fit.fl.4.6.lm=lm(lm.input, data=data.fl)
coef(fit.fl.4.6.lm)
summary(fit.fl.4.6.lm)

# put two sets of output together: notice the prediction equations can be different.
comp=data.frame(var.names,coef(fit.fl.4.6.lm) )
names(comp) =c("estimates from LASSO", "lm estimates")
comp


#3) Elastic net

####### For Elastic Net: alpha should be closed to 1 so that it will do model selection 
#yet benefit from Ridge reg. (A little weight for l2 loss)
fit.fl.lambda=glmnet(X.fl, Y, alpha=.99) 


fit.fl.cv=cv.glmnet(X.fl, Y, alpha=.99, nfolds=10 )  
plot(fit.fl.cv)

log(fit.fl.cv$lambda.min)  # lambda with min mse
log(fit.fl.cv$lambda.1se)  # largest lambda whose mse is within one se of the optimal model.
fit.fl.cv$lambda.min
fit.fl.cv$lambda.1se

# the optimal lambda is about 100 after running a few runs of cv.glmnet
set.seed(10)
fit.fl.final=glmnet(X.fl, Y, alpha=.99, lambda=100)  # the final elastic net fit
beta.final=coef(fit.fl.final)
beta.final=beta.final[which(beta.final !=0),]
beta.final=as.matrix(beta.final)
beta.final
rownames(beta.final)


# The remaining part is to fit a lm model. 
fit.final=lm(violentcrimes.perpop~pct.pop.underpov
             +male.pct.divorce
             +pct.kids2parents
             +pct.youngkids2parents
             +num.kids.nvrmarried
             +pct.kids.nvrmarried
             +pct.people.dense.hh
             +med.yr.house.built
             +pct.house.nophone, data.fl)

summary(fit.final)    # Still some var's are not significant 
#### Need to final tune the model and land one you think it is acceptable!!!!!!


### One more step: regsubset
fit.final=regsubsets(violentcrimes.perpop~pct.pop.underpov
                       +male.pct.divorce
                       +pct.kids2parents
                       +pct.youngkids2parents
                       +num.kids.nvrmarried
                       +pct.kids.nvrmarried
                       +pct.people.dense.hh
                       +med.yr.house.built
                       +pct.house.nophone, nvmax=15, method="exhau",  data.fl)
                  
summary(fit.final)

### 4) Final model through regsubsets()

### The following function inputs a data frame. Output 
# A final model which achieve the following
#  i) All subset selection results
#  ii) Report the largest p-vlues with each model
#  iii) Return the final model who's largest p-value < alpha


# Input: the data frame and the regsubset output= fit.final
# Output: the final model variable names

finalmodel <- function(data.fl, fit.final) # fit.final is an object form regsubsets
{
  p=fit.final$np-1  #  number of  predictors from fit.final
  var.names=c(names(data.fl)[dim(data.fl)[2]], names(coef(fit.final, p))[-1]) # collect all predictors and y
  data1=data.fl[, var.names]  # a subset
 
  temp.input=as.formula(paste(names(data1)[1], "~",
                   paste(names(data1)[-1], collapse = "+"),
                   sep = ""))      # get lm() formula
  
  try.1=lm(temp.input, data=data1)  # fit the current model
  largestp=max(coef(summary(try.1))[2:p+1, 4]) # largest p-values of all the predictors
 
   while(largestp > .05)   #stop if all the predictors are sig at .05 level
  
     {p=p-1  # otherwise move to the next smaller model
  
      var.names=c(names(data.fl)[dim(data.fl)[2]], names(coef(fit.final, p))[-1])
      data1=data.fl[, var.names]
  
      temp.input=as.formula(paste(names(data1)[1], "~",
                              paste(names(data1)[-1], collapse = "+"),
                              sep = ""))      # get lm() formula
  
      try.1=lm(temp.input, data=data1)  # fit the current model
      largestp=max(coef(summary(try.1))[2:p+1, 4]) # largest p-values of all the predictors
      }
  
  finalmodel=var.names
  finalmodel
}

finalmodel(data.fl,fit.final )  # output the final model with all the predictors being significant at level of .05.


c(finalmodel(data.fl,fit.final ))


############### END of the lecture ####################
#######################################################





##### Check a few things
# i) glmnet should output OLS estimates when lambda=0
# ii) the scaling is done properly internally with glmnet()


data5=data.fl[c("violentcrimes.perpop", "num.kids.nvrmarried" , "pct.kids.nvrmarried" , "med.yr.house.built" ,  "pct.house.nophone") ]
dim(data5)
Y=data5[, 1]
X=as.matrix(data5[, 2:5])
plot(cv.glmnet(X,Y, alpha=1))

fit.lasso.20=glmnet(X,Y, alpha=1, lambda=20)  # best lambda
coef(fit.lasso.20)
fit.lasso.0=glmnet(X,Y, alpha=1, lambda=0)    # glmnet when lambda=0
coef(fit.lasso.0)
fit.lm=lm(violentcrimes.perpop~num.kids.nvrmarried+pct.kids.nvrmarried+med.yr.house.built+pct.house.nophone, data5)
coef(fit.lm)                                  # OLS estimates

output=cbind(coef(fit.lasso.20), coef(fit.lasso.0), as.matrix(coef(fit.lm)))  # they all look fine
colnames(output) = c("glmnet.20", "glmnet.0", "OLS")
output







############### Need to get this cleaned #############################
######################################################################



### Appendix I: State effect?
### Will we get the same results by pulling all the states together. You may play with it
# I am skipping this.
# Question: what is major difference between this model
# and the one we run using FL alone?????
names(data2)
dim(data2)
data3=na.omit(data2) # sorry that I simply ignore the community with missing values now
dim(data3)

# Take a subset
data4=data3[, c("violentcrimes.perpop", 
                "pct.pop.underpov",
                "male.pct.divorce",
                "pct.kids2parents",
                "pct.youngkids2parents",
                "num.kids.nvrmarried",
                "pct.kids.nvrmarried",
                "pct.people.dense.hh",
                "med.yr.house.built",
                "pct.house.nophone",
                "pct.house.no.plumb",
                "state")]

#Is state an important variable? Does interactions between state and important factors exist?
fit1.state.ninteraction=lm(violentcrimes.perpop~., data4)  # including state but no interactions
summary(fit1.state.ninteraction)  # Rsq=.66

fit1.state.interaction=lm(violentcrimes.perpop~.+ .:state, data4) # adding interactions of state
      # vs. each predictor  #Rsq=.78
summary(fit1.state.interaction)

anova(fit1.state.ninteraction, fit1.state.interaction)

# Although we reject the null hypothesis that no interactions between
# state to other variables, I don't think we should include the interactions.
# Note: the above analysis is a quick one without much of the cares.


