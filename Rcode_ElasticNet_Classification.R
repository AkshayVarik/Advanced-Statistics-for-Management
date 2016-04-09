

#### Elastic net extended to classifications
# Case study: ART effects.

# 1) Data exploration
# 2) Elastic net in classification
# 3) Final findings

# An almost same analysis is prepared in a .rmd file. You will find explanations there, especially
# about elastic net in classifications.


# Case study: ART effects 
# Garci Garza, a former student,  did a project "HIV Progression". The goal of the study
# is to identify important factors that relate to patient improvement after AntiRetroviral Therapy (ART). 
# In particular we would like to identify the relevant positions of protease (PR) nucleotide 
# sequences and/or reverse transcriptase (RT) nucleotide sequence that may affect the improvement.
# Among 1000 HIV+ patients, two control variables, a viral load (number of HIV virus count in log 10 scale.)
# and CD4 counts (healthy people range from 500-1200 cells/mm^3) are available.
# The response variable is a binary status: 1=improved, 0=no improvement
# Note: y=1 implies that VL is dropped by at least 100 times. In log 10 scale means it is reduced by at least 2.

# Data: HIV_Seq_Clean.csv
# Garci spent a huge amount of time to study the subject and to clean the data.
# We will use his cleaned data and only use information about PR. This has been converted
# to a set of 100 or so categorical variables.  

#### Libraries needed
library(leaps)
library(pROC)
library(glmnet)
library(MASS)
library(dplyr)
library(car)



### 1) ART data exploration

rm(list=ls()) # Remove all the existing variables
#setwd(dir)
data=read.csv("HIV_Seq_Clean.csv", header=T)
dim(data)  # 920, some people got dropped due to missing values. 79 predictors, mount to p=230 var's
names(data)
str(data)
sum(is.na(data))   #no missing value any more

levels(as.factor(data$y))
# 1= improvement, 0= little/no improvement.
data$y=as.factor(data$y)  # set y as a binary variable.

### Data summary/exploration: see details in Garcia's report. 
summary(data)

# VLoad. There is no notion of normal load. People who are not infected have no viral load at all.
hist(data$VL.t0, xlab = "log 10 VLoad", col="red")
# we see the VL is high in this sample in general.

# CD4. 
summary(data$CD4.t0)
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 0.0   124.5   243.0   273.0   379.0  1589.0 

hist(data$CD4.t0, breaks=30, col="blue")
boxplot(data$CD4.t0)
mean(data$CD4.t0 < 200) #[1] 0.4195652
mean(data$CD4.t0 < 500) #[1] 0.8706522

# 42% of people with CD4 < 200 and 87% less than the normal lower bound. 

# Association between VL and CD4. 
plot(data$CD4.t0, data$VL.t0, 
     col=data$y,
     pch=as.numeric(data$y))
legend("bottomright", legend=c("0", "1"),
       lty=c(1,1), lwd=c(2,2), col=unique(data$y))

cor(data$CD4.t0, data$VL.t0)
# Not surprising at all that the two measurements are negatively correlated.
# It seems that the improvement mainly appear among the ones with high VL.t0.



# We move to exam PR's
summary(data)
# Notice many variables: PR2, PR3, PR5, PR6, PR7, PR8 ... There is no variability
# We should really take them out of our analysis!


### 2) Elastic net for classifications

# Goal:  Try to find important variables using elastic net

# Step I:  Get design matrix and response
X=model.matrix(y~., data)[,-1]   # a categorical variable is coded with indicator functions
dim(X)
Y=data[, 80]


# Step II: Select a model with a few important variables through elasticnet

# 10 fold cv to min misclassification errors

set.seed(10)   # for cross validation
#fit1.cv.class=cv.glmnet(X,Y, alpha=1, family="binomial", type.measure="class")
fit1.cv=cv.glmnet(X, Y, alpha=1, family="binomial", nfolds = 10)  

# type.measure by default is "deviance" for binomial. But it can also be set as "class", or "auc"

plot(fit1.cv)

# A quick review of the output:
names(fit1.cv); fit1.cv$name
data.frame(fit1.cv$lambda, fit1.cv$nzero)[1:30, ]
# Notice that there is a jump in nzero.

# We could chose lambda=fit1.cv$lambda.1se, lambda.min or any values of the lambda between the 
# two values.

# lambda=fit1.cv$lambda.1se
fit1.1se=glmnet(X, Y, alpha=1, family="binomial", lambda=fit1.cv$lambda.1se)
fit1.1se.beta=coef(fit1.1se)
beta=fit1.1se.beta[which(fit1.1se.beta !=0),] # non zero beta's
beta=as.matrix(beta); beta
rownames(beta)

# lambda=fit1.cv$lambda[fit1.cv$nzero==11], and repeat the above lines.
fit.nzero=glmnet(X, Y, alpha=1, family="binomial", lambda=fit1.cv$lambda[fit1.cv$nzero==11])
fit.nzero.beta=coef(fit.nzero)
beta=fit.nzero.beta[which(fit.nzero.beta !=0),] # non zero beta's
beta=as.matrix(beta); beta
rownames(beta)  

beta  # Logit equation obtained with type.maeasure=deviance, nzero=11, set.seed(10)

# (Intercept) -7.37247060
# PR10I       -0.63598341
# PR10L        0.46674647
# PR12R        0.94675562
# PR20K        0.04907370
# PR35N        0.27908839
# PR63L        0.07617845
# PR63P       -0.05406314
# PR72T        0.01291300
# PR82V        0.59579018
# PR92K        0.83617336
# VL.t0        1.16020091

# We could use the above logit function to estimate the Prob(y=1). But we don't have 
# varibility estimates here. 

##### I have to confess that I can't reproduce exactly what I did once that PR10, 12, 63, 82 and 92 were
# retained. I see if we use "class" as the criterion and use fit.1se we get those PR's



# Step III: fit the final model
# Since we are trying to locate a set of important features, I chose to use 
# glm() 

fit.logit.1=glm(y~PR10+PR12+PR63+PR82+PR92+VL.t0+CD4.t0,
              family=binomial, data=data)
summary(fit.logit.1)
library(car)
Anova(fit.logit.1)

# dropping CD4
fit.logit.2=glm(y~PR10+PR12+PR63+PR82+PR92+VL.t0,
                family=binomial, data=data)
Anova(fit.logit.2)

# see if we could drop both PR12 and PR63

anova(glm(y~PR10+PR82+PR92+VL.t0,
          family=binomial, data=data),
      glm(y~PR10+PR12+PR63+PR82+PR92+VL.t0,
          family=binomial, data=data),
      test="Chisq")

# Analysis of Deviance Table
# 
# Model 1: y ~ PR10 + PR82 + PR92 + VL.t0
# Model 2: y ~ PR10 + PR12 + PR63 + PR82 + PR92 + VL.t0
# Resid. Df Resid. Dev Df Deviance Pr(>Chi)
# 1       902     702.34                     
# 2       882     675.30 20   27.038   0.1342

# We have no evidence to reject the null that PR12 and PR63 play no roles. 


fit.logit.3=glm(y~PR10+PR82+PR92+VL.t0,
                family=binomial, data=data)
Anova(fit.logit.3)

# Analysis of Deviance Table (Type II tests)
# Response: y
# LR Chisq Df Pr(>Chisq)    
#   PR10    45.255  4  3.519e-09 ***
#   PR82    24.437  8   0.001935 ** 
#   PR92    15.806  4   0.003290 ** 
#   VL.t0  131.626  1  < 2.2e-16 ***

summary(fit.logit.3)

### 3) Findings

#If you trust our analyses we will then concentrate on PR10, PR82 and PR92. I will give
# a quick exploration and see what we can say those three positions. 


summary(fit.logit.3)

# Coefficients:
#   Estimate Std. Error z value Pr(>|z|)    
# (Intercept) -2.512e+01  1.022e+03  -0.025 0.980385    
# PR10I       -2.506e+00  6.626e-01  -3.781 0.000156 ***
# PR10L        2.316e-03  4.526e-01   0.005 0.995917    
# PR10R       -1.081e+00  1.366e+00  -0.791 0.428891    
# PR10V       -1.107e+00  6.703e-01  -1.652 0.098587 .  
# PR82C       -1.451e+01  2.400e+03  -0.006 0.995175    
# PR82F        3.006e+00  1.031e+00   2.915 0.003557 ** 
# PR82I       -1.377e+01  7.350e+02  -0.019 0.985049    
# PR82L       -1.298e+01  2.400e+03  -0.005 0.995684    
# PR82M       -1.416e+01  2.400e+03  -0.006 0.995290    
# PR82S       -1.001e+01  2.400e+03  -0.004 0.996671    
# PR82T        8.952e-01  1.210e+00   0.740 0.459286    
# PR82V        1.555e+00  4.423e-01   3.516 0.000438 ***
# PR92H        1.863e+01  1.022e+03   0.018 0.985449    
# PR92K        1.778e+01  1.022e+03   0.017 0.986114    
# PR92Q        1.571e+01  1.022e+03   0.015 0.987726    
# PR92R        1.540e+00  1.324e+03   0.001 0.999072    
# VL.t0        1.530e+00  1.494e-01  10.235  < 2e-16 ***


# 0) People with a high VL tend to be beneficial from the ART.

# i) PR10
table(data$PR10)
#  F   I   L   R   V 
# 49 162 647   7  55 

# There is a good amount of variabilities among the sample. 
# The chance of improving with PR10F seems to be higher. 

# ii) PR82
table(data$PR82)
#   A   C   F   I   L   M   S   T   V 
# 144   1  11   8   1   1   1  19 734 

# PR82: V, F are better than A

# iii) PR92

# None of the PR92 categories show difference to the base, how come we reject the
# null that PR92 is a significant variable?

table(data$PR92)
# E   H   K   Q   R 
# 4   2  14 894   6 

# PR92_E is the base. With only 4 observations in that category, it's hard to detect difference between
# "E" and any other level.

# If you switch "Q" as the base for example, you will see some small p-values.

data$PR92= relevel(data$PR92, "Q")  # set "Q" as the base
table(data$PR92)

summary(glm(y~PR10+PR82+PR92+VL.t0, family=binomial, data=data))

# Given most of the people have PR92Q, I won't take this discovery too seriously.



##########END######################

### Remarks
# 1) We may try to use other lambda's and we could have landed a different final model.
# 2) We could try to start with the following subset which only includes PR's with some 
# variabilities. My guess is that it will end up with same sets of PR's in the final model. 

summary(data)
var.interesting=c("PR10", "PR12", "PR13", "PR19", "PR20","PR24", "PR30", "PR32", "PR33",
                  "PR35", "PR36", "PR37", "PR41" ,"PR43"  ,"PR46" ,"PR53" ,"PR54" ,"PR55" 
                  ,"PR57" ,"PR58" ,"PR60" ,"PR61"    ,"PR62","PR63" ,"PR64"   ,"PR69"    
                  ,"PR70"    ,"PR71" ,"PR72"     ,"PR73"    ,"PR74" ,"PR77"  ,"PR82" ,"PR84"
                  ,"PR85" ,"PR88" ,"PR89" ,"PR90","PR92","PR93" ,"VL.t0","CD4.t0" ,"y")

data1=data[, var.interesting]
summary(data1)


#### Appendix 1: Comparison between LASSO and Logistic fits.

# As we know that LASSO shrinks coefficients towards 0. We expect the estimates from LASSO and logistic
# differ. How does this affect the prediction. Let's do a comparison.
# I am able to output in sample errors for both LASSO and glm. But only report cv errors for LASSO. 

fit.lasso=cv.glmnet(X,Y, alpha=1, family="binomial", type.measure="class")  # choose lambda by min mis-classification
  plot(fit.lasso)
  cv.error.lasso=fit.lasso$cvm[fit.lasso$lambda==fit.lasso$lambda.1se]  # output min cv errors.
  cv.error.lasso # this is a good estimate of testing miss-classification error, about .19
  
  
  #lasso.prob=predict(fit.lasso, X,  s="lambda.1se", type="response")  # Prob(y=1)
  lasso.predict=predict(fit.lasso, X,  s="lambda.1se", type="class")  # classification with .5 as the threshold
  lasso.error=mean(data$y != lasso.predict)
  lasso.error
  
  
  #### Output the non-zero variables
  
  beta.lasso=coef(fit.lasso, s="lambda.1se")   # output lasso estimates
  # beta.lasso=coef(fit.lasso, s=exp(-4.8))  #another way to do it
  beta=beta.lasso[which(beta.lasso !=0),] # non zero beta's
  beta=as.matrix(beta);
  beta=rownames(beta)  
  beta=substr(beta, 1, 4)
  beta1=levels(as.factor(beta))[-1]
 
  
  glm.input=as.formula(paste("y", "~", paste(beta1[-length(beta1)],collapse = "+"), "+VL.t0"))
  glm.input
  

  fit.glm=glm(glm.input, family="binomial", data=data)
  summary(fit.glm)
  
  glm.predict=factor(ifelse(fit.glm$fitted > .5, "1", "0") )
  glm.error=mean(data$y != glm.predict)
  glm.error
  
  
  #### Puting three errors.
  
  print(c(glm.error, lasso.error, cv.error.lasso ))
  
  
  
  
  
  ##### This doesn't work due to lack of levels in the formula. Get it back.
  fit.glm=glm(y~PR82, data, family = "binomial")
  cost <- function(r, pi = 0) mean(abs(r-pi) > 0.5)
  cv.glm(data, fit.glm, cost, K=6)$delta
  #####
  