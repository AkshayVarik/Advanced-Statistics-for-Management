
### Logistic Regression

# Topics:
#  1) Link function
#  2) Fit the data through ML estimation
#  3) Wald tests/intervals and Maximum Likelihood Ratio test
#  4) Model selection through backward selection
#  5) Appendix 1: log odd ratio as a function of SBP


# Case Study: Framingham Heart Study Data:
#  1) Identify Coronary Heart Disease risk factors
#   Famous study: http://www.framinghamheartstudy.org/risk-functions/
#  2) Predict Prob(HD=1) for a person who is
#         AGE    SEX  SBP DBP CHOL FRW CIG
#         45  FEMALE  100  80  180 110   5
#  3) Data: 1406 health professionals. Conditions gathered at the beginning 
#          of the study (early 50th). Both the original subjects and their next generations
#          have been included in the study. HD=0: No Heart Disease
  



######### Set up working directory###########
getwd()
rm(list=ls()) # Remove all the existing variables
#dir=c("/Users/lindazhao/Desktop/Dropbox/STAT471/Data") # school
dir=c("E:/Data Mining- STAT 571")   # my laptop
setwd("E:/Data Mining- STAT 571")
ls()
### Read Framingham.dat data

data = read.csv("Framingham.dat", sep=",", header=T, as.is=T)
str(data) 
names(data)

### Renames, setting the variables with correct natures...
names(data)[1]="HD"
data$HD=as.factor(data$HD)
data$SEX=as.factor(data$SEX)
str(data)
tail(data, 1)    # The last row is for prediction
data.new=data[1407,] # The female whose HD will be predicted.
data=data[-1407,]  # take out the last row 
summary(data) # HD: 311 of "0" and 1095 "1" 
# Notice missing values in FRW and CIG

sum(is.na(data)) # Total number of missing cells. 

#### For simplicity we start with a simple question:
#### How does HD relate to SBP?


# Plot SBP vs. HD and some summaries

tapply(data$SBP, data$HD, mean) # on average SBP seems to be higher among HD=1 

#Box plot: there are several ways to do so
plot(data$HD, data$SBP) # back to back box plots. SBP seems to be higher when HD=1
boxplot(data$SBP~data$HD) # another way of producing the boxplots

# Next we explore how proportions of HD=1 relate to SBP

# Standard plots:
plot(data$SBP, data$HD, col=data$HD)
legend("topright", legend=c("0", "1"), lty=c(1,1), lwd=c(2,2), col=unique(data$HD))
# Problems: many observations are overplotted so they are not visible. 
# Solutions: use jitter() to spread out the obs'n with similar SBP values

plot(jitter(as.numeric(data$HD), factor=.5) ~ data$SBP,
     pch=4,
     col=data$HD,
     ylab="HD",
     xlab="SBP") # still hard to tell
legend("topright", legend=c("0", "1"),
       lty=c(1,1), lwd=c(2,2), col=unique(data$HD))

# Alternatively we could use stripchart() as following:
stripchart(data$SBP~data$HD, method="jitter", 
           col=c("black", "red"), pch=4,
           ylab="HD", xlab="HB") 
legend("topright", legend=c("0", "1"),
       lty=c(1,1), lwd=c(2,2), col=unique(data$HD))
  
# Use plot(1:25, pch=1:25) to find out the symbols being used

##All the plots above do not show the proportion of "1"'s vs. "0"'s as a function of SBP.



##### Logistic Regression of HD vs. SBP

# We model P(Y=1|SBP)=exp(beta0+beta1 SBP)/(1+exp(beta0+beta1 SBP))
#  This is same as
#   - log (P(Y=1|SBP)/P(Y=0|SBP))=beta0+beta1 SBP
#   - We assume that log odds is monotonic as a func of SBP
# Maximum likelihood method is used to est. unknown parameters: beta0, beta1

#### glm() will be used to do logistic regression. It is very similar to lm() but some output might be different

fit1=glm(HD~SBP, data, family=binomial(logit))  #  the default is logit link
summary(fit1)

# Coefficients:
#                 Estimate Std. Error z value Pr(>|z|)    
#   (Intercept) -3.66342    0.34602 -10.587  < 2e-16 ***
#   SBP          0.01590    0.00221   7.196 6.22e-13 ***


### To see the prob function estimated by glm  -3.66+.0159 SBP
par(mfrow=c(1,1))
plot(data$SBP, fit1$fitted.values, pch=16, 
     xlab="SBP",
     ylab="Prob of P(Y=1|SBP")

### Alternatively, we can plot the prob through exp(-3.66+.0159 SBP)/(1+exp(-3.66+.0159 SBP))
x=seq(100, 300, by=1)
y=exp(-3.66+0.0159*x)/(1+exp(-3.66+0.0159*x))
plot(x,y, pch=16, lwd=1,
     xlab="SBP",
     ylab="Prob of P(Y=1|SBP)" )


#### Confidence intervals for the coefficients:

summary(fit1)
confint.default(fit1)   # usual z-intervals for the coeff's
confint(fit1)           # inverting the likelihood ratio tests. Both are similar in this case.

#### Likelihood Ratio Tests for the model

summary(fit1)  # deviance is -2log(lik)
chi.sq= 1485.9-1432.8     # get the Chi-square stat
pchisq(chi.sq, 1, lower.tail=FALSE)  # p-value: from the likelihood Ratio test

# or use anova/Anova{car}

anova(fit1, test="Chisq") # to test if the model is useful: null hypothesis is all (but the intercept) coeff's are 0

#       Df Deviance   Resid. Df Resid. Dev  Pr(>Chi)    
# NULL                  1405     1485.9              
# SBP   1    53.08      1404     1432.8 3.203e-13 ***

library(car)
Anova(fit1)   # Similar to the F-test in lm() set up.

# Response: HD
# LR Chisq Df Pr(>Chisq)    
# SBP    53.08  1  3.203e-13 ***


### What does a Chisq dis'n look like?
# - chisq_1=z^2
# - chisq_2=z1^2+z2^2, z1 and z2 are independently chosen z-scores
par(mfrow=c(2,1))
hist(rchisq(10000, 2), freq=FALSE, breaks=20) # chisq_2
hist(rchisq(10000, 20), freq=FALSE, breaks=20) # get familiar with a Chi-Squared disn
          # When DF is getting larger, Chi-Squared dis is approx. normal (why?)
par(mfrow=c(1,1 ))


### To do a prediction: 
fit1.predict=predict(fit1, data.new, type="response", interval="confidence", se.fit=T)
fit1.predict
names(fit1.predict)

### Multiple Logistic Regression. 
# To avoid problems, we will delete all cases with missing values
data=na.omit(data)

fit2=glm(HD~., data, family=binomial)   # with all the predictors
summary(fit2)

# Coefficients:
#             Estimate Std. Error z value Pr(>|z|)    
# (Intercept)   -9.334797   1.036630  -9.005  < 2e-16 ***
#   AGE          0.062491   0.014995   4.167 3.08e-05 ***
#   SEXMALE      0.906102   0.157639   5.748 9.03e-09 ***
#   SBP          0.014838   0.003886   3.818 0.000135 ***
#   DBP          0.002875   0.007620   0.377 0.705941    
#   CHOL         0.004459   0.001505   2.962 0.003053 ** 
#   FRW          0.005795   0.004055   1.429 0.152957    
#   CIG          0.012309   0.006087   2.022 0.043150 *  
# 
#   Null deviance: 1469.3  on 1392  degrees of freedom
# Residual deviance: 1343.1  on 1385  degrees of freedom

# AIC: 1359.1
# 
# Number of Fisher Scoring iterations: 4

#### Note:
# Residual Devianve=-2loglik
# AIC=-2loglik + 2 times d,  where d= total number of the parameters. In this case it is 8
# AIC is used to choose model. 

# To test if the model is useful, i.e. H0: all coeff's are 0

chi.sq=1469-1343
chi.sq  #126
pvalue=pchisq(chi.sq, 7, lower.tail=FALSE)
pvalue

# Or we use anova to get the Chi-sq test
fit0=glm(HD~1, data, family=binomial) # get the null statistics, i.e., set all coeff's =0
summary(fit0)

anova(fit0, fit2, test="Chisq") #It works only if two fits use same samples. 


### Use Likelihood Ratio test to see if DBP and FRW are not needed in fit2

fit4=update(fit2, .~. -CIG -FRW)
summary(fit4)
chi.sq.2=fit4$deviance-fit2$deviance
pchisq(chi.sq.2, 2, lower.tail=FALSE) # 0.06194729

### Or use anova... 
anova(fit4, fit2, test="Chisq")    # Something is wrong here!!!!!!!!!

# Model 1: HD ~ AGE + SEX + SBP + DBP + CHOL
# Model 2: HD ~ AGE + SEX + SBP + DBP + CHOL + FRW + CIG
# Resid. Df Resid. Dev Df Deviance Pr(>Chi)  
# 1      1387     1348.7                       
# 2      1385     1343.1  2   5.5629  0.06195 .


##### Model selection: Backward selection is the easiest

summary(fit2)
fit3=update(fit2, .~. -DBP)# Backward selection by kicking DBP out
summary(fit3)

fit4=update(fit3, .~. -FRW)
summary(fit4)

fit5=update(fit4, .~. -CIG)
summary(fit5)    # CIG didn't really add much to decrease the res. dev

### prediction for the subject
fit4.predict=predict(fit4, data.new, type="response")
fit4.predict    # Notice the difference between the predictions among fit4 and fit1!!!




#### Appendix 1: Show that logistic regression model of HD vs SBP is a 
# good model

# Let's find the proportion of "1" / "0" when group people by SBP
attach(data)
hist(data$SBP)  
bk=quantile(SBP, seq(from=0, to=1, by=.10))
#bk=quantile(SBP, c(0, .25, .5, .75, 1))
SBPN=cut(SBP, bk)  # Divide SBP to a categorical variable 

total=table(HD, SBPN)
total
HD.SBP=prop.table(total, 2)
plot(x=c(90, 300), y=c(-2.5, 1), type="n")
plot(bk[-11], log(HD.SBP[2, ] / HD.SBP[1, ]), pch=16,
     xlim=c(80, 200),
     xlab="SBP",
     ylab="log (p(HD=1)/p(HD=0))")



